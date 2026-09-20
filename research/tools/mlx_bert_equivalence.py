"""Convert a BERT encoder prefix separately, then validate with a torch-free MLX process."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def official_segments(args):
    segments = {}
    if args.frontend_json is not None:
        capture = json.loads(args.frontend_json.read_text())
        if capture["backend"] != "official":
            raise ValueError("Expected an official frontend capture")
        for case in capture["cases"]:
            for index, segment in enumerate(case["cleaned_segments"]):
                if segment["language"] == "zh":
                    segments[f"{case['key']}-{index}"] = dict(
                        segment, source=str(args.frontend_json), source_sha256=sha256(args.frontend_json))
    for trace_path in args.trace:
        trace = json.loads(trace_path.read_text())
        if trace["backend"] != "official":
            raise ValueError("Expected an official trace")
        for event in trace["events"]:
            if event["stage"] != "text.clean_text_inf" or event["args"][1].replace("all_", "") != "zh":
                continue
            phones, word2ph, normalized = event["result"]
            segments[f"{trace_path.stem}-segment-{event['index']}"] = {
                "normalized": normalized, "word2ph": word2ph, "phones": phones,
                "input": event["args"][0], "language": "zh", "source": str(trace_path),
                "source_sha256": sha256(trace_path), "source_event": event["index"],
            }
    unique, seen = {}, set()
    for name, segment in segments.items():
        identity = (segment["normalized"], tuple(segment["word2ph"]))
        if identity not in seen:
            unique[name] = segment
            seen.add(identity)
    if not unique:
        raise ValueError("No captured official Chinese input segments found")
    return unique


def prepare(args):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertForMaskedLM
    from sakuratts.frontend.bert_features import BertFeatures

    torch.set_num_threads(2)
    run = args.references.resolve() / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-mlx-bert-cpu")
    run.mkdir(parents=True)
    print(f"RUN_DIRECTORY={run}", flush=True)
    source = args.source_model
    provenance = {}
    if source is None:
        torch.manual_seed(30919)
        config = BertConfig(vocab_size=97, hidden_size=32, num_hidden_layers=6, num_attention_heads=4,
                            intermediate_size=48, max_position_embeddings=64, layer_norm_eps=1e-12,
                            hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        original = BertForMaskedLM(config).eval()
        source = run / "synthetic-source"
        original.save_pretrained(source)
        inputs = {
            "single": {"input_ids": np.array([[2, 7, 11, 3]], dtype=np.int64)},
            "padding_and_types": {
                "input_ids": np.array([[2, 7, 11, 3, 0, 0], [2, 8, 13, 18, 22, 3]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=np.int64),
                "token_type_ids": np.array([[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=np.int64),
            },
            "explicit_positions": {
                "input_ids": np.array([[2, 11, 13, 3]], dtype=np.int64),
                "position_ids": np.array([[4, 5, 6, 7]], dtype=np.int64),
            },
        }
    else:
        provenance = official_segments(args)
        source = source.resolve()
        original = AutoModelForMaskedLM.from_pretrained(source, torch_dtype=torch.float32, local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        inputs = {name: dict(tokenizer(segment["normalized"], return_tensors="np"))
                  for name, segment in provenance.items()}
        write_json(run / "input-provenance.json", provenance)
    package = run / "package"
    package.mkdir()
    cfg = original.config
    report = {"status": "prepared", "source_kind": "real" if args.source_model else "synthetic",
              "command": [sys.executable, *sys.argv], "package": str(package), "cases": [],
              "validation_tolerance": {"rtol": 1e-4, "atol": 1e-5},
              "torch_threads": torch.get_num_threads(), "model_lifecycle": "full model gold, release, then pruned model validation/export",
              "scope": "FP32 encoder-prefix conversion and features; not end-to-end audio or tokenizer replacement"}
    gelu_inputs = torch.tensor([-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0], dtype=torch.float32)
    np.savez(run / "gelu-official.npz", inputs=gelu_inputs.numpy(),
             output=torch.nn.functional.gelu(gelu_inputs, approximate="none").numpy())
    tokenizer_dir = args.references.resolve() / "models/shared/chinese-roberta-wwm-ext-large"
    if (tokenizer_dir / "tokenizer.json").is_file():
        from tokenizers import Tokenizer
        auto = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        standalone = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        texts = json.loads((PROJECT / "benchmarks/cases/speech_regressions.json").read_text())["cases"]
        texts.extend({"id": name, "text": segment["normalized"]} for name, segment in provenance.items())
        tokenizer_audit = {"purpose": "tokenizer JSON loading only; raw corpus, no G2P or language support claim",
                           "tokenizer_sha256": sha256(tokenizer_dir / "tokenizer.json"),
                           "tokenizers": metadata.version("tokenizers"), "cases": []}
        for case in texts:
            expected_tokens = dict(auto(case["text"]))
            encoded = standalone.encode(case["text"], add_special_tokens=True)
            actual_tokens = {"input_ids": encoded.ids, "token_type_ids": encoded.type_ids,
                             "attention_mask": encoded.attention_mask}
            tokenizer_audit["cases"].append({"id": case["id"], "text": case["text"],
                                              "auto": expected_tokens, "standalone": actual_tokens,
                                              "exact_equal": expected_tokens == actual_tokens})
        write_json(run / "tokenizer-audit.json", tokenizer_audit)
        report["tokenizer_json_matches_auto"] = all(case["exact_equal"] for case in tokenizer_audit["cases"])
    with torch.inference_mode():
        for name, arrays in inputs.items():
            torch_inputs = {key: torch.from_numpy(value) for key, value in arrays.items()}
            full = original(**torch_inputs, output_hidden_states=True)
            expected = full.hidden_states[-3]
            gold = {f"layer_{i}": tensor.numpy() for i, tensor in enumerate(full.hidden_states[:cfg.num_hidden_layers - 1])}
            case = {"id": name, "shape": list(expected.shape)}
            if name in provenance:
                word2ph = provenance[name]["word2ph"]
                characters = expected[0].cpu()[1:-1]
                if len(characters) != len(word2ph) or sum(word2ph) != len(provenance[name]["phones"]):
                    raise ValueError("Captured official word2ph does not align with tokenizer or phonemes")
                gold["phones"] = torch.cat([characters[i].repeat(count, 1) for i, count in enumerate(word2ph)]).T.numpy()
                case.update(word2ph=word2ph, normalized=provenance[name]["normalized"], phone_shape=list(gold["phones"].shape))
            np.savez(run / f"{name}-inputs.npz", **arrays)
            np.savez(run / f"{name}-official.npz", **gold)
            report["cases"].append(case)
            del full, expected, gold
    del original
    gc.collect()
    print("PHASE=full_model_released_loading_pruned_model", flush=True)
    model = BertFeatures.from_pretrained(source)
    if len(model.bert.encoder.layer) != cfg.num_hidden_layers - 2:
        raise ValueError("Retained layers do not select original hidden_states[-3]")
    with torch.inference_mode():
        for case in report["cases"]:
            arrays = inputs[case["id"]]
            pruned = model(**{key: torch.from_numpy(value) for key, value in arrays.items()}).numpy()
            with np.load(run / f"{case['id']}-official.npz", allow_pickle=False) as gold:
                case["pruned_torch_exact"] = bool(np.array_equal(pruned, gold[f"layer_{cfg.num_hidden_layers - 2}"]))
            if not case["pruned_torch_exact"]:
                raise AssertionError("Existing pruned PyTorch features no longer exactly match the full model")
            del pruned
    weights = {name: value.detach().cpu().numpy() for name, value in model.bert.state_dict().items()}
    np.savez(package / "weights.npz", **weights)
    config = {key: getattr(cfg, key) for key in ("hidden_size", "num_attention_heads", "intermediate_size",
              "max_position_embeddings", "type_vocab_size", "vocab_size", "layer_norm_eps", "hidden_act", "position_embedding_type")}
    config.update(source_num_hidden_layers=cfg.num_hidden_layers, retained_layers=len(model.bert.encoder.layer))
    manifest = {
        "format": "sakuratts-bert-features-fp32-v1", "config": config,
        "source": str(source), "source_files": [{"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                                                     for p in sorted(source.iterdir()) if p.is_file()],
        "feature_layer": "original hidden_states[-3]; no CLS/SEP removal or word2ph expansion inside model",
        "weights": {"file": "weights.npz", "sha256": sha256(package / "weights.npz"),
                    "bytes": (package / "weights.npz").stat().st_size,
                    "parameters": sum(value.size for value in weights.values()), "tensors": len(weights)},
        "conversion_python": sys.version, "torch": torch.__version__, "transformers": metadata.version("transformers"),
    }
    write_json(package / "manifest.json", manifest)
    for source_file in (Path(__file__), PROJECT / "src/sakuratts/backends/mlx/bert.py", PROJECT / "src/sakuratts/frontend/bert_features.py", PROJECT / "src/sakuratts/_internal/weight_storage.py"):
        shutil.copy2(source_file, run / source_file.name)
    write_json(run / "prepared.json", report)
    print(f"PREPARED={run}", flush=True)


def validate(args):
    import mlx.core as mx
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    from sakuratts.backends.mlx.bert import MLXBertFeatures

    run = args.run.resolve()
    destination = run / f"mlx-{args.device}-result.json"
    if destination.exists() or list(run.glob(f"*-mlx-{args.device}.npz")):
        raise FileExistsError("Validation artifacts already exist; prepare a new run to preserve them")
    shutil.copy2(Path(__file__), run / f"validation-{args.device}-research.tools.py")
    shutil.copy2(PROJECT / "src/sakuratts/backends/mlx/bert.py", run / f"validation-{args.device}-runtime.py")
    shutil.copy2(PROJECT / "src/sakuratts/_internal/weight_storage.py", run / f"validation-{args.device}-weight_storage.py")
    prepared = json.loads((run / "prepared.json").read_text())
    model = MLXBertFeatures.load(run / "package")
    report = {"status": "running", "device": str(mx.default_device()), "mlx": metadata.version("mlx"),
              "numpy": np.__version__, "python": sys.version, "command": [sys.executable, *sys.argv],
              "runtime_imported_torch": "torch" in sys.modules,
              "runtime_imported_transformers": "transformers" in sys.modules,
              "tolerance": prepared["validation_tolerance"], "cases": []}
    # Check exact-GELU primitives on the selected backend before model comparison.
    with np.load(run / "gelu-official.npz", allow_pickle=False) as gold:
        gelu_input = mx.array(gold["inputs"])
        gelu_expected = gold["output"].copy()
    gelu = gelu_input * 0.5 * (1 + mx.erf(gelu_input * (2 ** -0.5)))
    mx.eval(gelu)
    report["erf_gelu_values"] = np.array(gelu).tolist()
    report["erf_gelu_max_abs_error"] = float(np.max(np.abs(np.array(gelu) - gelu_expected)))
    report["erf_gelu_passed"] = bool(np.allclose(np.array(gelu), gelu_expected, rtol=1e-6, atol=1e-6))
    for case in prepared["cases"]:
        name = case["id"]
        with np.load(run / f"{name}-inputs.npz", allow_pickle=False) as data:
            inputs = {key: data[key].copy() for key in data.files}
        started = time.perf_counter()
        output, states = model(**inputs, return_intermediates=True)
        mx.eval(*states)
        elapsed = time.perf_counter() - started
        actual = {f"layer_{index}": np.array(value) for index, value in enumerate(states)}
        if "word2ph" in case:
            actual["phones"] = np.repeat(np.array(output)[0, 1:-1], case["word2ph"], axis=0).T
        np.savez(run / f"{name}-mlx-{args.device}.npz", **actual)
        checks = []
        with np.load(run / f"{name}-official.npz", allow_pickle=False) as gold:
            for key, value in actual.items():
                expected = gold[key]
                error = value.astype(np.float64) - expected.astype(np.float64)
                checks.append({"layer": key, "shape": list(value.shape),
                               "max_abs_error": float(np.max(np.abs(error))),
                               "rmse": float(np.sqrt(np.mean(error ** 2))),
                               "passed": bool(np.allclose(value, expected, **report["tolerance"]))})
        report["cases"].append({"id": name, "seconds_including_diagnostics": elapsed, "layers": checks})
    passed = report["erf_gelu_passed"] and all(row["passed"] for case in report["cases"] for row in case["layers"])
    report["status"] = "completed" if passed else "numerical_mismatch"
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "completed":
        raise AssertionError("MLX BERT feature comparison exceeded the preset tolerance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    converter = subparsers.add_parser("prepare")
    converter.add_argument("--references", type=Path, required=True)
    converter.add_argument("--source-model", type=Path)
    converter.add_argument("--frontend-json", type=Path)
    converter.add_argument("--trace", type=Path, action="append", default=[])
    validator = subparsers.add_parser("validate")
    validator.add_argument("--run", type=Path, required=True)
    validator.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else validate(args)


if __name__ == "__main__":
    main()
