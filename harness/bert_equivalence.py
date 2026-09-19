"""Compare full masked-LM and dependency-pruned Chinese BERT on saved inputs.

Uses normalized text and word2ph from the official frontend audit verbatim.
Timing excludes capture, phone expansion, serialization and tokenizer work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from sakuratts.bert_features import BertFeatures


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memory(device):
    result = {}
    if platform.system() == "Darwin":
        import resource
        result["process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if device == "mps":
        result["mps_allocated_bytes_at_boundary"] = torch.mps.current_allocated_memory()
        result["mps_driver_bytes_at_boundary"] = torch.mps.driver_allocated_memory()
    return result


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def phone_features(hidden, word2ph):
    # Match TextPreprocessor.get_bert_feature: move to CPU, strip CLS/SEP,
    # then repeat each character by the unchanged official word2ph list.
    characters = hidden[0].cpu()[1:-1]
    if len(characters) != len(word2ph):
        raise ValueError("Official normalized text and tokenizer alignment differ")
    return torch.cat([characters[i].repeat(count, 1) for i, count in enumerate(word2ph)]).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--frontend-json", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"), default=["cpu"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--load-note", default="Other machine load was not controlled")
    parser.add_argument("--export", action="store_true", help="Save pruned FP32 weights in this run directory")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("repeat and threads must be positive; warmup must be nonnegative")
    root = args.references.resolve()
    frontend_path = args.frontend_json.resolve()
    source = json.loads(frontend_path.read_text(encoding="utf-8"))
    if source["backend"] != "official":
        raise ValueError("Expected normalized inputs captured from the official frontend")
    output = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-bert-equivalence")
    output.mkdir(parents=True)
    print(f"RUN_DIRECTORY={output}", flush=True)
    model_path = root / "models/shared/chinese-roberta-wwm-ext-large"
    report = {
        "status": "running", "purpose": "bert_dependency_pruning_not_audio_quality_acceptance",
        "command": [sys.executable, *sys.argv], "dtype": "float32", "torch": torch.__version__,
        "platform": platform.platform(), "threads": args.threads,
        "load_note": args.load_note,
        "frontend_source": str(frontend_path), "frontend_sha256": sha256(frontend_path),
        "official_commit": source["source_commit"],
        "timing_scope": "pretokenized BERT forward; synchronize before and after each timed call; no capture or phone expansion",
        "memory_scope": "MPS execution boundaries, not peak; unified memory, not NVIDIA VRAM; RSS is process lifetime high-water mark",
        "variants": [],
    }
    shutil.copy2(frontend_path, output / "official-frontend.json")
    shutil.copy2(__file__, output / "bert_equivalence.py")
    module_path = Path(__file__).resolve().parents[1] / "src/sakuratts/bert_features.py"
    shutil.copy2(module_path, output / "bert_features.py")
    report["harness_sha256"] = sha256(Path(__file__))
    report["module_sha256"] = sha256(module_path)
    report["source_files"] = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                              for p in sorted(model_path.iterdir()) if p.is_file()]
    report["source_directory_file_bytes"] = sum(p["bytes"] for p in report["source_files"])
    write_json(output / "result.json", report)
    try:
        torch.set_num_threads(args.threads)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        cases = []
        seen = set()
        for case in source["cases"]:
            for index, segment in enumerate(case["cleaned_segments"]):
                if segment["language"] != "zh":
                    continue
                identity = (segment["normalized"], tuple(segment["word2ph"]))
                if identity in seen:
                    continue
                seen.add(identity)
                inputs = tokenizer(segment["normalized"], return_tensors="pt")
                cases.append({"id": f"{case['key']}-{index}", "normalized": segment["normalized"],
                              "word2ph": segment["word2ph"], "inputs": inputs})
        if not cases:
            raise ValueError("No Chinese segments in the official frontend capture")
        write_json(output / "inputs.json", [dict(c, inputs={k: v.tolist() for k, v in c["inputs"].items()}) for c in cases])
        for device in args.devices:
            if device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is unavailable")
            for variant in ("original", "pruned"):
                item = {"device": device, "variant": variant, "memory_before_load": memory(device), "cases": []}
                report["variants"].append(item)
                started = time.perf_counter()
                if variant == "original":
                    model = AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=torch.float32, local_files_only=True).eval()
                    original_layers = model.config.num_hidden_layers
                    retained_layers = original_layers
                else:
                    model = BertFeatures.from_pretrained(model_path)
                    original_layers = model.source_num_hidden_layers
                    retained_layers = model.bert.config.num_hidden_layers
                item["original_layers"] = original_layers
                item["retained_layers"] = retained_layers
                item["attention_implementation"] = model.bert.config._attn_implementation
                item["parameter_count"] = sum(p.numel() for p in model.parameters())
                item["parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
                item["state_dict_payload_bytes_including_tied_aliases"] = sum(
                    p.numel() * p.element_size() for p in model.state_dict().values())
                if variant == "pruned" and args.export and not (output / "pruned-fp32").exists():
                    model.bert.save_pretrained(output / "pruned-fp32")
                    item["export_files"] = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                                            for p in sorted((output / "pruned-fp32").iterdir()) if p.is_file()]
                    item["export_note"] = "FP32 export; source checkpoint storage precision may differ, so raw file ratio is not a same-precision saving"
                model.to(device)
                synchronize(device)
                item["load_seconds_including_optional_export"] = time.perf_counter() - started
                item["memory_after_load"] = memory(device)
                def forward(inputs):
                    if variant == "original":
                        return model(**inputs, output_hidden_states=True).hidden_states[-3]
                    return model(**inputs)
                with torch.inference_mode():
                    for case in cases:
                        inputs = {k: v.to(device) for k, v in case["inputs"].items()}
                        synchronize(device)
                        hidden = forward(inputs)
                        synchronize(device)
                        prefix = f"{device}-{variant}-{case['id']}"
                        np.save(output / f"{prefix}-hidden.npy", hidden.cpu().numpy())
                        phones = phone_features(hidden, case["word2ph"])
                        np.save(output / f"{prefix}-phones.npy", phones.numpy())
                        entry = {"id": case["id"], "hidden_shape": list(hidden.shape),
                                 "phone_shape": list(phones.shape), "memory_after_capture": memory(device)}
                        del hidden, phones
                        for _ in range(args.warmup):
                            temporary = forward(inputs)
                            del temporary
                        synchronize(device)
                        elapsed = []
                        for _ in range(args.repeat):
                            synchronize(device)
                            started = time.perf_counter()
                            temporary = forward(inputs)
                            synchronize(device)
                            elapsed.append(time.perf_counter() - started)
                            del temporary
                        entry.update(seconds=elapsed, warmup=args.warmup, median_seconds=statistics.median(elapsed),
                                     memory_after_timing=memory(device))
                        item["cases"].append(entry)
                        del inputs
                del model
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
                synchronize(device)
                item["memory_after_unload_and_empty_cache"] = memory(device)
                write_json(output / "result.json", report)
                print(json.dumps(item, ensure_ascii=False), flush=True)
        comparisons = []
        for device in args.devices:
            for case in cases:
                for kind in ("hidden", "phones"):
                    original = np.load(output / f"{device}-original-{case['id']}-{kind}.npy")
                    pruned = np.load(output / f"{device}-pruned-{case['id']}-{kind}.npy")
                    comparisons.append({"device": device, "id": case["id"], "kind": kind,
                                        "exact_equal": bool(np.array_equal(original, pruned)),
                                        "max_abs_error": float(np.max(np.abs(original - pruned))),
                                        "mean_abs_error": float(np.mean(np.abs(original - pruned)))})
        report["comparisons"] = comparisons
        report["status"] = "completed" if all(c["exact_equal"] for c in comparisons) else "numerical_difference"
        write_json(output / "result.json", report)
        print(f"COMPLETED={output} status={report['status']}", flush=True)
        if report["status"] != "completed":
            raise RuntimeError("Dependency pruning changed the selected BERT features")
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        write_json(output / "result.json", report)
        raise


if __name__ == "__main__":
    main()
