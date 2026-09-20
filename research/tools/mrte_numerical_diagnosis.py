#!/usr/bin/env python3
"""Trace MRTE on fixed official/native encoder outputs without changing inputs.

Official runs call the pinned upstream module with hooks. MLX runs use its
existing conv/attention operators and check the uninstrumented multihead path.
NumPy FP64 is an independent matrix/softmax oracle. No runtime is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sovits_fixed_conditions import load_acoustic, sha256, source_inventory


def compare(actual, expected):
    if actual.shape != expected.shape:
        return {"within_tolerance": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    actual64, expected64 = actual.astype(np.float64), expected.astype(np.float64)
    difference = actual64 - expected64
    limit = 1e-4 + 1e-5 * np.abs(expected64)
    index = np.unravel_index(np.argmax(np.abs(difference) - limit), difference.shape)
    return {"shape": list(actual.shape), "atol": 1e-4, "rtol": 1e-5,
            "all_finite": bool(np.isfinite(actual).all()), "array_equal": bool(np.array_equal(actual, expected)),
            "max_abs": float(np.max(np.abs(difference))), "rms": float(np.sqrt(np.mean(difference ** 2))),
            "outside_tolerance_count": int(np.count_nonzero(np.abs(difference) > limit)),
            "within_tolerance": bool(np.isfinite(actual).all() and np.all(np.abs(difference) <= limit)),
            "worst_margin": {"coordinate": [int(i) for i in index], "actual": float(actual64[index]),
                             "expected": float(expected64[index]), "error": float(difference[index]),
                             "allowed_error": float(limit[index])}}


def official_trace(model, data, device):
    import torch

    module = model.enc_p.mrte
    cross = module.cross_attention
    captured, handles = {}, []

    def save(name, tensor):
        captured[name] = tensor.detach().cpu().numpy().copy()

    for name, layer in (("c_pre", module.c_pre), ("text_pre", module.text_pre), ("query", cross.conv_q),
                        ("key", cross.conv_k), ("value", cross.conv_v), ("attention_output", cross.conv_o)):
        handles.append(layer.register_forward_hook(lambda owner, args, output, key=name: save(key, output)))
    handles.append(cross.conv_o.register_forward_pre_hook(lambda owner, args: save("context", args[0])))
    handles.append(module.c_post.register_forward_pre_hook(lambda owner, args: save("conditioned", args[0])))
    try:
        convert = lambda value: torch.from_numpy(value.copy()).to(device=device, dtype=torch.float32)
        ssl, text, ge = (convert(data[name]) for name in ("ssl", "text", "ge512"))
        mask = torch.ones((1, 1, ssl.shape[-1]), device=device)
        text_mask = torch.ones((1, 1, text.shape[-1]), device=device)
        with torch.inference_mode():
            output = module(ssl, mask, text, text_mask, ge)
            save("mrte", output)
            save("probabilities", cross.attn)
            heads = cross.n_heads
            query, key = (convert(captured[name]).reshape(1, heads, cross.k_channels, -1).transpose(2, 3)
                          for name in ("query", "key"))
            scores = (query / math.sqrt(cross.k_channels)) @ key.transpose(-2, -1)
            save("scores", scores)
        return captured
    finally:
        for handle in handles:
            handle.remove()


def mlx_trace(model, data):
    import mlx.core as mx
    from sakuratts.backends.mlx.encoder import attention

    ssl, text, ge = (mx.array(data[name].transpose(0, 2, 1)) for name in ("ssl", "text", "ge512"))
    prefix = "enc_p.mrte"
    captured = {}
    ssl = model.conv(ssl, prefix + ".c_pre")
    text = model.conv(text, prefix + ".text_pre")
    captured.update(c_pre=ssl, text_pre=text)
    cross = prefix + ".cross_attention"
    spec = model.modules[cross]
    heads, features = spec["n_heads"], spec["channels"] // spec["n_heads"]
    qkv = []
    for name, suffix, value in (("query", "q", ssl), ("key", "k", text), ("value", "v", text)):
        projected = model.conv(value, cross + ".conv_" + suffix)
        captured[name] = projected
        qkv.append(projected.reshape(1, -1, heads, features).transpose(0, 2, 1, 3))
    query, key, value = qkv
    mask = mx.ones((1, 1, ssl.shape[1], text.shape[1]), dtype=mx.float32)
    output, probability = attention(query, key, value, mask)
    context = output.transpose(0, 2, 1, 3).reshape(1, ssl.shape[1], spec["channels"])
    attended = model.conv(context, cross + ".conv_o")
    conditioned = attended + ssl + ge
    output = model.conv(conditioned, prefix + ".c_post")
    captured.update(context=context, attention_output=attended, conditioned=conditioned, mrte=output)
    captured = {name: tensor.transpose(0, 2, 1) for name, tensor in captured.items()}
    captured.update(probabilities=probability, scores=(query / math.sqrt(features)) @ key.swapaxes(-1, -2))
    # This explicit diagnostic expansion must match the existing runtime call.
    direct = model.conv(model.multihead(ssl, text, mask, cross) + ssl + ge, prefix + ".c_post")
    mx.eval(*captured.values(), direct)
    actual = {name: np.asarray(value).copy() for name, value in captured.items()}
    if not np.array_equal(actual["mrte"], np.asarray(direct.transpose(0, 2, 1))):
        raise AssertionError("Instrumented MRTE differs from the existing MLX multihead path")
    return actual


def numpy_trace(manifest, weights, data):
    """Independent all-FP64 MRTE, with 1x1 convolutions as affine matrices."""
    def conv(x, prefix):
        weight = weights[prefix + ".weight"]
        if weight.shape[-1] != 1:
            raise ValueError("The MRTE oracle covers 1x1 projections only")
        return x @ weight[:, :, 0].T + weights[prefix + ".bias"]

    ssl, text, ge = (data[name].astype(np.float64).transpose(0, 2, 1) for name in ("ssl", "text", "ge512"))
    prefix = "enc_p.mrte"
    ssl, text = conv(ssl, prefix + ".c_pre"), conv(text, prefix + ".text_pre")
    captured = {"c_pre": ssl, "text_pre": text}
    cross = prefix + ".cross_attention"
    spec = manifest["modules"][cross]
    heads, features = spec["n_heads"], spec["channels"] // spec["n_heads"]
    qkv = []
    for name, suffix, value in (("query", "q", ssl), ("key", "k", text), ("value", "v", text)):
        projected = conv(value, cross + ".conv_" + suffix)
        captured[name] = projected
        qkv.append(projected.reshape(1, -1, heads, features).transpose(0, 2, 1, 3))
    query, key, value = qkv
    scores = (query / math.sqrt(features)) @ key.swapaxes(-1, -2)
    probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    context = (probability @ value).transpose(0, 2, 1, 3).reshape(1, ssl.shape[1], spec["channels"])
    attended = conv(context, cross + ".conv_o")
    conditioned = attended + ssl + ge
    output = conv(conditioned, prefix + ".c_post")
    captured.update(context=context, attention_output=attended, conditioned=conditioned, mrte=output)
    captured = {name: value.transpose(0, 2, 1) for name, value in captured.items()}
    captured.update(probabilities=probability, scores=scores)
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--official-conditions", type=Path, required=True)
    parser.add_argument("--native-acoustic", type=Path, required=True)
    parser.add_argument("--official-mrte-run", type=Path)
    parser.add_argument("--case", default="ja-punctuation")
    parser.add_argument("--backend", choices=("official", "mlx", "numpy"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "gpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.backend == "mlx" and args.device == "mps" or args.backend != "mlx" and args.device == "gpu":
        parser.error("Use cpu/mps for official, cpu/gpu for MLX, or cpu for NumPy")
    if args.backend == "numpy" and args.device != "cpu":
        parser.error("NumPy oracle is CPU only")
    root = args.references.resolve()
    run = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-mrte-diagnosis-{args.backend}-{args.device}")
    run.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    files = ("research/tools/mrte_numerical_diagnosis.py", "research/tools/sovits_fixed_conditions.py",
             "src/sakuratts/backends/mlx/encoder.py", "src/sakuratts/_internal/weight_storage.py")
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    result = {"status": "running", "backend": args.backend, "device": args.device, "case": args.case,
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_sha256": {name: sha256(run / "source" / name) for name in files},
              "scope": "Fixed-stage MRTE error attribution; not full synthesis, performance or quality acceptance", "variants": {}}
    try:
        manifest = json.loads((args.package / "manifest.json").read_text())
        official = json.loads((args.official_conditions / "result.json").read_text())
        native = json.loads((args.native_acoustic / "result.json").read_text())
        if (official["backend"] != "official" or manifest["source"]["checkpoint_sha256"] != official["checkpoint_sha256"]
                or native["official_manifest_sha256"] != sha256(args.official_conditions / "result.json")
                or manifest["source"]["official_commit"] != official["upstream_source"]["commit"]):
            raise ValueError("Source identities differ")
        sources = {}
        for label, parent in (("official", official), ("native", native)):
            case = parent["cases"][args.case]
            if sha256(case["arrays_file"]) != case["arrays_sha256"]:
                raise ValueError("Source array hash changed")
            with np.load(case["arrays_file"], allow_pickle=False) as archive:
                sources[label] = {name: archive[("native_" if label == "native" else "") + name].copy()
                                  for name in ("ssl_encoded", "text_encoded", "mrte")}
                if label == "official":
                    ge512 = archive["ge_projected"].transpose(0, 2, 1).copy()
        result.update(package=str(args.package), package_manifest_sha256=sha256(args.package / "manifest.json"),
                      official_conditions=str(args.official_conditions), official_manifest_sha256=sha256(args.official_conditions / "result.json"),
                      native_acoustic=str(args.native_acoustic), native_manifest_sha256=sha256(args.native_acoustic / "result.json"))
        comparator = None
        if args.official_mrte_run:
            comparator = json.loads((args.official_mrte_run / "result.json").read_text())
            if (comparator["backend"] != "official" or comparator["case"] != args.case
                    or comparator["official_manifest_sha256"] != result["official_manifest_sha256"]
                    or comparator["native_manifest_sha256"] != result["native_manifest_sha256"]):
                raise ValueError("MRTE comparison run has different inputs")
            result.update(official_mrte_run=str(args.official_mrte_run), official_mrte_manifest_sha256=sha256(args.official_mrte_run / "result.json"))
        if args.backend == "official":
            import torch
            torch.set_num_threads(args.threads)
            result["upstream_source"] = source_inventory(root / "GPT-SoVITS", "official")
            model, audit, _ = load_acoustic(root / "GPT-SoVITS", "official", Path(official["checkpoint"]), args.device)
            result["model_audit"] = audit
        elif args.backend == "mlx":
            import mlx.core as mx
            from sakuratts.backends.mlx.encoder import MLXSoVITSEncoder
            mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
            model = MLXSoVITSEncoder.load(args.package)
        else:
            from sakuratts._internal.weight_storage import read_fp32, validate_storage
            path = args.package / manifest["weights"]["file"]
            if sha256(path) != manifest["weights"]["sha256"]:
                raise ValueError("Package weights changed")
            weights = {}
            with np.load(path, allow_pickle=False) as archive:
                validate_storage(manifest, archive.files)
                for name in archive.files:
                    if name.startswith("enc_p.mrte."):
                        weights[name] = read_fp32(archive, manifest, name).astype(np.float64)
        for ssl_source, text_source in (("official", "official"), ("native", "native"), ("official", "native"), ("native", "official")):
            variant = f"{ssl_source}_ssl_{text_source}_text"
            data = {"ssl": sources[ssl_source]["ssl_encoded"], "text": sources[text_source]["text_encoded"], "ge512": ge512}
            if args.backend == "official":
                actual = official_trace(model, data, args.device)
            elif args.backend == "mlx":
                with mx.stream(mx.cpu if args.device == "cpu" else mx.gpu):
                    actual = mlx_trace(model, data)
            else:
                actual = numpy_trace(manifest, weights, data)
            arrays_file = run / f"{variant}.npz"
            np.savez(arrays_file, **actual, input_ssl=data["ssl"], input_text=data["text"], ge512=ge512)
            item = {"arrays_file": str(arrays_file), "arrays_sha256": sha256(arrays_file),
                    "mrte_vs_official_pipeline": compare(actual["mrte"], sources["official"]["mrte"]),
                    "mrte_vs_native_pipeline": compare(actual["mrte"], sources["native"]["mrte"])}
            if comparator:
                expected_case = comparator["variants"][variant]
                if sha256(expected_case["arrays_file"]) != expected_case["arrays_sha256"]:
                    raise ValueError("MRTE reference arrays changed")
                with np.load(expected_case["arrays_file"], allow_pickle=False) as archive:
                    item["same_inputs_vs_official"] = {name: compare(value, archive[name]) for name, value in actual.items()}
            result["variants"][variant] = item
        result["status"] = "diagnosis_completed"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        result["torch_imported"] = "torch" in sys.modules
        result["mlx_imported"] = "mlx.core" in sys.modules
        (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(run), "status": result["status"], "error": result.get("error"),
                      "mrte": {name: item["mrte_vs_official_pipeline"] for name, item in result["variants"].items()}}, indent=2))
    return 0 if result["status"] == "diagnosis_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
