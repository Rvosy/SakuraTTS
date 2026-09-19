"""Replay one BERT layer and its operations with fixed saved inputs.

Prepare uses PyTorch on CPU; validate uses the torch-free MLX environment.
Each preparation gets a new directory and validation never overwrites results.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def compare(actual, expected, tolerance):
    error = actual.astype(np.float64) - expected.astype(np.float64)
    close = np.isclose(actual, expected, **tolerance)
    return {"shape": list(actual.shape), "max_abs_error": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "passed": bool(close.all()),
            "mismatch_count": int((~close).sum()),
            "first_mismatch_coordinates": np.argwhere(~close)[:10].tolist()}


def prepare(args):
    import torch
    from transformers import BertConfig
    from transformers.models.bert.modeling_bert import BertLayer

    torch.set_num_threads(2)
    source = args.run.resolve()
    package = source / "package"
    manifest = json.loads((package / "manifest.json").read_text())
    prefix = f"encoder.layer.{args.layer}."
    if not 0 <= args.layer < manifest["config"]["retained_layers"]:
        raise ValueError("Layer must identify a retained encoder layer, counting from zero")
    if digest(package / "weights.npz") != manifest["weights"]["sha256"]:
        raise ValueError("Source package weights do not match the manifest")
    output = source.parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-bert-layer-diagnostic")
    output.mkdir()
    print(f"RUN_DIRECTORY={output}", flush=True)
    with np.load(package / "weights.npz", allow_pickle=False) as data:
        weights = {name: data[name].copy() for name in data.files if name.startswith(prefix)}
    np.savez(output / "weights.npz", **weights)
    cfg = BertConfig(**manifest["config"])
    # The complete CPU reference selected SDPA; all dropout is disabled by eval.
    cfg._attn_implementation = "sdpa"
    model = BertLayer(cfg).eval()
    model.load_state_dict({name[len(prefix):]: torch.from_numpy(value) for name, value in weights.items()}, strict=True)
    with np.load(source / f"{args.case}-inputs.npz", allow_pickle=False) as data:
        attention_mask = data["attention_mask"].copy()
    mask = np.where(attention_mask[:, None, None, :] != 0, np.float32(0), np.finfo(np.float32).min)
    torch_mask = None if np.all(attention_mask == 1) else torch.from_numpy(mask)
    np.save(output / "attention-mask.npy", mask)
    with np.load(source / f"{args.case}-official.npz", allow_pickle=False) as data:
        official_input = data[f"layer_{args.layer}"].copy()
        saved_output = data[f"layer_{args.layer + 1}"].copy()
    with np.load(source / f"{args.case}-mlx-gpu.npz", allow_pickle=False) as data:
        gpu_input = data[f"layer_{args.layer}"].copy()
    stages = {}
    hooks = []

    def capture(name):
        def hook(module, inputs, result):
            value = result[0] if isinstance(result, tuple) else result
            stages[name] = value.detach().numpy().copy()
            if name in ("attention_norm", "output_norm"):
                stages[name + "_input"] = inputs[0].detach().numpy().copy()
        return hook

    modules = dict(query_linear=model.attention.self.query, key_linear=model.attention.self.key,
                   value_linear=model.attention.self.value, attention_context=model.attention.self,
                   attention_dense=model.attention.output.dense, attention_norm=model.attention.output.LayerNorm,
                   intermediate_dense=model.intermediate.dense, gelu=model.intermediate,
                   output_dense=model.output.dense, output_norm=model.output.LayerNorm)
    for name, module in modules.items():
        hooks.append(module.register_forward_hook(capture(name)))
    report = {"status": "prepared", "command": [sys.executable, *sys.argv], "source_run": str(source),
              "case": args.case, "layer_index_zero_based": args.layer, "config": manifest["config"],
              "source_manifest_sha256": digest(package / "manifest.json"), "torch": torch.__version__,
              "transformers": metadata.version("transformers"), "torch_threads": torch.get_num_threads(),
              "attention_implementation": "sdpa", "tolerance": {"rtol": 1e-4, "atol": 1e-5}, "cases": []}
    with torch.inference_mode():
        for name, values in (("official_input", official_input), ("mlx_gpu_input", gpu_input)):
            stages.clear()
            result = model(torch.from_numpy(values), attention_mask=torch_mask)[0].numpy()
            np.save(output / f"{name}-input.npy", values)
            np.savez(output / f"{name}-official-stages.npz", **stages)
            entry = {"id": name}
            if name == "official_input":
                entry["reproduces_saved_layer_exact"] = bool(np.array_equal(result, saved_output))
                entry["saved_layer_comparison"] = compare(result, saved_output, report["tolerance"])
            report["cases"].append(entry)
    for hook in hooks:
        hook.remove()
    shutil.copy2(__file__, output / "prepare-harness.py")
    write_json(output / "prepared.json", report)
    print(f"PREPARED={output}", flush=True)


def validate(args):
    import mlx.core as mx
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    from sakuratts.mlx_bert import MLXBertFeatures

    run = args.run.resolve()
    result_path = run / f"mlx-{args.device}-result.json"
    if result_path.exists() or list(run.glob(f"*-mlx-{args.device}-*.npz")):
        raise FileExistsError("Validation artifacts already exist")
    prepared = json.loads((run / "prepared.json").read_text())
    model = MLXBertFeatures(prepared["config"], mx.load(run / "weights.npz"))
    layer = prepared["layer_index_zero_based"]
    prefix = f"encoder.layer.{layer}."
    mask = mx.array(np.load(run / "attention-mask.npy"))
    report = {"device": str(mx.default_device()), "mlx": metadata.version("mlx"),
              "command": [sys.executable, *sys.argv], "tolerance": prepared["tolerance"],
              "runtime_imported_torch": "torch" in sys.modules, "cases": [],
              "scope": "single layer and independent operations from fixed official inputs; diagnostic only"}
    shutil.copy2(__file__, run / f"validation-{args.device}-harness.py")
    shutil.copy2(PROJECT / "src/sakuratts/mlx_bert.py", run / f"validation-{args.device}-runtime.py")
    shutil.copy2(PROJECT / "src/sakuratts/weight_storage.py", run / f"validation-{args.device}-weight_storage.py")
    for case in prepared["cases"]:
        name = case["id"]
        x = mx.array(np.load(run / f"{name}-input.npy"))
        _, stages = model.encoder_layer(x, mask, layer, return_stages=True)
        mx.eval(*stages.values())
        arrays = {key: np.array(value) for key, value in stages.items()}
        np.savez(run / f"{name}-mlx-{args.device}-layer.npz", **arrays)
        with np.load(run / f"{name}-official-stages.npz") as data:
            gold = {key: data[key].copy() for key in data.files}
        checks = {key: compare(value, gold[key], report["tolerance"]) for key, value in arrays.items()}
        # Each operation below receives official input, excluding upstream error.
        fixed = {key: mx.array(value) for key, value in gold.items()}
        qkv = [fixed[key].reshape(x.shape[0], x.shape[1], model.heads, model.head_width).transpose(0, 2, 1, 3)
               for key in ("query_linear", "key_linear", "value_linear")]
        local = {"attention_context": mx.fast.scaled_dot_product_attention(
                    *qkv, scale=model.head_width ** -0.5, mask=mask)
                    .transpose(0, 2, 1, 3).reshape(x.shape),
                 "attention_dense": model._linear(fixed["attention_context"], prefix + "attention.output.dense"),
                 "attention_norm": model._norm(fixed["attention_norm_input"], prefix + "attention.output.LayerNorm"),
                 "intermediate_dense": model._linear(fixed["attention_norm"], prefix + "intermediate.dense"),
                 "gelu": fixed["intermediate_dense"] * 0.5 * (1 + mx.erf(fixed["intermediate_dense"] * (2 ** -0.5))),
                 "output_dense": model._linear(fixed["gelu"], prefix + "output.dense"),
                 "output_norm": model._norm(fixed["output_norm_input"], prefix + "output.LayerNorm")}
        mx.eval(*local.values())
        local_arrays = {key: np.array(value) for key, value in local.items()}
        np.savez(run / f"{name}-mlx-{args.device}-operations.npz", **local_arrays)
        local_checks = {key: compare(value, gold[key], report["tolerance"]) for key, value in local_arrays.items()}
        report["cases"].append({"id": name, "layer_stages": checks, "fixed_operation_inputs": local_checks})
    report["status"] = "completed" if all(check["passed"] for case in report["cases"]
        for section in ("layer_stages", "fixed_operation_inputs") for check in case[section].values()) else "numerical_mismatch"
    write_json(result_path, report)
    print(f"RESULT={result_path} status={report['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    converter = sub.add_parser("prepare")
    converter.add_argument("--run", type=Path, required=True)
    converter.add_argument("--case", required=True)
    converter.add_argument("--layer", type=int, required=True)
    validator = sub.add_parser("validate")
    validator.add_argument("--run", type=Path, required=True)
    validator.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else validate(args)


if __name__ == "__main__":
    main()
