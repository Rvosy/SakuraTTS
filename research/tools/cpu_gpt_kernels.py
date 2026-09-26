"""Fixed-history CPU GPT probe for BLAS weight layout and vector kernels."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from cpu_gpt_profile import CPUGPT, replay_inputs, replay, summarize, array_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=["c-matmul", "f-matmul", "f-dot", "c-dot", "c-einsum", "c-fastnorm"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    root, inputs, tokens, identity = replay_inputs(args.model.resolve(), args.result.resolve())
    model = CPUGPT.load(root, threads=args.threads)
    gemv, norm = model._gemv, model._norm
    reference = None
    result = {"identity": identity, "threads": args.threads, "variants": {}}
    try:
        for variant in args.variants:
            order, kernel = variant.split("-", 1)
            if order not in ("c", "f"):
                raise ValueError("Expected c or f layout")
            for name, value in model.weights.items():
                if name.endswith(".weight") and value.ndim == 2:
                    model.weights[name] = np.asarray(value, order=order.upper())
            model._gemv, model._norm = gemv, norm
            if kernel in ("dot", "einsum"):
                def vector(x, prefix, out):
                    weights = model.weights[prefix + ".weight"]
                    if kernel == "dot":
                        np.dot(weights, x, out=out)
                    else:
                        np.einsum("ij,j->i", weights, x, optimize=False, out=out)
                    bias = model.weights.get(prefix + ".bias")
                    if bias is not None:
                        out += bias
                model._gemv = vector
            elif kernel.startswith("fastnorm"):
                def vector_norm(x, prefix):
                    if x.ndim != 1:
                        return norm(x, prefix)
                    x -= x.sum() / np.float32(x.size)
                    x /= np.sqrt(np.dot(x, x) / np.float32(x.size) + model.epsilon)
                    x *= model.weights[prefix + ".weight"]
                    x += model.weights[prefix + ".bias"]
                    return x
                model._norm = vector_norm
            elif kernel != "matmul":
                raise ValueError("Unknown kernel")
            matmul = np.matmul
            def attention_matmul(left, right, *args, **kwargs):
                if left.ndim == right.ndim == 3 and right.shape[-1] == 1:
                    out = kwargs["out"]
                    np.einsum("hij,hj->hi", left, right[:, :, 0], out=out[:, :, 0], optimize=False)
                    return out
                return matmul(left, right, *args, **kwargs)
            with patch.object(np, "matmul", attention_matmul if "attention" in kernel else matmul):
                replay(model, inputs, tokens)
                rows = []
                for _ in range(args.repeats):
                    row, logits = replay(model, inputs, tokens)
                    rows.append(row)
            if reference is None:
                reference = logits
            row = {"measurements": rows, "median": summarize(rows),
                "max_abs_logits_difference": float(np.max(np.abs(logits - reference))),
                "logits_rms_difference": float(np.sqrt(np.mean((logits - reference) ** 2))),
                "argmax_disagreements": int(np.count_nonzero(np.argmax(logits, axis=1) != np.argmax(reference, axis=1))),
                "logits_sha256": array_digest(logits)}
            result["variants"][f"{len(result['variants']):02d}-{variant}"] = row
            print(json.dumps({"variant": variant, **{key: value for key, value in row.items() if key != "measurements"}}), flush=True)
    finally:
        model.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
