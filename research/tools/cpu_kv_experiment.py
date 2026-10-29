"""Compare CPU KV storage with cpu_memory.py in fresh processes.

Pass --kv-layout contiguous, layers or mmap, followed by the normal cpu_memory.py
arguments. This experiment does not change capacity, model artifacts, sampling
or product defaults. Use --source to select a frozen implementation. Allocation
growth is intentionally isolated here until RSS and latency justify its cost.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kv-layout", choices=("contiguous", "layers", "mmap"), required=True)
    args, remaining = parser.parse_known_args()
    source = Path(remaining[remaining.index("--source") + 1]) if "--source" in remaining else Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    from sakuratts.backends.cpu.onnx_gpt import ONNXCPUGPT
    from cpu_memory import main as measure

    original_run = ONNXCPUGPT._run

    def allocate(gpt):
        if args.kv_layout == "mmap":
            if gpt.keys is None:
                import mmap
                shape = (gpt.layers, gpt.capacity + 1, gpt.heads, gpt.head_dim)
                size = gpt.layers * (gpt.capacity + 1) * gpt.width * np.dtype(gpt.tensor_dtype).itemsize
                def buffer():
                    return np.ndarray(shape, gpt.tensor_dtype,
                        buffer=mmap.mmap(-1, size, access=mmap.ACCESS_COPY))
                gpt.keys, gpt.values = buffer(), buffer()
                gpt.keys[:, 0] = gpt.values[:, 0] = 0
                gpt.decode_mask = np.zeros((1, 1, gpt.capacity + 1), gpt.tensor_dtype)
                gpt.decode_mask[:, :, 0] = -np.inf
            return
        gpt.keys = [None] * gpt.layers if args.kv_layout == "layers" else None
        gpt.values = [None] * gpt.layers if args.kv_layout == "layers" else None
        gpt.length = 0
        gpt.decode_mask = np.zeros((1, 1, gpt.capacity + 1), gpt.tensor_dtype)
        gpt.decode_mask[:, :, 0] = -np.inf

    def run(gpt, hidden, mask):
        if args.kv_layout == "mmap":
            return original_run(gpt, hidden, mask)
        end = gpt.length + hidden.shape[0]
        capacity = min(gpt.capacity, ((end + 255) // 256) * 256)
        try:
            if args.kv_layout == "contiguous":
                if gpt.keys is None or end >= gpt.keys.shape[1]:
                    shape = (gpt.layers, capacity + 1, gpt.heads, gpt.head_dim)
                    for name in ("keys", "values"):
                        old = getattr(gpt, name)
                        value = np.empty(shape, gpt.tensor_dtype)
                        if old is None:
                            value[:, 0] = 0
                        else:
                            value[:, :gpt.length + 1] = old[:, :gpt.length + 1]
                        setattr(gpt, name, value)
                        del old
                return original_run(gpt, hidden, mask)
            if gpt.keys[0] is None or end >= gpt.keys[0].shape[0]:
                for cache in (gpt.keys, gpt.values):
                    for i in range(gpt.layers):
                        value = np.empty((capacity + 1, gpt.heads, gpt.head_dim), gpt.tensor_dtype)
                        if cache[i] is None:
                            value[0] = 0
                        else:
                            value[:gpt.length + 1] = cache[i][:gpt.length + 1]
                        cache[i] = value
            feeds = {"hidden": np.asarray(hidden, dtype=gpt.tensor_dtype),
                     "mask": np.asarray(mask, dtype=gpt.tensor_dtype)}
            for layer in range(gpt.layers):
                feeds[f"past_key.{layer}"] = gpt.keys[layer][:gpt.length + 1]
                feeds[f"past_value.{layer}"] = gpt.values[layer][:gpt.length + 1]
            logits, keys, values = gpt.session.run(None, feeds)
            for layer in range(gpt.layers):
                gpt.keys[layer][gpt.length + 1:end + 1] = keys[layer]
                gpt.values[layer][gpt.length + 1:end + 1] = values[layer]
            gpt.length = end
            return logits.astype(np.float32, copy=False)
        except BaseException:
            gpt.release_request_state()
            raise

    ONNXCPUGPT._allocate_state = allocate
    ONNXCPUGPT._run = run
    sys.argv[1:] = remaining
    measure()
    output = Path(remaining[remaining.index("--output") + 1]) / "result.json"
    result = json.loads(output.read_text(encoding="utf-8"))
    result["kv_experiment"] = {"layout": args.kv_layout,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
