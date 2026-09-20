"""Check FP32 prefill across request orders and request-state recreation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from windows_gpt_precision import load_reference_prompt, metrics
from sakuratts._internal.reference_condition import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = json.loads(args.captures.read_text(encoding="utf-8"))
    captures, identities = {}, {}
    for case, raw in mapping.items():
        path = (args.captures.parent/raw).resolve(strict=True)
        with np.load(path, allow_pickle=False) as archive:
            captures[case] = {key: archive[key] for key in
                ("gpt_all_phones", "gpt_all_bert", "sampled_tokens", "raw_logits")}
        identities[case] = {"path": str(path), "sha256": sha256_file(path)}
    prompt, reference_hash = load_reference_prompt(args.reference)
    from sakuratts.backends.cuda.gpt import CUDAGPT

    names = list(captures)
    # Original order with repeated requests, reverse order, then fixed shuffled order.
    order = [name for name in names for _ in range(6)] + names[::-1]*2
    order += np.random.default_rng(20260920).choice(names, size=32).tolist()
    report = {"passed": False, "cases": [], "captures": identities,
              "reference_archive_sha256": reference_hash,
              "executor_sha256": sha256_file(ROOT / "src/sakuratts/backends/cuda/gpt.py"),
              "scope": "Prefill and one decode only; not a complete-request timing benchmark.",
              "atol": 1e-4, "rtol": 1e-5}
    for attention, chunk in (("baseline", 256), ("split-kv", 256), ("split-kv", 512)):
        model = CUDAGPT.load(args.gpt, precision="fp32", attention=attention, attention_chunk_size=chunk)
        try:
            for index, case in enumerate(order):
                arrays = captures[case]
                rebuilt = index > 0 and index % 11 == 0
                if rebuilt:
                    model.release_request_state()
                actual = model.prefill(arrays["gpt_all_phones"][None], prompt, arrays["gpt_all_bert"].T[None])
                prefill = metrics(actual[0], arrays["raw_logits"][0])
                decoded = model.decode(int(arrays["sampled_tokens"].reshape(-1)[0]))
                decode = metrics(decoded[0], arrays["raw_logits"][1])
                report["cases"].append({"attention": attention, "chunk_size": chunk,
                    "case": case, "index": index, "rebuilt": rebuilt,
                    "prefill": prefill, "first_decode": decode,
                    "state_addresses": {name: value.data.ptr for name, value in model.workspace.items()},
                    "passed": prefill["strict_passed"] and decode["strict_passed"]})
        finally:
            model.close()
    report["passed"] = all(row["passed"] for row in report["cases"])
    (args.output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "failures": [row for row in report["cases"] if not row["passed"]]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
