"""Measure the same complete requests returned by the public Engine."""

import json
from pathlib import Path
import time

from sakuratts.engine import Engine


def run(args, *, experimental=None):
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    report = {"scope": "Complete non-streaming PCM; first request includes lazy GPU loads; no quality claim",
              "experimental": experimental or {}, "requests": []}
    with Engine.load(args.model, experimental=experimental) as engine:
        report["open_ms"] = (time.perf_counter() - started) * 1000
        report["model"] = engine.model.info()
        for index in range(args.repeats):
            audio = engine.synthesize(args.text, reference=args.reference, seed=args.seed)
            report["requests"].append({"run": index, **audio.report})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    complete = all(row["status"] == "completed" for row in report["requests"])
    print(json.dumps({"report": str(output), "completed": complete}))
    return 0 if complete else 2
