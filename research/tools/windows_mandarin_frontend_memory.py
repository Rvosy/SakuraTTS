"""Measure the public target-text path and its Windows process tree in fresh processes."""

import argparse
import json
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import threading
import time


TEXT = {
    "short": "今天下午三点，我们在图书馆门口见。",
    "long": "明天下午三点，我们在银行门口见面。去之前先看一下天气预报，如果下雨，记得带伞。",
    "dictionary": "森林。",
}


def worker(args):
    runpy.run_path(str(args.bundle / "launcher.py"))["configure"]()
    if args.source:
        sys.path.insert(0, str(args.source))
    from sakuratts.model import Model
    from sakuratts.runtime.portable import model_config
    from sakuratts.TTS_infer_pack.frontend import load_frontend

    model = Model.load(args.model)
    config = model_config(model.runtime_config)
    folder = model.path.parent / config["frontend"]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["chinese_g2p"]["directory"] = str(args.resources)
    frontend = load_frontend(model.path, config, folder, manifest)
    try:
        print(json.dumps({"phase": "ready"}), flush=True)
        for line in sys.stdin:
            label = line.strip()
            start = time.perf_counter_ns()
            parts = frontend.text.prepare_target(TEXT[label], "zh", "cut0")
            elapsed = time.perf_counter_ns() - start
            processor = frontend.text.processors["zh"]
            print(json.dumps({"phase": "call", "ms": elapsed / 1e6,
                  "phones": sum(len(part["phones"]) for part in parts),
                  "g2pw_loaded": "session" in vars(processor.phones.g2pw)}), flush=True)
    finally:
        frontend.close()


def tree_memory(process, psutil):
    working_set = private = 0
    for item in [process, *process.children(recursive=True)]:
        try:
            info = item.memory_info()
        except psutil.NoSuchProcess:
            continue
        working_set += info.wset
        private += info.private
    return {"working_set": working_set, "private": private}


def measure(args, first):
    import psutil

    def receive(phase):
        for line in child.stdout:
            if line.startswith("{"):
                row = json.loads(line)
                if row.get("phase") == phase:
                    return row
        raise RuntimeError(f"Worker exited before {phase}, return code {child.poll()}")

    command = [str(args.bundle / "runtime/main/python.exe"), "-I", "-X", "utf8", "-B", "-u",
               str(Path(__file__).resolve()), "--worker", "--bundle", str(args.bundle),
               "--model", str(args.model), "--resources", str(args.resources)]
    if args.source:
        command += ["--source", str(args.source)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8") as child:
        try:
            receive("ready")
            process = psutil.Process(child.pid)
            before = tree_memory(process, psutil)
            peak = before.copy()
            stop = threading.Event()

            def sample():
                while not stop.is_set():
                    for name, value in tree_memory(process, psutil).items():
                        peak[name] = max(peak[name], value)
                    stop.wait(.01)

            sampler = threading.Thread(target=sample)
            sampler.start()
            calls = []
            try:
                # Separate dictionary-only cold start from a cold request requiring both models.
                labels = [first] if first == "dictionary" else [first, *(["short", "long"] * 6)]
                for label in labels:
                    child.stdin.write(label + "\n")
                    child.stdin.flush()
                    row = receive("call")
                    calls.append(dict(label=label, **row, memory=tree_memory(process, psutil)))
            finally:
                stop.set()
                sampler.join()
            for name in peak:
                peak[name] = max(peak[name], *(row["memory"][name] for row in calls))
            return {"before": before, "peak": peak, "calls": calls}
        finally:
            child.stdin.close()
            if child.wait():
                raise RuntimeError("Target frontend worker failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "model", "resources"):
        parser.add_argument("--" + name, required=True, type=lambda p: Path(p).resolve(strict=True))
    parser.add_argument("--source", type=lambda p: Path(p).resolve(strict=True),
                        help="Use this checkout with the same packaged dependencies")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if not args.output:
        parser.error("--output is required for measurements")
    data = {first: measure(args, first) for first in ("short", "dictionary")}
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    calls = data["short"]["calls"]
    for label in ("short", "long"):
        hot = [r["ms"] for r in calls[3:] if r["label"] == label]
        print(label, round(statistics.median(hot), 2), "hot ms")


if __name__ == "__main__":
    main()
