"""Run a precision/device matrix sequentially in isolated measured processes."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import cpu_amd_benchmark as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cases", nargs="+", choices=("short", "long"), default=["short", "long"])
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    selected = args.variants or list(matrix)
    args.output.mkdir(parents=True, exist_ok=True)
    status = {}
    for name in selected:
        row = matrix[name]
        if Path(name).name != name:
            raise ValueError("Variant must be a directory basename")
        options = args.output / (name + "-options.json")
        with options.open("x", encoding="utf-8") as stream:
            json.dump(row.get("options", {}), stream, indent=2)
        command = ["--engines", row["backend"], "--python", str(args.python.resolve()),
                   "--model", row["model"], "--character", str(args.character),
                   "--cases", *args.cases, "--seed", "1234", "--warmups", "1",
                   "--repeats", str(args.repeats), "--output", str(args.output / name),
                   "--experimental", str(options), "--timeout", "900"]
        if row.get("profile"):
            command += ["--profile", row["profile"]]
        print(json.dumps({"variant": name, "state": "starting"}), flush=True)
        status[name] = benchmark.main(command)
        (args.output / "matrix-status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if status[name]:
            return status[name]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
