"""Exercise the public CLI with Python file/network auditing in every worker.

This is a development check, not an OS sandbox. Native library I/O is outside
Python audit coverage; loaded native module paths are recorded separately.
"""

import argparse
import atexit
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def install_audit(output, forbidden):
    opened, violations = set(), []
    paths = [os.path.normcase(str(Path(path).resolve())) for path in forbidden]

    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            violations.append({"event": event})
            raise RuntimeError("Network access is disabled during isolation validation")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            opened.add(path)
            if any(path == root or path.startswith(root + os.sep) for root in paths):
                violations.append({"event": event, "path": path})
                raise RuntimeError("Runtime tried to read a development-only directory: " + path)

    sys.addaudithook(audit)
    original = subprocess.Popen

    class AuditedPopen(original):
        def __init__(self, command, *args, **kwargs):
            if isinstance(command, (list, tuple)):
                command = list(command)
                if len(command) >= 3 and Path(command[0]).name.lower() == "python.exe" and command[1] == "-B":
                    command[2:2] = [str(Path(__file__).resolve()), "--child"]
            super().__init__(command, *args, **kwargs)

    subprocess.Popen = AuditedPopen

    def save():
        modules = {name: str(module.__file__) for name, module in list(sys.modules.items())
                   if getattr(module, "__file__", None)}
        record = {"python": sys.executable, "argv": sys.argv,
                  "torch_imported": "torch" in sys.modules,
                  "violations": violations, "python_open_paths": sorted(opened),
                  "module_paths": modules,
                  "scope": "Python audit events and imported modules; not native-library file I/O tracing"}
        (output / ("process-%d.json" % os.getpid())).write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    atexit.register(save)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        output = Path(os.environ["SAKURATTS_ISOLATION_OUTPUT"])
        install_audit(output, json.loads(os.environ["SAKURATTS_ISOLATION_FORBIDDEN"]))
        script = sys.argv[2]
        sys.argv = sys.argv[2:]
        runpy.run_path(script, run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--forbid", action="append", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(SAKURATTS_ISOLATION_OUTPUT=str(args.output.resolve()),
                      SAKURATTS_ISOLATION_FORBIDDEN=json.dumps(args.forbid),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    absent = {name: importlib.util.find_spec(name) is None
              for name in ("torch", "torchaudio", "transformers", "onnx", "mlx")}
    (args.output / "dependencies.json").write_text(json.dumps(absent, indent=2) + "\n", encoding="utf-8")
    if not all(absent.values()):
        raise RuntimeError("Use the clean runtime environment without development dependencies")
    install_audit(args.output.resolve(), args.forbid)
    sys.argv = ["sakuratts", "synthesize", "--config", args.config,
                "--text", "おはよう。今日もよろしくね。", "--output", str(args.output / "standalone.wav")]
    runpy.run_module("sakuratts", run_name="__main__")


if __name__ == "__main__":
    main()
