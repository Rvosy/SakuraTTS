#!/usr/bin/env python3
"""Run reference_smoke in a child and record its actual process exit.

Accepts the same arguments and uses this interpreter. A result written before
interpreter teardown cannot prove clean process termination; keep both records.
No retries, error filtering, or forced child termination are performed.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    script = Path(__file__).resolve()
    command = [sys.executable, "-u", str(script.with_name("reference_smoke.py")), *sys.argv[1:]]
    started = time.perf_counter()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    output, log = None, None
    early_lines = []
    for line in child.stdout:
        print(line, end="", flush=True)
        if output is None:
            early_lines.append(line)
            if line.startswith("RUN_DIRECTORY="):
                output = Path(line.rstrip().split("=", 1)[1])
                log = (output / "process.stdout-stderr.log").open("x")
                log.writelines(early_lines)
                early_lines.clear()
        else:
            log.write(line)
        if log:
            log.flush()
    returncode = child.wait()
    if log:
        log.close()
    if output is not None:
        shutil.copy2(script, output / script.name)
        manifest_path = output / "result.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        result = {
            "recorded_at": datetime.now(timezone.utc).isoformat(), "pid": child.pid,
            "command": command, "returncode": returncode,
            "signal": -returncode if returncode < 0 else None,
            "child_reported_status": manifest.get("status"),
            "clean_completion": returncode == 0 and manifest.get("status") == "completed",
            "elapsed_seconds_including_startup_and_teardown": time.perf_counter() - started,
            "scope": "Process lifecycle, not inference timing or a resource measurement",
            "launcher_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "log_sha256": hashlib.sha256((output / "process.stdout-stderr.log").read_bytes()).hexdigest(),
        }
        (output / "process-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return returncode if returncode >= 0 else 128 - returncode


if __name__ == "__main__":
    raise SystemExit(main())
