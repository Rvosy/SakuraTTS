"""Run the frozen two-arm replay in disposable containers, never in production."""

import json
from pathlib import Path
import subprocess
import sys


def main(directory, arms=("candidate", "baseline")):
    work = Path(directory).resolve()
    if work.name != "sakura-voice-candidate-20261026" or not (work / "private/frozen-inputs.json").is_file():
        raise ValueError("Expected the isolated voice candidate snapshot")
    model = "gpt-6-luna-reply"
    for arm in arms:
        if arm not in ("candidate", "baseline"):
            raise ValueError("Expected candidate or baseline")
        log = work / "private" / f"{arm}-replay.log"
        if log.exists():
            raise FileExistsError(log)
        command = [
            "docker", "run", "--rm", "--name", "sakura-voice-replay-" + arm,
            "--network", "newapi-migration_private", "--memory", "512m", "--memory-swap", "768m",
            "--cpus", "0.5", "-v", str(work) + ":/eval", "-e", "MAIBOT_INSTANCE_DIR=/eval/" + arm,
            "-e", "PYTHONPATH=/MaiMBot", "-e", "TZ=Asia/Shanghai", "-w", "/eval/" + arm,
            "--entrypoint", "/MaiMBot/.venv/bin/python",
            *(["-v", str(work / "host/maisaka_generator_base.py") + ":/MaiMBot/src/chat/replyer/maisaka_generator_base.py:ro"] if arm == "candidate" else []),
            "maibot-sakura-support:1.3.14-memory-attribution",
            "/eval/tools/run_reply_replay.py", "--model", model, "--structured",
        ]
        print("START", arm, flush=True)
        with log.open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{arm} exited with {result.returncode}; see {log}")
        data = json.loads((work / "private/results" / f"{arm}-{model}.json").read_text())
        print("DONE", arm, "records", len(data["records"]),
              "errors", sum("error" in row for row in data["records"]), flush=True)
    print("BATCH_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ("candidate", "baseline"))
