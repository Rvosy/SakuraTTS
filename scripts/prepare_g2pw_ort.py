"""Serialize CPU graph optimizations to a new local ORT-format candidate."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback

os.environ["ORT_DISABLE_TELEMETRY"] = "1"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    source_hash = sha256(source)
    output = args.output_root.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-g2pw-ort-cpu")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "prepare_g2pw_ort.py")
    report = dict(status="running", format="sakuratts-g2pw-ort-candidate-v1",
                  command=[sys.executable, *sys.argv], source=str(source), source_sha256=source_hash,
                  source_bytes=source.stat().st_size, converter_sha256=sha256(__file__),
                  scope="Local CPU optimized graph, same weights and model input contract. Original source preserved. Separate numerical verification required; not a portable deployment promise.")
    try:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        options.add_session_config_entry("session.save_model_format", "ORT")
        target = output / "g2pw.ort"
        options.optimized_model_filepath = str(target)
        start = time.perf_counter()
        session = ort.InferenceSession(str(source), sess_options=options, providers=["CPUExecutionProvider"])
        report["load_and_serialize_seconds"] = time.perf_counter() - start
        report.update(runtime=dict(version=metadata.version("onnxruntime"), build=ort.get_build_info(),
                                   system=platform.system(), machine=platform.machine(), providers=session.get_providers()),
                      inputs=[dict(name=item.name, type=item.type, shape=item.shape) for item in session.get_inputs()],
                      outputs=[dict(name=item.name, type=item.type, shape=item.shape) for item in session.get_outputs()],
                      model_file=target.name, model_sha256=sha256(target), model_bytes=target.stat().st_size,
                      reload_graph_optimization="ORT_DISABLE_ALL (serialized optimizations already applied)")
        del session
        if sha256(source) != source_hash:
            raise ValueError("Source model changed during preparation")
        report["status"] = "converted_unvalidated"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    report["process_lifetime_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    report["torch_imported"] = "torch" in sys.modules
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), status=report["status"], error=report.get("error"))))
    return 0 if report["status"] == "converted_unvalidated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
