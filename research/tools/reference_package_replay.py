#!/usr/bin/env python3
"""Fresh-process bitwise reload and rejection checks for prepared references."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from sakuratts.module.reference_condition import ARRAY_DTYPES, PreparedReference, sha256_file


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def worker(args):
    package, source = args.package.resolve(), args.prepared_run.resolve()
    reference = PreparedReference.load(package)
    source_arrays = {}
    with np.load(source / "prepared-reference.npz", allow_pickle=False) as original:
        source_arrays.update({name: original[name] for name in ("reference_phones", "prompt_semantic", "reference_bert")})
    with np.load(source / "prepared-acoustic.npz", allow_pickle=False) as original:
        source_arrays.update({name: original[name] for name in ("ge", "ge512")})
    equality = {name: {"bytes_equal": source_arrays[name].tobytes(order="C") == getattr(reference, name).tobytes(order="C"),
                       "shape_equal": source_arrays[name].shape == getattr(reference, name).shape,
                       "dtype_equal": source_arrays[name].dtype == getattr(reference, name).dtype,
                       "read_only": not getattr(reference, name).flags.writeable}
                for name in ARRAY_DTYPES}
    checks = []

    def reject(name, path):
        try:
            PreparedReference.load(path)
        except (ValueError, KeyError, zipfile.BadZipFile) as error:
            checks.append({"case": name, "rejected": True, "error": str(error)})
        else:
            checks.append({"case": name, "rejected": False})

    def variant(name):
        target = args.result.parent / "rejections" / name
        shutil.copytree(package, target)
        return target

    damaged = variant("damaged-archive")
    with (damaged / "conditions.npz").open("r+b") as stream:
        stream.seek(100)
        byte = stream.read(1)
        stream.seek(100)
        stream.write(bytes([byte[0] ^ 1]))
    reject("damaged-archive", damaged)

    def altered_archive(name, arrays):
        target = variant(name)
        np.savez(target / "conditions.npz", **arrays)
        reject(name, target)

    missing = {key: value for key, value in source_arrays.items() if key != "ge"}
    altered_archive("missing-required-array", missing)
    misaligned = dict(source_arrays, reference_bert=source_arrays["reference_bert"][:, :-1])
    altered_archive("misaligned-reference-bert", misaligned)
    imports = {name: name in sys.modules for name in ("torch", "transformers", "mlx", "onnxruntime")}
    passed = all(all(row.values()) for row in equality.values()) and all(row["rejected"] for row in checks) and not any(imports.values())
    report = {"status": "passed" if passed else "failed", "command": [sys.executable, *sys.argv],
              "relocated_package": str(package), "array_checks": equality, "rejection_checks": checks,
              "imports": imports, "scope": "NumPy array validation only; no generation, model execution, ASR or listening"}
    write_json(args.result, report)
    return 0 if passed else 1


def run(args):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.references.resolve() / "runs" / (timestamp + "-reference-condition-reload")
    (output / "source/research/tools").mkdir(parents=True)
    (output / "source/sakuratts").mkdir(parents=True)
    shutil.copy2(__file__, output / "source/research/tools/reference_package_replay.py")
    shutil.copy2(PROJECT / "sakuratts/module/reference_condition.py", output / "source/sakuratts/module/reference_condition.py")
    relocated = output / "relocated-package"
    relocated.mkdir()
    for name in ("manifest.json", "conditions.npz"):
        shutil.copy2(args.package / name, relocated / name)
    command = [sys.executable, str(output / "source/research/tools/reference_package_replay.py"), "--worker",
               "--package", str(relocated), "--prepared-run", str(args.prepared_run.resolve()),
               "--result", str(output / "worker-result.json")]
    with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    report = {"status": "passed" if result.returncode == 0 else "failed", "command": command,
              "exit_code": result.returncode, "run": str(output), "source_package": str(args.package.resolve()),
              "source_manifest_sha256": sha256_file(args.package / "manifest.json"),
              "reader_sha256": sha256_file(output / "source/sakuratts/module/reference_condition.py"),
              "harness_sha256": sha256_file(output / "source/research/tools/reference_package_replay.py")}
    write_json(output / "result.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=PROJECT.parent / "SakuraTTS-References")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return worker(args) if args.worker else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
