#!/usr/bin/env python3
"""Fresh-process bitwise reload and rejection checks for prepared references."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from sakuratts._internal.reference_condition import ARRAY_DTYPES, PreparedReference, sha256_array, sha256_file


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def worker(args):
    package, source = args.package.resolve(), args.prepared_run.resolve()
    manifest = read_json(package / "manifest.json")
    expected = dict(manifest["identity"])
    reference = PreparedReference.load(package, **expected, manifest_sha256=args.manifest_sha256)
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

    def reject(name, path=package, **overrides):
        arguments = dict(expected, **overrides)
        try:
            PreparedReference.load(path, **arguments)
        except ValueError as error:
            checks.append({"case": name, "rejected": True, "error": str(error)})
        else:
            checks.append({"case": name, "rejected": False})

    for field in ("gpt_checkpoint_sha256", "sovits_checkpoint_sha256", "audio_sha256", "official_commit"):
        reject("wrong-" + field, **{field: "0" * len(expected[field])})
    reject("missing-model-identity", gpt_checkpoint_sha256=None)
    reject("wrong-reference-text", reference_text=expected["reference_text"] + " changed")
    reject("wrong-reference-language", reference_language="not-the-reference-language")
    reject("wrong-manifest-hash", manifest_sha256="0" * 64)

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

    def altered_archive(name, arrays, update_array_metadata=False):
        target = variant(name)
        updated = read_json(target / "manifest.json")
        np.savez(target / "conditions.npz", **arrays)
        updated["archive"].update(bytes=(target / "conditions.npz").stat().st_size,
                                  sha256=sha256_file(target / "conditions.npz"))
        if update_array_metadata:
            updated["arrays"] = {key: dict(dtype=str(value.dtype), shape=list(value.shape), bytes=value.nbytes,
                                           sha256_raw_c_order=sha256_array(value)) for key, value in arrays.items()}
        write_json(target / "manifest.json", updated)
        reject(name, target)

    changed = {key: value.copy() for key, value in source_arrays.items()}
    changed["prompt_semantic"][0] += 1
    altered_archive("changed-array-with-refreshed-archive-hash", changed)
    missing = {key: value for key, value in source_arrays.items() if key != "ge"}
    altered_archive("missing-required-array", missing, True)
    misaligned = dict(source_arrays, reference_bert=source_arrays["reference_bert"][:, :-1])
    altered_archive("misaligned-reference-bert", misaligned, True)
    imports = {name: name in sys.modules for name in ("torch", "transformers", "mlx", "onnxruntime")}
    passed = all(all(row.values()) for row in equality.values()) and all(row["rejected"] for row in checks) and not any(imports.values())
    report = {"status": "passed" if passed else "failed", "command": [sys.executable, *sys.argv],
              "relocated_package": str(package), "array_checks": equality, "rejection_checks": checks,
              "imports": imports, "scope": "NumPy archive/identity validation only; no generation, model execution, ASR or listening"}
    write_json(args.result, report)
    return 0 if passed else 1


def run(args):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.references.resolve() / "runs" / (timestamp + "-reference-condition-reload")
    (output / "source/harness").mkdir(parents=True)
    (output / "source/src/sakuratts").mkdir(parents=True)
    shutil.copy2(__file__, output / "source/research/tools/reference_package_replay.py")
    shutil.copy2(PROJECT / "src/sakuratts/_internal/reference_condition.py", output / "source/src/sakuratts/_internal/reference_condition.py")
    relocated = output / "relocated-package"
    relocated.mkdir()
    for name in ("manifest.json", "conditions.npz"):
        shutil.copy2(args.package / name, relocated / name)
    command = [sys.executable, str(output / "source/research/tools/reference_package_replay.py"), "--worker",
               "--package", str(relocated), "--prepared-run", str(args.prepared_run.resolve()),
               "--manifest-sha256", sha256_file(args.package / "manifest.json"),
               "--result", str(output / "worker-result.json")]
    with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    report = {"status": "passed" if result.returncode == 0 else "failed", "command": command,
              "exit_code": result.returncode, "run": str(output), "source_package": str(args.package.resolve()),
              "source_manifest_sha256": sha256_file(args.package / "manifest.json"),
              "reader_sha256": sha256_file(output / "source/src/sakuratts/_internal/reference_condition.py"),
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
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return worker(args) if args.worker else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
