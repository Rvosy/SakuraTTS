"""Inspect installed metadata for a future CPU G2PW runner; never import ORT."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=PROJECT.parent / "SakuraTTS-References")
    args = parser.parse_args()
    references = args.references.resolve()
    output = references / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-g2pw-dependency-audit")
    output.mkdir(parents=True)
    candidate_python = references / ".venv-mlx-macos/bin/python"
    command = [str(candidate_python), "-c", "from importlib import metadata; import json; print(json.dumps({d.metadata['Name']:d.version for d in metadata.distributions()}))"]
    current = json.loads(subprocess.check_output(command, text=True))
    normalized = {canonicalize_name(name): version for name, version in current.items()}
    packages = []
    names = ["onnxruntime", "flatbuffers", "protobuf", "numpy", "packaging",
             "opencc-python-reimplemented", "pypinyin"]
    for name in names:
        distribution = metadata.distribution(name)
        paths = {distribution.locate_file(path).resolve() for path in distribution.files or []}
        packages.append(dict(name=distribution.metadata["Name"], version=distribution.version,
                             requires=distribution.requires or [], wheel=distribution.read_text("WHEEL"),
                             candidate_version=normalized.get(canonicalize_name(name)),
                             installed_reference_logical_bytes=sum(path.stat().st_size for path in paths if path.is_file())))
    ort_requirements = []
    for text in metadata.distribution("onnxruntime").requires or []:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        version = normalized.get(canonicalize_name(requirement.name))
        ort_requirements.append(dict(requirement=text, candidate_version=version,
                                     candidate_satisfies=version is not None and version in requirement.specifier))
    additions = [item for item in packages
                 if canonicalize_name(item["name"]) in {"onnxruntime", "flatbuffers", "protobuf"}
                 and item["candidate_version"] is None]
    model = references / "GPT-SoVITS/GPT_SoVITS/text/G2PWModel/g2pW.onnx"
    report = dict(command=[sys.executable, *sys.argv], candidate_inventory_command=command,
                  scope="Installed distribution metadata and file sizes only. No install, download, native-library import or model load.",
                  candidate_inventory=current, packages=packages, ort_default_requirements=ort_requirements,
                  planned_fixed_additions=[f"{item['name']}=={item['version']}" for item in additions],
                  reference_file_bytes_of_planned_additions=sum(item["installed_reference_logical_bytes"] for item in additions),
                  model=dict(path=str(model), bytes=model.stat().st_size),
                  runtime_imported_onnxruntime="onnxruntime" in sys.modules,
                  runtime_imported_torch="torch" in sys.modules,
                  runtime_imported_transformers="transformers" in sys.modules,
                  package_changes_performed=False)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(Path(__file__), output / "audit_g2pw_dependencies.py")
    print(f"RUN_DIRECTORY={output}")
    print(json.dumps({key: report[key] for key in (
        "planned_fixed_additions", "reference_file_bytes_of_planned_additions", "package_changes_performed")}))


if __name__ == "__main__":
    main()
