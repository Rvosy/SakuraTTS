#!/usr/bin/env python3
"""Build a developer preview ZIP from a filtered source snapshot; never upload."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile


ROOT_FILES = {"README.md", "LICENSE", "MANIFEST.in", "pyproject.toml", "AGENTS.md", "uv.lock", "start-server.bat", "api.py"}
SOURCE_DIRS = ("src/sakuratts", "scripts", "tools", "requirements", "research", "benchmarks", "tests", "docs", "examples")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".ps1", ".yaml", ".yml", ".rst", ".ini", ".cfg"}
EXCLUDED_DIRS = {"build", "dist", "__pycache__", "node_modules"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def is_link(path):
    attrs = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def source_files(root):
    """Only select known text trees, without following links or local environments."""
    for path in sorted(root.iterdir()):
        if (path.name in ROOT_FILES or (path.name.startswith("requirements") and path.suffix == ".txt")):
            if path.is_file() and not is_link(path):
                yield path
    for relative in SOURCE_DIRS:
        directory = root / relative
        parents = [root.joinpath(*Path(relative).parts[:i]) for i in range(1, len(Path(relative).parts) + 1)]
        if not directory.is_dir() or any(is_link(path) for path in parents):
            continue
        for current, directories, files in os.walk(directory, followlinks=False):
            current = Path(current)
            directories[:] = sorted(name for name in directories
                if not name.startswith(".") and name not in EXCLUDED_DIRS
                and not name.endswith(".egg-info") and not is_link(current / name))
            for name in sorted(files):
                path = current / name
                if not name.startswith(".") and path.suffix in TEXT_SUFFIXES and not is_link(path):
                    yield path


def stage_source(root, destination):
    inventory = {}
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Source is not UTF-8 text: " + relative) from error
        if b"\0" in data:
            raise ValueError("Source contains binary data: " + relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        inventory[relative] = {"bytes": len(data), "sha256": sha256(data)}
    required = ROOT_FILES - {"AGENTS.md", "uv.lock"}
    required |= {"docs/preview-release.md", "requirements/windows-runtime.txt", "src/sakuratts/__init__.py"}
    missing = sorted(required - inventory.keys())
    if missing:
        raise ValueError("Required preview sources are missing: " + ", ".join(missing))
    return dict(sorted(inventory.items()))


def git_state(root):
    unavailable = {"available": False, "head": None, "dirty": None}
    if not (root / ".git").exists():
        return unavailable
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return unavailable
    return {"available": True, "head": head, "dirty": bool(dirty)}


def verify_distributions(wheel, sdist, inventory):
    """Require the distributions to preserve the selected source and licenses."""
    def verify(data, source_name, artifact):
        if {"bytes": len(data), "sha256": sha256(data)} != inventory[source_name]:
            raise ValueError(artifact + " source mismatch: " + source_name)

    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = {member.name: member for member in members}
        if len(names) != len(members):
            raise ValueError("sdist contains duplicate members")
        prefix = sdist.name.removesuffix(".tar.gz") + "/"
        for relative in inventory:
            member = names.get(prefix + relative)
            if member is None or not member.isfile():
                raise ValueError("sdist source missing: " + relative)
            verify(archive.extractfile(member).read(), relative, "sdist")
    product = {name.removeprefix("src/"): name for name in inventory
               if name.startswith("src/sakuratts/") and name.endswith(".py")}
    licenses = [name for name in inventory if name == "LICENSE" or
                (name.startswith("docs/third-party/") and Path(name).suffix in (".txt", ".md"))]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate members")
        actual_product = {name for name in names if name.startswith("sakuratts/") and not name.endswith("/")}
        if actual_product != set(product):
            raise ValueError("wheel product files differ from source snapshot: " +
                             ", ".join(sorted(actual_product.symmetric_difference(product))))
        metadata_roots = {name.split("/")[0] for name in names if name.split("/")[0].endswith(".dist-info")}
        if len(metadata_roots) != 1:
            raise ValueError("wheel must contain exactly one dist-info directory")
        metadata_root = metadata_roots.pop()
        for name, relative in product.items():
            verify(archive.read(name), relative, "wheel")
        for relative in licenses:
            name = metadata_root + "/licenses/" + relative
            if name not in names:
                raise ValueError("wheel license missing: " + relative)
            verify(archive.read(name), relative, "wheel")
    return {"hashes_passed": True, "sdist_source_files": len(inventory),
            "wheel_python_files": len(product), "wheel_license_files": len(licenses)}


def build_preview(root, output, *, offline=False):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite the preview output directory")
    if any(output == root / name or root / name in output.parents for name in SOURCE_DIRS):
        raise ValueError("Preview output must be outside the selected source directories")
    uv_version = subprocess.run(["uv", "--version"], check=True, capture_output=True,
                                text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="sakuratts-preview-") as temporary:
        temporary = Path(temporary)
        source, dist = temporary / "source", temporary / "dist"
        inventory = stage_source(root, source)
        version = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        if not re.fullmatch(r"[A-Za-z0-9._+-]+", version):
            raise ValueError("Unsupported package version for a preview filename")
        revision = git_state(root)
        command = ["uv", "build", "--sdist", "--wheel", "--python", sys.executable,
                   "--no-python-downloads", "--out-dir", str(dist), str(source)]
        if offline:
            command.append("--offline")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        subprocess.run(command, cwd=source, env=environment, check=True)
        wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise ValueError("Build must produce exactly one wheel and one source distribution")
        distribution_validation = verify_distributions(wheels[0], sdists[0], inventory)
        payload = {"dist/" + path.name: path.read_bytes() for path in wheels + sdists}
        payload["QUICKSTART.md"] = (source / "docs/preview-release.md").read_bytes()
        payload["requirements/windows-runtime.txt"] = (source / "requirements/windows-runtime.txt").read_bytes()
        manifest = {"format": "sakuratts-developer-preview-v1", "version": version,
                    "git": revision, "python": sys.version.split()[0], "uv": uv_version,
                    "offline_build": offline, "source_files": inventory,
                    "distribution_validation": distribution_validation,
                    "payload_files": {name: {"bytes": len(data), "sha256": sha256(data)}
                                      for name, data in sorted(payload.items())},
                    "scope": "Python wheel and source only; no models, CUDA DLLs, Python runtime or upload"}
        payload["release-manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        payload["SHA256SUMS"] = "".join(sha256(data) + "  " + name + "\n"
                                        for name, data in sorted(payload.items())).encode("utf-8")
        filename = "sakuratts-" + version + "-preview.zip"
        archive = temporary / filename
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in sorted(payload.items()):
                bundle.writestr(name, data)
        checksum = sha256(archive.read_bytes())
        output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(archive, output / filename)
        (output / (filename + ".sha256")).write_text(checksum + "  " + filename + "\n", encoding="utf-8")
    return {"archive": filename, "sha256": checksum, "version": version, "source_file_count": len(inventory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="A new output directory")
    parser.add_argument("--offline", action="store_true", help="Build using only cached build dependencies")
    args = parser.parse_args()
    result = build_preview(Path(__file__).resolve().parents[1], args.output, offline=args.offline)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
