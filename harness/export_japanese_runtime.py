#!/usr/bin/env python3
"""Export an experimental Japanese runtime at its final local path.

Uses only the standard library. Copies an explicit clean environment and four
converted packages; never downloads, loads models or follows provenance paths.
The resulting venv is created at the output path, not copied from the old venv.
Export success does not establish isolated synthesis or general relocatability.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE_FORMATS = {
    "frontend": "sakuratts-japanese-frontend-resources-v1",
    "reference": "sakuratts-prepared-reference-v1",
    "gpt": "sakuratts-gpt-fp32-v1",
    "sovits": "sakuratts-sovits-decode-fp32-v1",
}
IGNORED = {"__pycache__", ".DS_Store"}
INSTALL_TOOL_ENTRIES = {"pip", "setuptools", "pkg_resources", "_distutils_hack",
                        "distutils-precedence.pth"}
PROBE = """
import json, sys, sysconfig
print(json.dumps({
    'version': list(sys.version_info[:3]), 'executable': sys.executable,
    'base_executable': sys._base_executable, 'prefix': sys.prefix,
    'base_prefix': sys.base_prefix, 'exec_prefix': sys.exec_prefix,
    'base_exec_prefix': sys.base_exec_prefix, 'sys_path': sys.path,
    'paths': sysconfig.get_paths(),
    'build_paths': {k: sysconfig.get_config_var(k)
                    for k in ('prefix', 'exec_prefix', 'BINDIR', 'LIBDIR', 'LIBPL')},
}))
"""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def probe(python, *, no_site=False):
    command = [str(python), "-I", "-B"]
    if no_site:
        command.append("-S")
    return json.loads(subprocess.check_output(command + ["-c", PROBE], text=True))


def normalized_name(name):
    return name.lower().replace("_", "-").replace(".", "-")


def check_distributions(site_packages):
    expected = {}
    for line in (PROJECT / "requirements-mlx-japanese.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = line.split("==")
            expected[normalized_name(name)] = version
    installed = {normalized_name(d.metadata["Name"]): d.version
                 for d in metadata.distributions(path=[str(site_packages)])}
    if ({name: installed.get(name) for name in expected} != expected
            or set(installed) - set(expected) - {"pip", "setuptools"}):
        raise ValueError("Use the clean environment pinned by requirements-mlx-japanese.txt")
    return installed


def install_tool_entries(site_packages):
    """Only omit named installers; keep runtime distribution metadata intact."""
    entries = {name for name in INSTALL_TOOL_ENTRIES if (site_packages / name).exists()}
    for pattern in ("pip-*.dist-info", "setuptools-*.dist-info"):
        entries.update(path.name for path in site_packages.glob(pattern))
    return entries


def copy_tree(source, target, origins, *, output, exclude=()):
    """Preserve in-tree links; reject links that would retain an old dependency."""
    source_root, target_root = source, target
    excluded = set(exclude)

    def copy_entry(old, new):
        if old.relative_to(source_root).as_posix() in excluded:
            return
        if old.name in IGNORED or old.suffix in {".pyc", ".pyo"}:
            return
        relative = new.relative_to(output).as_posix()
        if old.is_symlink():
            resolved = old.resolve(strict=True)
            if not resolved.is_relative_to(source_root):
                raise ValueError(f"Source symlink leaves the copied tree: {old} -> {resolved}")
            mapped = target_root / resolved.relative_to(source_root)
            new.symlink_to(os.path.relpath(mapped, new.parent), target_is_directory=resolved.is_dir())
            origins[relative] = {"path": str(old), "link_target": os.readlink(old),
                                 "resolved_target": str(resolved)}
        elif old.is_dir():
            new.mkdir(exist_ok=False)
            for child in sorted(old.iterdir()):
                copy_entry(child, new / child.name)
        elif old.is_file():
            before = sha256(old)
            shutil.copy2(old, new)
            if sha256(new) != before:
                raise ValueError(f"Copy differs from source: {old}")
            origins[relative] = {"path": str(old), "sha256": before}
        else:
            raise ValueError(f"Unsupported source entry: {old}")

    copy_entry(source, target)


def inventory(output, origins):
    files, totals = {}, {"regular_file_bytes": 0, "symlink_text_bytes": 0}
    for directory, directories, filenames in os.walk(output, followlinks=False):
        root = Path(directory)
        names = filenames + [name for name in directories if (root / name).is_symlink()]
        for name in sorted(names):
            path = root / name
            relative = path.relative_to(output).as_posix()
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                size = path.lstat().st_size
                entry = {"type": "symlink", "logical_bytes": size, "target": os.readlink(path),
                         "resolved_target": str(resolved), "target_within_bundle": resolved.is_relative_to(output)}
                if resolved.is_file():
                    entry["target_file_bytes"] = resolved.stat().st_size
                totals["symlink_text_bytes"] += size
            else:
                size = path.stat().st_size
                entry = {"type": "file", "logical_bytes": size, "sha256": sha256(path)}
                totals["regular_file_bytes"] += size
                if relative in origins and entry["sha256"] != origins[relative].get("sha256"):
                    raise ValueError(f"Copied file changed before inventory: {relative}")
            if relative in origins:
                entry["source"] = origins[relative]
            files[relative] = entry
    totals["logical_bytes_including_symlink_text"] = sum(totals.values())
    totals["file_and_symlink_count"] = len(files)
    return dict(sorted(files.items())), totals


def export(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    output = args.output.resolve()
    venv = args.venv.resolve(strict=True)
    if not (venv / "pyvenv.cfg").is_file():
        raise ValueError("--venv must identify an existing virtual environment")
    original_python = probe(venv / "bin/python", no_site=True)
    if original_python["version"][:2] != [3, 11] or sys.platform != "darwin":
        raise ValueError("This experiment currently supports macOS with CPython 3.11")
    base = Path(original_python["base_prefix"]).resolve(strict=True)
    base_python = Path(original_python["base_executable"]).resolve(strict=True)
    if not base_python.is_relative_to(base):
        raise ValueError("Base interpreter executable is outside its base prefix")
    site_packages = venv / "lib/python3.11/site-packages"
    installed = check_distributions(site_packages)
    site_exclusions = set() if args.include_install_tools else install_tool_entries(site_packages)
    base_exclusions = set()
    if not args.include_install_tools:
        base_exclusions = {"lib/python3.11/ensurepip", "bin/pip", "bin/pip3", "bin/pip3.11"}
        base_exclusions.update("lib/python3.11/site-packages/" + name
                               for name in install_tool_entries(base / "lib/python3.11/site-packages"))
    packages = {name: getattr(args, name + "_package").resolve(strict=True) for name in PACKAGE_FORMATS}
    manifests = {name: read_json(path / "manifest.json") for name, path in packages.items()}
    for name, expected in PACKAGE_FORMATS.items():
        if manifests[name]["format"] != expected:
            raise ValueError(f"Unexpected {name} package format")
    inputs = [PROJECT, venv, base, *packages.values()]
    if any(output.is_relative_to(path) for path in inputs):
        raise ValueError("Output must be outside the checkout and all input trees")
    # Create the environment in its final location: venv records absolute paths.
    output.mkdir(parents=True, exist_ok=False)
    origins = {}
    copy_tree(PROJECT / "src", output / "src", origins, output=output)
    (output / "scripts").mkdir()
    copy_tree(PROJECT / "scripts/synthesize_japanese.py", output / "scripts/synthesize_japanese.py",
              origins, output=output)
    copy_tree(PROJECT / "requirements-mlx-japanese.txt", output / "requirements-mlx-japanese.txt",
              origins, output=output)
    copy_tree(PROJECT / "docs/third-party", output / "third-party", origins, output=output)
    copy_tree(Path(__file__).resolve(), output / "exporter.py", origins, output=output)
    (output / "resources").mkdir()
    for name, path in packages.items():
        copy_tree(path, output / "resources" / name, origins, output=output)
    if args.python_mode == "bundled":
        copy_tree(base, output / "python", origins, output=output, exclude=base_exclusions)
        base_python = output / "python" / base_python.relative_to(base)
        copied_probe = probe(base_python, no_site=True)
        for key in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
            if Path(copied_probe[key]).resolve() != output / "python":
                raise ValueError(f"Copied base Python did not relocate {key}")
        for path in [*copied_probe["sys_path"], *copied_probe["paths"].values()]:
            if not Path(path).resolve().is_relative_to(output / "python"):
                raise ValueError(f"Copied base Python retains an external runtime path: {path}")
    else:
        copied_probe = None
    command = [str(base_python), "-I", "-B", "-m", "venv", "--without-pip", str(output / "venv")]
    subprocess.run(command, check=True)
    new_site = output / "venv/lib/python3.11/site-packages"
    # venv creates this directory empty; never replace an existing package.
    if any(new_site.iterdir()):
        raise ValueError("New venv unexpectedly contains packages")
    new_site.rmdir()
    copy_tree(site_packages, new_site, origins, output=output, exclude=site_exclusions)
    exported_distributions = check_distributions(new_site)
    runtime_probe = probe(output / "venv/bin/python")
    if Path(runtime_probe["prefix"]).resolve() != output / "venv":
        raise ValueError("New interpreter did not activate its virtual environment")
    if args.python_mode == "bundled":
        for path in [*runtime_probe["sys_path"], *runtime_probe["paths"].values()]:
            if not Path(path).resolve().is_relative_to(output):
                raise ValueError(f"New venv retains an external runtime path: {path}")
    for name in ("output", "cache", "tmp"):
        (output / name).mkdir()
    files, totals = inventory(output, origins)
    manifest = {
        "format": "sakuratts-japanese-runtime-export-v1", "status": "exported_not_synthesis_validated",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "output": str(output),
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "python_mode": args.python_mode, "base_python_bundled": args.python_mode == "bundled",
        "python": {"source_probe": original_python, "copied_base_probe": copied_probe,
                   "runtime_probe": runtime_probe, "venv_creation_argv": command},
        "distributions": exported_distributions, "source_distributions": installed,
        "include_install_tools": args.include_install_tools,
        "omitted_install_tool_entries": {
            "site_packages": sorted(site_exclusions),
            "bundled_python": sorted(base_exclusions) if args.python_mode == "bundled" else [],
        },
        "packages": {name: {"path": f"resources/{name}", "format": data["format"],
                            "manifest_sha256": sha256(output / "resources" / name / "manifest.json")}
                     for name, data in manifests.items()},
        "files": files, "totals": totals,
        "boundaries": [
            "No frontend, model import or synthesis performed by this exporter; OS isolation still unverified.",
            "Venv is created at this final absolute output path; moving it again requires recreation.",
            "Apple arm64/macOS 26 MLX wheels and macOS system libraries remain platform dependencies.",
            "CPython build_paths can retain the build prefix; runtime prefixes are reported separately.",
            "Source paths and copied provenance are evidence only; no historical run or original checkpoint is copied.",
            "Install tools are optional; the default inference bundle omits pip/setuptools and bundled ensurepip. Re-export to change dependencies.",
            "Logical sizes exclude this manifest, filesystem allocation, system libraries and future output/cache files; symlink target bytes are not added twice.",
            ("Base CPython is copied; independent operation still requires denied-old-path synthesis verification."
             if args.python_mode == "bundled" else "Base CPython remains external and its files are excluded from bundle size."),
        ],
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PACKAGE_FORMATS:
        parser.add_argument("--" + name + "-package", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-mode", choices=("bundled", "external"), default="bundled",
                        help="Copy base CPython, or explicitly keep it as an external dependency")
    parser.add_argument("--include-install-tools", action="store_true",
                        help="Keep pip/setuptools and bundled ensurepip for the full-environment comparison")
    args = parser.parse_args()
    result = export(args)
    print(json.dumps({"output": result["output"], "status": result["status"], "totals": result["totals"]}))
