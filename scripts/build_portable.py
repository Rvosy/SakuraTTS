"""Assemble a model-free Windows bundle from explicit local inputs, offline."""

import argparse
import base64
import csv
from email.parser import Parser
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tomllib
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

BACKEND_EXTRAS = {("windows-x64", "cuda"): "nvidia"}
LANGUAGE_EXTRAS = {"ja": "japanese"}
SERVICE_EXTRAS = {"http": "server"}


def read_recipe(path):
    """Select implemented payloads before inspecting any large local inputs."""
    recipe = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if (recipe["target"], recipe["backend"]) not in BACKEND_EXTRAS:
        raise ValueError("Portable assembly currently implements only windows-x64 with backend cuda")
    if not recipe["languages"] or set(recipe["languages"]) - LANGUAGE_EXTRAS.keys():
        raise ValueError("Portable assembly currently implements only the ja language component")
    if set(recipe["services"]) - SERVICE_EXTRAS.keys():
        raise ValueError("Unknown service component; supported services: http")
    return {key: recipe[key] for key in ("target", "backend", "languages", "services", "workers")}


def main_requirements(project, recipe):
    """Platform, language and service extras are independent package choices."""
    extras = [BACKEND_EXTRAS[recipe["target"], recipe["backend"]],
              *(LANGUAGE_EXTRAS[name] for name in recipe["languages"]),
              *(SERVICE_EXTRAS[name] for name in recipe["services"])]
    requirements = list(project["dependencies"])
    for extra in extras:
        requirements.extend(project["optional-dependencies"][extra])
    return requirements


def launch_files(recipe):
    names = ["launcher.py", "check_runtime.py", "sakuratts.bat", "check-runtime.bat", "README.md"]
    if "http" in recipe["services"]:
        names.append("start-server.bat")
    return names


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def regular(path):
    attrs = getattr(path.lstat(), "st_file_attributes", 0)
    if path.is_symlink() or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise ValueError("Linked build input is not allowed: " + str(path))
    return path.is_file()


def tree(path):
    for child in sorted(path.iterdir()):
        if child.name in ("__pycache__", ".git", "site-packages"):
            continue
        regular(child)
        if child.is_dir():
            yield from tree(child)
        elif child.suffix not in (".pyc", ".pyo"):
            yield child


def distributions(site):
    result = {}
    for metadata in sorted(site.glob("*.dist-info/METADATA")):
        values = Parser().parsestr(metadata.read_text(encoding="utf-8"))
        name = canonicalize_name(values["Name"])
        if name in result:
            raise ValueError("Multiple installed distributions named " + name)
        result[name] = (metadata.parent, values)
    return result


def dependency_names(installed, roots, python_version):
    env = dict(default_environment(), python_version=python_version,
               python_full_version=python_version + ".0", sys_platform="win32",
               platform_system="Windows", platform_machine="AMD64", extra="")
    selected, visited, pending = set(), set(), list(roots)
    while pending:
        requirement = Requirement(pending.pop())
        name = canonicalize_name(requirement.name)
        if name not in installed:
            raise ValueError("Local runtime dependency is missing: " + name)
        if not requirement.specifier.contains(installed[name][1]["Version"], prereleases=True):
            raise ValueError("Local runtime dependency version mismatch: " + str(requirement))
        visit = (name, tuple(sorted(requirement.extras)))
        if visit in visited:
            continue
        visited.add(visit)
        selected.add(name)
        for value in installed[name][1].get_all("Requires-Dist", []):
            req = Requirement(value)
            if req.marker is None or any(req.marker.evaluate(dict(env, extra=extra)) for extra in ("", *requirement.extras)):
                pending.append(str(req))
    return sorted(selected)


class Plan:
    def __init__(self):
        self.files = {}
        self.components = {}
        self.shared_files = {}
        self.release = {}
        self.workers = {}

    def add(self, source, destination, component, expected=None):
        source = Path(source)
        destination = PurePosixPath(destination)
        if destination.is_absolute() or ".." in destination.parts or ":" in str(destination) or "\\" in str(destination):
            raise ValueError("Unsafe bundle destination: " + str(destination))
        if not regular(source):
            raise FileNotFoundError(source)
        checksum = digest(source)
        if expected is not None and checksum != expected:
            raise ValueError("Local input checksum differs from its manifest: " + str(source))
        row = {"source": str(source.resolve()), "sha256": checksum,
               "bytes": source.stat().st_size, "component": component}
        key = str(destination)
        if key in self.files and self.files[key]["sha256"] != checksum:
            raise ValueError("Conflicting bundle files: " + key)
        self.files[key] = row

    def package(self, site, directory, metadata, target):
        name, version = metadata["Name"], metadata["Version"]
        component = target + ":" + name
        self.components[component] = {"name": name, "version": version, "source": "local-installed-RECORD"}
        with (directory / "RECORD").open(encoding="utf-8", newline="") as stream:
            for relative, checksum, _ in csv.reader(stream):
                if not relative or "\\" in relative or PurePosixPath(relative).is_absolute() or ":" in relative:
                    raise ValueError("Unsafe RECORD entry: " + relative)
                parts = PurePosixPath(relative).parts
                if ".." in parts:
                    continue  # Environment Scripts/include are not library payloads.
                if (relative.endswith((".pyc", ".pyo", ".pth")) or "__pycache__" in parts
                        or parts[-1] in ("direct_url.json", "uv_cache.json", "uv_build.json")):
                    continue
                path = site.joinpath(*parts)
                if site.resolve() not in path.resolve(strict=True).parents:
                    raise ValueError("RECORD input escaped site-packages: " + relative)
                expected = None
                if checksum:
                    algorithm, encoded = checksum.split("=", 1)
                    if algorithm != "sha256":
                        raise ValueError("Unsupported RECORD checksum: " + algorithm)
                    expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
                self.add(path, target + "/" + relative, component, expected)


def trim_main_runtime(plan):
    """Omit installation tools, desktop demos and package tests from the server."""
    stdlib = "runtime/main/Lib/"
    numpy = stdlib + "site-packages/numpy/"
    excluded = tuple(stdlib + name + "/" for name in ("test", "idlelib", "tkinter", "ensurepip"))
    for name in list(plan.files):
        if (name.startswith(excluded)
                or (name.startswith(numpy) and "tests" in PurePosixPath(name).parts)):
            del plan.files[name]


def make_plan(args):
    plan = Plan()
    recipe = read_recipe(getattr(args, "recipe", None) or args.root / "packaging/recipes/windows-nvidia-ja.toml")
    plan.release = {key: recipe[key] for key in ("target", "backend", "languages", "services")}
    plan.release["preparation"] = getattr(args, "preparation", None) is not None
    plan.workers = recipe["workers"]
    # Copy the base interpreter, never a venv trampoline or pyvenv.cfg.
    for name in ("python.exe", "python3.dll", "python311.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"):
        plan.add(args.python_base / name, "runtime/main/" + name, "cpython-3.11")
    for directory in ("Lib", "DLLs"):
        for source in tree(args.python_base / directory):
            plan.add(source, "runtime/main/" + source.relative_to(args.python_base).as_posix(), "cpython-3.11")
    main = distributions(args.main_site)
    project = tomllib.loads((args.root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for name in dependency_names(main, main_requirements(project, recipe), "3.11"):
        plan.package(args.main_site, *main[name], "runtime/main/Lib/site-packages")
    trim_main_runtime(plan)

    # The private acoustic ABI is copied only through its existing hash manifest.
    worker = json.loads((args.worker / "runtime-manifest.json").read_text(encoding="utf-8"))
    for name, row in worker["files"].items():
        plan.add(args.worker / name, "runtime/acoustic/" + name, "acoustic-worker", row["sha256"])

    # The worker searches the main NVIDIA wheel directories as well as its own cuda/.
    main_dlls = {Path(name).name: (name, row) for name, row in plan.files.items()
                 if name.startswith("runtime/main/Lib/site-packages/nvidia/") and name.endswith(".dll")}
    for name, row in list(plan.files.items()):
        same = main_dlls.get(Path(name).name)
        if name.startswith("runtime/acoustic/cuda/") and same and same[1]["sha256"] == row["sha256"]:
            plan.shared_files[name] = same[0]
            del plan.files[name]
    plan.add(args.ffmpeg, "runtime/bin/ffmpeg.exe", "ffmpeg-local")
    for name in ("GPT-SoVITS-LICENSE.txt", "OpenJTalk-dictionary-COPYING.txt",
                 "pyopenjtalk-LICENSE.md", "SudachiDict-LEGAL.txt", "VITS-LICENSE.txt", "LGPL-3.0.txt", "GPL-3.0.txt"):
        plan.add(args.root / "docs/third-party" / name, "licenses/sakuratts/" + name, "third-party-notices")
    plan.add(args.root / "LICENSE", "licenses/SakuraTTS-LICENSE.txt", "sakuratts")
    for name in ("fp32.json", "fp16.json", "low-vram.json", "minimum-vram.json"):
        plan.add(args.root / "examples" / name, "configs/" + name, "inference-profiles")
    preparation = getattr(args, "preparation", None)
    if preparation is not None:
        add_preparation(plan, preparation)
    for name in ("acoustic", "frontend"):
        if plan.workers[name] not in plan.files:
            raise ValueError("Worker is missing from the selected release payload: " + name)
    return plan


def add_preparation(plan, source):
    """Include only the preparation component's verified release inventory."""
    source = Path(source).resolve(strict=True)
    manifest_path = source / "preparation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "sakuratts-preparation-bundle-v1":
        raise ValueError("Unsupported preparation component manifest")
    marker = json.loads((source / "preparation.json").read_text(encoding="utf-8"))
    if marker.get("format") != "sakuratts-preparation-v1":
        raise ValueError("Unsupported preparation component marker")
    for field in ("python", "official_source", "language_model", "cnhubert"):
        relative = marker[field]
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or ":" in relative or "\\" in relative:
            raise ValueError("Unsafe preparation component path: " + relative)
        resolved = (source / path).resolve(strict=True)
        if source not in resolved.parents:
            raise ValueError("Preparation component path escaped its root")
        if ((resolved.is_file() and relative not in manifest["files"])
                or (resolved.is_dir() and not any(name.startswith(relative.rstrip("/") + "/")
                                                 for name in manifest["files"]))):
            raise ValueError("Preparation component resource is missing from its inventory: " + relative)
    if "preparation.json" not in manifest["files"] or marker["python"] not in manifest["files"]:
        raise ValueError("Preparation component inventory is incomplete")
    for name, row in manifest["files"].items():
        if source not in (source / name).resolve(strict=True).parents:
            raise ValueError("Preparation inventory input escaped its root: " + name)
        plan.add(source / name, "runtime/preparation/" + name,
                 "preparation:" + row.get("component", "resource"), row["sha256"])
    plan.add(manifest_path, "runtime/preparation/preparation-manifest.json", "preparation-inventory")
    for name, component in manifest.get("components", {}).items():
        plan.components["preparation:" + name] = component


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def assemble(args, plan):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory: " + str(output))
    output.mkdir(parents=True)
    for index, (name, row) in enumerate(plan.files.items()):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["source"], path)
        if index % 2000 == 0:
            print(f"Copied {index}/{len(plan.files)} files", flush=True)
    # Only our independently built product wheel is allowed to install sakuratts.
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            parts = PurePosixPath(name).parts
            if ".." in parts or name.startswith("/") or ":" in name or "\\" in name or name.endswith(".pth"):
                raise ValueError("Unsafe wheel entry: " + name)
            target = output / "runtime/main/Lib/site-packages" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    write(output / "runtime/main/python311._pth", ".\nDLLs\nLib\nLib/site-packages\n")
    # Marker read only by the explicit portable launcher.
    write(output / "runtime/portable.json", json.dumps({"format": "sakuratts-portable-v1",
        "has_preparation": getattr(args, "preparation", None) is not None,
        "workers": plan.workers, "release": plan.release}))
    templates = args.root / "scripts/portable"
    for name in launch_files(plan.release):
        shutil.copyfile(templates / name, output / name)
    for directory in ("models", "configs", "logs", "cache"):
        (output / directory).mkdir(exist_ok=True)
        write(output / directory / ".keep", "")
    write(output / "configs/tts_infer.example.yaml", "custom:\n  version: v2ProPlus\n  device: cuda\n  is_half: false\n  t2s_weights_path: models/your-gpt.ckpt\n  vits_weights_path: models/your-sovits.pth\n")
    notice = subprocess.run([str(args.ffmpeg), "-L"], capture_output=True, text=True, check=True)
    write(output / "licenses/FFmpeg-build-and-license.txt", notice.stderr + notice.stdout)
    inventory = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            name = path.relative_to(output).as_posix()
            row = plan.files.get(name, {})
            checksum = digest(path)
            if row and checksum != row["sha256"]:
                raise ValueError("Input changed while copying: " + name)
            inventory[name] = {"bytes": path.stat().st_size, "sha256": checksum,
                               "component": row.get("component", "product-or-generated")}
    manifest = {"format": "sakuratts-portable-bundle-v1", "network_used": False,
                "release": plan.release, "workers": plan.workers,
                "synthesis_models_included": False, "personal_references_included": False,
                "wheel_sha256": digest(args.wheel), "components": plan.components, "shared_files": plan.shared_files, "files": inventory,
                "bytes": sum(row["bytes"] for row in inventory.values())}
    write(output / "bundle-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "files": len(inventory), "bytes": manifest["bytes"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python-base", "main-site", "ffmpeg", "worker", "wheel", "output", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preparation", type=Path,
                        help="Optional verified offline component built by build_preparation.py")
    parser.add_argument("--recipe", type=Path,
                        help="Release recipe TOML (default: packaging/recipes/windows-nvidia-ja.toml)")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.root = Path(__file__).resolve().parents[1]
    plan = make_plan(args)
    # Local audit keeps source paths outside the public archive.
    write(args.audit, json.dumps({"release": plan.release, "workers": plan.workers, "files": plan.files, "components": plan.components, "shared_files": plan.shared_files}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"planned_files": len(plan.files), "bytes": sum(row["bytes"] for row in plan.files.values())}), flush=True)
    if not args.plan_only:
        assemble(args, plan)


if __name__ == "__main__":
    main()
