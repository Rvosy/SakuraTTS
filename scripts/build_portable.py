"""Assemble a model-free portable bundle from explicit local inputs, offline."""

import argparse
import csv
from email.parser import Parser
import hashlib
import importlib.util
import json
import re
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tomllib
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags, parse_tag
from packaging.utils import canonicalize_name

_spec = importlib.util.spec_from_file_location("macos_runtime", Path(__file__).with_name("macos_runtime.py"))
macos = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(macos)

BACKEND_EXTRAS = {("windows-x64", "cuda"): "nvidia",
                  ("windows-x64", "directml"): "directml",
                  ("windows-x64", "cpu"): "cpu",
                  ("macos-arm64", "mlx"): "mlx"}
LANGUAGE_EXTRAS = {"ja": "japanese", "en": "english"}
SERVICE_EXTRAS = {"http": "server"}


def read_recipe(path):
    """Select implemented payloads before inspecting any large local inputs."""
    recipe = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if (recipe["target"], recipe["backend"]) not in BACKEND_EXTRAS:
        raise ValueError("Portable assembly supports windows-x64 with backend cuda, directml or cpu, and macos-arm64 with mlx")
    if "ja" not in recipe["languages"] or set(recipe["languages"]) - LANGUAGE_EXTRAS.keys():
        raise ValueError("Portable language components require ja, with optional en")
    if set(recipe["services"]) - SERVICE_EXTRAS.keys():
        raise ValueError("Unknown service component; supported services: http")
    selected = {key: recipe[key] for key in ("target", "backend", "languages", "services")}
    backends = recipe.get("backends", ["cpu", "directml"] if recipe["backend"] == "directml" else [recipe["backend"]])
    if (not isinstance(backends, list) or not backends or recipe["backend"] not in backends
            or len(set(backends)) != len(backends)
            or any((recipe["target"], backend) not in BACKEND_EXTRAS for backend in backends)):
        raise ValueError("Bundle backends must include the default and match the target platform")
    selected["backends"] = backends
    if recipe["target"] == "macos-arm64":
        selected["minimum_macos"] = recipe["minimum_macos"]
    return selected | {"workers": recipe.get("workers", {})}


def main_requirements(project, recipe):
    """Platform, language and service extras are independent package choices."""
    backends = recipe["backends"]
    # DirectML supplies the CPU provider too. Never install two ORT distributions
    # over the same onnxruntime module; CUDA acoustics keep their private worker.
    extras = [*(BACKEND_EXTRAS[recipe["target"], backend] for backend in backends
                if not (backend == "cpu" and "directml" in backends)),
              *("japanese-text" if name == "ja" and "directml" in backends else LANGUAGE_EXTRAS[name]
                for name in recipe["languages"]),
              *(SERVICE_EXTRAS[name] for name in recipe["services"])]
    requirements = list(project["dependencies"])
    visited = set()
    while extras:
        extra = extras.pop()
        if extra in visited:
            continue
        visited.add(extra)
        for value in project["optional-dependencies"][extra]:
            requirement = Requirement(value)
            if canonicalize_name(requirement.name) == canonicalize_name(project["name"]):
                extras.extend(requirement.extras)
            else:
                requirements.append(value)
    return requirements


def launch_files(recipe):
    if recipe["target"] == "macos-arm64":
        return ["launcher.py", "check_runtime.py", "sakuratts.command", "check-runtime.command", "README-macos.md"] + (
            ["start-server.command"] if "http" in recipe["services"] else [])
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


def windows_wheel_compatible(directory, python_version):
    """Reject native libraries built for another interpreter or platform."""
    wheel = directory / "WHEEL"
    if not wheel.is_file():
        raise ValueError("Windows runtime input requires wheel metadata: " + str(directory))
    tags = {tag for line in wheel.read_text(encoding="utf-8").splitlines() if line.startswith("Tag: ")
            for tag in parse_tag(line[5:])}
    version = tuple(map(int, python_version.split(".")))
    interpreter = "cp" + python_version.replace(".", "")
    supported = set(cpython_tags(version, abis=[interpreter], platforms=["win_amd64"]))
    supported.update(compatible_tags(version, interpreter=interpreter, platforms=["win_amd64"]))
    if not tags & supported:
        raise ValueError(f"{directory.name} has no Windows x64 wheel compatible with Python {python_version}")


def dependency_names(installed, roots, python_version, target="windows-x64"):
    env = dict(default_environment(), python_version=python_version,
               python_full_version=python_version + ".0", sys_platform="win32",
               platform_system="Windows", platform_machine="AMD64", extra="")
    if target == "macos-arm64":
        env.update(sys_platform="darwin", platform_system="Darwin", platform_machine="arm64", os_name="posix")
    else:
        env["os_name"] = "nt"
    selected, visited, pending = set(), set(), list(roots)
    while pending:
        requirement = Requirement(pending.pop())
        if requirement.marker is not None and not requirement.marker.evaluate(env):
            continue
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
                req.marker = None
                pending.append(str(req))
    return sorted(selected)


class Plan:
    def __init__(self):
        self.files = {}
        self.components = {}
        self.shared_files = {}
        self.release = {}
        self.workers = {}
        self.python_stem = "python311"
        self.python_paths = [".", "DLLs", "Lib", "Lib/site-packages"]
        self.site_target = "runtime/main/Lib/site-packages"

    def add(self, source, destination, component):
        source = Path(source)
        destination = PurePosixPath(destination)
        if destination.is_absolute() or ".." in destination.parts or ":" in str(destination) or "\\" in str(destination):
            raise ValueError("Unsafe bundle destination: " + str(destination))
        if not regular(source):
            raise FileNotFoundError(source)
        checksum = digest(source)
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
            for relative, _, _ in csv.reader(stream):
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
                self.add(path, target + "/" + relative, component)


def trim_main_runtime(plan):
    """Omit installation tools, desktop demos and package tests from the server."""
    stdlib = "runtime/main/Lib/"
    numpy = stdlib + "site-packages/numpy/"
    excluded = tuple(stdlib + name + "/" for name in ("test", "idlelib", "tkinter", "ensurepip"))
    for name in list(plan.files):
        if (name.startswith(excluded)
                or (name.startswith(numpy) and "tests" in PurePosixPath(name).parts)):
            del plan.files[name]


def add_interpreter(plan, base, target="", vc_runtime=None):
    """Copy a base or embeddable interpreter without its installed packages."""
    candidates = [path for path in base.glob("python3*.dll") if re.fullmatch(r"python3\d+", path.stem)]
    if len(candidates) != 1:
        raise ValueError("Provide one explicit Windows CPython interpreter (python3X.dll)")
    stem = candidates[0].stem
    prefix = target.rstrip("/") + "/" if target else ""
    for name in ("python.exe", "python3.dll", stem + ".dll", "LICENSE.txt"):
        plan.add(base / name, prefix + name, "cpython")
    if vc_runtime is None:
        for name in ("vcruntime140.dll", "vcruntime140_1.dll"):
            plan.add(base / name, prefix + name, "cpython")
    else:
        # Use the redistributable CRT input, not DLLs copied from System32.
        for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
            if not (vc_runtime / name).is_file():
                raise FileNotFoundError(vc_runtime / name)
        for source in sorted(vc_runtime.glob("*.dll")):
            plan.add(source, prefix + source.name, "msvc-runtime")
    if (base / (stem + ".zip")).is_file():
        plan.add(base / (stem + ".zip"), prefix + stem + ".zip", "cpython")
        for source in sorted(base.iterdir()):
            if source.suffix.lower() in (".pyd", ".dll") and prefix + source.name not in plan.files:
                plan.add(source, prefix + source.name, "cpython")
        paths = [stem + ".zip", ".", "Lib/site-packages"]
    else:
        for directory in ("Lib", "DLLs"):
            for source in tree(base / directory):
                plan.add(source, prefix + source.relative_to(base).as_posix(), "cpython")
        paths = [".", "DLLs", "Lib", "Lib/site-packages"]
    return stem, "3." + stem.removeprefix("python3"), paths


def make_plan(args):
    plan = Plan()
    recipe = read_recipe(args.recipe or args.root / "packaging/recipes/windows-x64.toml")
    plan.release = {key: recipe[key] for key in ("target", "backend", "languages", "services")}
    plan.release["preparation"] = args.preparation is not None
    plan.release["backends"] = recipe["backends"]
    plan.workers = recipe["workers"]
    apple = recipe["target"] == "macos-arm64"
    if apple:
        version, plan.site_target = macos.add_interpreter(plan, args.python_base, "runtime/main")
        plan.release.update(minimum_macos=recipe["minimum_macos"], python_executable="runtime/main/bin/python3")
    else:
        if args.vc_runtime is None:
            raise ValueError("Windows assembly requires --vc-runtime")
        plan.python_stem, version, plan.python_paths = add_interpreter(plan, args.python_base, "runtime/main", args.vc_runtime)
    main = distributions(args.main_site)
    ort_packages = set(main) & {"onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"}
    if len(ort_packages) > 1:
        raise ValueError("ORT distributions share module files; use a clean input environment with only one: " + ", ".join(sorted(ort_packages)))
    project = tomllib.loads((args.root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    plan.release["version"] = project["version"]
    commit = subprocess.run(["git", "-C", str(args.root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(args.root), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    plan.release.update(source_commit=commit, source_dirty=bool(dirty), python=version)
    for name in dependency_names(main, main_requirements(project, recipe), version, recipe["target"]):
        if apple:
            macos.wheel_compatible(main[name][0], recipe["minimum_macos"], version)
        else:
            windows_wheel_compatible(main[name][0], version)
        plan.package(args.main_site, *main[name], plan.site_target)
    trim_main_runtime(plan)

    # The private acoustic ABI is copied only through its existing hash manifest.
    if plan.workers:
        if args.worker is None:
            raise ValueError("The selected recipe requires --worker")
        worker = json.loads((args.worker / "runtime-manifest.json").read_text(encoding="utf-8"))
        for name, row in worker["files"].items():
            plan.add(args.worker / name, "runtime/acoustic/" + name, "acoustic-worker")

    # The worker searches the main NVIDIA wheel directories as well as its own cuda/.
    main_dlls = {Path(name).name: (name, row) for name, row in plan.files.items()
                 if name.startswith("runtime/main/Lib/site-packages/nvidia/") and name.endswith(".dll")}
    for name, row in list(plan.files.items()):
        same = main_dlls.get(Path(name).name)
        if name.startswith("runtime/acoustic/cuda/") and same and same[1]["sha256"] == row["sha256"]:
            plan.shared_files[name] = same[0]
            del plan.files[name]
    plan.add(args.ffmpeg, "runtime/bin/ffmpeg" if apple else "runtime/bin/ffmpeg.exe", "ffmpeg-local")
    for name in ("GPT-SoVITS-LICENSE.txt", "OpenJTalk-dictionary-COPYING.txt",
                 "pyopenjtalk-LICENSE.md", "SudachiDict-LEGAL.txt", "VITS-LICENSE.txt", "LGPL-3.0.txt", "GPL-3.0.txt"):
        plan.add(args.root / "docs/third-party" / name, "licenses/sakuratts/" + name, "third-party-notices")
    plan.add(args.root / "LICENSE", "licenses/SakuraTTS-LICENSE.txt", "sakuratts")
    profiles = []
    if "cuda" in recipe["backends"]:
        profiles.extend(("fp32.json", "fp16.json", "low-vram.json", "minimum-vram.json"))
    if "cpu" in recipe["backends"]:
        profiles.append("cpu.json")
    if "directml" in recipe["backends"]:
        profiles.append("directml.json")
    for name in profiles:
        plan.add(args.root / "examples" / name, "configs/" + name, "inference-profiles")
    if apple:
        plan.release["binary_audit"] = macos.audit(plan, recipe["minimum_macos"], "runtime/main/bin/python3")
    if args.preparation is not None:
        add_preparation(plan, args.preparation)
    for name, path in plan.workers.items():
        if path not in plan.files:
            raise ValueError("Worker is missing from the selected release payload: " + name)
    return plan


def add_preparation(plan, source):
    """Include only the preparation component's verified release inventory."""
    source = Path(source).resolve(strict=True)
    manifest_path = source / "preparation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "sakuratts-preparation-bundle-v1":
        raise ValueError("Unsupported preparation component manifest")
    release = manifest.get("release", {})
    target = plan.release.get("target", "windows-x64")
    preparation_target = release.get("target", "windows-x64")
    if preparation_target != target:
        raise ValueError("Preparation component target " + preparation_target + " does not match bundle target " + target)
    if target == "macos-arm64":
        if tuple(map(int, release["minimum_macos"].split("."))) > tuple(map(int, plan.release["minimum_macos"].split("."))):
            raise ValueError("Preparation requires a newer macOS version than the main runtime")
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
                 "preparation:" + row.get("component", "resource"))
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
        shutil.copy2(row["source"], path)
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
            target = output / plan.site_target / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    apple = plan.release["target"] == "macos-arm64"
    if not apple:
        write(output / ("runtime/main/" + plan.python_stem + "._pth"), "\n".join(plan.python_paths) + "\n")
    # Marker read only by the explicit portable launcher.
    write(output / "runtime/portable.json", json.dumps({"format": "sakuratts-portable-v1",
        "has_preparation": plan.release["preparation"],
        "workers": plan.workers, "release": plan.release}))
    templates = args.root / "scripts/portable"
    for name in launch_files(plan.release):
        destination = output / ("README.md" if name == "README-macos.md" else name)
        shutil.copy2(templates / name, destination)
        if name.endswith(".command"):
            destination.chmod(0o755)
    for directory in ("models", "configs", "logs", "cache"):
        (output / directory).mkdir(exist_ok=True)
        write(output / directory / ".keep", "")
    backends = plan.release.get("backends", [plan.release["backend"]])
    for backend in backends:
        config = ("custom:\n  device: " + backend + "\n  is_half: false\n"
                  "  t2s_weights_path: models/your-gpt.ckpt\n  vits_weights_path: models/your-sovits.pth\n"
                  "sakuratts:\n  backend: " + backend + "\n")
        write(output / ("configs/tts_infer." + backend + ".example.yaml"), config)
        if backend == backends[0]:
            write(output / "configs/tts_infer.example.yaml", config)
    notice = subprocess.run([str(args.ffmpeg), "-L"], capture_output=True, text=True, check=True)
    write(output / "licenses/FFmpeg-build-and-license.txt", notice.stderr + notice.stdout)
    inventory = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            name = path.relative_to(output).as_posix()
            row = plan.files.get(name, {})
            checksum = digest(path)
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
    for name in ("python-base", "main-site", "ffmpeg", "wheel", "output", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vc-runtime", type=Path, help="Required Microsoft redistributable CRT for Windows")
    parser.add_argument("--worker", type=Path, help="Verified acoustic/frontend worker component (CUDA recipe)")
    parser.add_argument("--preparation", type=Path,
                        help="Optional verified offline component built by build_preparation.py")
    parser.add_argument("--recipe", type=Path,
                        help="Release recipe TOML (default: packaging/recipes/windows-x64.toml)")
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
