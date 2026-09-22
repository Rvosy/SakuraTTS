"""Build the optional offline preparation component from explicit local inputs.

The output is copied under runtime/preparation by build_portable.py. Installed
distributions are selected by dependency metadata and copied through RECORD;
the original environment, user voices and synthesis base models are excluded.
"""

import argparse
import ast
import base64
import csv
from email.parser import Parser
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil

from packaging.utils import canonicalize_name

_spec = importlib.util.spec_from_file_location("_portable_build_helpers", Path(__file__).with_name("build_portable.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
Plan, digest, regular, tree, write = (_helpers.Plan, _helpers.digest, _helpers.regular, _helpers.tree, _helpers.write)


def runtime_file(relative):
    """Keep executable/data payloads; preparation never builds native extensions."""
    path = PurePosixPath(relative)
    return not (set(path.parts) & {"include", "test", "tests"}
                or path.suffix in (".h", ".hpp", ".pxd", ".pyx", ".lib", ".a")
                or relative in ("onnxruntime/capi/onnxruntime_providers_cuda.dll",
                                "onnxruntime/capi/onnxruntime_providers_tensorrt.dll"))


class PreparationPlan(Plan):
    def __init__(self, record_overrides=None):
        super().__init__()
        self.record_overrides = record_overrides or {}
        self.applied_overrides = {}

    def package(self, site, directory, metadata, target):
        component = target + ":" + metadata["Name"]
        self.components[component] = {"name": metadata["Name"], "version": metadata["Version"], "source": "local-installed-RECORD"}
        with (directory / "RECORD").open(encoding="utf-8", newline="") as stream:
            for relative, checksum, _ in csv.reader(stream):
                parts = PurePosixPath(relative).parts
                if not relative or "\\" in relative or PurePosixPath(relative).is_absolute() or ":" in relative:
                    raise ValueError("Unsafe RECORD entry: " + relative)
                if ".." in parts or "__pycache__" in parts or relative.endswith((".pyc", ".pyo", ".pth")):
                    continue
                if parts[-1] in ("direct_url.json", "uv_cache.json", "uv_build.json"):
                    continue
                if not runtime_file(relative):
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
                if relative in self.record_overrides:
                    override = self.record_overrides[relative]
                    if (not isinstance(override, dict) or not isinstance(override.get("reason"), str)
                            or not override["reason"].strip() or not isinstance(override.get("sha256"), str)
                            or len(override["sha256"]) != 64 or override.get("record_sha256") != expected):
                        raise ValueError("RECORD override requires matching upstream hash, current sha256 and reason: " + relative)
                    self.applied_overrides[relative] = {**override, "record_sha256": expected}
                    expected = override["sha256"]
                self.add(path, target + "/" + relative, component, expected)

# These imports are required by the upstream reference-only TTS constructor,
# including modules imported eagerly even when their synthesis branch is unused.
ROOT_REQUIREMENTS = (
    "torch", "torchaudio", "transformers", "onnx", "soundfile", "librosa",
    "x-transformers", "fast-langdetect", "split-lang", "pytorch-lightning",
    "peft", "ffmpeg-python", "cn2an", "pypinyin", "jieba-fast", "jieba",
    "rotary-embedding-torch", "PyYAML", "tqdm", "matplotlib",
)
SOURCE_DIRECTORIES = (
    "GPT_SoVITS/AR", "GPT_SoVITS/BigVGAN", "GPT_SoVITS/eres2net",
    "GPT_SoVITS/f5_tts", "GPT_SoVITS/feature_extractor", "GPT_SoVITS/module",
    "GPT_SoVITS/text", "GPT_SoVITS/TTS_infer_pack", "tools/i18n",
    "tools/AP_BWE_main/datasets1", "tools/AP_BWE_main/models",
)
SOURCE_FILES = (
    "GPT_SoVITS/process_ckpt.py", "GPT_SoVITS/sv.py", "GPT_SoVITS/utils.py", "tools/audio_sr.py",
    "GPT_SoVITS/text/opencpop-strict.txt", "GPT_SoVITS/text/ja_userdic/userdict.csv",
    "GPT_SoVITS/text/ja_userdic/userdict.md5", "GPT_SoVITS/text/ja_userdic/user.dict",
    "LICENSE", "tools/AP_BWE_main/LICENSE",
)
# Public analysis resources only. No GPT, SoVITS, vocoder or upsampler weights.
AUXILIARY_FILES = (
    "GPT_SoVITS/pretrained_models/chinese-hubert-base/config.json",
    "GPT_SoVITS/pretrained_models/chinese-hubert-base/preprocessor_config.json",
    "GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin",
    "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    "GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin",
)
MARKER = {
    "format": "sakuratts-preparation-v1", "python": "python.exe", "official_source": "official",
    "cnhubert": "official/GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "language_model": "official/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin",
    "default_device": "cpu", "default_precision": "fp32",
}


def installed_distributions(site):
    result = _helpers.distributions(site)
    # Older offline distributions may contain directory eggs (not executable
    # .pth files). They are explicit sys.path entries in our isolated interpreter.
    for metadata_file in sorted(site.glob("*.egg/EGG-INFO/PKG-INFO")):
        regular(metadata_file.parent.parent)
        metadata = Parser().parsestr(metadata_file.read_text(encoding="utf-8"))
        name = canonicalize_name(metadata["Name"])
        if name in result:
            raise ValueError("Multiple installed distributions named " + name)
        requirements = metadata_file.with_name("requires.txt")
        if requirements.exists():
            for line in requirements.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if line.startswith("["):
                    raise ValueError("Build input egg has unsupported optional dependency metadata: " + name)
                metadata["Requires-Dist"] = line
        result[name] = metadata_file.parent, metadata
    return result


def add_interpreter(plan, base):
    candidates = sorted(path for path in base.glob("python3*.dll") if path.name != "python3.dll")
    if len(candidates) != 1 or candidates[0].stem not in ("python39", "python311"):
        raise ValueError("Preparation requires one explicit CPython 3.9 or 3.11 Windows interpreter")
    stem = candidates[0].stem
    for name in ("python.exe", "python3.dll", stem + ".dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"):
        plan.add(base / name, name, "cpython")
    if (base / (stem + ".zip")).is_file():
        plan.add(base / (stem + ".zip"), stem + ".zip", "cpython")
        # Windows embeddable CPython keeps its extension modules beside python.
        for source in sorted(base.iterdir()):
            if source.suffix.lower() in (".pyd", ".dll"):
                plan.add(source, source.name, "cpython")
        paths = [stem + ".zip", ".", "Lib/site-packages"]
    else:
        for directory in ("Lib", "DLLs"):
            for source in tree(base / directory):
                plan.add(source, source.relative_to(base).as_posix(), "cpython")
        paths = [".", "DLLs", "Lib", "Lib/site-packages"]
    return stem, "3.9" if stem == "python39" else "3.11", paths


def add_official_sources(plan, source):
    for name in SOURCE_DIRECTORIES:
        for path in tree(source / name):
            if path.suffix == ".py" or path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                plan.add(path, "official/" + path.relative_to(source).as_posix(), "gpt-sovits-source")
    for name in SOURCE_FILES:
        plan.add(source / name, "official/" + name, "gpt-sovits-source")
    for path in sorted((source / "tools/i18n/locale").glob("*.json")):
        plan.add(path, "official/" + path.relative_to(source).as_posix(), "gpt-sovits-source")
    for name in AUXILIARY_FILES:
        plan.add(source / name, "official/" + name, "public-analysis-resource")


def frontend_requirement(installed, python_version):
    if "pyopenjtalk" in installed:
        if python_version != "3.9":
            raise ValueError("Classic pyopenjtalk requires CPython 3.9 to match the portable frontend worker ABI")
        return "pyopenjtalk==0.3.4"
    return "pyopenjtalk-plus"


def preparation_distributions(site, runtime_site=None):
    installed = installed_distributions(site)
    origins = dict.fromkeys(installed, site)
    if runtime_site is not None:
        overrides = installed_distributions(runtime_site)
        installed.update(overrides)
        origins.update(dict.fromkeys(overrides, runtime_site))
    # The preparation worker always runs on CPU. A CUDA PyTorch wheel adds
    # several gigabytes which cannot be used by this component.
    version = ast.parse((origins["torch"] / "torch/version.py").read_text(encoding="utf-8"))
    cuda = next((node.value for node in version.body if
                 (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "cuda")
                 or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "cuda"
                                                          for target in node.targets))), None)
    if not isinstance(cuda, ast.Constant) or cuda.value is not None:
        raise ValueError("Preparation requires CPU PyTorch; supply CPU wheels with --runtime-site")
    return installed, origins


def make_plan(args):
    overrides = json.loads(args.record_overrides.read_text(encoding="utf-8")) if args.record_overrides else {}
    plan = PreparationPlan(overrides)
    stem, version, paths = add_interpreter(plan, args.python_base)
    installed, origins = preparation_distributions(args.site, getattr(args, "runtime_site", None))
    frontend = frontend_requirement(installed, version)
    # A local GPU wheel also contains the CPU backend. Omit its optional GPU
    # providers above rather than downloading another copy of the CPU runtime.
    onnxruntime = "onnxruntime" if "onnxruntime" in installed else "onnxruntime-gpu"
    names = _helpers.dependency_names(installed, [*ROOT_REQUIREMENTS, frontend, onnxruntime], version)
    for name in names:
        directory, metadata = installed[name]
        site = origins[name]
        if directory.name == "EGG-INFO":
            egg = directory.parent
            component = "Lib/site-packages:" + metadata["Name"]
            plan.components[component] = {"name": metadata["Name"], "version": metadata["Version"], "source": "local-directory-egg"}
            for path in tree(egg):
                relative = path.relative_to(site).as_posix()
                if path.suffix != ".pth" and runtime_file(relative):
                    plan.add(path, "Lib/site-packages/" + relative, component)
            paths.append("Lib/site-packages/" + egg.name)
        else:
            plan.package(site, directory, metadata, "Lib/site-packages")
    if set(plan.record_overrides) != set(plan.applied_overrides):
        raise ValueError("RECORD override does not match a selected distribution file")
    # pyopenjtalk classic downloads its dictionary after wheel installation, so
    # the original wheel RECORD does not describe this required public resource.
    frontend_site = origins[canonicalize_name(frontend.split("==")[0])]
    dictionary = frontend_site / "pyopenjtalk/open_jtalk_dic_utf_8-1.11"
    if not (dictionary / "sys.dic").is_file() or not (dictionary / "COPYING").is_file():
        raise FileNotFoundError("Local OpenJTalk dictionary and COPYING are required: " + str(dictionary))
    for path in tree(dictionary):
        plan.add(path, "Lib/site-packages/" + path.relative_to(frontend_site).as_posix(), "openjtalk-dictionary")
    add_official_sources(plan, args.official_source)
    return plan, stem, paths


def license_inventory(plan):
    result = {}
    for name, component in plan.components.items():
        files = [path for path, row in plan.files.items() if row["component"] == name and
                 (Path(path).name.upper().startswith(("LICENSE", "COPYING", "NOTICE")) or Path(path).name in ("METADATA", "PKG-INFO"))]
        result[name] = {**component, "notices": sorted(files)}
    result["cpython"] = {"notices": ["LICENSE.txt"]}
    result["gpt-sovits-source"] = {"notices": sorted(path for path in plan.files if path.startswith("official/") and Path(path).name.upper().startswith(("LICENSE", "NOTICE", "COPYING")))}
    result["openjtalk-dictionary"] = {"notices": ["Lib/site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11/COPYING"]}
    # The local upstream bundle does not supply weight license texts. Record the
    # gap rather than attributing the upstream source's MIT license to weights.
    result["public-analysis-resource"] = {
        "files": list(AUXILIARY_FILES), "notices": [],
        "license_status": "Model-specific redistribution notices are not supplied by these local inputs; review before public distribution.",
    }
    return {"format": "sakuratts-preparation-licenses-v1", "components": result}


def assemble(args, plan, stem, paths):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory: " + str(output))
    output.mkdir(parents=True)
    inventory = {}
    for index, (name, row) in enumerate(plan.files.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["source"], target)
        if digest(target) != row["sha256"]:
            raise ValueError("Input changed while copying: " + name)
        inventory[name] = {key: row[key] for key in ("bytes", "sha256", "component")}
        if index % 2000 == 0:
            print(f"Copied {index}/{len(plan.files)} files", flush=True)
    # The local upstream config can contain personal checkpoint paths. The
    # preparer hashes this placeholder but supplies its own explicit job config.
    generated = {stem + "._pth": "\n".join(paths) + "\n",
                 "official/GPT_SoVITS/configs/tts_infer.yaml": "{}\n",
                 "preparation.json": json.dumps(MARKER, indent=2) + "\n",
                 "licenses.json": json.dumps(license_inventory(plan), indent=2) + "\n"}
    for name, content in generated.items():
        path = output / name
        write(path, content)
        inventory[name] = {"bytes": path.stat().st_size, "sha256": digest(path), "component": "generated"}
    manifest = {"format": "sakuratts-preparation-bundle-v1", "network_used": False,
                "synthesis_models_included": False, "personal_references_included": False,
                "auxiliary_analysis_models_included": True,
                "source_sha256": digest(args.official_source / "GPT_SoVITS/TTS_infer_pack/TTS.py"),
                "record_overrides": getattr(plan, "applied_overrides", {}),
                "components": plan.components, "files": inventory,
                "bytes": sum(row["bytes"] for row in inventory.values())}
    write(output / "preparation-manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "files": len(inventory), "bytes": manifest["bytes"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python-base", "site", "official-source", "output", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--runtime-site", type=Path,
                        help="CPU torch, torchaudio and onnxruntime site-packages overriding --site")
    parser.add_argument("--record-overrides", type=Path,
                        help="Explicit JSON mapping of locally patched RECORD paths to sha256 and reason")
    args = parser.parse_args()
    plan, stem, paths = make_plan(args)
    write(args.audit, json.dumps({"files": plan.files, "components": plan.components}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"planned_files": len(plan.files), "bytes": sum(row["bytes"] for row in plan.files.values())}), flush=True)
    if not args.plan_only:
        assemble(args, plan, stem, paths)


if __name__ == "__main__":
    main()
