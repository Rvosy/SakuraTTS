#!/usr/bin/env python3
"""Prepare one Japanese V2Pro reference from raw audio with fixed official code.

This development tool always prepares conditions only; it never synthesizes a
target or trains a model. Torch stays in the worker process. --check-only reads
and hashes inputs/resources without importing any inference backend.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sakuratts.reference_condition import ARRAY_DTYPES, FORMAT, PreparedReference, sha256_array, sha256_file, validate_arrays

COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
OFFICIAL_FILES = (
    "GPT_SoVITS/TTS_infer_pack/TTS.py", "GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py",
    "GPT_SoVITS/TTS_infer_pack/text_segmentation_method.py", "GPT_SoVITS/process_ckpt.py",
    "GPT_SoVITS/module/models.py", "GPT_SoVITS/module/modules.py", "GPT_SoVITS/module/commons.py",
    "GPT_SoVITS/module/mel_processing.py", "GPT_SoVITS/feature_extractor/cnhubert.py",
    "GPT_SoVITS/sv.py", "GPT_SoVITS/eres2net/kaldi.py", "GPT_SoVITS/eres2net/ERes2NetV2.py",
    "GPT_SoVITS/text/__init__.py", "GPT_SoVITS/text/cleaner.py", "GPT_SoVITS/text/japanese.py",
    "GPT_SoVITS/text/symbols.py", "GPT_SoVITS/text/symbols2.py", "GPT_SoVITS/text/LangSegmenter/langsegmenter.py",
)
LOCAL_FILES = ("scripts/prepare_japanese_reference.py", "harness/prepared_reference.py",
               "harness/prepared_acoustic.py", "src/sakuratts/reference_condition.py")
DISTRIBUTIONS = ("torch", "torchaudio", "transformers", "numpy", "librosa", "soundfile", "soxr", "audioread",
                 "scipy", "numba", "llvmlite", "pyopenjtalk-plus", "SudachiPy", "SudachiDict-core",
                 "onnxruntime", "split-lang", "fast-langdetect", "fasttext-predict", "budoux", "pydantic")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def package_directory(distribution, directory):
    return Path(metadata.distribution(distribution).locate_file(directory)).resolve(strict=True)


def distribution_identity(name):
    dist = metadata.distribution(name)
    records = [path for path in dist.files or () if str(path).endswith(".dist-info/RECORD")]
    return {"version": dist.version, "installed_record": file_identity(dist.locate_file(records[0])) if records else None}


def validate_official_sources(repo):
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("Official checkout is not the fixed GPT-SoVITS commit")
    sources = {}
    for name in OFFICIAL_FILES:
        path = repo / name
        expected = subprocess.check_output(["git", "-C", str(repo), "show", COMMIT + ":" + name])
        if path.read_bytes() != expected:
            raise ValueError("Official source differs from the fixed commit: " + name)
        sources[name] = file_identity(path)
    return sources


def preflight(options):
    paths = {key: Path(options[key]).resolve(strict=True) for key in
             ("references", "official_source", "gpt_checkpoint", "sovits_checkpoint", "audio", "cnhubert",
              "sv_checkpoint", "main_dictionary", "user_dictionary", "language_model")}
    if not options["text"].strip("\n").strip():
        raise ValueError("Reference transcript must contain text")
    if paths["language_model"].name != "lid.176.bin":
        raise ValueError("The fixed official language router requires lid.176.bin")
    sources = validate_official_sources(paths["official_source"])
    user = paths["official_source"] / "GPT_SoVITS/text/ja_userdic"
    if hashlib.md5((user / "userdict.csv").read_bytes()).hexdigest() != (user / "userdict.md5").read_text(encoding="utf-8"):
        raise ValueError("Official user dictionary needs rebuilding; preparation will not modify the source checkout")
    if sha256_file(paths["user_dictionary"]) != sha256_file(user / "user.dict"):
        raise ValueError("This preparation scope requires the existing fixed official user dictionary")
    for name in ("config.json", "preprocessor_config.json"):
        if not (paths["cnhubert"] / name).is_file():
            raise FileNotFoundError(paths["cnhubert"] / name)
    if not any((paths["cnhubert"] / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")):
        raise ValueError("CNHuBERT local checkpoint is missing")
    pyopenjtalk = package_directory("pyopenjtalk-plus", "pyopenjtalk")
    resource_paths = set()
    for directory in (paths["cnhubert"], paths["main_dictionary"],
                      package_directory("SudachiPy", "sudachipy/resources"),
                      package_directory("SudachiDict-core", "sudachidict_core/resources")):
        resource_paths.update(path.resolve() for path in directory.rglob("*") if path.is_file())
    resource_paths.update((paths["sv_checkpoint"], paths["user_dictionary"], paths["language_model"],
                           user / "user.dict", user / "userdict.csv", user / "userdict.md5",
                           pyopenjtalk / "yomi_model/nani_enc.onnx", pyopenjtalk / "yomi_model/nani_model.onnx"))
    resources = {str(path): file_identity(path) for path in sorted(resource_paths)}
    inputs = {key: file_identity(paths[key]) for key in ("gpt_checkpoint", "sovits_checkpoint", "audio")}
    result = {"options": dict(options, **{key: str(value) for key, value in paths.items()}), "inputs": inputs,
              "resources": resources, "official_commit": COMMIT, "official_sources": sources,
              "official_status": subprocess.check_output(["git", "-C", str(paths["official_source"]), "status", "--short"], text=True),
              "local_sources": {name: file_identity(PROJECT / name) for name in LOCAL_FILES},
              "distributions": {name: distribution_identity(name) for name in DISTRIBUTIONS}}
    if options.get("comparison_package"):
        identity = {key + "_sha256": value["sha256"] for key, value in inputs.items()}
        reference = PreparedReference.load(options["comparison_package"], **identity, reference_text=options["text"],
                                           reference_language="ja", official_commit=COMMIT)
        result["comparison_manifest_sha256"] = sha256_file(Path(options["comparison_package"]) / "manifest.json")
        result["comparison_array_hashes"] = {name: sha256_array(getattr(reference, name)) for name in ARRAY_DTYPES}
    return result


def verify_files(prepared):
    for group in ("inputs", "resources", "official_sources", "local_sources"):
        for spec in prepared[group].values():
            if sha256_file(spec["path"]) != spec["sha256"]:
                raise ValueError("Preparation input/resource changed: " + spec["path"])


def array_identity(array):
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256_raw_c_order": sha256_array(array)}


@contextlib.contextmanager
def observe_audio(tts_module, events):
    """Record the actual decoder dispatch and outputs without choosing a backend."""
    import librosa.core.audio as librosa_audio
    import torchaudio
    from torchaudio._backend.utils import get_available_backends

    restored = []

    def replace(owner, name, function, *, static=False):
        original = owner.__dict__[name]
        restored.append((owner, name, original))
        setattr(owner, name, staticmethod(function) if static else function)

    for name in ("__soundfile_load", "__audioread_load"):
        original = getattr(librosa_audio, name)

        def loader(*args, _function=original, _name=name, **kwargs):
            try:
                array, rate = _function(*args, **kwargs)
            except Exception as error:
                events.append({"stage": "librosa." + _name, "status": "error", "error": str(error)})
                raise
            events.append({"stage": "librosa." + _name, "status": "completed", "sample_rate": rate,
                           "array": array_identity(array)})
            return array, rate
        replace(librosa_audio, name, loader)
    original_resample = librosa_audio.resample

    def librosa_resample(array, **kwargs):
        output = original_resample(array, **kwargs)
        events.append({"stage": "librosa.resample", "arguments": kwargs,
                       "input": array_identity(array), "output": array_identity(output)})
        return output
    replace(librosa_audio, "resample", librosa_resample)
    for name, backend in get_available_backends().items():
        original = backend.load

        def backend_load(*args, _function=original, _name=name, **kwargs):
            array, rate = _function(*args, **kwargs)
            events.append({"stage": "torchaudio.load", "backend": _name, "sample_rate": rate,
                           "array": array_identity(array)})
            return array, rate
        replace(backend, "load", backend_load, static=True)
    original_resample_torch = tts_module.resample

    def torch_resample(array, sr0, sr1, device):
        output = original_resample_torch(array, sr0, sr1, device)
        transform = tts_module.resample_transform_dict[f"{sr0}-{sr1}-{device}"]
        events.append({"stage": "official.torchaudio.Resample", "source_rate": sr0, "target_rate": sr1,
                       "device": str(device), "method": transform.resampling_method,
                       "lowpass_filter_width": transform.lowpass_filter_width, "rolloff": transform.rolloff,
                       "beta": transform.beta, "input": array_identity(array), "output": array_identity(output)})
        return output
    replace(tts_module, "resample", torch_resample)
    try:
        yield
    finally:
        for owner, name, original in reversed(restored):
            setattr(owner, name, original)


def loaded_sources(repo, output):
    """Hash imported implementations and snapshot actual official Python files."""
    files = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).is_file():
            path = Path(name).resolve()
            if str(path).startswith(str(repo) + os.sep) or "site-packages" in path.parts:
                files[str(path)] = file_identity(path)
                if path.is_relative_to(repo):
                    relative = path.relative_to(repo)
                    if path.suffix == ".py":
                        expected = subprocess.check_output(["git", "-C", str(repo), "show", COMMIT + ":" + str(relative)])
                        if path.read_bytes() != expected:
                            raise ValueError("Imported official source differs from fixed commit: " + str(relative))
                    destination = output / "source/official-imported" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
    return files


def native_library_identity():
    """Capture loaded audio/Torch libraries on the currently supported Mac host."""
    if sys.platform != "darwin":
        raise NotImplementedError("Complete native-library identity is currently implemented for macOS only")
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    paths = []
    for index in range(dyld._dyld_image_count()):
        raw = dyld._dyld_get_image_name(index)
        if raw:
            path = Path(os.fsdecode(raw))
            if path.is_file() and ("/torch/lib/" in str(path) or any(part in path.name.lower() for part in
                    ("libsndfile", "libsoxr", "libavcodec", "libavformat", "libavutil", "libswresample", "libsox", "libtorio"))):
                paths.append(path)
    return {str(path.resolve()): file_identity(path) for path in sorted(set(paths))}


def worker(run):
    prepared = read_json(run / "preflight.json")
    options = prepared["options"]
    repo = Path(options["official_source"])
    result = {"status": "running", "prepare_only": True, "target_synthesis_calls": 0}
    try:
        verify_files(prepared)
        os.environ.update(ORT_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          OPEN_JTALK_DICT_DIR=options["main_dictionary"], PYTHONDONTWRITEBYTECODE="1",
                          language="en_US", version="v2")
        os.environ["HF_HOME"] = str(Path(options["references"]) / ".cache/huggingface")
        os.environ["NUMBA_CACHE_DIR"] = str(run / "cache/numba")
        os.environ["MPLCONFIGDIR"] = str(run / "cache/matplotlib")
        sys.dont_write_bytecode = True
        os.chdir(repo)
        sys.path[:0] = [str(PROJECT / "harness"), str(repo / "GPT_SoVITS"), str(repo)]
        import_started = time.perf_counter()
        import torch
        import soundfile
        import soxr
        import importlib
        import fast_langdetect
        import pyopenjtalk
        from pyopenjtalk.yomi_model import nani_predict
        from prepared_reference import prepare_reference
        from prepared_acoustic import prepare_acoustic
        tts = importlib.import_module("TTS_infer_pack.TTS")
        import sv

        if Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR)).resolve() != Path(options["main_dictionary"]):
            raise ValueError("Loaded OpenJTalk dictionary differs from the explicit input")
        if nani_predict.enc_session is None or nani_predict.model_session is None:
            raise RuntimeError("Both official Nani sessions must be available")
        importlib.import_module("text.japanese")
        pyopenjtalk.update_global_jtalk_with_user_dict(options["user_dictionary"])
        fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
            fast_langdetect.infer.LangDetectConfig(cache_dir=Path(options["language_model"]).parent))
        sv.sv_path = options["sv_checkpoint"]
        torch.set_num_threads(4)
        if options["device"] == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")

        def synchronize():
            if options["device"] == "mps":
                torch.mps.synchronize()

        def memory_snapshot():
            import resource
            snapshot = {"rss_at_boundary_bytes": int(subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024,
                "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            if options["device"] == "mps":
                snapshot.update(mps_allocated_bytes_at_boundary=torch.mps.current_allocated_memory(),
                                mps_driver_bytes_at_boundary=torch.mps.driver_allocated_memory())
            return snapshot

        result["cost"] = {"imports_and_frontend_initialization_seconds": time.perf_counter() - import_started,
                          "memory_before_model_load": memory_snapshot(),
                          "scope": "One fresh preparation process; phase timings include synchronization and artifact saves, not normal synthesis latency; MPS/RSS boundary snapshots are not phase peaks or NVIDIA VRAM"}

        class ReferenceOnlyTTS(tts.TTS):
            def _init_models(self):
                self.init_vits_weights(self.configs.vits_weights_path)
                self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)

        # TTS_Config creates GPT_SoVITS/configs relative to cwd even for a dict.
        # Keep that directory and all later save_configs calls in this new run.
        os.chdir(run)
        config = tts.TTS_Config({"custom": {"device": options["device"], "is_half": False, "version": "v2Pro",
            "t2s_weights_path": options["gpt_checkpoint"], "vits_weights_path": options["sovits_checkpoint"],
            "cnhuhbert_base_path": options["cnhubert"]}})
        config.configs_path = str(run / "tts-prepare.yaml")
        os.chdir(repo)
        started = time.perf_counter()
        engine = ReferenceOnlyTTS(config)
        synchronize()
        result["cost"]["model_load_seconds"] = time.perf_counter() - started
        result["cost"]["memory_after_model_load"] = memory_snapshot()
        if (engine.t2s_model is not None or engine.bert_model is not None or engine.bert_tokenizer is not None
                or engine.configs.version != "v2Pro" or str(engine.configs.device) != options["device"]):
            raise ValueError("Reference-only loading departed from the declared scope")
        segments = []
        original_clean = engine.text_preprocessor.clean_text_inf

        def clean(text, language, version="v2Pro"):
            if language != "ja":
                raise NotImplementedError("This tool accepts Japanese reference segments only: " + language)
            phones, alignment, normalized = original_clean(text, language, version)
            segments.append({"text": text, "language": language, "phones": phones, "normalized_text": normalized})
            return phones, alignment, normalized
        engine.text_preprocessor.clean_text_inf = clean

        reference = {"path": options["audio"], "text": options["text"], "language": "ja"}
        identity = {"inputs": {spec["path"]: spec["sha256"] for spec in prepared["inputs"].values()},
                    "source_commit": COMMIT, "source_status": prepared["official_status"], "precision": "float32"}
        events = []
        with observe_audio(tts, events):
            started = time.perf_counter()
            result["reference_preparation"] = prepare_reference(engine, reference, run, identity, synchronize)
            result["cost"]["reference_preparation_and_save_seconds"] = time.perf_counter() - started
            result["cost"]["memory_after_reference_auxiliary_release"] = memory_snapshot()
            started = time.perf_counter()
            result["acoustic_preparation"] = prepare_acoustic(engine, run, synchronize)
            result["cost"]["acoustic_preparation_and_save_seconds"] = time.perf_counter() - started
            result["cost"]["memory_after_acoustic_preparation_release"] = memory_snapshot()
        result.update(status="completed", version="v2Pro", precision="float32", actual_device=str(engine.configs.device),
                      model_sampling_rate=engine.configs.sampling_rate, semantic_zero_padding_samples=int(engine.configs.sampling_rate * 0.3),
                      torch_version=torch.__version__, torch_git_version=torch.version.git_version, python=sys.version,
                      loaded_models=["official SoVITS", "CNHuBERT", "ERes2NetV2"],
                      gpt_loaded=False, chinese_bert_loaded=False, reference_segments=segments, audio_events=events,
                      audio_library_versions={"libsndfile": soundfile.__libsndfile_version__, "libsoxr": soxr.__libsoxr_version__},
                      nani_providers={"encoder": nani_predict.enc_session.get_providers(), "model": nani_predict.model_session.get_providers()})
        if any(event.get("backend") == "ffmpeg" for event in events):
            from torio.utils import ffmpeg_utils
            result["audio_library_versions"]["ffmpeg_libraries"] = ffmpeg_utils.get_versions()
        engine = None
        import gc
        gc.collect()
        if options["device"] == "mps":
            torch.mps.empty_cache()
        synchronize()
        result["cost"]["memory_after_model_release"] = memory_snapshot()
        verify_files(prepared)
        result["imported_sources"] = loaded_sources(repo, run)
        result["native_libraries"] = native_library_identity()
        result["platform"] = {"system": os.uname().sysname, "release": os.uname().release,
                              "machine": os.uname().machine, "macos": subprocess.check_output(["sw_vers"], text=True)}
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    write_json(run / "worker-result.json", result)
    return 0 if result["status"] == "completed" else 1


def export_package(prepared, run, process):
    options = prepared["options"]
    metadata_json = read_json(run / "prepared-reference.json")
    worker_result = read_json(run / "worker-result.json")
    if process["returncode"] != 0 or worker_result["status"] != "completed":
        raise ValueError("A reference package requires a successful preparation process exit")
    arrays = {}
    with np.load(run / "prepared-reference.npz", allow_pickle=False) as archive:
        arrays.update({name: archive[name] for name in ("reference_phones", "prompt_semantic", "reference_bert")})
    with np.load(run / "prepared-acoustic.npz", allow_pickle=False) as archive:
        arrays.update({name: archive[name] for name in ("ge", "ge512")})
    validate_arrays(arrays)
    if np.count_nonzero(arrays["reference_bert"]):
        raise AssertionError("Official Japanese reference BERT must remain FP32 zero features")
    comparison = None
    if options.get("comparison_package"):
        previous = PreparedReference.load(options["comparison_package"],
            gpt_checkpoint_sha256=prepared["inputs"]["gpt_checkpoint"]["sha256"],
            sovits_checkpoint_sha256=prepared["inputs"]["sovits_checkpoint"]["sha256"],
            manifest_sha256=prepared["comparison_manifest_sha256"])
        metrics = {}
        for name, value in arrays.items():
            expected = getattr(previous, name)
            same_structure = value.shape == expected.shape and value.dtype == expected.dtype
            exact = same_structure and value.tobytes() == expected.tobytes()
            row = {"shape_equal": value.shape == expected.shape, "dtype_equal": value.dtype == expected.dtype,
                   "array_bytes_equal": exact, "actual": array_identity(value), "expected": array_identity(expected)}
            if same_structure and np.issubdtype(value.dtype, np.floating):
                difference = np.abs(value.astype(np.float64) - expected.astype(np.float64))
                tolerance = 1e-4 + 1e-5 * np.abs(expected.astype(np.float64))
                row.update(atol=1e-4, rtol=1e-5, max_abs=float(difference.max()),
                           rms=float(np.sqrt(np.mean(difference ** 2))),
                           outside_tolerance_count=int(np.count_nonzero(difference > tolerance)),
                           within_fp32_tolerance=bool(np.all(difference <= tolerance)))
            metrics[name] = row
        comparison = {name: row["array_bytes_equal"] for name, row in metrics.items()}
        write_json(run / "comparison.json", {"all_array_bytes_equal": all(comparison.values()), "arrays": metrics,
                   "scope": "Integer phones/tokens require exact equality; FP32 tolerance is reported separately and does not replace byte equality"})
        if not all(comparison.values()):
            raise AssertionError("Fresh preparation differs from the comparison package; evidence retained")
    identity = {key + "_sha256": spec["sha256"] for key, spec in prepared["inputs"].items()}
    identity.update(official_commit=COMMIT, reference_text=options["text"], reference_language="ja")
    stamp = run.name.split("-japanese-reference-")[0]
    destination = Path(options["references"]) / "models/converted" / (stamp + "-v2pro-japanese-reference")
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(destination / "conditions.npz", **arrays)
    evidence_names = ("preflight.json", "worker-result.json", "process-result.json", "prepared-reference.json",
                      "prepared-reference.npz", "prepared-acoustic.npz", "process.stdout-stderr.log")
    manifest = {"format": FORMAT, "model_family": "v2Pro", "created_at_utc": stamp,
        "scope": "Fresh raw-audio preparation of one Japanese reference with fixed official FP32 V2Pro; no synthesis or new voice quality claim",
        "identity": identity, "reference": {"prompt_text": metadata_json["prompt_text"], "normalized_text": metadata_json["normalized_text"]},
        "preparation": {"precision": "float32", "device": worker_result["actual_device"],
                        "torch_version": worker_result["torch_version"], "torch_git_version": worker_result["torch_git_version"],
                        "reference_bert_all_zero": True, "single_reference_list_mean_preserved": True,
                        "gpt_loaded": False, "chinese_bert_loaded": False, "prepare_only": True},
        "resources": prepared["resources"], "environment": {"distributions": prepared["distributions"],
                        "platform": worker_result["platform"], "audio_library_versions": worker_result["audio_library_versions"],
                        "native_libraries": worker_result["native_libraries"], "nani_providers": worker_result["nani_providers"]},
        "implementations": {"official_commit": COMMIT, "official_sources": prepared["official_sources"],
                            "imported_sources": worker_result["imported_sources"], "local_sources": prepared["local_sources"]},
        "audio_preprocessing": {"events": worker_result["audio_events"],
                                "semantic_zero_padding_samples": worker_result["semantic_zero_padding_samples"],
                                "model_sampling_rate": worker_result["model_sampling_rate"]},
        "archive": {"file": "conditions.npz", "bytes": (destination / "conditions.npz").stat().st_size,
                    "sha256": sha256_file(destination / "conditions.npz")},
        "arrays": {name: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.nbytes,
                          "sha256_raw_c_order": sha256_array(value)} for name, value in arrays.items()},
        "provenance": {"preparation_run": str(run), "resource_identity_scope": "Captured for this new preparation, never reconstructed for a historical run",
                       "distribution_record_scope": "Installed distribution metadata; actual imported implementations and loaded audio/Torch libraries are hashed separately",
                       "process_exit_code": process["returncode"], "source_files_sha256": {name: sha256_file(run / name) for name in evidence_names},
                       "comparison_package": options.get("comparison_package"), "comparison_manifest_sha256": prepared.get("comparison_manifest_sha256"),
                       "comparison_array_bytes_equal": comparison}}
    write_json(destination / "manifest.json", manifest)
    restored = PreparedReference.load(destination, **identity, manifest_sha256=sha256_file(destination / "manifest.json"))
    for name, value in arrays.items():
        if value.tobytes() != getattr(restored, name).tobytes():
            raise AssertionError("New package roundtrip changed: " + name)
    return {"package": str(destination), "manifest_sha256": sha256_file(destination / "manifest.json"),
            "comparison_array_bytes_equal": comparison, "roundtrip_array_bytes_equal": True}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-run":
        return worker(Path(sys.argv[2]).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "official-source", "gpt-checkpoint", "sovits-checkpoint", "audio", "cnhubert",
                 "sv-checkpoint", "main-dictionary", "user-dictionary", "language-model"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--comparison-package", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This first preparation tool captures native-library identity on macOS only")
    options = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    prepared = preflight(options)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.references.resolve() / "runs" / (stamp + "-japanese-reference-" + ("check" if args.check_only else "prepare"))
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "preflight.json", prepared)
    for group, prefix in ((prepared["local_sources"], "local"), (prepared["official_sources"], "official")):
        for name, spec in group.items():
            target = run / "source" / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(spec["path"], target)
    summary = {"status": "checked" if args.check_only else "running", "run": str(run),
               "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], "prepare_only": True,
               "inference_backends_imported_in_parent": {name: name in sys.modules for name in ("torch", "transformers", "mlx", "onnxruntime")}}
    if args.check_only:
        write_json(run / "result.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    print("RUN_DIRECTORY=" + str(run), flush=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-run", str(run)]
    start = time.perf_counter()
    with (run / "process.stdout-stderr.log").open("x") as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = child.wait()
    process = {"command": command, "pid": child.pid, "returncode": returncode,
               "signal": -returncode if returncode < 0 else None, "elapsed_seconds": time.perf_counter() - start,
               "log_sha256": sha256_file(run / "process.stdout-stderr.log"),
               "scope": "Actual child exit including interpreter teardown, not inference timing"}
    write_json(run / "process-result.json", process)
    try:
        verify_files(prepared)
        summary.update(export_package(prepared, run, process), status="completed")
    except Exception:
        summary.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    write_json(run / "result.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
