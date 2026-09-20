"""Admission for self-contained, explicitly selected vocoder chunk packages."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from sakuratts.backends.onnx.sovits import FP16_EXECUTION_OPTIONS, INPUT_NAMES, _package_file
from sakuratts.backends.onnx.vocoder_receptive_field import VocoderReceptiveField

FORMAT = "sakuratts-sovits-chunked-v1"
SCREEN_FORMAT = "sakuratts-vocoder-chunk-screen-v1"
CHUNK_LIMITS = {"max_abs_error": .005, "rmse": .0005, "minimum_snr_db": 45.,
    "max_spectral_convergence": .01, "max_active_log_spectral_rms_db": .3,
    "seam_radius_samples": 640, "seam_max_abs_error": .005, "seam_rmse": .0005}
ORIGINAL_TOLERANCE = {"atol": 1e-4, "rtol": 1e-5}
SCREEN_CHECKS = {"source_verified", "artifact_files_verified", "split_full_bitwise_equal",
    "chunk_engineering_passed", "seams_passed", "repeats_bitwise_equal",
    "ordinary_inputs_verified", "boundary_inputs_verified"}


def identity_sha256(manifest):
    identity = {key: value for key, value in manifest.items() if key != "validation"}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def package_file(root, spec):
    name = spec.get("file")
    if (not isinstance(name, str) or Path(name).name != name or name in ("", ".", "..")
            or type(spec.get("bytes")) is not int or spec["bytes"] < 1):
        raise ValueError("Chunk package files require a basename and positive byte count")
    resolved = (root / name).resolve(strict=True)
    if resolved.parent != root or not resolved.is_file():
        raise ValueError("Chunk package files must be contained in their own package")
    return _package_file(root, spec)


def _interfaces(manifest):
    config, interfaces = manifest["config"], manifest["interfaces"]
    channels, condition = config["model"]["inter_channels"], config["model"]["gin_channels"]
    dtype = "FLOAT16" if manifest["dtype"] == "float16" else "FLOAT"
    latent = ("decoder_input", dtype, [None, channels, None])
    ge = ("ge", "FLOAT", [1, condition, 1])
    expected = {"latent": {
        "inputs": [("codes", "INT64", [1, 1, None]), ("phones", "INT64", [1, None]), ge,
                   ("ge512", "FLOAT", [1, 512, 1]), ("noise", "FLOAT", [1, channels, None]),
                   ("noise_scale", "FLOAT", [])], "outputs": [latent]},
        "vocoder": {"inputs": [latent, ge], "outputs": [("waveform", "FLOAT", [None, 1, None])]}}
    if set(interfaces) != set(expected) or set(manifest["inputs"]) != set(INPUT_NAMES):
        raise ValueError("Unexpected split graph or public input names")
    for kind, roles in expected.items():
        if set(interfaces[kind]) != set(roles):
            raise ValueError("Unexpected split graph interface roles")
        for role, specs in roles.items():
            actual = interfaces[kind][role]
            if len(actual) != len(specs):
                raise ValueError("Unexpected split graph interface arity")
            for item, (name, dtype, shape) in zip(actual, specs):
                if (item.get("name") != name or item.get("type") != dtype
                        or len(item.get("shape", [])) != len(shape)
                        or any(expected_dim is not None and actual_dim != expected_dim
                               for actual_dim, expected_dim in zip(item["shape"], shape))):
                    raise ValueError("Split graph interface differs from the public tensor contract")
    for name, dtype, shape in expected["latent"]["inputs"]:
        item = manifest["inputs"][name]
        if (item.get("dtype") != {"FLOAT": "float32", "INT64": "int64"}[dtype]
                or len(item.get("shape", [])) != len(shape)
                or any(dim is not None and observed != dim for observed, dim in zip(item["shape"], shape))):
            raise ValueError("Split package public tensor declaration is inconsistent")


def read_chunked_manifest(package, *, diagnostic=False, allow_experimental_fp16=False,
                          acoustic_chunk_frames=None, acoustic_arena_shrink=False):
    root = Path(package).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("Expected a self-contained chunked acoustic package")
    if type(acoustic_chunk_frames) is not int or acoustic_chunk_frames < 0:
        raise ValueError("Chunk packages require an explicit nonnegative acoustic_chunk_frames")
    if diagnostic or acoustic_arena_shrink is not True:
        raise ValueError("Chunk packages require arena shrinkage and do not support intermediate capture")
    dtype, config, precision = manifest["dtype"], manifest["config"], manifest.get("precision") or {}
    if dtype not in ("float16", "float32") or config["model"]["version"] not in ("v2Pro", "v2ProPlus"):
        raise ValueError("Expected V2Pro/V2ProPlus FP32 or experimental FP16 acoustics")
    if (config["semantic_upsample_factor"] != 2 or config["semantic_hz"] != 25
            or any(type(v) is not int or v < 1 for v in config["model"]["upsample_rates"])):
        raise ValueError("Chunk packages require the supported 25 Hz acoustic conversion")
    if dtype == "float16" and (not allow_experimental_fp16 or precision.get("profile") != "fp16-mixed-v1"
            or precision.get("keep_io_types") is not True or precision.get("input_dtype") != "float32"
            or precision.get("output_dtype") != "float32" or precision.get("conv_transpose_lowering_method") != "polyphase"):
        raise ValueError("FP16 chunks require explicit selection of the screened polyphase FP16 package")
    settings = manifest["settings"]
    if (settings.get("execution_options") != FP16_EXECUTION_OPTIONS
            or settings.get("ort_graph_optimization_level") != precision.get("ort_graph_optimization_level", "ORT_ENABLE_ALL")
            or settings.get("ort_graph_optimization_level") not in ("ORT_ENABLE_ALL", "ORT_ENABLE_BASIC", "ORT_DISABLE_ALL")
            or type(settings.get("ort_use_deterministic_compute")) is not bool
            or settings["ort_use_deterministic_compute"] != precision.get("ort_use_deterministic_compute", False)):
        raise ValueError("Chunk session settings differ from the validated execution policy")
    cut = manifest["cut"]
    if (cut.get("exported_value") != "decoder_input" or cut.get("shared_initializer_bytes") != 0
            or cut.get("shared_initializer_names") != []
            or cut.get("internal_dtype") != ("FLOAT16" if dtype == "float16" else "FLOAT")):
        raise ValueError("Unexpected chunk boundary precision or duplicated weights")
    _interfaces(manifest)
    for role in ("graphs", "weights"):
        if set(manifest[role]) != {"latent", "vocoder"}:
            raise ValueError("Chunk packages own exactly two graph and weight partitions")
        for spec in manifest[role].values():
            package_file(root, spec)
    planner = VocoderReceptiveField.from_json(package_file(root, manifest["rf"]))
    provenance = manifest["provenance"]
    original_graph = (precision["source_graphs"]["diagnostic"] if dtype == "float16"
                      else provenance["source_graph"])
    if (planner.source["graph_sha256"] != provenance["rf_original_graph_sha256"]
            or planner.source["graph_sha256"] != original_graph["sha256"]
            or planner.samples_per_frame != math.prod(config["model"]["upsample_rates"])
            or planner.samples_per_frame * config["semantic_upsample_factor"] * config["semantic_hz"] != config["sample_rate"]
            or (planner.input_name, planner.output_name, planner.condition_name) != ("decoder_input", "waveform", "ge")):
        raise ValueError("Chunk dependency plan or sample ratio differs from the source model")
    validation = manifest.get("validation", {})
    if validation.get("kind") != SCREEN_FORMAT or validation.get("passed") is not True:
        raise ValueError("The chunk package has not passed its own waveform and seam screening")
    report = json.loads(package_file(root, validation).read_text(encoding="utf-8"))
    chunks = report.get("chunk_frames", [])
    if (report.get("format") != SCREEN_FORMAT or type(report.get("version")) is not int or report["version"] != 1
            or report.get("package_identity_sha256") != identity_sha256(manifest)
            or report.get("passed") is not True or report.get("quality_accepted") is not False
            or report.get("acoustic_arena_shrink") is not True or report.get("settings") != settings
            or report.get("limits") != CHUNK_LIMITS or report.get("original_tolerance") != ORIGINAL_TOLERANCE
            or not isinstance(chunks, list) or not chunks or len(set(chunks)) != len(chunks)
            or any(type(size) is not int or size < 0 for size in chunks)
            or acoustic_chunk_frames not in chunks
            or any(report.get("checks", {}).get(key) is not True for key in SCREEN_CHECKS)
            or not report.get("evidence")):
        raise ValueError("Chunk screening identity, coverage or execution options do not match this request")
    return manifest, None
