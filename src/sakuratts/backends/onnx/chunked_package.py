"""Admission for self-contained, explicitly selected vocoder chunk packages."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from sakuratts.backends.onnx.sovits import INPUT_NAMES, _package_file
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
    dtype, config = manifest["dtype"], manifest["config"]
    if dtype not in ("float16", "float32") or config["model"]["version"] not in ("v2Pro", "v2ProPlus"):
        raise ValueError("Expected V2Pro/V2ProPlus FP32 or experimental FP16 acoustics")
    if (config["semantic_upsample_factor"] != 2 or config["semantic_hz"] != 25
            or any(type(v) is not int or v < 1 for v in config["model"]["upsample_rates"])):
        raise ValueError("Chunk packages require the supported 25 Hz acoustic conversion")
    if dtype == "float16" and not allow_experimental_fp16:
        raise ValueError("FP16 acoustic packages require allow_experimental_fp16=True")
    _interfaces(manifest)
    planner = VocoderReceptiveField.from_json(package_file(root, manifest["rf"]))
    if (planner.samples_per_frame != math.prod(config["model"]["upsample_rates"])
            or planner.samples_per_frame * config["semantic_upsample_factor"] * config["semantic_hz"] != config["sample_rate"]
            or (planner.input_name, planner.output_name, planner.condition_name) != ("decoder_input", "waveform", "ge")):
        raise ValueError("Chunk dependency plan or sample ratio differs from the source model")
    return manifest, None
