"""Named execution presets; importing them does not load a compute backend."""

_CPU = {"threads": 8, "gpt_threads": 4, "gpt_backend": "onnx", "gpt_precision": "int8",
        "gpt_prefill_query_chunk_size": 0, "enable_cpu_mem_arena": False, "policy": "resident"}
_DIRECTML = {"threads": 2, "gpt_threads": 4, "gpt_backend": "directml", "gpt_precision": "fp16",
             "capacity": 1280, "gpt_prefill_query_chunk_size": 0, "enable_cpu_mem_arena": False,
             "allow_experimental_acoustic_fp16": True, "policy": "resident"}
_CUDA_FP16 = {"gpt_precision": "fp16", "allow_experimental_acoustic_fp16": True,
              "acoustic_chunk_frames": 256, "policy": "release-state"}
_PROFILES = {
    "cpu": {"int8": dict(_CPU)},
    "directml": {"fp16": dict(_DIRECTML)},
    "cuda": {
        "fp32": {"gpt_precision": "fp32", "policy": "resident"},
        "fp16": dict(_CUDA_FP16),
        "low-memory": {**_CUDA_FP16, "acoustic_session_policy": "staged"},
        "minimum-memory": {**_CUDA_FP16, "acoustic_session_policy": "staged", "policy": "staged"},
    },
    "mlx": {
        "fp32": {"gpt_precision": "fp32", "policy": "resident"},
        "low-memory": {"gpt_precision": "fp32", "policy": "release-state"},
        "minimum-memory": {"gpt_precision": "fp32", "policy": "staged"},
    },
}
_DEFAULT_PROFILES = {"cpu": "int8", "directml": "fp16"}


def available_profiles():
    """Return a new mapping so callers cannot change preset defaults."""
    return {backend: list(profiles) for backend, profiles in _PROFILES.items()}


def resolve_profile(backend, profile, overrides=None):
    """Return the effective preset and merged options at a configuration boundary."""
    profile = _DEFAULT_PROFILES.get(backend) if profile is None else profile
    if profile is None:
        return None, overrides
    profiles = _PROFILES.get(backend, {})
    if not isinstance(profile, str) or profile not in profiles:
        raise ValueError(f"Profile {profile!r} is not supported by {backend}; choose from: "
                         + ", ".join(profiles))
    options = {**profiles[profile], **(overrides or {})}
    return profile, options


def uses_staged_policy(profile, overrides=None):
    # Every minimum-memory preset stages GPT and acoustics. Explicit options win.
    default = "staged" if profile == "minimum-memory" else "resident"
    return (overrides or {}).get("policy", default) == "staged"


def validate_runtime_precision(backend, profile, runtime):
    """A precision preset must not silently run a mismatched acoustic package."""
    expected = "fp32" if profile in ("fp32", "int8") else (
        "fp16" if profile == "fp16" or backend == "cuda" and profile in ("low-memory", "minimum-memory") else None)
    if expected is not None and runtime.acoustic_precision != expected:
        raise ValueError(f"Profile {profile} requires a {expected} acoustic package; "
                         f"the selected model contains {runtime.acoustic_precision}. "
                         "Selecting a profile does not convert model weights.")
