"""One Japanese request using an offline prepared V2Pro reference.

This composition has no file or backend imports. Models and the real text
frontend are supplied by the caller. Saved target features/tokens are never an
input; explicit random draws/noise are optional hooks for controlled replay.
"""

from dataclasses import dataclass
import time

import numpy as np

from .generation import SemanticGeneration, generate_semantic
from .reference_condition import PreparedReference


@dataclass
class PreparedText:
    text: str
    language: str
    target: dict
    seconds: float


@dataclass
class SpeechResult:
    sample_rate: int
    pcm: np.ndarray
    waveform: np.ndarray
    target: dict
    generation: SemanticGeneration
    timings: dict


def single_fragment_pcm(waveform, sample_rate, fragment_interval=0.3):
    """Pinned official non-streaming normalization, trailing silence and cast."""
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1).copy()
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Expected finite nonempty waveform samples")
    maximum = np.max(np.abs(audio))
    if maximum > 1:
        audio /= maximum
    silence = np.zeros(int(sample_rate * fragment_interval), dtype=np.float32)
    return (np.concatenate((audio, silence)) * 32768).astype(np.int16)


def _validate_models(reference, gpt, sovits):
    gpt_manifest, acoustic_manifest = gpt.weight_manifest, sovits.encoder.manifest
    identity = reference.manifest["identity"]
    if identity["reference_language"] != "ja":
        raise ValueError("Only a prepared Japanese reference is currently supported")
    for name, manifest in (("gpt", gpt_manifest), ("sovits", acoustic_manifest)):
        if (manifest["source"]["checkpoint_sha256"] != identity[name + "_checkpoint_sha256"]
                or manifest["source"]["official_commit"] != identity["official_commit"]):
            raise ValueError(f"Loaded {name} model differs from the prepared reference")
    config = acoustic_manifest["config"]
    if config["model"]["version"] != "v2Pro" or reference.manifest["model_family"] != "v2Pro":
        raise ValueError("Only the validated V2Pro architecture is supported")


def prepare_text(text, language, frontend):
    """Prepare target features before loading GPT/acoustic models, if desired.

    After this returns, the caller may release its Japanese frontend components. The result
    contains only target data and does not retain the frontend or its models.
    """
    if language not in ("ja", "all_ja"):
        raise ValueError("Only Japanese ja/all_ja requests are currently supported")
    start = time.perf_counter()
    targets = frontend.prepare_target(text, language, split_method="cut0")
    if len(targets) != 1:
        raise NotImplementedError("The current synthesis entry requires exactly one cut0 fragment")
    target = targets[0]
    target_phones = np.asarray(target["phones"], dtype=np.int64)
    target_bert = np.asarray(target["bert_features"])
    if target_bert.dtype != np.float32 or target_bert.shape != (1024, target_phones.size):
        raise ValueError("Target BERT must be FP32 and aligned to every target phone")
    return PreparedText(text, language, target, time.perf_counter() - start)


def synthesize_prepared(prepared: PreparedText, reference: PreparedReference, *, gpt, sovits,
                        early_stop_num, top_k=15, top_p=1.0, temperature=1.0,
                        repetition_penalty=1.35, speed=1.0, noise_scale=0.5,
                        fragment_interval=0.3, rng=None, semantic_random_draw=None, acoustic_noise=None):
    """Generate from this request's independently prepared target features.

    The caller owns model loading and precision. NumPy's RNG is not seed-
    equivalent to Torch; controlled replay supplies semantic_random_draw and
    an acoustic_noise array. No reference encoder is loaded by either entry.
    """
    if prepared.language not in ("ja", "all_ja"):
        raise ValueError("Only Japanese ja/all_ja requests are currently supported")
    if speed != 1.0 or top_p != 1.0 or fragment_interval < 0:
        raise ValueError("Require speed=1, top_p=1 and a nonnegative fragment interval")
    _validate_models(reference, gpt, sovits)
    config = sovits.encoder.manifest["config"]
    rng = np.random.default_rng() if rng is None else rng
    start = time.perf_counter()
    target = prepared.target
    target_phones = np.asarray(target["phones"], dtype=np.int64)
    target_bert = np.asarray(target["bert_features"])
    phones = np.concatenate((reference.reference_phones, target_phones))[None, :]
    bert = np.concatenate((reference.reference_bert, target_bert), axis=1).T[None, :, :]
    frontend_done = time.perf_counter()
    generated = generate_semantic(
        gpt, phones, reference.prompt_semantic[None, :], bert, eos=gpt.config["eos"],
        top_k=top_k, top_p=top_p, temperature=temperature,
        repetition_penalty=repetition_penalty, early_stop_num=early_stop_num,
        rng=rng, random_draw=semantic_random_draw,
    )
    semantic_done = time.perf_counter()
    semantic = generated.semantic
    noise_shape = (1, config["model"]["inter_channels"],
                   semantic.shape[-1] * config["semantic_upsample_factor"])
    if acoustic_noise is None:
        acoustic_noise = rng.standard_normal(noise_shape, dtype=np.float32)
    else:
        acoustic_noise = np.asarray(acoustic_noise)
    if (acoustic_noise.dtype != np.float32 or acoustic_noise.shape != noise_shape
            or not np.isfinite(acoustic_noise).all()):
        raise ValueError(f"Acoustic noise must be finite FP32 with generated-history shape {noise_shape}")
    waveform = sovits.decode(semantic, target_phones[None, :], reference.ge, reference.ge512,
                             acoustic_noise, noise_scale=noise_scale, speed=speed)
    acoustic_done = time.perf_counter()
    waveform = np.asarray(waveform).copy()
    pcm = single_fragment_pcm(waveform, sovits.sample_rate, fragment_interval)
    finished = time.perf_counter()
    return SpeechResult(sovits.sample_rate, pcm, waveform, target, generated, {
        "frontend_seconds": prepared.seconds,
        "condition_seconds": frontend_done - start,
        "semantic_seconds": semantic_done - frontend_done,
        "acoustic_seconds": acoustic_done - semantic_done,
        "output_copy_pcm_seconds": finished - acoustic_done,
        "prepared_inference_seconds": finished - start,
        "compute_seconds": prepared.seconds + finished - start,
    })


def synthesize(text, language, reference: PreparedReference, *, frontend, gpt, sovits, **parameters):
    """Convenience entry when the caller already owns all required components.

    Use prepare_text then synthesize_prepared to separate frontend-model and
    synthesis-model residency. Both paths execute the same request functions.
    """
    _validate_models(reference, gpt, sovits)
    start = time.perf_counter()
    prepared = prepare_text(text, language, frontend)
    result = synthesize_prepared(prepared, reference, gpt=gpt, sovits=sovits, **parameters)
    result.timings["request_seconds"] = time.perf_counter() - start
    return result
