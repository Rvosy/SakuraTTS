"""One Japanese request using an offline prepared V2Pro reference.

This composition has no file or backend imports. Models and the real text
frontend are supplied by the caller. Saved target features/tokens are never an
input; explicit random draws/noise are optional hooks for controlled replay.
"""

from dataclasses import dataclass
import time

import numpy as np

from .generation import SemanticGeneration, check_cancelled, generate_semantic
from .reference_condition import PreparedReference


@dataclass
class PreparedText:
    text: str
    language: str
    target: dict
    seconds: float


@dataclass(frozen=True)
class PreparedSemantic:
    """One request between semantic and acoustic execution, without models.

    The reference and RNG stay bound to the request, so acoustic execution
    cannot accidentally substitute another reference or restart sampling.
    Do not mutate the prepared inputs or advance this RNG between phases.
    """
    prepared: PreparedText
    reference: PreparedReference
    reference_identity: dict
    target_phones: np.ndarray
    generation: SemanticGeneration
    rng: np.random.Generator
    timings: dict


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


def _validate_model(reference, name, manifest):
    identity = reference.manifest["identity"]
    if identity["reference_language"] != "ja":
        raise ValueError("Only a prepared Japanese reference is currently supported")
    if (manifest["source"]["checkpoint_sha256"] != identity[name + "_checkpoint_sha256"]
            or manifest["source"]["official_commit"] != identity["official_commit"]):
        raise ValueError(f"Loaded {name} model differs from the prepared reference")
    if (reference.manifest["model_family"] != "v2Pro"
            or name == "sovits" and manifest["config"]["model"]["version"] != "v2Pro"):
        raise ValueError("Only the validated V2Pro architecture is supported")


def _validate_models(reference, gpt, sovits):
    _validate_model(reference, "gpt", gpt.weight_manifest)
    _validate_model(reference, "sovits", sovits.encoder.manifest)


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


def generate_prepared_semantic(prepared: PreparedText, reference: PreparedReference, *, gpt,
                               early_stop_num, top_k=15, top_p=1.0, temperature=1.0,
                               repetition_penalty=1.35, rng=None, semantic_random_draw=None,
                               release_gpt_state=False, cancel_requested=None):
    """Generate semantics without loading an acoustic model.

    release_gpt_state discards GPT request KV after semantic generation, also
    on semantic failure, while retaining weights. The supplied GPT must expose
    release_request_state(); subsequent decode requires a new prefill. After
    return the caller may unload GPT entirely before loading SoVITS.
    The cancellation predicate is not retained in the returned request. Pass it
    explicitly to synthesize_acoustic to keep cancellation enabled in that phase.
    """
    if prepared.language not in ("ja", "all_ja"):
        raise ValueError("Only Japanese ja/all_ja requests are currently supported")
    _validate_model(reference, "gpt", gpt.weight_manifest)
    rng = np.random.default_rng() if rng is None else rng
    start = time.perf_counter()
    target = prepared.target
    target_phones = np.array(target["phones"], dtype=np.int64, copy=True)
    target_phones.setflags(write=False)
    prepared = PreparedText(prepared.text, prepared.language,
                            dict(target, phones=target_phones.tolist()), prepared.seconds)
    target_bert = np.asarray(target["bert_features"])
    phones = np.concatenate((reference.reference_phones, target_phones))[None, :]
    bert = np.concatenate((reference.reference_bert, target_bert), axis=1).T[None, :, :]
    frontend_done = time.perf_counter()
    try:
        generated = generate_semantic(
            gpt, phones, reference.prompt_semantic[None, :], bert, eos=gpt.config["eos"],
            top_k=top_k, top_p=top_p, temperature=temperature,
            repetition_penalty=repetition_penalty, early_stop_num=early_stop_num,
            rng=rng, random_draw=semantic_random_draw, cancel_requested=cancel_requested,
        )
    finally:
        if release_gpt_state:
            gpt.release_request_state()
    semantic_done = time.perf_counter()
    return PreparedSemantic(prepared, reference, dict(reference.manifest["identity"]),
                            target_phones, generated, rng, {
        "frontend_seconds": prepared.seconds,
        "condition_seconds": frontend_done - start,
        "semantic_seconds": semantic_done - frontend_done,
    })


def synthesize_acoustic(request: PreparedSemantic, *, sovits, speed=1.0, noise_scale=0.5,
                        fragment_interval=0.3, acoustic_noise=None, cancel_requested=None):
    """Decode this request with its bound reference and remaining RNG state.

    Model loading/unloading and caller time between phases are excluded from
    the returned compute timings. The caller measures complete request time.
    Cancellation is checked before noise generation and after decode returns its
    evaluated waveform; a cancelled request never returns PCM, including partial PCM.
    """
    if speed != 1.0 or fragment_interval < 0:
        raise ValueError("Require speed=1 and a nonnegative fragment interval")
    reference = request.reference
    if reference.manifest["identity"] != request.reference_identity:
        raise ValueError("Reference identity changed between semantic and acoustic execution")
    _validate_model(reference, "sovits", sovits.encoder.manifest)
    sovits.validate_reference(reference)
    check_cancelled(cancel_requested, "before_acoustic")
    config = sovits.encoder.manifest["config"]
    start = time.perf_counter()
    semantic = request.generation.semantic
    target = request.prepared.target
    target_phones = request.target_phones
    noise_shape = (1, config["model"]["inter_channels"],
                   semantic.shape[-1] * config["semantic_upsample_factor"])
    if acoustic_noise is None:
        acoustic_noise = request.rng.standard_normal(noise_shape, dtype=np.float32)
    else:
        acoustic_noise = np.asarray(acoustic_noise)
    if (acoustic_noise.dtype != np.float32 or acoustic_noise.shape != noise_shape
            or not np.isfinite(acoustic_noise).all()):
        raise ValueError(f"Acoustic noise must be finite FP32 with generated-history shape {noise_shape}")
    waveform = sovits.decode(semantic, target_phones[None, :], reference.ge, reference.ge512,
                             acoustic_noise, noise_scale=noise_scale, speed=speed)
    check_cancelled(cancel_requested, "after_acoustic")
    acoustic_done = time.perf_counter()
    waveform = np.asarray(waveform).copy()
    pcm = single_fragment_pcm(waveform, sovits.sample_rate, fragment_interval)
    finished = time.perf_counter()
    inference_seconds = (request.timings["condition_seconds"] + request.timings["semantic_seconds"]
                         + finished - start)
    return SpeechResult(sovits.sample_rate, pcm, waveform, target, request.generation, {
        **request.timings,
        "acoustic_seconds": acoustic_done - start,
        "output_copy_pcm_seconds": finished - acoustic_done,
        "prepared_inference_seconds": inference_seconds,
        "compute_seconds": request.prepared.seconds + inference_seconds,
    })


def synthesize_prepared(prepared: PreparedText, reference: PreparedReference, *, gpt, sovits,
                        early_stop_num, top_k=15, top_p=1.0, temperature=1.0,
                        repetition_penalty=1.35, speed=1.0, noise_scale=0.5,
                        fragment_interval=0.3, rng=None, semantic_random_draw=None, acoustic_noise=None,
                        release_gpt_state=False, cancel_requested=None):
    """Compose both phases when the caller owns both loaded models.

    NumPy's RNG is not seed-equivalent to Torch. Explicit draws/noise support
    controlled replay. No reference encoder is loaded by either phase.
    """
    if speed != 1.0 or top_p != 1.0 or fragment_interval < 0:
        raise ValueError("Require speed=1, top_p=1 and a nonnegative fragment interval")
    _validate_models(reference, gpt, sovits)
    request = generate_prepared_semantic(
        prepared, reference, gpt=gpt, early_stop_num=early_stop_num, top_k=top_k, top_p=top_p,
        temperature=temperature, repetition_penalty=repetition_penalty, rng=rng,
        semantic_random_draw=semantic_random_draw, release_gpt_state=release_gpt_state,
        cancel_requested=cancel_requested,
    )
    return synthesize_acoustic(request, sovits=sovits, speed=speed, noise_scale=noise_scale,
                               fragment_interval=fragment_interval, acoustic_noise=acoustic_noise,
                               cancel_requested=cancel_requested)


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
