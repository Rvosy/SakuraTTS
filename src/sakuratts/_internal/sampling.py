"""NumPy FP32 reference for pinned GPT-SoVITS non-streaming sampling.

No PyTorch dependency. NumPy RNG streams are not equivalent to PyTorch streams.
Top-p ties use stable token order here; the upstream torch.sort is not stable,
and FP32 reduction order can change a cumulative threshold decision. Ties and
numerically borderline top-p values can therefore select different tokens;
the equivalence harness records those differences.
"""

from dataclasses import dataclass

import numpy as np


def _softmax(logits):
    exponentials = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True, dtype=np.float32)


def logits_to_probs(logits, previous_tokens=None, *, temperature=1.0,
                    top_k=None, top_p=None, repetition_penalty=1.0):
    """Match the official operation order and repetition-only input mutation.

    Input is a writable FP32 matrix (batch, vocabulary). As in the upstream,
    repeated history entries apply the penalty once, not once per occurrence.
    """
    if logits.ndim != 2 or logits.dtype != np.float32:
        raise ValueError("Expected a writable FP32 (batch, vocabulary) array")
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = np.asarray(previous_tokens, dtype=np.int64)
        score = np.take_along_axis(logits, previous_tokens, axis=1)
        score = np.where(score < 0, score * np.float32(repetition_penalty),
                         score / np.float32(repetition_penalty))
        np.put_along_axis(logits, previous_tokens, score, axis=1)
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-logits, axis=-1, kind="stable")
        sorted_logits = np.take_along_axis(logits, order, axis=1)
        remove_sorted = np.cumsum(_softmax(sorted_logits), axis=-1, dtype=np.float32) > top_p
        remove_sorted[:, 0] = False
        remove = np.empty_like(remove_sorted)
        np.put_along_axis(remove, order, remove_sorted, axis=1)
        logits = np.where(remove, np.float32(-np.inf), logits)
    logits = logits / np.float32(max(temperature, 1e-5))
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("The official top-k sampling path requires top_k > 0 or None")
        k = min(top_k, logits.shape[-1])
        pivot = np.partition(logits, logits.shape[-1] - k, axis=-1)[:, -k, None]
        logits = np.where(logits < pivot, np.float32(-np.inf), logits)
    return _softmax(logits)


def sample(logits, previous_tokens=None, *, exponential_noise=None, rng=None, **kwargs):
    """Return (int32 token column, probabilities), optionally using shared noise."""
    probabilities = logits_to_probs(logits, previous_tokens, **kwargs)
    if exponential_noise is None:
        if rng is None:
            rng = np.random.default_rng()
        exponential_noise = rng.exponential(size=probabilities.shape).astype(np.float32)
    noise = np.asarray(exponential_noise, dtype=np.float32)
    if noise.shape != probabilities.shape or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Exponential noise must be finite, strictly positive, and match probabilities")
    token = np.argmax(probabilities / noise, axis=-1, keepdims=True).astype(np.int32)
    return token, probabilities


def exclude_initial_eos(logits, step_index, eos):
    """The official idx=0..10 path removes the final EOS column as a view."""
    if eos != logits.shape[-1] - 1:
        raise ValueError("The pinned official path requires EOS in the final vocabulary column")
    return logits[:, :-1] if step_index < 11 else logits


@dataclass
class StopResult:
    history: np.ndarray
    stopped: bool
    reasons: tuple[str, ...]
    returned_index: int

    def official_suffix(self):
        """Preserve caller [-idx:] exactly, including -0 selecting everything."""
        return self.history[-self.returned_index:]


def finish_nonstream_step(history, sampled_token, penalized_logits, *, eos,
                          step_index, prefix_length, early_stop_num=-1, reference_free=False):
    """Single-request transition from infer_panel_naive, including its limits.

    penalized_logits is the sampling input after repetition mutation, before
    top-p/temperature/top-k. EOS by argmax also removes a non-EOS sampled token.
    The idx return and strict > early-stop check are preserved, not corrected.
    """
    history = np.concatenate((np.asarray(history, dtype=np.int64), [sampled_token]))
    reasons = []
    if early_stop_num != -1 and len(history) - prefix_length > early_stop_num:
        reasons.append("early_stop_num")
    argmax_eos = int(np.argmax(penalized_logits)) == eos
    sample_eos = sampled_token == eos
    if argmax_eos or sample_eos:
        if argmax_eos:
            reasons.append("argmax_eos")
        if sample_eos:
            reasons.append("sample_eos")
        history = history[:-1]
    if step_index == 1499:
        reasons.append("iteration_limit")
    if reasons and len(history) == 0:
        history = np.zeros(1, dtype=np.int64)
    return StopResult(history, bool(reasons), tuple(reasons), 0 if reference_free else step_index)
