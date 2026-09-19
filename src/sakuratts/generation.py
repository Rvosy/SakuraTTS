"""Single-request semantic generation using own history and official stopping.

The model provides prefill/decode; this loop never accepts target tokens.
Explicit random draws and an optional observer support reproducible diagnosis.
Current validated sampling scope is top_p=1 with a nonempty reference prefix.
"""

from dataclasses import dataclass

import numpy as np

from .sampling import StopResult, exclude_initial_eos, finish_nonstream_step, sample


class SynthesisCancelled(RuntimeError):
    """The caller requested cancellation at this completed compute boundary."""

    def __init__(self, stage):
        self.stage = stage
        super().__init__(f"Synthesis cancelled at {stage}")


def check_cancelled(cancel_requested, stage):
    if cancel_requested is not None and cancel_requested():
        raise SynthesisCancelled(stage)


@dataclass
class SemanticGeneration:
    sampled_tokens: np.ndarray
    stop: StopResult

    @property
    def semantic(self):
        return self.stop.official_suffix()[None, None, :]


def generate_semantic(model, phones, prompt, bert, *, eos, top_k=15, top_p=1.0,
                      temperature=1.0, repetition_penalty=1.35, early_stop_num=-1,
                      rng=None, random_draw=None, observer=None, cancel_requested=None):
    """Generate with NumPy sampling; model capacity errors remain explicit.

    random_draw(index, probability_shape) may return saved exponential noise.
    observer(index, raw_logits, token, probabilities, stop) runs after the state
    transition. Normal calls omit both hooks and do not retain per-step logits.
    cancel_requested is polled around model calls and after each sampling step.
    Cancellation raises without returning partial tokens or interrupting kernels.
    """
    prompt = np.asarray(prompt)
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError("Expected one nonempty reference semantic prefix")
    if top_p != 1.0:
        raise ValueError("Top-p below 1 has unresolved compatibility boundaries")
    if rng is None:
        rng = np.random.default_rng()
    history = prompt[0].copy()
    sampled = []
    check_cancelled(cancel_requested, "before_prefill")
    logits = model.prefill(phones, prompt, bert)
    check_cancelled(cancel_requested, "after_prefill")
    for index in range(1500):
        raw = np.asarray(logits).copy()
        active = exclude_initial_eos(raw.copy() if observer is not None else raw, index, eos)
        noise = None if random_draw is None else random_draw(index, active.shape)
        token, probabilities = sample(
            active, history[None, :], exponential_noise=noise, rng=rng,
            top_k=top_k, top_p=top_p, temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        actual_token = int(token[0, 0])
        sampled.append(actual_token)
        stop = finish_nonstream_step(
            history, actual_token, active[0], eos=eos, step_index=index,
            prefix_length=prompt.shape[1], early_stop_num=early_stop_num,
        )
        history = stop.history
        if observer is not None:
            observer(index, raw, actual_token, probabilities, stop)
        check_cancelled(cancel_requested, "semantic_step")
        if stop.stopped:
            return SemanticGeneration(np.asarray(sampled, dtype=np.int64), stop)
        logits = model.decode(actual_token)
        check_cancelled(cancel_requested, "after_decode")
    raise AssertionError("The official iteration limit did not produce a stop")
