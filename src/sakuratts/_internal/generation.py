"""Single-request semantic generation using own history and official stopping.

The model provides prefill/decode; this loop never accepts target tokens.
Explicit random draws and an optional observer support reproducible diagnosis.
Current validated sampling scope is top_p=1 with a nonempty reference prefix.
"""

from dataclasses import dataclass
import logging
import sys
import time

import numpy as np

from sakuratts._internal.sampling import StopResult, exclude_initial_eos, finish_nonstream_step, sample
from sakuratts._internal.logging import set_stage, terminal_progress_enabled

logger = logging.getLogger("sakuratts.inference")


class _SemanticProgress:
    """Display completed sampling steps without inspecting GPU state or RNG."""

    def __init__(self, prefix_length):
        self.prefix_length = prefix_length

    def __enter__(self):
        self.enabled = logger.isEnabledFor(logging.INFO)
        self.bar = None
        self.count = 0
        self.reasons = ()
        self.started = self.last_log = time.perf_counter()
        set_stage("语义预测")
        if self.enabled:
            logger.info("预测语义Token", extra={"block": "stage"})
            logger.debug("开始语义预测，最多 1500 步")
            if terminal_progress_enabled(logger):
                try:
                    from tqdm import tqdm
                except ImportError:
                    pass  # Library-only installations can use plain logs.
                else:
                    self.bar = tqdm(total=None, desc="    GPT   ", unit="it", file=sys.stderr,
                        mininterval=0.2, dynamic_ncols=True, ascii=True, leave=False,
                        bar_format="{desc}  {n_fmt} it · {rate_fmt} · {elapsed}")
        return self

    def update(self, count, stop):
        if not self.enabled:
            return
        self.count, self.reasons = count, stop.reasons
        if self.bar is not None:
            self.bar.update(1)
        now = time.perf_counter()
        if now - self.last_log >= 1.0 and not stop.stopped:
            logger.log(logging.INFO if self.bar is None else logging.DEBUG,
                       "GPT     %d it · %.1f it/s · %.2f s", count,
                       count / max(now - self.started, 1e-9), now - self.started,
                       extra={"block": "progress"})
            self.last_log = now

    def __exit__(self, error_type, error, traceback):
        if self.bar is not None:
            self.bar.close()
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started
        if error is not None:
            state = "取消" if isinstance(error, SynthesisCancelled) else "失败"
            logger.debug("语义预测%s | 已采样 %d 步 | %.3f s | %s", state, self.count, elapsed, error)
            return
        labels = {"argmax_eos": "argmax EOS", "sample_eos": "采样 EOS",
                  "early_stop_num": "长度上限", "iteration_limit": "1500 步上限"}
        logger.debug("语义预测结束 | 采样 %d 次 | %.3f s | %.1f it/s | 停止原因: %s",
                    self.count, elapsed, self.count / max(elapsed, 1e-9),
                    ", ".join(labels.get(reason, reason) for reason in self.reasons))
        limited = bool(set(self.reasons) & {"early_stop_num", "iteration_limit"})
        eos = bool(set(self.reasons) & {"argmax_eos", "sample_eos"})
        logger.info("GPT     %d it · %.1f it/s · %.3f s", self.count,
                    self.count / max(elapsed, 1e-9), elapsed, extra={"block": "progress"})
        # Match upstream batch logging: length after sampling, before EOS removal.
        logger.log(logging.WARNING if limited else logging.INFO,
                   "T2S Decoding %s [%d -> %d]%s", "EOS" if eos else "STOP",
                   self.prefix_length, self.prefix_length + self.count,
                   " · 达到长度上限" if limited else "",
                   extra={"block": "progress"})


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
    with _SemanticProgress(prompt.shape[1]) as progress:
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
            progress.update(index + 1, stop)
            if observer is not None:
                observer(index, raw, actual_token, probabilities, stop)
            check_cancelled(cancel_requested, "semantic_step")
            if stop.stopped:
                return SemanticGeneration(np.asarray(sampled, dtype=np.int64), stop)
            logits = model.decode(actual_token)
            check_cancelled(cancel_requested, "after_decode")
    raise AssertionError("The official iteration limit did not produce a stop")
