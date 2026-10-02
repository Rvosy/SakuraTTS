"""Public, synchronous inference API shared by the CLI and HTTP service."""

from dataclasses import dataclass
import io
from pathlib import Path
from threading import Lock
import wave

from .model import Model
from .runtime.pcm import pcm_s16le_bytes


class BusyError(RuntimeError):
    """An engine already has an active request."""


@dataclass
class Audio:
    pcm: object
    sample_rate: int
    report: dict

    def wav_bytes(self):
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm_s16le_bytes(self.pcm))
        return stream.getvalue()

    def save(self, path):
        path = Path(path)
        if path.suffix.lower() != ".wav":
            raise ValueError("Choose a .wav output path")
        data = self.wav_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        return path


class Engine:
    """One model, one active request. Close explicitly or use a with block."""

    def __init__(self, model, runtime, *, profile=None):
        self.model = model
        self._runtime = runtime
        self.profile = profile
        self._lock = Lock()
        self._closed = False

    @classmethod
    def load(cls, path, *, backend=None, profile=None, experimental=None, load_references=True):
        model = path if isinstance(path, Model) else Model.load(path)
        from .backends import create_runtime
        from .profiles import resolve_profile, validate_runtime_precision
        selected = model.backend if backend is None else backend
        profile, options = resolve_profile(selected, profile, experimental)
        runtime = create_runtime(model, backend=backend,
            experimental=options, load_references=load_references)
        try:
            if profile is not None:
                validate_runtime_precision(selected, profile, runtime)
        except BaseException as error:
            try:
                runtime.close()
            except BaseException as cleanup_error:
                error.add_note(f"Runtime cleanup failed after profile validation: {cleanup_error!r}")
            raise
        return cls(model, runtime, profile=profile)

    def synthesize(self, text, *, reference=None, seed=1234, language="ja",
                   split_method="cut0", top_k=15, top_p=1., temperature=1.,
                   repetition_penalty=1.35, early_stop_num=2700, cancel_requested=None,
                   fragment_interval=0.3, on_fragment=None, collect_audio=True, split_bucket=False):
        if not self._lock.acquire(blocking=False):
            raise BusyError("This engine already has an active request")
        try:
            if self._closed:
                raise RuntimeError("Engine is closed")
            pcm, report = self._runtime.synthesize(text, reference=reference, seed=seed,
                language=language, split_method=split_method, top_k=top_k, top_p=top_p,
                temperature=temperature, repetition_penalty=repetition_penalty,
                early_stop_num=early_stop_num, cancel_requested=cancel_requested,
                fragment_interval=fragment_interval, on_fragment=on_fragment, collect_audio=collect_audio,
                split_bucket=split_bucket)
            if self.profile is not None:
                report["profile"] = self.profile
            return Audio(pcm, report["sample_rate"], report)
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            if not self._closed:
                self._runtime.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load(path, **kwargs):
    return Engine.load(path, **kwargs)


def __getattr__(name):
    # Preserve the earlier inference imports without a circular dependency.
    if name == "Inference":
        from .TTS_infer_pack.TTS import Inference
        return Inference
    if name == "read_inference_configuration":
        from .TTS_infer_pack.config import read_inference_configuration
        return read_inference_configuration
    raise AttributeError("module 'sakuratts.engine' has no attribute " + repr(name))
