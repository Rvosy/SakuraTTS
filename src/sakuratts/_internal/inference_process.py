"""Lazy synchronous proxy for a separately owned inference process tree."""

from array import array
import json
import logging
import math
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time

from ..engine import Audio, read_inference_configuration
from ..model import Model
from .cancellation import SynthesisCancelled
from .pcm import pcm_from_s16le
from .process_tree import ProcessTree

MAX_METADATA = 4 * 1024 * 1024
MAX_PCM = 128 * 1024
logger = logging.getLogger("sakuratts.engine")


def _read_exact(stream, size):
    result = bytearray(size)
    view = memoryview(result)
    while view:
        count = stream.readinto(view)
        if not count:
            raise EOFError("Inference worker closed its output")
        view = view[count:]
    return result


def read_frame(stream):
    metadata_size, pcm_size = struct.unpack("<II", _read_exact(stream, 8))
    if not 0 < metadata_size <= MAX_METADATA or not 0 <= pcm_size <= MAX_PCM or pcm_size % 2:
        raise ValueError("Invalid inference IPC frame length")
    metadata = json.loads(_read_exact(stream, metadata_size).decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Inference IPC metadata must be an object")
    return metadata, bytes(_read_exact(stream, pcm_size))


def write_frame(stream, metadata, pcm=b""):
    encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not 0 < len(encoded) <= MAX_METADATA or len(pcm) > MAX_PCM or len(pcm) % 2:
        raise ValueError("Invalid inference IPC frame length")
    for data in (struct.pack("<II", len(encoded), len(pcm)), encoded, pcm):
        view = memoryview(data)
        while view:
            count = stream.write(view)
            if not count:
                raise EOFError("Inference worker closed its input")
            view = view[count:]
    stream.flush()


class _RemoteError(Exception):
    def __init__(self, metadata):
        self.metadata = metadata


class ProcessInference:
    """All operations run serially on the service's single inference thread."""

    def __init__(self, model=None, *, tts_config=None, backend=None, experimental=None,
                 startup_timeout=120, operation_timeout=300):
        for value in (startup_timeout, operation_timeout):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Inference timeouts must be finite and positive")
        self.preparation = "runtime_init" if (experimental or {}).get("policy") == "staged" else "model_load"
        resolved_model, settings = read_inference_configuration(model, tts_config=tts_config)
        configured = resolved_model is not None or bool(settings.get("gpt_checkpoint") and settings.get("sovits_checkpoint"))
        if tts_config and not configured:
            raise ValueError("TTS configuration requires both t2s_weights_path and vits_weights_path, or sakuratts.model")
        self._configuration = {"model": self._model_value(model),
            "tts_config": str(tts_config) if tts_config else None, "experimental": dict(experimental or {})}
        if backend is not None:
            self._configuration["backend"] = backend
        # Validate serialization before creating any process, including user options.
        json.dumps(self._configuration, allow_nan=False)
        self.configured = configured
        self.startup_timeout, self.operation_timeout = startup_timeout, operation_timeout
        self._process = self._tree = self._reader = self._writer = self._stderr_reader = None
        self._messages = self._reader_stop = None
        self._outbound = self._write_errors = None
        self._info = self._snapshot = None
        self._serial = 0

    @staticmethod
    def _model_value(model):
        if isinstance(model, Model):
            return {"path": str(model.path), "manifest": model.manifest}
        return str(model) if model is not None else None

    @property
    def pid(self):
        return self._process.pid if self.alive else None

    @property
    def alive(self):
        return self._process is not None and self._process.poll() is None

    def info(self):
        return dict(self._info) if self.alive and self._info is not None else None

    def _command(self):
        # Private override point for hardware-free subprocess integration tests.
        return [sys.executable, "-m", "sakuratts._internal.inference_worker"]

    @staticmethod
    def _read_output(stream, messages, stop):
        try:
            while not stop.is_set():
                message = read_frame(stream)
                while not stop.is_set():
                    try:
                        messages.put(message, timeout=.05)
                        break
                    except queue.Full:
                        pass
        except Exception as error:
            # Queued tracebacks retain this frame, its queue, and the finished
            # thread's native handles until cyclic garbage collection runs.
            error.__traceback__ = error.__context__ = error.__cause__ = None
            while not stop.is_set():
                try:
                    messages.put(error, timeout=.05)
                    break
                except queue.Full:
                    pass

    @staticmethod
    def _read_stderr(stream):
        try:
            while line := stream.readline(8192):
                logger.debug("Inference worker: %s", line.decode("utf-8", errors="replace").rstrip())
        except (OSError, ValueError):
            pass

    @staticmethod
    def _write_commands(stream, outbound, errors, stop):
        try:
            while not stop.is_set():
                try:
                    message = outbound.get(timeout=.05)
                except queue.Empty:
                    continue
                write_frame(stream, message)
        except Exception as error:
            error.__traceback__ = error.__context__ = error.__cause__ = None
            errors.put_nowait(error)

    def wake(self):
        if self.alive:
            return
        if not self.configured:
            raise ValueError("Model weights are not configured")
        self._dispose()
        tree = self._tree = ProcessTree()
        try:
            environment = os.environ.copy()
            root = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            process = self._process = subprocess.Popen(self._command(), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=environment,
                **tree.popen_options())
            # The child does not import inference backends before its first frame.
            tree.bind(process)
            self._messages, self._reader_stop = queue.Queue(maxsize=2), threading.Event()
            self._outbound, self._write_errors = queue.Queue(maxsize=2), queue.Queue(maxsize=1)
            self._reader = threading.Thread(target=self._read_output,
                args=(process.stdout, self._messages, self._reader_stop), daemon=True,
                name="sakuratts-inference-ipc")
            self._writer = threading.Thread(target=self._write_commands,
                args=(process.stdin, self._outbound, self._write_errors, self._reader_stop), daemon=True,
                name="sakuratts-inference-input")
            self._stderr_reader = threading.Thread(target=self._read_stderr, args=(process.stderr,),
                daemon=True, name="sakuratts-inference-stderr")
            self._reader.start()
            self._writer.start()
            self._stderr_reader.start()
            self._exchange("initialize", configuration=self._configuration, snapshot=self._snapshot,
                timeout=self.startup_timeout)
        except BaseException:
            self._dispose()
            raise

    def _exchange(self, operation, *, timeout=None, on_fragment=None, cancel_requested=None, **payload):
        self._serial += 1
        serial = self._serial
        deadline = time.monotonic() + (timeout or self.operation_timeout)
        cancellation_sent = False
        audio_pcm = array("h")
        fragment_pcm = array("h")
        fragment_rate = None
        from .logging import request_id
        try:
            self._outbound.put_nowait({"id": serial, "operation": operation,
                "request_id": request_id.get(), **payload})
            while True:
                try:
                    write_error = self._write_errors.get_nowait()
                except queue.Empty:
                    pass
                else:
                    raise write_error
                if cancel_requested is not None and cancel_requested() and not cancellation_sent:
                    self._outbound.put_nowait({"id": serial, "operation": "cancel"})
                    cancellation_sent = True
                    audio_pcm = array("h")
                    fragment_pcm = array("h")
                    deadline = min(deadline, time.monotonic() + 5)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Inference {operation} timed out")
                try:
                    message = self._messages.get(timeout=min(.05, remaining))
                except queue.Empty:
                    if not self.alive:
                        raise RuntimeError("Inference worker exited unexpectedly")
                    continue
                if isinstance(message, Exception):
                    raise message
                metadata, pcm = message
                if metadata.get("type") == "log":
                    logging.getLogger(metadata.get("logger", "sakuratts.engine")).log(
                        metadata.get("level", logging.INFO), "%s", metadata.get("message", ""),
                        extra=metadata.get("extra", {}))
                    continue
                if metadata.get("id") != serial:
                    raise ValueError("Inference worker returned an unexpected request identity")
                kind = metadata.get("type")
                if kind in ("fragment", "audio"):
                    rate = metadata.get("sample_rate")
                    if not isinstance(rate, int) or isinstance(rate, bool) or not 8000 <= rate <= 384000:
                        raise ValueError("Inference worker returned an invalid sample rate")
                    block = pcm_from_s16le(pcm)
                    if kind == "fragment":
                        if not isinstance(metadata.get("end_of_fragment"), bool):
                            raise ValueError("Inference worker omitted the fragment boundary")
                        if fragment_rate is not None and fragment_rate != rate:
                            raise ValueError("Inference worker changed sample rate within a fragment")
                        fragment_rate = rate
                        if not cancellation_sent:
                            fragment_pcm.extend(block)
                        if metadata["end_of_fragment"]:
                            if on_fragment is not None and not cancellation_sent:
                                complete = fragment_pcm
                                fragment_pcm = array("h")
                                fragment_rate = None
                                try:
                                    on_fragment(complete, rate)
                                finally:
                                    del complete
                            fragment_pcm = array("h")
                            fragment_rate = None
                    elif kind == "audio" and not cancellation_sent:
                        audio_pcm.extend(block)
                    continue
                if kind == "error":
                    raise _RemoteError(metadata)
                if kind != "result" or pcm:
                    raise ValueError("Inference worker returned an invalid response")
                if fragment_rate is not None:
                    raise ValueError("Inference worker ended with an incomplete audio fragment")
                self._info = metadata.get("info")
                self._snapshot = metadata.get("snapshot", self._snapshot)
                if cancellation_sent:
                    # The worker may finish while cancellation is in flight. Its
                    # completed response leaves the transport safe to reuse.
                    raise _RemoteError(dict(metadata, error_type="SynthesisCancelled", message="Synthesis cancelled"))
                if operation == "tts":
                    report = metadata["report"]
                    return Audio(audio_pcm, metadata["sample_rate"], report)
                return None
        except _RemoteError as error:
            metadata = error.metadata
            kind = metadata.get("error_type")
            self._info = metadata.get("info")
            if self._info is not None:
                self._snapshot = metadata.get("snapshot", self._snapshot)
            if self._info is None or operation in ("initialize", "shutdown"):
                self._dispose()
            if kind == "SynthesisCancelled":
                raise SynthesisCancelled(metadata.get("message", "Synthesis cancelled")) from None
            error_type = {"ValueError": ValueError, "NotImplementedError": NotImplementedError,
                "FileNotFoundError": FileNotFoundError}.get(kind, RuntimeError)
            raise error_type(metadata.get("message", "Inference worker failed")) from None
        except BaseException:
            self._dispose()
            raise

    def tts(self, request, *, on_fragment=None, cancel_requested=None):
        self.wake()
        return self._exchange("tts", request=request, streaming=on_fragment is not None,
            on_fragment=on_fragment, cancel_requested=cancel_requested)

    def set_weights(self, kind, path):
        if kind not in ("gpt", "sovits"):
            raise ValueError("Unknown weight kind")
        self.wake()
        self._exchange("set_weights", kind=kind, path=str(path))

    def set_reference_audio(self, path):
        self.wake()
        self._exchange("set_reference_audio", path=str(path))

    def _dispose(self):
        # Keep ownership if termination fails so callers can retry cleanup.
        if self._tree is not None:
            self._tree.close()
        if self._reader_stop is not None:
            self._reader_stop.set()
        process = self._process
        if process is not None:
            for reader in (self._reader, self._writer, self._stderr_reader):
                if reader is not None:
                    reader.join(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        self._tree = self._process = self._reader = self._writer = self._stderr_reader = None
        self._messages = self._reader_stop = None
        self._outbound = self._write_errors = None
        self._info = None

    def sleep(self):
        if self.alive:
            try:
                self._exchange("shutdown", timeout=min(5, self.operation_timeout))
            finally:
                self._dispose()
        else:
            self._dispose()

    def close(self):
        self.sleep()
