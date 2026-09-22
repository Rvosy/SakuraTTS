"""Private complete-inference worker. The first IPC frame is its startup gate."""

import logging
import os
from pathlib import Path
import queue
import signal
import sys
import threading

from .inference_process import MAX_PCM, read_frame, write_frame


def main(inference_factory=None):
    # Keep the wire descriptor private; Python and native stdout both become stderr.
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def send(metadata, pcm=b""):
        with lock:
            write_frame(wire, metadata, pcm)

    class WireLogging(logging.Handler):
        def emit(self, record):
            try:
                send({"type": "log", "level": record.levelno, "logger": record.name,
                    "message": self.format(record),
                    "extra": {key: getattr(record, key) for key in ("block", "text") if hasattr(record, key)}})
            except (OSError, ValueError):
                pass

    handler = WireLogging()
    logging.getLogger("sakuratts").addHandler(handler)
    logging.getLogger("sakuratts").setLevel(logging.DEBUG)
    commands = queue.Queue(maxsize=1)
    state = {"id": None, "cancel": threading.Event()}
    shutdown_complete = threading.Event()

    def read_commands():
        try:
            while True:
                metadata, pcm = read_frame(sys.stdin.buffer)
                if pcm:
                    raise ValueError("Commands cannot include PCM")
                if metadata.get("operation") == "cancel":
                    if metadata.get("id") == state["id"]:
                        state["cancel"].set()
                    continue
                cancellation = threading.Event()
                state.update(id=metadata.get("id"), cancel=cancellation)
                commands.put((metadata, cancellation))
        except (EOFError, OSError, ValueError) as error:
            if isinstance(error, EOFError) and shutdown_complete.is_set():
                return
            # On POSIX, pipe EOF is also the parent-death notification. On Windows
            # the parent's noninheritable Job Object owns the complete tree.
            if os.name != "nt":
                os.killpg(os.getpgrp(), signal.SIGKILL)
            os._exit(1)

    initial, initial_pcm = read_frame(sys.stdin.buffer)
    if initial_pcm or initial.get("operation") != "initialize":
        raise ValueError("Inference worker requires an initialization command")
    # Complete NumPy's native Windows initialization before another thread
    # holds a blocking stdin read; concurrent startup can deadlock in the CRT.
    # The startup gate above must still precede imports, so the parent owns this
    # process through its Job Object before any inference initialization starts.
    import numpy as np
    state["id"] = initial.get("id")
    commands.put((initial, state["cancel"]))
    control_thread = threading.Thread(target=read_commands, daemon=True, name="sakuratts-control")
    control_thread.start()
    inference = None

    def snapshot():
        model = inference.model
        return {"model": {"path": str(model.path), "manifest": model.manifest} if model is not None else None,
            "settings": inference.settings, "reference_audio": inference.reference_audio}

    def decode_model(value):
        if isinstance(value, dict):
            from ..model import Model
            return Model(Path(value["path"]), value["manifest"])
        return value

    def audio_blocks(serial, kind, pcm, rate, cancelled):
        pcm = np.asarray(pcm)
        if pcm.ndim != 1 or pcm.dtype != np.dtype("int16"):
            raise ValueError("Inference audio must be mono int16 PCM")
        if kind == "fragment" and not pcm.size:
            send({"id": serial, "type": kind, "sample_rate": rate, "end_of_fragment": True})
        for start in range(0, pcm.size, MAX_PCM // 2):
            if cancelled():
                from .cancellation import SynthesisCancelled
                raise SynthesisCancelled("Synthesis cancelled")
            block = pcm[start:start + MAX_PCM // 2].astype("<i2", copy=False).tobytes()
            metadata = {"id": serial, "type": kind, "sample_rate": rate}
            if kind == "fragment":
                metadata["end_of_fragment"] = start + MAX_PCM // 2 >= pcm.size
            send(metadata, block)

    try:
        while True:
            message, cancellation = commands.get()
            operation, serial = message.get("operation"), message.get("id")
            from .logging import request_id
            request_id.set(message.get("request_id", "-"))
            try:
                if operation == "initialize":
                    if inference is not None:
                        raise ValueError("Inference worker is already initialized")
                    if inference_factory is None:
                        from functools import partial
                        from ..engine import Inference
                        inference_factory = partial(Inference, _allow_staged=True)
                    configuration = message["configuration"]
                    saved = message.get("snapshot")
                    options = {"backend": configuration["backend"]} if "backend" in configuration else {}
                    if saved is None:
                        inference = inference_factory(decode_model(configuration["model"]),
                            tts_config=configuration["tts_config"], experimental=configuration["experimental"], **options)
                    else:
                        inference = inference_factory(experimental=configuration["experimental"], **options)
                        inference.settings = saved["settings"]
                        if saved["model"] is not None:
                            inference._activate(decode_model(saved["model"]))
                        if saved.get("reference_audio"):
                            inference.set_reference_audio(saved["reference_audio"])
                    if inference.info() is None:
                        raise ValueError("Model weights are not configured")
                elif inference is None:
                    raise ValueError("Inference worker has not been initialized")
                elif operation == "shutdown":
                    inference.close()
                    inference = None
                    shutdown_complete.set()
                    send({"id": serial, "type": "result", "info": None})
                    break
                elif operation == "set_weights":
                    if message["kind"] not in ("gpt", "sovits"):
                        raise ValueError("Unknown weight kind")
                    inference.set_weights(message["kind"], message["path"])
                elif operation == "set_reference_audio":
                    inference.set_reference_audio(message["path"])
                elif operation == "tts":
                    streaming = message["streaming"]
                    def fragment(pcm, rate):
                        audio_blocks(serial, "fragment", pcm, rate, cancellation.is_set)
                    result = inference.tts(message["request"],
                        on_fragment=fragment if streaming else None, cancel_requested=cancellation.is_set)
                    if not streaming:
                        audio_blocks(serial, "audio", result.pcm, result.sample_rate, cancellation.is_set)
                    send({"id": serial, "type": "result", "sample_rate": result.sample_rate,
                        "report": result.report, "info": inference.info(), "snapshot": snapshot()})
                    continue
                else:
                    raise ValueError("Unknown inference operation")
                send({"id": serial, "type": "result", "info": inference.info(), "snapshot": snapshot()})
            except Exception as error:
                logging.getLogger("sakuratts.engine").debug("Inference operation failed", exc_info=True)
                info = inference.info() if inference is not None else None
                reply = {"id": serial, "type": "error", "error_type": type(error).__name__,
                    "message": str(error), "info": info}
                if info is not None:
                    reply["snapshot"] = snapshot()
                send(reply)
                if operation in ("initialize", "shutdown"):
                    break
    finally:
        try:
            if inference is not None:
                inference.close()
        finally:
            # The parent closes stdin after a shutdown result; initialization
            # failures are reaped by its existing process-tree cleanup.
            control_thread.join()
            logging.getLogger("sakuratts").removeHandler(handler)
            wire.close()


if __name__ == "__main__":
    main()
