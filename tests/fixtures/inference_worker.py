"""Hardware-free inference implementation run behind the production worker loop."""

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from sakuratts.engine import Audio
from sakuratts.model import Model
from sakuratts._internal.generation import SynthesisCancelled
from sakuratts._internal.inference_worker import main


class FakeInference:
    def __init__(self, model=None, *, tts_config=None, experimental=None, backend=None):
        self.settings = {"loads": 1, "backend": backend}
        self.reference_audio = None
        self.model = None
        self.children = []
        if model is not None:
            self._activate(model if isinstance(model, Model) else Model(Path(model), {"name": "initial"}))
        if tts_config:
            config = json.loads(Path(tts_config).read_text(encoding="utf-8"))
            if config.get("sleep"):
                time.sleep(config["sleep"])
            if config.get("error"):
                raise ValueError("Invalid fake configuration")
            if config.get("prepare_child"):
                from sakuratts._internal.logging import run_conversion
                logger = logging.getLogger("sakuratts.converter")
                logger.setLevel(config.get("preparation_log_level", "DEBUG"))
                run_conversion([sys.executable, "-c",
                    "import sys; assert sys.stdin.buffer.read() == b''; print('ready')"], env=dict(os.environ))
            if config.get("child_file"):
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self.children.append(child)
                Path(config["child_file"]).write_text(str(child.pid))
            self._activate(Model(Path(tts_config), {"name": "configured"}))

    def _activate(self, model):
        self.model = model
        self.reference_audio = None

    def info(self):
        return {"name": self.model.name} if self.model else None

    def tts(self, request, *, on_fragment=None, cancel_requested=None):
        if request.get("crash"):
            os._exit(9)
        if request.get("hang"):
            time.sleep(120)
        if request.get("error"):
            raise ValueError("Invalid fake request")
        count = request.get("chunks", 3)
        pcm = np.arange(request.get("samples", 16), dtype=np.int16)
        for _ in range(count):
            time.sleep(request.get("delay", 0))
            if cancel_requested and cancel_requested():
                raise SynthesisCancelled("Fake cancelled")
            if on_fragment:
                on_fragment(pcm, 32000)
        return Audio(np.empty(0, np.int16) if on_fragment else np.tile(pcm, count), 32000,
            {"sample_rate": 32000, "name": self.model.name if self.model else None,
             "reference": self.reference_audio, "pid": os.getpid(), "backend": self.settings["backend"]})

    def set_weights(self, kind, path):
        if path == "bad":
            raise ValueError("Fake weight switch failed")
        if path == "broken":
            self.model = None
            raise ValueError("Fake activation failed")
        self.model = Model(self.model.path, dict(self.model.manifest, name=path))
        self.settings[kind] = path
        self.reference_audio = None

    def set_reference_audio(self, path):
        self.reference_audio = path

    def close(self):
        # Intentionally leave descendants alive to exercise process-tree ownership.
        self.model = None


if __name__ == "__main__":
    main(FakeInference)
