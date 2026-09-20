"""Explicit classic OpenJTalk frontend profile in an independent CPU process.

The existing pyopenjtalk-plus frontend keeps its own behavior. This profile
matches voices prepared with the classic pyopenjtalk 0.3.4 implementation.
"""

import os
from pathlib import Path
import subprocess

from sakuratts._internal.protocol import read_message, write_message
from sakuratts.frontend.japanese import JapaneseG2P


class ClassicJapaneseG2P:
    normalize = staticmethod(JapaneseG2P.normalize)

    def __init__(self, python, module_directory, main_dictionary, user_dictionary):
        self.python = Path(python).resolve(strict=True)
        self.module_directory = Path(module_directory).resolve(strict=True)
        self.main_dictionary = Path(main_dictionary).resolve(strict=True)
        self.user_dictionary = Path(user_dictionary).resolve(strict=True)
        if not self.python.is_file() or not self.user_dictionary.is_file():
            raise ValueError("Classic frontend Python and user dictionary must be files")
        if not self.module_directory.is_dir() or not self.main_dictionary.is_dir():
            raise ValueError("Classic frontend module and main dictionary directories are required")
        self.process = None
        self.runtime = None
        self._closed = False
        self._start()

    def _start(self):
        command = [str(self.python), "-B", str(Path(__file__).resolve().parents[1] / "_internal/classic_japanese_worker.py"),
                   "--module-directory", str(self.module_directory),
                   "--dictionary", str(self.main_dictionary), "--user-dictionary", str(self.user_dictionary)]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
                           OPEN_JTALK_DICT_DIR=str(self.main_dictionary))
        environment.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            self.runtime, arrays = read_message(self.process.stdout)
            if self.runtime.get("status") != "ready" or arrays:
                raise RuntimeError(self.runtime.get("error", "Classic frontend did not become ready"))
            if self.runtime.get("implementation") != "pyopenjtalk-classic" or self.runtime.get("version") != "0.3.4":
                raise RuntimeError("Classic frontend requires pyopenjtalk 0.3.4")
        except BaseException:
            self._stop()
            raise

    def g2p(self, normalized_text):
        if self._closed:
            raise RuntimeError("ClassicJapaneseG2P is closed")
        if not isinstance(normalized_text, str):
            raise TypeError("Japanese G2P input must be a string")
        if self.process is None:
            self._start()
        try:
            write_message(self.process.stdin, {"command": "g2p", "text": normalized_text})
            response, arrays = read_message(self.process.stdout)
            if response.get("status") != "ok":
                raise RuntimeError(response.get("error", "Classic Japanese G2P failed"))
            phones = response.get("phones")
            if arrays or not isinstance(phones, list) or any(not isinstance(phone, str) for phone in phones):
                raise ValueError("Classic frontend returned an invalid phone sequence")
            return phones
        except BaseException:
            self._stop()
            raise

    def _stop(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    write_message(process.stdin, {"command": "close"})
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    process.wait(timeout=5)
        finally:
            process.stdin.close()
            process.stdout.close()

    def close(self):
        self._closed = True
        self._stop()
