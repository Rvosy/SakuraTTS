"""One CPU reference preparer owned by the active inference session."""

import logging
import os
from pathlib import Path
import subprocess
import threading

from ..runtime.process_tree import ProcessTree
from ..runtime.protocol import read_message, write_message


class ReferencePreparer:
    def __init__(self):
        self.process = None
        self.tree = None
        self.reader = None

    @staticmethod
    def _stderr(stream):
        for line in iter(stream.readline, b''):
            logging.getLogger('sakuratts.prepare').debug('%s', line.decode('utf-8', errors='replace').rstrip())

    def run(self, command, *, env):
        try:
            if self.process is None:
                self.tree = ProcessTree() if os.name == 'nt' else None
                self.process = subprocess.Popen([command[0], '-B', str(Path(__file__).with_name('prepare_resources.py')), '--serve'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=env, bufsize=0, **(ProcessTree.popen_options() if self.tree else {}))
                if self.tree:
                    self.tree.bind(self.process)
                self.reader = threading.Thread(target=self._stderr, args=(self.process.stderr,), daemon=True)
                self.reader.start()
            # converter owns the interpreter and script; only CLI arguments cross the wire.
            write_message(self.process.stdin, {'arguments': command[3:]})
            result, _ = read_message(self.process.stdout)
            if result['status'] != 'ok':
                raise RuntimeError(result['error'])
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.tree is not None:
            self.tree.close()
        elif self.process is not None:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.reader is not None:
            self.reader.join(timeout=5)
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()
        self.process = self.tree = self.reader = None
