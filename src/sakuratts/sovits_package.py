"""Read a V2Pro acoustic weight archive once, independently of its backend."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np

from .weight_storage import read_fp32, validate_storage


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SoVITSPackage:
    def __init__(self, manifest, archive):
        self.manifest = manifest
        self._archive = archive

    @classmethod
    @contextmanager
    def open(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["format"] != "sakuratts-sovits-decode-fp32-v1"
                or manifest["config"]["model"]["version"] != "v2Pro"
                or manifest["dtype"] != "float32"):
            raise ValueError("Expected the current V2Pro FP32 acoustic package")
        path = directory / manifest["weights"]["file"]
        if sha256(path) != manifest["weights"]["sha256"]:
            raise ValueError("Acoustic weights checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(manifest["tensor_sources"]):
                raise ValueError("Acoustic archive differs from the declared tensor set")
            validate_storage(manifest, archive.files)
            yield cls(manifest, archive)

    def tensors(self, *prefixes, names=(), exclude=()):
        """Yield selected exact FP32 arrays; the caller owns backend placement.

        Each component consumes a disjoint prefix. Arrays are streamed rather
        than retained as a second full CPU copy of the acoustic weights.
        """
        for name in self.manifest["tensor_sources"]:
            if name not in exclude and (name in names or name.startswith(prefixes)):
                yield name, read_fp32(self._archive, self.manifest, name)
