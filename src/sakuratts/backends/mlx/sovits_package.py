"""Read a V2Pro acoustic weight archive once, independently of its backend."""

from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from sakuratts._internal.reference_condition import sha256_file as sha256
from sakuratts._internal.weight_storage import read_fp32



def validate_manifest(manifest):
    if manifest.get("dtype") != "float32":
        raise ValueError("MLX acoustic execution requires FP32; FP16 is not implemented")
    if manifest.get("config", {}).get("model", {}).get("version") != "v2Pro":
        raise ValueError("MLX acoustic execution supports only V2Pro; V2ProPlus is not implemented")
    if manifest.get("format") != "sakuratts-sovits-decode-fp32-v1":
        raise ValueError("MLX requires a native sakuratts-sovits-decode-fp32-v1 package, not ONNX")


class SoVITSPackage:
    def __init__(self, manifest, archive):
        self.manifest = manifest
        self._archive = archive

    @classmethod
    @contextmanager
    def open(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        validate_manifest(manifest)
        path = directory / manifest["weights"]["file"]
        with np.load(path, allow_pickle=False) as archive:
            yield cls(manifest, archive)

    def tensors(self, *prefixes, names=(), exclude=()):
        """Yield selected exact FP32 arrays; the caller owns backend placement.

        Each component consumes a disjoint prefix. Arrays are streamed rather
        than retained as a second full CPU copy of the acoustic weights.
        """
        for name in self._archive.files:
            if name not in exclude and (name in names or name.startswith(prefixes)):
                yield name, read_fp32(self._archive, name)
