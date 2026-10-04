"""Portable single-reference V2Pro/V2ProPlus conditions; NumPy only.

This reader supports original target text with an offline prepared reference.
It cannot prepare a new reference from raw audio or rebuild missing conditions.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


FORMAT = "sakuratts-prepared-reference-v1"
ARRAY_DTYPES = {
    "reference_phones": "int64", "prompt_semantic": "int64",
    "reference_bert": "float32", "ge": "float32", "ge512": "float32",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def validate_arrays(arrays):
    for name, dtype in ARRAY_DTYPES.items():
        array = arrays[name]
        if array.dtype != np.dtype(dtype):
            raise ValueError(f"Reference dtype mismatch: {name}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite reference array: {name}")
    phones, prompt = arrays["reference_phones"], arrays["prompt_semantic"]
    if phones.ndim != 1 or phones.size == 0 or (phones < 0).any():
        raise ValueError("Expected nonempty one-dimensional reference phones")
    if prompt.ndim != 1 or prompt.size == 0 or (prompt < 0).any():
        raise ValueError("Expected nonempty one-dimensional reference semantic tokens")
    if arrays["reference_bert"].shape != (1024, phones.size):
        raise ValueError("Reference BERT and phone alignment mismatch")
    if arrays["ge"].shape != (1, 1024, 1) or arrays["ge512"].shape != (1, 512, 1):
        raise ValueError("Expected ge [1,1024,1] and ge512 [1,512,1]")


@dataclass(frozen=True)
class PreparedReference:
    manifest: dict
    reference_phones: np.ndarray
    prompt_semantic: np.ndarray
    reference_bert: np.ndarray
    ge: np.ndarray
    ge512: np.ndarray

    @classmethod
    def load(cls, package):
        """Read the reference arrays; provenance metadata is descriptive."""
        package = Path(package)
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format"] != FORMAT or manifest["model_family"] not in ("v2Pro", "v2ProPlus"):
            raise ValueError("Unsupported reference condition format or model family")
        with np.load(package / manifest["archive"]["file"], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in ARRAY_DTYPES}
        validate_arrays(arrays)
        for array in arrays.values():
            array.setflags(write=False)
        return cls(manifest=manifest, **arrays)


@dataclass(frozen=True)
class BoundAcousticReference:
    """Immutable CPU conditions for one acoustic model.

    The byte-backed arrays cannot be made writable. This object never owns a
    model or the caller's mutable manifest, and switching requires a new model.
    """
    ge: np.ndarray
    ge512: np.ndarray

    @staticmethod
    def _condition(value, name, shape):
        value = np.asarray(value)
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Expected finite FP32 {name} with shape {shape}")
        return value

    @classmethod
    def from_reference(cls, reference, model_manifest):
        if not isinstance(reference, PreparedReference):
            raise TypeError("Binding requires a PreparedReference")
        family = model_manifest["config"]["model"]["version"]
        if family not in ("v2Pro", "v2ProPlus") or reference.manifest["model_family"] != family:
            raise ValueError("Reference family and acoustic architecture must match (V2Pro/V2ProPlus)")
        snapshots = {}
        for name in ("ge", "ge512"):
            shape = tuple(model_manifest["inputs"][name]["shape"])
            value = cls._condition(getattr(reference, name), name, shape)
            snapshots[name] = np.frombuffer(value.tobytes(order="C"), dtype=np.float32).reshape(shape)
        return cls(snapshots["ge"], snapshots["ge512"])

    def validate_conditions(self, ge, ge512):
        for name, value in (("ge", ge), ("ge512", ge512)):
            expected = getattr(self, name)
            value = self._condition(value, name, expected.shape)
            if not np.array_equal(value.view(np.uint32), expected.view(np.uint32)):
                raise ValueError(f"Bound acoustic reference {name} differs; load a new model for this reference")

    def validate_reference(self, reference):
        if not isinstance(reference, PreparedReference):
            raise TypeError("Binding requires a PreparedReference")
        self.validate_conditions(reference.ge, reference.ge512)
