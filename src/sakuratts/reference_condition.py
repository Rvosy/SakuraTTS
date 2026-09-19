"""Portable single-reference V2Pro conditions; NumPy only, no model loading.

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
    if set(arrays) != set(ARRAY_DTYPES):
        raise ValueError("Reference package must contain exactly the five required arrays")
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
        raise ValueError("Expected the selected V2Pro ge/ge512 dimensions")


@dataclass(frozen=True)
class PreparedReference:
    manifest: dict
    reference_phones: np.ndarray
    prompt_semantic: np.ndarray
    reference_bert: np.ndarray
    ge: np.ndarray
    ge512: np.ndarray

    @classmethod
    def load(cls, package, *, gpt_checkpoint_sha256, sovits_checkpoint_sha256,
             reference_text=None, reference_language=None, audio_sha256=None,
             official_commit=None, manifest_sha256=None):
        """Load immutable arrays after checking model and optional caller identity.

        reference_text means the original reference transcript, not target text
        or the punctuated prompt. Provenance paths are descriptive and never read.
        """
        if not gpt_checkpoint_sha256 or not sovits_checkpoint_sha256:
            raise ValueError("Both GPT and SoVITS checkpoint identities are required")
        package = Path(package)
        manifest_path = package / "manifest.json"
        if manifest_sha256 is not None and sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("Reference manifest SHA-256 mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["format"] != FORMAT or manifest["model_family"] != "v2Pro":
            raise ValueError("Unsupported reference condition format or model family")
        identity = manifest["identity"]
        expected = {
            "gpt_checkpoint_sha256": gpt_checkpoint_sha256,
            "sovits_checkpoint_sha256": sovits_checkpoint_sha256,
            "reference_text": reference_text, "reference_language": reference_language,
            "audio_sha256": audio_sha256, "official_commit": official_commit,
        }
        for field, value in expected.items():
            if value is not None and identity[field] != value:
                raise ValueError(f"Reference identity mismatch: {field}")
        archive_info = manifest["archive"]
        if archive_info["file"] != "conditions.npz":
            raise ValueError("Unsupported reference archive filename")
        archive_path = package / archive_info["file"]
        if archive_path.stat().st_size != archive_info["bytes"] or sha256_file(archive_path) != archive_info["sha256"]:
            raise ValueError("Reference archive SHA-256 or size mismatch")
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        validate_arrays(arrays)
        if set(manifest["arrays"]) != set(arrays):
            raise ValueError("Reference array metadata does not cover the archive")
        for name, array in arrays.items():
            spec = manifest["arrays"][name]
            if (spec["dtype"] != str(array.dtype) or spec["shape"] != list(array.shape)
                    or spec["bytes"] != array.nbytes or spec["sha256_raw_c_order"] != sha256_array(array)):
                raise ValueError(f"Reference array metadata or SHA-256 mismatch: {name}")
            array.setflags(write=False)
        return cls(manifest=manifest, **arrays)


@dataclass(frozen=True)
class BoundAcousticReference:
    """Immutable reference identity and CPU conditions for one acoustic model.

    The byte-backed arrays cannot be made writable. This object never owns a
    model or the caller's mutable manifest, and switching requires a new model.
    """
    identity_json: str
    ge: np.ndarray
    ge512: np.ndarray
    ge_sha256: str
    ge512_sha256: str

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
        identity = reference.manifest["identity"]
        if (reference.manifest["model_family"] != "v2Pro"
                or model_manifest["config"]["model"]["version"] != "v2Pro"
                or identity["reference_language"] != "ja"):
            raise ValueError("Binding requires the validated V2Pro Japanese reference")
        source = model_manifest["source"]
        if (identity["sovits_checkpoint_sha256"] != source["checkpoint_sha256"]
                or identity["official_commit"] != source["official_commit"]):
            raise ValueError("Loaded sovits model differs from the prepared reference")
        snapshots = {}
        for name, shape in (("ge", (1, 1024, 1)), ("ge512", (1, 512, 1))):
            value = cls._condition(getattr(reference, name), name, shape)
            snapshots[name] = np.frombuffer(value.tobytes(order="C"), dtype=np.float32).reshape(shape)
        return cls(json.dumps(identity, sort_keys=True, ensure_ascii=False),
                   snapshots["ge"], snapshots["ge512"],
                   sha256_array(snapshots["ge"]), sha256_array(snapshots["ge512"]))

    def validate_conditions(self, ge, ge512):
        for name, value in (("ge", ge), ("ge512", ge512)):
            expected = getattr(self, name)
            value = self._condition(value, name, expected.shape)
            if sha256_array(value) != getattr(self, name + "_sha256"):
                raise ValueError(f"Bound acoustic reference {name} differs; load a new model for this reference")

    def validate_reference(self, reference):
        if not isinstance(reference, PreparedReference):
            raise TypeError("Binding requires a PreparedReference")
        if (reference.manifest["model_family"] != "v2Pro"
                or json.dumps(reference.manifest["identity"], sort_keys=True, ensure_ascii=False) != self.identity_json):
            raise ValueError("Bound acoustic reference identity differs; load a new model for this reference")
        self.validate_conditions(reference.ge, reference.ge512)
