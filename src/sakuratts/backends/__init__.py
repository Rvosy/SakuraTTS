"""Explicit runtime selection without importing optional compute libraries."""

from importlib import import_module
from typing import Protocol


_IMPLEMENTATIONS = {"cuda": ".cuda"}
SUPPORTED_BACKENDS = tuple(_IMPLEMENTATIONS)


def available_backends():
    """Implemented runtime adapters, independent of installed device drivers."""
    return list(SUPPORTED_BACKENDS)


class Runtime(Protocol):
    """GPT-SoVITS runtime surface used by Engine and the HTTP lifecycle."""

    name: str
    gpt_precision: str
    acoustic_precision: str
    packages: dict
    manifests: dict
    frontend: object

    def load(self): ...
    def synthesize(self, text, **options): ...
    def close(self): ...


def require_backend(name):
    """Reject unavailable implementations before starting conversion or workers."""
    if name not in SUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"Backend {name!r} is not implemented; available: {', '.join(SUPPORTED_BACKENDS)}")
    return name


def create_runtime(model, *, backend=None, experimental=None, load_references=True) -> Runtime:
    """Keep selection explicit; never substitute a different device backend."""
    name = require_backend(model.backend if backend is None else backend)
    implementation = import_module(_IMPLEMENTATIONS[name], __name__)
    return implementation.create_runtime(model,
        experimental=experimental, load_references=load_references)
