"""CPU runtime assembly without importing compute libraries at discovery time."""


def create_runtime(model, *, experimental=None, load_references=True):
    from .engine import CPUEngine
    return CPUEngine(model, load_references=load_references, **dict(experimental or {}))
