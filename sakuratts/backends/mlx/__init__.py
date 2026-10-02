"""Private implementation components; use sakuratts.Engine for synthesis."""


def create_runtime(model, *, experimental=None, load_references=True):
    from .engine import MLXEngine
    return MLXEngine(model, load_references=load_references, **dict(experimental or {}))
