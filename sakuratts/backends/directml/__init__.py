"""DirectML runtime assembly for Windows GPUs, including AMD integrated graphics."""


def create_runtime(model, *, experimental=None, load_references=True):
    from .engine import DirectMLEngine
    return DirectMLEngine(model, load_references=load_references, **dict(experimental or {}))
