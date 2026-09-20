"""SakuraTTS public API. Importing the package loads no optional backend."""

__all__ = ["Engine", "Audio", "Model", "BusyError", "load", "start_server"]


def __getattr__(name):
    if name == "Model":
        from .model import Model
        return Model
    if name in ("Engine", "Audio", "BusyError", "load"):
        from . import engine
        return getattr(engine, name)
    raise AttributeError("module 'sakuratts' has no attribute " + repr(name))


def start_server(model=None, *, host="127.0.0.1", port=9880, tts_config=None, experimental=None):
    from .server import start_server as run
    return run(model, host=host, port=port, tts_config=tts_config, experimental=experimental)
