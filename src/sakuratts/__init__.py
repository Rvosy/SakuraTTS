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


def start_server(model=None, *, host="127.0.0.1", port=9880, tts_config=None, experimental=None,
                 runtime_mode="direct", idle_sleep_seconds=60., wake_timeout_seconds=120.,
                 operation_timeout_seconds=300.):
    from .server import start_server as run
    runtime_options = {}
    if (runtime_mode != "direct" or idle_sleep_seconds != 60.
            or wake_timeout_seconds != 120. or operation_timeout_seconds != 300.):
        runtime_options = {"runtime_mode": runtime_mode, "idle_sleep_seconds": idle_sleep_seconds,
                           "wake_timeout_seconds": wake_timeout_seconds,
                           "operation_timeout_seconds": operation_timeout_seconds}
    return run(model, host=host, port=port, tts_config=tts_config, experimental=experimental,
               **runtime_options)
