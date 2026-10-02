"""Cancellation shared by the service and inference backends without imports."""


class SynthesisCancelled(RuntimeError):
    """The caller requested cancellation at this completed compute boundary."""

    def __init__(self, stage):
        self.stage = stage
        super().__init__(f"Synthesis cancelled at {stage}")


def check_cancelled(cancel_requested, stage):
    if cancel_requested is not None and cancel_requested():
        raise SynthesisCancelled(stage)
