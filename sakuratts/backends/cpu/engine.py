"""CPU model execution with the shared ORT assembly."""

from ..ort import ORTEngine


class CPUEngine(ORTEngine):
    name = "cpu"
