"""DirectML models, with an explicitly selected optional CPU GPT path."""

from ..ort import ORTEngine


class DirectMLEngine(ORTEngine):
    name = "directml"
