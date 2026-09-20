"""Integer dependency planning for the original local 1D ONNX vocoder.

This module does no inference. Plans slice real decoder_input frames, never pad
latent frames or fade samples. All intervals are half-open. Run a chunk with the
unchanged ge, then take waveform[..., crop_start:crop_end]. Numerical agreement
still requires validation: changing a convolution's shape can change its GPU
algorithm even when every mathematical dependency is present.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


def ceil_div(value, divisor):
    return -((-value) // divisor)


def conv_input_interval(start, end, kernel, dilation=1, stride=1, pad_left=0):
    return start * stride - pad_left, (end - 1) * stride - pad_left + dilation * (kernel - 1) + 1


def transpose_input_interval(start, end, kernel, stride, pad_left=0):
    # Each input i contributes to i*stride + k - pad_left, 0 <= k < kernel.
    return ceil_div(start + pad_left - kernel + 1, stride), (end - 1 + pad_left) // stride + 1


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class TemporalOperation:
    name: str
    kind: str
    inputs: tuple
    output: str
    kernel: int = 1
    dilation: int = 1
    stride: int = 1
    pad_left: int = 0


class VocoderReceptiveField:
    FORMAT = "sakuratts.vocoder-receptive-field"
    VERSION = 1

    def __init__(self, operations, scales, *, input_name="decoder_input", output_name="waveform",
                 condition_name="ge", source=None):
        self.operations = tuple(operations)
        self.scales = dict(scales)
        self.input_name, self.output_name = input_name, output_name
        self.condition_name, self.source = condition_name, source
        self.samples_per_frame = self.scales[output_name]

    def to_dict(self):
        """Offline graph-derived specification; runtime needs only this JSON.

        The graph hash identifies the original ONNX protobuf file, not external
        weight bytes. Coefficients never prune dependencies. A lowered vocoder
        must separately establish its correspondence to this original graph.
        """
        return {"format": self.FORMAT, "version": self.VERSION,
                "source": dict(self.source), "input_name": self.input_name,
                "output_name": self.output_name, "condition_name": self.condition_name,
                "samples_per_frame": self.samples_per_frame, "scales": dict(self.scales),
                "operations": [{**asdict(op), "inputs": list(op.inputs)} for op in self.operations]}

    @classmethod
    def from_dict(cls, value):
        """Load a validated dependency DAG without importing ONNX or NumPy."""
        fields = {"format", "version", "source", "input_name", "output_name", "condition_name",
                  "samples_per_frame", "scales", "operations"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Unexpected receptive-field specification fields")
        if value["format"] != cls.FORMAT or type(value["version"]) is not int or value["version"] != cls.VERSION:
            raise ValueError("Unsupported receptive-field specification version")
        source = value["source"]
        if not isinstance(source, dict):
            raise ValueError("Missing source graph identity")
        digest = source.get("graph_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Expected the source graph SHA-256")
        names = [value[key] for key in ("input_name", "output_name", "condition_name")]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 3:
            raise ValueError("Expected distinct input, output and condition names")
        scales, records = value["scales"], value["operations"]
        if not isinstance(scales, dict) or not isinstance(records, list) or not records:
            raise ValueError("Expected temporal scales and operations")
        for name, scale in scales.items():
            if not isinstance(name, str) or not name or _integer(scale, "scale") < 1:
                raise ValueError("Expected named positive temporal scales")
        input_name, output_name, condition_name = names
        if scales.get(input_name) != 1 or condition_name in scales:
            raise ValueError("The input has scale one and the condition is broadcast")
        if (_integer(value["samples_per_frame"], "samples_per_frame") < 1
                or scales.get(output_name) != value["samples_per_frame"]):
            raise ValueError("Output sample scale mismatch")
        known, operations = {input_name}, []
        for record in records:
            if not isinstance(record, dict) or set(record) != set(TemporalOperation.__dataclass_fields__):
                raise ValueError("Unexpected temporal operation fields")
            name, kind, inputs, output = (record[key] for key in ("name", "kind", "inputs", "output"))
            if not isinstance(name, str) or not isinstance(kind, str):
                raise ValueError("Invalid operation name or kind")
            if (not isinstance(inputs, list) or not inputs or any(not isinstance(x, str) or x not in known for x in inputs)
                    or not isinstance(output, str) or output in known or output not in scales):
                raise ValueError("Operations must be a topologically sorted temporal DAG")
            kernel, dilation, stride, pad = (record[key] for key in ("kernel", "dilation", "stride", "pad_left"))
            for number, key in ((kernel, "kernel"), (dilation, "dilation"), (stride, "stride"), (pad, "pad_left")):
                _integer(number, key)
            if min(kernel, dilation, stride) < 1 or pad < 0:
                raise ValueError("Invalid convolution dependency parameters")
            expected_scale = scales[inputs[0]]
            if kind == "Conv":
                valid = len(inputs) == 1 and stride == 1 and pad <= dilation * (kernel - 1)
            elif kind == "ConvTranspose":
                valid = len(inputs) == 1 and dilation == 1 and kernel >= stride and pad <= kernel - stride
                expected_scale *= stride
            elif kind in ("LeakyRelu", "Tanh", "Identity", "Add", "Div"):
                valid = (len(inputs) in ((1, 2) if kind == "Add" else (1,))
                         and (kernel, dilation, stride, pad) == (1, 1, 1, 0))
            else:
                valid = False
            if not valid or scales[output] != expected_scale or any(scales[x] != scales[inputs[0]] for x in inputs):
                raise ValueError(f"Unsupported temporal operation or scale: {name}")
            operations.append(TemporalOperation(**{**record, "inputs": tuple(inputs)}))
            known.add(output)
        if set(scales) != known or output_name not in known:
            raise ValueError("Temporal scale names do not match the DAG")
        # JSON round-trip detaches nested provenance supplied by the caller and
        # rejects non-JSON source metadata before it reaches a runtime report.
        source = json.loads(json.dumps(source, allow_nan=False))
        return cls(operations, scales, input_name=input_name, output_name=output_name,
                   condition_name=condition_name, source=source)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def required_intervals(self, start, end, total_frames=None):
        """Structural dependency hull at every temporal tensor, in global indices."""
        _integer(start, "start")
        _integer(end, "end")
        if end <= start:
            raise ValueError("Require a nonempty output interval")
        if total_frames is not None:
            _integer(total_frames, "total_frames")
            if total_frames < 1 or not 0 <= start < end <= total_frames * self.samples_per_frame:
                raise ValueError("Output core must lie inside the true utterance")
        required = {self.output_name: (start, end)}
        for operation in reversed(self.operations):
            if operation.output not in required:
                continue
            lo, hi = required[operation.output]
            if operation.kind == "Conv":
                interval = conv_input_interval(lo, hi, operation.kernel, operation.dilation,
                                               operation.stride, operation.pad_left)
            elif operation.kind == "ConvTranspose":
                interval = transpose_input_interval(lo, hi, operation.kernel, operation.stride, operation.pad_left)
            else:
                interval = lo, hi
            for name in operation.inputs:
                a, b = interval
                if total_frames is not None:
                    a, b = max(0, a), min(total_frames * self.scales[name], b)
                if a >= b:
                    continue
                previous = required.get(name)
                required[name] = (min(a, previous[0]), max(b, previous[1])) if previous else (a, b)
        return required

    def plan(self, total_frames, core_start_frame, core_end_frame):
        for value, name in ((total_frames, "total_frames"), (core_start_frame, "core_start_frame"),
                            (core_end_frame, "core_end_frame")):
            _integer(value, name)
        if not 0 <= core_start_frame < core_end_frame <= total_frames:
            raise ValueError("Require a nonempty core inside the true latent length")
        start, end = core_start_frame * self.samples_per_frame, core_end_frame * self.samples_per_frame
        required = self.required_intervals(start, end, total_frames)
        # Include every intermediate dependency domain, not just the final latent
        # hull: missing internal positions can also contain bias/condition values.
        input_start = min(lo // self.scales[name] for name, (lo, _) in required.items())
        input_end = max(ceil_div(hi, self.scales[name]) for name, (_, hi) in required.items())
        assert 0 <= input_start <= core_start_frame < core_end_frame <= input_end <= total_frames
        return {"input_start": input_start, "input_end": input_end,
                "core_start_frame": core_start_frame, "core_end_frame": core_end_frame,
                "core_sample_start": start, "core_sample_end": end,
                "crop_start": start - input_start * self.samples_per_frame,
                "crop_end": end - input_start * self.samples_per_frame,
                "samples_per_frame": self.samples_per_frame,
                "left_halo_frames": core_start_frame - input_start,
                "right_halo_frames": input_end - core_end_frame}

    def plan_chunks(self, total_frames, core_frames):
        _integer(core_frames, "core_frames")
        _integer(total_frames, "total_frames")
        if core_frames < 1 or total_frames < 1:
            raise ValueError("Lengths must be positive")
        return [self.plan(total_frames, start, min(start + core_frames, total_frames))
                for start in range(0, total_frames, core_frames)]

    def phase_audit(self):
        phases = []
        for phase in range(self.samples_per_frame):
            required = self.required_intervals(phase, phase + 1)
            lo = min(a // self.scales[name] for name, (a, _) in required.items())
            hi = max(ceil_div(b, self.scales[name]) for name, (_, b) in required.items())
            phases.append({"phase": phase, "input_start": lo, "input_end": hi})
        return {"samples_per_frame": self.samples_per_frame, "phases_checked": len(phases),
                "left_halo_frames": max(-row["input_start"] for row in phases),
                "right_halo_frames": max(row["input_end"] - 1 for row in phases), "phases": phases}
