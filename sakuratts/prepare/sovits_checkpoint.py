"""Load the shared GPT-SoVITS acoustic architecture for offline exporters.

This module belongs to the preparation environment; ordinary inference does
not import PyTorch or the upstream source tree.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


def plain(value):
    if type(value).__name__ == "HParams":
        return plain(vars(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


@dataclass
class SoVITSCheckpoint:
    model: object
    config: dict
    model_config: dict
    state: dict
    missing_keys: list[str]


def load_checkpoint(checkpoint, source):
    """Identify the family from the checkpoint and require all inference tensors."""
    sys.dont_write_bytecode = True
    source = Path(source).resolve()
    sys.path[:0] = [str(source / "GPT_SoVITS"), str(source)]
    from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
    from module.models import SynthesizerTrn

    _, family, lora = get_sovits_version_from_path_fast(str(checkpoint))
    if family not in ("v2Pro", "v2ProPlus") or lora:
        raise ValueError("Only non-LoRA V2Pro and V2ProPlus checkpoints are supported")
    original = load_sovits_new(str(checkpoint))
    config = plain(original["config"])
    if config["model"].get("version", family) != family:
        raise ValueError("Checkpoint header and model configuration disagree")
    model_config = dict(config["model"], version=family, semantic_frame_rate="25hz")
    model = SynthesizerTrn(config["data"]["filter_length"] // 2 + 1,
                           config["train"]["segment_size"] // config["data"]["hop_length"],
                           n_speakers=config["data"]["n_speakers"], **model_config).float().eval()
    state = original["weight"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(not key.startswith("enc_q.") for key in incompatible.missing_keys):
        raise ValueError(f"Unsupported checkpoint schema: {incompatible}")
    if len(model.quantizer.vq.layers) != 1 or not isinstance(model.quantizer.vq.layers[0].project_out, torch.nn.Identity):
        raise ValueError("Require a single codebook with identity output projection")
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    if model.gin_channels != 1024 or model.ge_to512.out_features != 512:
        raise ValueError("Prepared reference schema requires ge=1024 and ge512=512")
    return SoVITSCheckpoint(model, config, model_config, state, list(incompatible.missing_keys))
