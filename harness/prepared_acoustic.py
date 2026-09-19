"""Single-reference V2Pro acoustic lifetime experiment for the official harness.

The decode expression follows pinned GPT-SoVITS module/models.py (MIT; see
docs/third-party/GPT-SoVITS-LICENSE.txt). Only reference-dependent expressions
move to preparation; precision, noise, flow, and waveform expressions stay the
same. This adapter is not a general cache loader or a streaming implementation.
"""

from __future__ import annotations

import gc
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def decode_prepared(model, codes, text, refer, noise_scale=0.5, speed=1, sv_emb=None):
    if (type(refer) is not list or len(refer) != 1 or refer[0] is not model._prepared_spec
            or type(sv_emb) is not list or len(sv_emb) != 1 or sv_emb[0] is not model._prepared_embedding):
        raise ValueError("Prepared acoustic experiment requires its original single reference")
    ge = model._prepared_ge
    y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
    text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)
    quantized = model.quantizer.decode(codes)
    if model.semantic_frame_rate == "25hz":
        quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
    _, m_p, logs_p, y_mask, _, _ = model.enc_p(
        quantized, y_lengths, text, text_lengths, model._prepared_ge512, speed,
    )
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = model.flow(z_p, y_mask, g=ge, reverse=True)
    return model.dec((z * y_mask)[:, :, :], g=ge)


@torch.inference_mode()
def prepare_acoustic(engine, output, synchronize):
    from module import commons
    from prepared_reference import PreparedSpeaker

    if engine.configs.version != "v2Pro" or not isinstance(engine.sv_model, PreparedSpeaker):
        raise ValueError("Acoustic preparation requires the V2Pro prepared reference experiment")
    if len(engine.prompt_cache["refer_spec"]) != 1:
        raise ValueError("Acoustic preparation currently supports exactly one reference")
    model = engine.vits_model
    spec, audio = engine.prompt_cache["refer_spec"][0]
    spec = spec.to(dtype=engine.precision, device=engine.configs.device)
    embedding = engine.sv_model.compute_embedding3(audio)
    lengths = torch.LongTensor([spec.size(2)]).to(spec.device)
    mask = torch.unsqueeze(commons.sequence_mask(lengths, spec.size(2)), 1).to(spec.dtype)
    ge = model.ref_enc(spec[:, :704] * mask, mask)
    ge += model.sv_emb(embedding).unsqueeze(-1)
    ge = model.prelu(ge)
    # The official caller passes a one-element list and decode still averages it.
    ge = torch.stack([ge], 0).mean(0)
    ge512 = model.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
    synchronize()
    np.savez(output / "prepared-acoustic.npz", ge=ge.cpu().numpy(), ge512=ge512.cpu().numpy())
    model._prepared_spec = spec
    model._prepared_embedding = embedding
    model._prepared_ge = ge
    model._prepared_ge512 = ge512
    modules = ("enc_q", "ref_enc", "sv_emb", "prelu", "ge_to512")
    released = {name: sum(t.numel() * t.element_size() for t in getattr(model, name).parameters())
                for name in modules}
    for name in modules:
        delattr(model, name)
    model.decode = MethodType(decode_prepared, model)
    gc.collect()
    if str(engine.configs.device) == "mps":
        torch.mps.empty_cache()
    synchronize()
    return {"scope": "single fixed V2Pro reference; non-streaming decode only",
            "released_parameter_bytes": released,
            "cached_condition_bytes": sum(t.numel() * t.element_size() for t in (ge, ge512)),
            "enc_q_reason": "training posterior network is absent from the decode dependency graph",
            "other_modules_reason": "their outputs depend only on the prepared reference"}
