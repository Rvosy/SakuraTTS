"""V2Pro reference-lifetime experiment against the unmodified official engine.

This adapter deliberately accepts one fixed reference. It is a development
experiment, not a general-purpose TTS backend or a persistent cache loader.
"""

from __future__ import annotations

import gc
import json

import numpy as np
import torch


class PreparedSpeaker:
    def __init__(self, audio, embedding):
        self.audio = audio
        self.embedding = embedding

    def compute_embedding3(self, audio):
        if audio is not self.audio:
            raise ValueError("Prepared reference adapter cannot accept another audio tensor")
        return self.embedding


@torch.inference_mode()
def prepare_reference(engine, reference, output, identity, synchronize):
    if engine.configs.version != "v2Pro":
        raise ValueError("This lifecycle experiment has only been defined for v2Pro")
    engine.set_ref_audio(reference["path"])
    spec, audio = engine.prompt_cache["refer_spec"][0]
    embedding = engine.sv_model.compute_embedding3(audio)
    # Keep the exact upstream prompt punctuation and frontend semantics.
    from TTS_infer_pack.text_segmentation_method import splits

    prompt_text = reference["text"].strip("\n")
    if prompt_text[-1] not in splits:
        prompt_text += "." if reference["language"] == "en" else "。"
    phones, bert, norm = engine.text_preprocessor.segment_and_extract_feature_for_text(
        prompt_text, reference["language"], engine.configs.version
    )
    engine.prompt_cache.update(prompt_text=prompt_text, prompt_lang=reference["language"],
                               phones=phones, bert_features=bert, norm_text=norm)
    arrays = {
        "prompt_semantic": engine.prompt_cache["prompt_semantic"],
        "reference_spectrogram": spec, "speaker_audio_16k": audio,
        "speaker_embedding": embedding, "reference_bert": bert,
    }
    np.savez(output / "prepared-reference.npz", reference_phones=np.asarray(phones, dtype=np.int64),
             **{name: value.cpu().numpy() for name, value in arrays.items()})
    (output / "prepared-reference.json").write_text(json.dumps({
        "identity": identity, "reference": reference, "prompt_text": prompt_text,
        "normalized_text": norm, "model_version": engine.configs.version,
        "scope": "exact official preparation; single-reference experiment; not yet a reusable cache format",
    }, ensure_ascii=False, indent=2) + "\n")
    synchronize()
    engine.sv_model = PreparedSpeaker(audio, embedding)
    engine.cnhuhbert_model = None
    gc.collect()
    if str(engine.configs.device) == "mps":
        torch.mps.empty_cache()
    synchronize()
    return {"released_models": ["CNHuBERT", "ERes2Net"], "cached_speaker_embedding": True,
            "reference_artifact_bytes": (output / "prepared-reference.npz").stat().st_size}
