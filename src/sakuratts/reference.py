"""Transparent reference conditioning for the upstream audio-path API."""

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path

from ._internal.reference_condition import PreparedReference, sha256_array, sha256_file

logger = logging.getLogger("sakuratts.reference")


class ReferenceCache:
    def __init__(self, engine, settings):
        self.engine = engine
        self.settings = settings
        self.audio_cache = OrderedDict()
        runtime = engine._runtime
        self.identity = {kind + "_checkpoint_sha256": runtime.manifests[kind]["source"]["checkpoint_sha256"]
                         for kind in ("gpt", "sovits")}
        self.identity["official_commit"] = runtime.manifests["gpt"]["source"]["official_commit"]
        self.paths = [] if settings.get("cnhubert") else [engine.model.path.parent / path
                      for path in engine.model.runtime_config.get("references", {}).values()]

    def _audio(self, audio_path, prompt_text):
        audio_path = Path(audio_path).resolve(strict=True)
        audio_hash = sha256_file(audio_path)
        if audio_hash in self.audio_cache:
            self.audio_cache.move_to_end(audio_hash)
            logger.debug("参考音频条件 | 命中内存缓存 | SHA256=%s", audio_hash)
            return self.audio_cache[audio_hash]
        expected = dict(self.identity, audio_sha256=audio_hash)
        base = None
        # Existing prepared packages are optional cache entries, never defaults.
        for path in self.paths:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if all(manifest["identity"].get(k) == v for k, v in expected.items()):
                base = PreparedReference.load(path, **expected)
                logger.debug("参考音频条件 | 复用部署包缓存 | %s", path)
                break
        if base is None:
            from .converter import prepare_reference
            script = Path(__file__).parent / "_internal/conversion/prepare_windows_resources.py"
            resources = {"frontend": sha256_file(self.engine._runtime.packages["frontend"] / "manifest.json")}
            if self.settings.get("official_source"):
                source = Path(self.settings["official_source"]) / "GPT_SoVITS"
                for prefix, directory in (("hubert", Path(self.settings.get("cnhubert", source / "pretrained_models/chinese-hubert-base"))),
                                          ("speaker", source / "eres2net")):
                    for resource in sorted(directory.rglob("*")):
                        if resource.is_file() and "__pycache__" not in resource.parts:
                            resources[prefix + "/" + str(resource.relative_to(directory))] = sha256_file(resource)
                speaker = source / "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
                if speaker.is_file():
                    resources[str(speaker.relative_to(source))] = sha256_file(speaker)
            key = hashlib.sha256(json.dumps({**expected, "resources": resources, "preparer": sha256_file(script)},
                sort_keys=True).encode("utf-8")).hexdigest()
            path = Path(self.settings.get("cache_dir", ".cache/sakuratts")) / "references" / key
            if path.exists():
                logger.debug("参考音频条件 | 命中磁盘缓存 | %s", path)
            else:
                required = ("gpt_checkpoint", "sovits_checkpoint", "official_source", "python")
                missing = [name for name in required if not self.settings.get(name)]
                if missing:
                    raise ValueError("Raw reference preparation is not configured: " + ", ".join(missing)
                                     + ". Set these preparation paths in the sakuratts section of --tts-config.")
                for kind in ("gpt", "sovits"):
                    if sha256_file(self.settings[kind + "_checkpoint"]) != self.identity[kind + "_checkpoint_sha256"]:
                        raise ValueError("Reference preparation checkpoint differs from loaded " + kind)
                logger.info("正在提取参考音频特征  %s", audio_path.name)
                logger.debug("Preparing reference audio: %s", audio_path)
                prepare_reference(gpt=self.settings["gpt_checkpoint"], sovits=self.settings["sovits_checkpoint"],
                    audio=audio_path, text="参考音声。", frontend=self.engine._runtime.packages["frontend"],
                    official_source=self.settings["official_source"], python=self.settings["python"], output=path,
                    cnhubert=self.settings.get("cnhubert"))
            base = PreparedReference.load(path, **expected)
        self.audio_cache[audio_hash] = base
        if len(self.audio_cache) > 8:
            self.audio_cache.popitem(last=False)
        return base

    def prepare_audio(self, path):
        self._audio(path, "")

    def resolve(self, path, text, language):
        if language not in ("ja", "all_ja"):
            raise NotImplementedError("Reference language is not implemented: " + language)
        if not text or not text.strip():
            raise NotImplementedError("Inference without prompt_text is not implemented by the native backend")
        base = self._audio(path, text)
        from .frontend.text_frontend import splits
        import numpy as np
        prompt = text.strip("\n")
        if prompt[-1] not in splits:
            prompt += "。"
        target = self.engine._runtime.frontend.segment(prompt, language)
        phones = np.asarray(target["phones"], dtype=np.int64)
        bert = np.asarray(target["bert_features"], dtype=np.float32)
        phones.setflags(write=False)
        bert.setflags(write=False)
        # Acoustic/semantic audio features are independent of the transcript.
        # Rebuild text features with the active frontend on every request.
        manifest = dict(base.manifest, identity=dict(base.manifest["identity"],
            reference_text=text, reference_language=language),
            reference={"prompt_text": prompt, "normalized_text": target["norm_text"]})
        manifest["arrays"] = dict(base.manifest.get("arrays", {}))
        for name, array in (("reference_phones", phones), ("reference_bert", bert)):
            manifest["arrays"][name] = {"dtype": str(array.dtype), "shape": list(array.shape),
                "bytes": array.nbytes, "sha256_raw_c_order": sha256_array(array)}
        return replace(base, manifest=manifest, reference_phones=phones, reference_bert=bert)
