"""Public, synchronous inference API shared by the CLI and HTTP service."""

from dataclasses import dataclass
import io
from pathlib import Path
from threading import Lock
import wave

from .model import Model
from ._internal.pcm import pcm_s16le_bytes


class BusyError(RuntimeError):
    """An engine already has an active request."""


@dataclass
class Audio:
    pcm: object
    sample_rate: int
    report: dict

    def wav_bytes(self):
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm_s16le_bytes(self.pcm))
        return stream.getvalue()

    def save(self, path):
        path = Path(path)
        if path.suffix.lower() != ".wav":
            raise ValueError("Choose a .wav output path")
        data = self.wav_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        return path


class Engine:
    """One model, one active request. Close explicitly or use a with block."""

    def __init__(self, model, runtime):
        self.model = model
        self._runtime = runtime
        self._lock = Lock()
        self._closed = False

    @classmethod
    def load(cls, path, *, backend=None, experimental=None, load_references=True):
        model = path if isinstance(path, Model) else Model.load(path)
        from .backends import create_runtime
        return cls(model, create_runtime(model, backend=backend,
            experimental=experimental, load_references=load_references))

    def synthesize(self, text, *, reference=None, seed=1234, language="ja",
                   split_method="cut0", top_k=15, temperature=1.,
                   repetition_penalty=1.35, early_stop_num=2700, cancel_requested=None,
                   fragment_interval=0.3, on_fragment=None, collect_audio=True):
        if not self._lock.acquire(blocking=False):
            raise BusyError("This engine already has an active request")
        try:
            if self._closed:
                raise RuntimeError("Engine is closed")
            pcm, report = self._runtime.synthesize(text, reference=reference, seed=seed,
                language=language, split_method=split_method, top_k=top_k,
                temperature=temperature, repetition_penalty=repetition_penalty,
                early_stop_num=early_stop_num, cancel_requested=cancel_requested,
                fragment_interval=fragment_interval, on_fragment=on_fragment, collect_audio=collect_audio)
            return Audio(pcm, report["sample_rate"], report)
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            if not self._closed:
                self._runtime.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load(path, **kwargs):
    return Engine.load(path, **kwargs)


def read_inference_configuration(model=None, *, tts_config=None):
    """Read service settings without constructing frontends or GPU resources."""
    import json
    settings = {}
    if tts_config:
        path = Path(tts_config)
        if path.suffix.lower() == ".json":
            config = json.loads(path.read_text(encoding="utf-8"))
        else:
            import yaml
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("TTS configuration must be a mapping")
        if "format" in config:
            model = path
        else:
            settings = dict(config.get("sakuratts", {}))
            custom = config.get("custom", config)
            if "device" in custom:
                settings.setdefault("backend", custom["device"])
            if custom.get("is_half", False):
                raise NotImplementedError("Official is_half mode is not yet supported; use is_half: false")
            if custom.get("version", "v2ProPlus") != "v2ProPlus":
                raise NotImplementedError("The native service currently supports v2ProPlus")
            model = model or settings.get("model")
            for key, field in (("gpt", "t2s_weights_path"), ("sovits", "vits_weights_path")):
                if custom.get(field):
                    settings[key + "_checkpoint"] = custom[field]
            if custom.get("cnhuhbert_base_path"):
                settings["cnhubert"] = custom["cnhuhbert_base_path"]
    from ._internal.portable import preparation_settings
    return model, preparation_settings(settings)


class Inference:
    """Upstream API lifecycle, owned entirely by one service worker thread."""

    def __init__(self, model=None, *, tts_config=None, backend=None, experimental=None, _allow_staged=False):
        import logging
        if (experimental or {}).get("policy") == "staged" and not _allow_staged:
            raise ValueError("Direct HTTP loads both models at startup; staged policy requires managed mode or the low-level Engine")
        self.logger = logging.getLogger("sakuratts.engine")
        self.engine = None
        self.references = None
        self.model = None
        self.experimental = experimental
        self.reference_audio = None
        model, self.settings = read_inference_configuration(model, tts_config=tts_config)
        if backend is not None:
            self.settings["backend"] = backend
        if "backend" in self.settings:
            from .backends import require_backend
            require_backend(self.settings["backend"])
        try:
            if model is not None:
                self._activate(model if isinstance(model, Model) else Model.load(model))
                # Explicit upstream weight paths override the prepared configuration.
                for kind in ("gpt", "sovits"):
                    if self.settings.get(kind + "_checkpoint"):
                        self.set_weights(kind, self.settings[kind + "_checkpoint"])
            elif self.settings.get("gpt_checkpoint") and self.settings.get("sovits_checkpoint"):
                self._convert_initial()
            elif tts_config:
                raise ValueError("TTS configuration requires both t2s_weights_path and vits_weights_path, or sakuratts.model")
            if self.engine is not None:
                self._log_weights()
                self.logger.info("设备  %s · GPT %s / SoVITS %s", self.engine._runtime.name.upper(),
                                 self.engine._runtime.gpt_precision.upper(),
                                 self.engine._runtime.acoustic_precision.upper())
        except BaseException:
            self.close()
            raise

    def _activate(self, model):
        from .reference import ReferenceCache
        workers = {name: str(Path(self.settings[name]).resolve()) for name in ("acoustic_python", "frontend_python")
                   if self.settings.get(name)}
        if workers:
            model = Model(model.path, dict(model.manifest, **workers))
        self.logger.info("加载 GPT / SoVITS…")
        self.logger.debug("Loading GPT and SoVITS weights: %s", model.path)
        options = {"backend": self.settings["backend"]} if "backend" in self.settings else {}
        candidate = Engine.load(model, experimental=self.experimental, load_references=False, **options)
        try:
            references = ReferenceCache(candidate, self.settings)
            self.close()
            candidate._runtime.load()
        except BaseException:
            candidate.close()
            raise
        self.engine = candidate
        self.model = model
        self.references = references
        self.reference_audio = None
        self.logger.debug("Model weights loaded | %s | %s | %s", model.name,
                          candidate._runtime.name, ", ".join(model.languages))

    def _conversion_settings(self):
        missing = [key for key in ("official_source", "python") if not self.settings.get(key)]
        if missing:
            from ._internal.portable import bundle_root
            if bundle_root() is not None:
                raise ValueError("This bundle cannot convert original checkpoints without its preparation component. "
                                 "Use the complete bundle, or install the matching component in runtime/preparation.")
            raise ValueError("Checkpoint conversion requires sakuratts." + " and sakuratts.".join(missing)
                             + " in --tts-config")
        return {key: self.settings[key] for key in ("official_source", "python")}

    def _log_weights(self, *, announce=True):
        runtime = self.engine._runtime
        for kind in ("gpt", "sovits"):
            source = runtime.manifests[kind]["source"]
            path = (self.settings.get(kind + "_checkpoint") or source.get("checkpoint")
                    or runtime.packages[kind])
            self.logger.debug("当前 %s 权重 | %s | SHA256=%s", kind.upper(), path,
                              source["checkpoint_sha256"])
            if announce:
                self.logger.info("%-6s  %s", "GPT" if kind == "gpt" else "SoVITS", Path(path).name)

    def _convert_initial(self):
        import hashlib
        import json
        from .converter import convert
        from ._internal.portable import bundle_root
        from ._internal.reference_condition import sha256_file
        options = self._conversion_settings()
        identity = {kind: sha256_file(self.settings[kind + "_checkpoint"]) for kind in ("gpt", "sovits")}
        identity["source"] = sha256_file(Path(options["official_source"]) / "GPT_SoVITS/TTS_infer_pack/TTS.py")
        portable_root = bundle_root()
        if portable_root is None:
            identity["python"] = str(Path(options["python"]).resolve())
        else:
            identity["preparation"] = sha256_file(portable_root / "runtime/preparation/preparation-manifest.json")
        identity["converter"] = sha256_file(Path(__file__).with_name("converter.py"))
        for script in ("convert_gpt.py", "export_sovits_onnx.py", "prepare_windows_resources.py"):
            identity[script] = sha256_file(Path(__file__).parent / "_internal/conversion" / script)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        output = Path(self.settings.get("cache_dir", ".cache/sakuratts")) / "models" / key
        if not output.exists():
            self.logger.info("首次转换 GPT / SoVITS 权重，完成后将复用缓存")
            convert(gpt=self.settings["gpt_checkpoint"], sovits=self.settings["sovits_checkpoint"],
                output=output, name=Path(self.settings["sovits_checkpoint"]).stem,
                  acoustic_python=self.settings.get("acoustic_python"),
                  frontend_python=self.settings.get("frontend_python"),
                  language_model=self.settings.get("language_model"), **options)
        else:
            self.logger.info("复用 GPT / SoVITS 转换缓存")
        self._activate(Model.load(output))

    def set_weights(self, kind, weights_path):
        import hashlib
        import json
        from .converter import convert_checkpoint
        from ._internal.portable import bundle_root
        from ._internal.reference_condition import sha256_file
        path = Path(weights_path).resolve(strict=True)
        self.logger.debug("请求切换 %s 权重 | %s", kind.upper(), path)
        if path.is_file() and path.suffix.lower() in (".ckpt", ".pth"):
            digest = sha256_file(path)
            if self.engine and self.engine._runtime.manifests[kind]["source"]["checkpoint_sha256"] == digest:
                self.settings[kind + "_checkpoint"] = str(path)
                self.logger.debug("%s 权重与当前一致，继续复用", kind.upper())
                return
            options = self._conversion_settings()
            source_hash = sha256_file(Path(options["official_source"]) / "GPT_SoVITS/TTS_infer_pack/TTS.py")
            script = "convert_gpt.py" if kind == "gpt" else "export_sovits_onnx.py"
            converter_hash = sha256_file(Path(__file__).parent / "_internal/conversion" / script)
            identity = digest + source_hash + kind + converter_hash
            portable_root = bundle_root()
            if portable_root is not None:
                identity += sha256_file(portable_root / "runtime/preparation/preparation-manifest.json")
            key = hashlib.sha256(identity.encode()).hexdigest()
            converted = Path(self.settings.get("cache_dir", ".cache/sakuratts")) / "weights" / key
            if not converted.exists():
                self.logger.info("首次转换 %s 权重  %s", kind.upper(), path.name)
                convert_checkpoint(kind, path, converted, **options)
            checkpoint = str(path)
            path = converted
        else:
            if not path.is_dir():
                raise ValueError("weights_path must be an original checkpoint or a converted package directory")
            checkpoint = None
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        expected = "sakuratts-gpt-fp32-v1" if kind == "gpt" else "sakuratts-sovits-onnx-v1"
        if manifest.get("format") != expected:
            raise ValueError("Wrong model type for " + kind)
        if self.model is None:
            raise ValueError("Configure both initial model weights and frontend resources in --tts-config first")
        config = self.model.runtime_config
        root = self.model.path.parent
        for field in ("gpt", "sovits", "frontend", "acoustic_python", "frontend_python", "main_dictionary"):
            if field in config:
                config[field] = str((root / config[field]).resolve())
        config[kind] = str(path.resolve())
        config["references"] = {}
        config.pop("default_reference", None)
        config["format"] = "sakuratts-windows-config-v1"
        # A transient configuration avoids modifying the user's files.
        candidate = Model(self.model.path, config)
        previous = self.settings.get(kind + "_checkpoint")
        if checkpoint:
            self.settings[kind + "_checkpoint"] = checkpoint
        else:
            self.settings.pop(kind + "_checkpoint", None)
        try:
            self._activate(candidate)
        except BaseException:
            if previous is not None:
                self.settings[kind + "_checkpoint"] = previous
            else:
                self.settings.pop(kind + "_checkpoint", None)
            raise
        self.logger.info("%s 权重切换完成", kind.upper())
        self._log_weights()

    def set_reference_audio(self, path):
        if self.references is None:
            raise ValueError("Model weights are not loaded")
        self.references.prepare_audio(path)
        self.reference_audio = str(Path(path).resolve(strict=True))

    def tts(self, request, *, on_fragment=None, cancel_requested=None):
        import time
        from ._internal.logging import compact, request_id, set_stage
        request = dict(request)
        if request["seed"] == -1:
            import secrets
            request["seed"] = secrets.randbelow(2 ** 32)
        self.logger.info("请求 #%s  %s · %d 字 · seed=%d", request_id.get(), request["text_lang"],
                         len(request["text"]), request["seed"], extra={"block": "request"})
        self.logger.debug("请求参数: %r", request)
        if self.engine is None:
            raise ValueError("Model weights are not loaded; supply --tts-config or a prepared model configuration")
        started = time.perf_counter()
        self._log_weights(announce=False)
        self.logger.debug("参考音频 | %s | 语言=%s | 转写=%r", request["ref_audio_path"],
                         request["prompt_lang"], request["prompt_text"])
        set_stage("参考准备")
        reference = self.references.resolve(request["ref_audio_path"], request["prompt_text"], request["prompt_lang"])
        reference_ms = (time.perf_counter() - started) * 1000
        self.logger.info("参考    %s · 已准备", compact(Path(request["ref_audio_path"]).name))
        self.logger.debug("参考条件就绪 | %.3f s | %d 个音素 | %d 个参考语义 Token",
                         reference_ms / 1000, reference.reference_phones.size, reference.prompt_semantic.size)
        result = self.engine.synthesize(request["text"], reference=reference, seed=request["seed"],
            language=request["text_lang"], split_method=request["text_split_method"], top_k=request["top_k"],
            temperature=request["temperature"], repetition_penalty=request["repetition_penalty"],
            fragment_interval=request["fragment_interval"], on_fragment=on_fragment,
            collect_audio=on_fragment is None, cancel_requested=cancel_requested)
        result.report["reference_ms"] = reference_ms
        result.report["native_inference_ms"] = result.report["request_ms"]
        result.report["request_ms"] = (time.perf_counter() - started) * 1000
        return result

    def info(self):
        if self.engine is None:
            return None
        info = self.engine.model.info()
        info["backend"] = self.engine._runtime.name
        info.pop("references", None)
        info.pop("default_reference", None)
        return info

    def close(self):
        engine, self.engine = self.engine, None
        self.references = None
        if engine is not None:
            engine.close()
