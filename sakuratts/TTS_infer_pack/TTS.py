"""Model, reference and request orchestration shared by inference entry points."""

from pathlib import Path
from ..model import Model
from ..engine import Engine
from .config import read_inference_configuration


class Inference:
    """Upstream API lifecycle, owned entirely by one service worker thread."""

    def __init__(self, model=None, *, tts_config=None, backend=None, profile=None, experimental=None, _allow_staged=False):
        import logging
        self.logger = logging.getLogger("sakuratts.engine")
        self.engine = None
        self.references = None
        self.model = None
        self.experimental = experimental
        self.reference_audio = None
        model, self.settings = read_inference_configuration(model, tts_config=tts_config)
        if self.settings.get("runtime_options"):
            self.experimental = {**self.settings["runtime_options"], **(experimental or {})}
        if profile is not None:
            self.settings["profile"] = profile
        if backend is not None:
            self.settings["backend"] = backend
        from ..profiles import resolve_profile, uses_staged_policy
        if "backend" in self.settings:
            effective, _ = resolve_profile(self.settings["backend"], self.settings.get("profile"), self.experimental)
            if effective is not None:
                self.settings["profile"] = effective
        if uses_staged_policy(self.settings.get("profile"), self.experimental) and not _allow_staged:
            raise ValueError("Direct HTTP loads both models at startup; staged policy requires managed mode or the low-level Engine")
        if "backend" in self.settings:
            from ..backends import require_backend
            require_backend(self.settings["backend"])
            if self.settings["backend"] == "mlx":
                raise NotImplementedError("MLX currently supports prepared native V2Pro packages through Engine only; HTTP Inference is not supported")
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
        if self.settings.get("backend", model.backend) == "mlx":
            raise NotImplementedError("MLX currently supports prepared native V2Pro packages through Engine only; HTTP Inference is not supported")
        from .reference import ReferenceCache
        workers = {name: str(Path(self.settings[name]).resolve()) for name in ("acoustic_python", "frontend_python")
                   if self.settings.get(name)}
        if workers:
            model = Model(model.path, dict(model.manifest, **workers))
        self.logger.info("加载 GPT / SoVITS…")
        self.logger.debug("Loading GPT and SoVITS weights: %s", model.path)
        options = {"backend": self.settings["backend"]} if "backend" in self.settings else {}
        if "profile" in self.settings:
            options["profile"] = self.settings["profile"]
        candidate = Engine.load(model, experimental=self.experimental, load_references=False, **options)
        try:
            references = ReferenceCache(candidate, self.settings)
        except BaseException as error:
            try:
                candidate.close()
            except BaseException as cleanup_error:
                error.add_note(f"Candidate cleanup failed: {cleanup_error!r}")
            raise
        previous = self.engine
        try:
            # Keep the current frontend and reference cache until the new pair loads.
            if previous is not None:
                previous._runtime.unload()
            candidate._runtime.load()
        except BaseException as error:
            try:
                candidate.close()
            except BaseException as cleanup_error:
                error.add_note(f"Candidate cleanup failed: {cleanup_error!r}")
                try:
                    self.close()
                except BaseException as close_error:
                    error.add_note(f"Previous runtime cleanup failed: {close_error!r}")
                raise error
            if previous is not None:
                try:
                    previous._runtime.load()
                except BaseException as restore_error:
                    error.add_note(f"Previous model restore failed: {restore_error!r}")
                    self.logger.error("模型切换失败，旧模型也未能恢复: %s", restore_error)
                    try:
                        self.close()
                    except BaseException as cleanup_error:
                        error.add_note(f"Previous runtime cleanup failed: {cleanup_error!r}")
                else:
                    self.logger.warning("模型切换失败，已恢复原模型")
            raise
        try:
            self.close()
        except BaseException as error:
            try:
                candidate.close()
            except BaseException as cleanup_error:
                error.add_note(f"Candidate cleanup failed: {cleanup_error!r}")
            raise
        self.engine = candidate
        self.model = model
        self.references = references
        self.reference_audio = None
        self.logger.debug("Model weights loaded | %s | %s | %s", model.name,
                          candidate._runtime.name, ", ".join(model.languages))

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
        from ..prepare.cache import prepare_initial_model
        self._activate(prepare_initial_model(self.settings, self.experimental))

    def set_weights(self, kind, weights_path):
        import json
        from ..prepare.cache import prepare_checkpoint
        from ..module.reference_condition import sha256_file
        path = Path(weights_path).resolve(strict=True)
        self.logger.debug("请求切换 %s 权重 | %s", kind.upper(), path)
        if path.is_file() and path.suffix.lower() in (".ckpt", ".pth"):
            digest = sha256_file(path)
            if self.engine and self.engine._runtime.manifests[kind]["source"]["checkpoint_sha256"] == digest:
                self.settings[kind + "_checkpoint"] = str(path)
                self.logger.debug("%s 权重与当前一致，继续复用", kind.upper())
                return
            backend = self.settings.get("backend", self.model.backend if self.model is not None else "cuda")
            converted = prepare_checkpoint(kind, path, digest, self.settings,
                                           backend=backend, experimental=self.experimental)
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
        from ..runtime.logging import compact, request_id, set_stage
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
            top_p=request.get("top_p", 1.),
            temperature=request["temperature"], repetition_penalty=request["repetition_penalty"],
            fragment_interval=request["fragment_interval"], on_fragment=on_fragment,
            split_bucket=request["split_bucket"],
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
