"""Shared model ownership and synthesis lifecycle across explicit backends."""

import json
import logging
import math
import time

import numpy as np

from .reference_condition import PreparedReference
from .synthesis import prepare_text_request, generate_prepared_semantic, synthesize_acoustic
from .logging import set_stage

logger = logging.getLogger("sakuratts.inference")


class InferenceRuntime:
    """One active model pair, with backend-owned GPT and acoustic loaders."""

    def _prepare_model(self, config, *, load_references):
        from sakuratts.model import Model
        model = config if isinstance(config, Model) else Model.load(config)
        self.config_path = model.path
        from sakuratts._internal.portable import model_config
        self.config = model_config(model.runtime_config)
        root = self.config_path.parent
        self.packages = {key: (root / self.config[key]).resolve(strict=True)
                         for key in ("gpt", "sovits", "frontend")}
        self.manifests = {key: json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                          for key, path in self.packages.items()}
        self._validate_acoustic_package()
        self.references = {}
        for name, path in (self.config.get("references", {}) if load_references else {}).items():
            self.references[name] = PreparedReference.load(root / path)
        from sakuratts.frontend.runtime import load_frontend
        self.frontend_runtime = load_frontend(self.config_path, self.config,
            self.packages["frontend"], self.manifests["frontend"])
        self.frontend = self.frontend_runtime.text
        self.gpt = self.sovits = None
        self.busy = False

    def _validate_acoustic_package(self):
        """Check precision selection before starting the frontend."""
        acoustic_dtype = self.manifests["sovits"].get("dtype")
        if acoustic_dtype not in ("float32", "float16"):
            raise ValueError("Expected FP32 or FP16 acoustic weights")
        self.acoustic_precision = "fp16" if acoustic_dtype == "float16" else "fp32"
        if self.acoustic_precision == "fp16" and not self.allow_experimental_acoustic_fp16:
            raise ValueError("FP16 acoustic packages require allow_experimental_acoustic_fp16=True")

    def load(self):
        if self.policy == "staged":
            return
        self._load_gpt()
        self._load_sovits()

    def unload(self):
        """Idle unload is explicit; the next request includes the reload cost."""
        if self.gpt is not None:
            self.gpt.close()
            self.gpt = None
        if self.sovits is not None:
            self.sovits.close()
            self.sovits = None

    def close(self):
        self.unload()
        self.frontend_runtime.close()

    def synthesize(self, text, *, reference=None, seed=1234, language="ja", split_method="cut0",
                   top_k=15, top_p=1., temperature=1., repetition_penalty=1.35, early_stop_num=2700,
                   cancel_requested=None, random_inputs=None, fragment_interval=0.3,
                   on_fragment=None, collect_audio=True, split_bucket=False):
        if self.busy:
            raise RuntimeError("This engine already has an active request")
        if not collect_audio and on_fragment is None:
            raise ValueError("Streaming without collecting audio requires an on_fragment callback")
        if not text.strip() or seed < -1 or top_k<1 or early_stop_num < -1:
            raise ValueError("Require text, seed>=-1, top_k>=1 and early_stop_num>=-1")
        if not math.isfinite(fragment_interval) or fragment_interval < 0:
            raise ValueError("fragment_interval must be finite and nonnegative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if seed == -1:
            import secrets
            seed = secrets.randbelow(2 ** 32)
        if any(not math.isfinite(v) or v<=0 for v in (temperature,repetition_penalty)):
            raise ValueError("Temperature and repetition penalty must be finite and positive")
        if isinstance(reference, PreparedReference):
            ref = reference
            reference = ref.manifest["identity"]["audio_sha256"]
        else:
            reference = reference or self.config.get("default_reference", next(iter(self.references), None))
            if reference not in self.references:
                raise ValueError(f"Unknown reference {reference!r}; available: {list(self.references)}")
            ref = self.references[reference]
        self.busy = True
        started = time.perf_counter()
        parts, fragments = [], []
        pcm_samples = 0
        try:
            logger.debug("推理参数 | seed=%d | %s | %s | top_k=%d | temperature=%g | repetition_penalty=%g",
                        seed, language, split_method, top_k, temperature, repetition_penalty)
            logger.debug("切分文本并提取文本特征 | 输入: %r", text)
            logger.info("文本: %s", text, extra={"block": "text", "text": text})
            set_stage("文本处理")
            logger.info("提取文本Bert特征")
            request = prepare_text_request(text,language,self.frontend,split_method=split_method)
            logger.info("%d 段 · %d phones · %.3f s", len(request.fragments),
                        sum(len(part.target["phones"]) for part in request.fragments), request.seconds,
                        extra={"block": "progress"})
            if random_inputs is not None and len(random_inputs)!=len(request.fragments):
                raise ValueError("Replay random inputs must cover every prepared fragment")
            # api_v2 sorts even singleton batches; fragment streaming disables it.
            bucketed = split_bucket and on_fragment is None
            execution_order = list(range(len(request.fragments)))
            if bucketed:
                execution_order.sort(key=lambda index: len(request.fragments[index].target["norm_text"]))
            rng = np.random.default_rng(seed)
            for index in execution_order:
                prepared = request.fragments[index]
                logger.debug("分句 %d/%d | %d 个音素 | 前端处理后的文本: %r",
                            index + 1, len(request.fragments), len(prepared.target["phones"]),
                            prepared.target["norm_text"])
                if len(request.fragments) > 1:
                    logger.info("片段 [%d/%d]", index + 1, len(request.fragments))
                self._load_gpt()
                if self.policy != "staged":
                    self._load_sovits()
                semantic = generate_prepared_semantic(prepared,ref,gpt=self.gpt,rng=rng,
                    top_k=top_k,top_p=top_p,temperature=temperature,repetition_penalty=repetition_penalty,
                    early_stop_num=early_stop_num,release_gpt_state=self.policy=="release-state",
                    cancel_requested=cancel_requested,
                    semantic_random_draw=(None if random_inputs is None else
                        lambda step,shape: random_inputs[index]["draws"][step,:shape[-1]][None]))
                if self.policy == "staged":
                    self.gpt.close()
                    self.gpt = None
                    self._load_sovits()
                set_stage("声学合成")
                logger.info("合成音频", extra={"block": "stage"})
                logger.debug("合成音频 | 分句 %d/%d | %d 个有效语义 Token", index + 1,
                            len(request.fragments), semantic.generation.semantic.shape[-1])
                actual = synthesize_acoustic(semantic,sovits=self.sovits,cancel_requested=cancel_requested,
                    fragment_interval=fragment_interval,
                    acoustic_noise=None if random_inputs is None else random_inputs[index]["noise"])
                logger.debug("分句音频完成 | %.3f s | 音频 %.2f s | 采样率 %d Hz",
                            actual.timings["acoustic_seconds"], actual.pcm.size / actual.sample_rate,
                            actual.sample_rate)
                logger.info("SoVITS  %.3f s", actual.timings["acoustic_seconds"], extra={"block": "progress"})
                pcm_samples += actual.pcm.size
                if collect_audio:
                    parts.append((index, actual.pcm))
                if on_fragment is not None:
                    on_fragment(actual.pcm, actual.sample_rate)
                fragments.append({"index":index,"normalized_text":prepared.target["norm_text"],
                    "phones":prepared.target["phones"],"sampled_tokens":actual.generation.sampled_tokens.tolist(),
                    "semantic_tokens":actual.generation.semantic.reshape(-1).tolist(),
                    "stop_reasons":list(actual.generation.stop.reasons),
                    "returned_index":actual.generation.stop.returned_index,
                    "waveform_samples":actual.waveform.size,"pcm_samples":actual.pcm.size,
                    "timings":actual.timings,
                    "acoustic_transport":getattr(self.sovits,"last_transfer",None)})
                rate=actual.sample_rate
                if self.policy == "staged":
                    self.sovits.close()
                    self.sovits = None
            pcm = np.concatenate([part for _, part in sorted(parts)]) if collect_audio else np.empty(0, dtype=np.int16)
            fragments.sort(key=lambda fragment: fragment["index"])
            elapsed=time.perf_counter()-started
            limited=any(set(f["stop_reasons"]) & {"early_stop_num","iteration_limit"} for f in fragments)
            duration=sum(f["waveform_samples"] for f in fragments)/rate
            report={"status":"stopped_at_limit" if limited else "completed", "text":text,
                "reference":reference,"reference_identity":ref.manifest["identity"],
                "parameters":{"seed":seed,"rng":"numpy.default_rng (not Torch seed-equivalent)",
                    "language":language,"split_method":split_method,"split_bucket":bucketed,
                    "top_k":top_k,"top_p":top_p,
                    "temperature":temperature,"repetition_penalty":repetition_penalty,
                    "early_stop_num":early_stop_num,"noise_scale":0.5,"speed":1.,"fragment_interval":fragment_interval},
                "policy":self.policy,"capacity":self.capacity, **self._execution_report(),
                "request_ms":elapsed*1000,"frontend_ms":request.seconds*1000,
                "sample_rate":rate,"audio_seconds":duration,"pcm_seconds":pcm_samples/rate,
                "rtf":elapsed/duration,"fragments":fragments,"execution_order":execution_order,
                "timing_scope":"Original text through complete PCM, including missing model loads, transfers and sampling; file output separate",
                "precision": f"GPT {self.gpt_precision.upper()}, acoustic {self.acoustic_precision.upper()}; FP16 paths are experimental; public logits and acoustic I/O remain FP32",
                "gpt_precision":self.gpt_precision,"acoustic_precision":self.acoustic_precision,
                "acoustic_arena_shrink":self.acoustic_arena_shrink,
                "acoustic_chunk_frames":self.acoustic_chunk_frames,
                "acoustic_session_policy":self.acoustic_session_policy,
                "gpt_prefill_query_chunk_size":self.gpt_prefill_query_chunk_size,
                "frontend_profile":self.frontend_runtime.profile,
                "random_inputs":"fresh" if random_inputs is None else "explicit replay of draws and acoustic noise",
                "quality":{"human_listening":"not_run","asr":"not_run"}}
            return pcm,report
        except BaseException as error:
            cleanup = []
            if self.policy == "staged":
                gpt, sovits = self.gpt, self.sovits
                self.gpt = self.sovits = None
                if gpt is not None:
                    cleanup.append(("GPT close", gpt.close))
                if sovits is not None:
                    cleanup.append(("SoVITS close", sovits.close))
            else:
                if self.gpt is not None:
                    cleanup.append(("GPT request-state release", self.gpt.release_request_state))
                if self.sovits is not None and getattr(self.sovits,"process",False) is None:
                    sovits, self.sovits = self.sovits, None
                    cleanup.append(("failed SoVITS close", sovits.close))
            for label, operation in cleanup:
                try:
                    operation()
                except BaseException as cleanup_error:
                    error.add_note(f"{label} failed during cleanup: {cleanup_error!r}")
            raise
        finally:
            self.busy=False
