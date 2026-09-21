"""Independent Japanese text-to-WAV entry point for prepared CUDA models."""

from importlib import metadata
import json
import logging
import math
import os
from pathlib import Path
import time
import wave

import numpy as np

from sakuratts._internal.reference_condition import PreparedReference, sha256_file
from sakuratts._internal.synthesis import prepare_text_request, generate_prepared_semantic, synthesize_acoustic
from sakuratts._internal.logging import set_stage

logger = logging.getLogger("sakuratts.inference")


class NVIDIAEngine:
    """One active model pair and one synchronous request; caller owns lifetime."""
    def __init__(self, config, *, policy="resident", use_graph=True, capacity=2048,
                 gpt_precision="fp32", gpt_attention="baseline", gpt_attention_chunk_size=256,
                 allow_experimental_acoustic_fp16=False, acoustic_arena_shrink=True,
                 acoustic_chunk_frames=None, load_references=True, gpt_prefill_query_chunk_size=0,
                 acoustic_session_policy="resident"):
        if policy not in ("resident", "release-state", "staged"):
            raise ValueError("Unknown model policy")
        if gpt_precision not in ("fp32", "fp16"):
            raise ValueError("GPT precision must be fp32 or fp16")
        if type(gpt_prefill_query_chunk_size) is not int or gpt_prefill_query_chunk_size < 0:
            raise ValueError("GPT prefill query chunk size must be a nonnegative integer")
        self.gpt_prefill_query_chunk_size = gpt_prefill_query_chunk_size
        if not isinstance(acoustic_arena_shrink, bool):
            raise ValueError("acoustic_arena_shrink must be a bool")
        if acoustic_chunk_frames is not None and (type(acoustic_chunk_frames) is not int or acoustic_chunk_frames<0):
            raise ValueError("acoustic_chunk_frames must be a nonnegative integer or None")
        if acoustic_session_policy not in ("resident", "staged"):
            raise ValueError("acoustic_session_policy must be resident or staged")
        if acoustic_session_policy == "staged" and acoustic_chunk_frames is None:
            raise ValueError("staged acoustic_session_policy requires acoustic_chunk_frames")
        if gpt_attention not in ("baseline", "split-kv"):
            raise ValueError("GPT attention must be baseline or split-kv")
        if (not isinstance(gpt_attention_chunk_size, (int, np.integer))
                or gpt_attention_chunk_size not in (256, 512)):
            raise ValueError("GPT attention chunk size must be 256 or 512")
        self.gpt_precision = gpt_precision
        self.gpt_attention, self.gpt_attention_chunk_size = gpt_attention, gpt_attention_chunk_size
        self.allow_experimental_acoustic_fp16 = allow_experimental_acoustic_fp16
        self.acoustic_arena_shrink = acoustic_arena_shrink
        self.acoustic_chunk_frames = acoustic_chunk_frames
        self.acoustic_session_policy = acoustic_session_policy
        from sakuratts.model import Model
        model = config if isinstance(config, Model) else Model.load(config)
        self.config_path = model.path
        self.config = model.runtime_config
        if self.config.get("format") != "sakuratts-windows-config-v1":
            raise ValueError("Expected a sakuratts-windows-config-v1 configuration")
        root = self.config_path.parent
        self.packages = {key: (root / self.config[key]).resolve(strict=True)
                         for key in ("gpt", "sovits", "frontend")}
        self.manifests = {key: json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                          for key, path in self.packages.items()}
        acoustic_dtype = self.manifests["sovits"].get("dtype")
        if acoustic_dtype not in ("float32", "float16"):
            raise ValueError("Expected FP32 or screened experimental FP16 acoustic weights")
        self.acoustic_precision = "fp16" if acoustic_dtype == "float16" else "fp32"
        if self.acoustic_precision == "fp16" and not allow_experimental_acoustic_fp16:
            raise ValueError("FP16 acoustic packages require allow_experimental_acoustic_fp16=True")
        if (acoustic_chunk_frames is not None
                or self.manifests["sovits"].get("format", "sakuratts-sovits-onnx-v1") != "sakuratts-sovits-onnx-v1"):
            from sakuratts.backends.onnx.sovits import read_manifest
            self.manifests["sovits"],_ = read_manifest(self.packages["sovits"],
                allow_experimental_fp16=allow_experimental_acoustic_fp16,
                acoustic_arena_shrink=acoustic_arena_shrink, acoustic_chunk_frames=acoustic_chunk_frames,
                acoustic_session_policy=acoustic_session_policy)
        source = self.manifests["gpt"]["source"]["official_commit"]
        if (self.manifests["sovits"]["source"]["official_commit"] != source
                or self.manifests["frontend"]["official_commit"] != source):
            raise ValueError("GPT, acoustic and frontend source identities do not match")
        self.references = {}
        for name, path in (self.config.get("references", {}) if load_references else {}).items():
            self.references[name] = PreparedReference.load(root / path,
                gpt_checkpoint_sha256=self.manifests["gpt"]["source"]["checkpoint_sha256"],
                sovits_checkpoint_sha256=self.manifests["sovits"]["source"]["checkpoint_sha256"],
                reference_language="ja", official_commit=source)
        frontend_manifest = self.manifests["frontend"]
        if frontend_manifest["format"] != "sakuratts-japanese-frontend-resources-v1":
            raise ValueError("Unsupported frontend resource package")
        if not {"symbols-v2.json", "user.dict", "lid.176.bin"}.issubset(frontend_manifest["files"]):
            raise ValueError("Incomplete frontend resource package")
        for name,spec in frontend_manifest["files"].items():
            path = (self.packages["frontend"] / name).resolve(strict=True)
            if self.packages["frontend"] not in path.parents:
                raise ValueError("Frontend resource must remain inside its package")
            if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
                raise ValueError(f"Frontend resource checksum mismatch: {name}")
        from sakuratts.frontend.text_frontend import LanguageSegmenter, TextFrontend
        profile=frontend_manifest.get("japanese_g2p",{"implementation":"pyopenjtalk-plus"})
        self.japanese = self.segmenter = None
        try:
            if profile["implementation"]=="pyopenjtalk-classic":
                from sakuratts.frontend.classic_japanese import ClassicJapaneseG2P
                if profile.get("version") != "0.3.4" or not self.config.get("acoustic_python"):
                    raise ValueError("This classic frontend package requires version 0.3.4 and its prepared Python runtime")
                directories = {}
                for key in ("module_directory", "main_dictionary"):
                    path = (self.packages["frontend"] / profile[key]).resolve(strict=True)
                    if self.packages["frontend"] not in path.parents or not path.is_dir():
                        raise ValueError("Classic frontend directories must remain inside their package")
                    directories[key] = path
                self.japanese=ClassicJapaneseG2P(
                    self.config_path.parent/self.config["acoustic_python"],
                    directories["module_directory"], directories["main_dictionary"],
                    self.packages["frontend"]/"user.dict")
            elif profile["implementation"]=="pyopenjtalk-plus":
                from sakuratts.frontend.japanese import JapaneseG2P
                dictionary = self.config.get("main_dictionary")
                dictionary = (root / dictionary).resolve(strict=True) if dictionary else Path(
                    metadata.distribution("pyopenjtalk-plus").locate_file("pyopenjtalk/dictionary"))
                os.environ["OPEN_JTALK_DICT_DIR"] = str(dictionary)
                self.japanese = JapaneseG2P(dictionary, self.packages["frontend"] / "user.dict")
            else:
                raise ValueError("Unsupported Japanese frontend implementation")
            self.segmenter = LanguageSegmenter(self.packages["frontend"])
            symbols = json.loads((self.packages["frontend"] / "symbols-v2.json").read_text(encoding="utf-8"))
            self.frontend = TextFrontend(self.japanese, symbols, self.segmenter)
        except BaseException as error:
            for component in (self.segmenter, self.japanese):
                if component is not None:
                    try:
                        component.close()
                    except BaseException as cleanup_error:
                        error.add_note(f"Frontend construction cleanup failed: {cleanup_error!r}")
            raise
        self.policy, self.use_graph, self.capacity = policy, use_graph, capacity
        self.gpt = self.sovits = None
        self.busy = False

    def load(self):
        if self.policy == "staged":
            return
        self._load_gpt()
        self._load_sovits()

    def _load_gpt(self):
        if self.gpt is None:
            from sakuratts.backends.cuda.gpt import CUDAGPT
            self.gpt = CUDAGPT.load(self.packages["gpt"], capacity=self.capacity,
                                    use_graph=self.use_graph, precision=self.gpt_precision,
                                    attention=self.gpt_attention, attention_chunk_size=self.gpt_attention_chunk_size,
                                    prefill_query_chunk_size=self.gpt_prefill_query_chunk_size)

    def _load_sovits(self):
        if self.sovits is None:
            if self.config.get("acoustic_python"):
                from sakuratts.backends.onnx.process import ORTProcessSoVITS
                self.sovits = ORTProcessSoVITS(self.packages["sovits"],
                    self.config_path.parent / self.config["acoustic_python"],
                    allow_experimental_fp16=self.allow_experimental_acoustic_fp16,
                    acoustic_arena_shrink=self.acoustic_arena_shrink,
                    acoustic_chunk_frames=self.acoustic_chunk_frames,
                    acoustic_session_policy=self.acoustic_session_policy)
            else:
                from sakuratts.backends.onnx.sovits import ORTSoVITS
                self.sovits = ORTSoVITS.load(self.packages["sovits"],
                    allow_experimental_fp16=self.allow_experimental_acoustic_fp16,
                    acoustic_arena_shrink=self.acoustic_arena_shrink,
                    acoustic_chunk_frames=self.acoustic_chunk_frames,
                    acoustic_session_policy=self.acoustic_session_policy)

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
        self.japanese.close()
        self.segmenter.close()

    def synthesize(self, text, *, reference=None, seed=1234, language="ja", split_method="cut0",
                   top_k=15, temperature=1., repetition_penalty=1.35, early_stop_num=2700,
                   cancel_requested=None, random_inputs=None, fragment_interval=0.3,
                   on_fragment=None, collect_audio=True):
        if self.busy:
            raise RuntimeError("This engine already has an active request")
        if not collect_audio and on_fragment is None:
            raise ValueError("Streaming without collecting audio requires an on_fragment callback")
        if not text.strip() or seed < -1 or top_k<1 or early_stop_num < -1:
            raise ValueError("Require text, seed>=-1, top_k>=1 and early_stop_num>=-1")
        if not math.isfinite(fragment_interval) or fragment_interval < 0:
            raise ValueError("fragment_interval must be finite and nonnegative")
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
            rng = np.random.default_rng(seed)
            for index, prepared in enumerate(request.fragments):
                logger.debug("分句 %d/%d | %d 个音素 | 前端处理后的文本: %r",
                            index + 1, len(request.fragments), len(prepared.target["phones"]),
                            prepared.target["norm_text"])
                if len(request.fragments) > 1:
                    logger.info("片段 [%d/%d]", index + 1, len(request.fragments))
                self._load_gpt()
                if self.policy != "staged":
                    self._load_sovits()
                semantic = generate_prepared_semantic(prepared,ref,gpt=self.gpt,rng=rng,
                    top_k=top_k,temperature=temperature,repetition_penalty=repetition_penalty,
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
                    parts.append(actual.pcm)
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
            pcm = np.concatenate(parts) if collect_audio else np.empty(0, dtype=np.int16)
            elapsed=time.perf_counter()-started
            limited=any(set(f["stop_reasons"]) & {"early_stop_num","iteration_limit"} for f in fragments)
            duration=sum(f["waveform_samples"] for f in fragments)/rate
            report={"status":"stopped_at_limit" if limited else "completed", "text":text,
                "reference":reference,"reference_identity":ref.manifest["identity"],
                "parameters":{"seed":seed,"rng":"numpy.default_rng (not Torch seed-equivalent)",
                    "language":language,"split_method":split_method,"top_k":top_k,"top_p":1.,
                    "temperature":temperature,"repetition_penalty":repetition_penalty,
                    "early_stop_num":early_stop_num,"noise_scale":0.5,"speed":1.,"fragment_interval":fragment_interval},
                "policy":self.policy,"cuda_graph":self.use_graph,"capacity":self.capacity,
                "request_ms":elapsed*1000,"frontend_ms":request.seconds*1000,
                "sample_rate":rate,"audio_seconds":duration,"pcm_seconds":pcm_samples/rate,
                "rtf":elapsed/duration,"fragments":fragments,
                "timing_scope":"Original text through complete PCM, including missing model loads, transfers and sampling; file output separate",
                "precision": f"GPT {self.gpt_precision.upper()}, acoustic {self.acoustic_precision.upper()}; FP16 paths are experimental; public logits and acoustic I/O remain FP32",
                "gpt_precision":self.gpt_precision,"acoustic_precision":self.acoustic_precision,
                "acoustic_arena_shrink":self.acoustic_arena_shrink,
                "acoustic_chunk_frames":self.acoustic_chunk_frames,
                "acoustic_session_policy":self.acoustic_session_policy,
                "gpt_attention":self.gpt_attention,"gpt_attention_chunk_size":self.gpt_attention_chunk_size,
                "gpt_prefill_query_chunk_size":self.gpt_prefill_query_chunk_size,
                "frontend_profile":self.manifests["frontend"].get("japanese_g2p",{"implementation":"pyopenjtalk-plus"}),
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


def write_wav(path, pcm, sample_rate):
    path=Path(path)
    with path.open("xb") as stream:
        with wave.open(stream,"wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.astype("<i2",copy=False).tobytes())


def run_cli(args):
    output=Path(args.output).resolve()
    record=output.with_suffix(".json")
    if output.suffix.lower()!=".wav" or output.exists() or record.exists():
        raise ValueError("Choose a new .wav output path; neither WAV nor JSON may already exist")
    start=time.perf_counter()
    engine=NVIDIAEngine(args.config,policy=args.model_policy,use_graph=not args.no_cuda_graph,
        capacity=args.capacity,gpt_precision=args.gpt_precision,
                        gpt_attention=args.gpt_attention,gpt_attention_chunk_size=args.gpt_attention_chunk_size,
                        allow_experimental_acoustic_fp16=args.allow_experimental_acoustic_fp16,
                        acoustic_arena_shrink=args.acoustic_arena_shrink,
                        acoustic_chunk_frames=args.acoustic_chunk_frames)
    try:
        pcm,report=engine.synthesize(args.text,reference=args.reference,seed=args.seed,
            language=args.language,split_method=args.text_split_method,top_k=args.top_k,
            temperature=args.temperature,repetition_penalty=args.repetition_penalty,
            early_stop_num=args.early_stop_num)
        output.parent.mkdir(parents=True,exist_ok=True)
        write_wav(output,pcm,report["sample_rate"])
        report["startup_to_wav_ms"]=(time.perf_counter()-start)*1000
        report["startup_timing_scope"]="run_cli entry through WAV write; excludes Python startup, module imports, JSON write and shutdown"
        report["config_sha256"]=sha256_file(args.config)
        report["wav_sha256"]=sha256_file(output)
        import sys
        report["torch_imported"]="torch" in sys.modules
        record.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"status":report["status"],"audio":str(output),"record":str(record)},ensure_ascii=False))
        return 0 if report["status"]=="completed" else 2
    finally:
        engine.close()
