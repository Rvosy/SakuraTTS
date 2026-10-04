"""Independent Japanese text-to-WAV entry point for prepared CUDA models."""

import json
from pathlib import Path
import time
import wave

import numpy as np

from sakuratts.module.reference_condition import sha256_file
from sakuratts.TTS_infer_pack.runtime import InferenceRuntime


class NVIDIAEngine(InferenceRuntime):
    """One active model pair and one synchronous request; caller owns lifetime."""
    name = "cuda"

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
        self.policy, self.use_graph, self.capacity = policy, use_graph, capacity
        self._prepare_model(config, load_references=load_references)

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
                from sakuratts.runtime.ort_process import ORTProcessSoVITS
                self.sovits = ORTProcessSoVITS(self.packages["sovits"],
                    self.config_path.parent / self.config["acoustic_python"],
                    allow_experimental_fp16=self.allow_experimental_acoustic_fp16,
                    acoustic_arena_shrink=self.acoustic_arena_shrink,
                    acoustic_chunk_frames=self.acoustic_chunk_frames,
                    acoustic_session_policy=self.acoustic_session_policy)
            else:
                from sakuratts.module.sovits import ORTSoVITS
                self.sovits = ORTSoVITS.load(self.packages["sovits"],
                    allow_experimental_fp16=self.allow_experimental_acoustic_fp16,
                    acoustic_arena_shrink=self.acoustic_arena_shrink,
                    acoustic_chunk_frames=self.acoustic_chunk_frames,
                    acoustic_session_policy=self.acoustic_session_policy)

    def _execution_report(self):
        return {"backend": self.name, "gpt_device": "cuda", "acoustic_device": "cuda",
                "cuda_graph": self.use_graph, "gpt_attention": self.gpt_attention,
                "gpt_attention_chunk_size": self.gpt_attention_chunk_size}


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
