"""Use a private, persistent ORT process when its Python ABI differs."""

import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from sakuratts.runtime.protocol import read_message, write_message
from sakuratts.module.sovits import ORTSoVITS, read_manifest


class ORTProcessSoVITS:
    def __init__(self,package,python,*,diagnostic=False,allow_experimental_fp16=False,
                 acoustic_arena_shrink=False,acoustic_chunk_frames=None,acoustic_session_policy="resident"):
        if not isinstance(acoustic_arena_shrink, bool):
            raise ValueError("acoustic_arena_shrink must be a bool")
        package=Path(package).resolve(strict=True)
        python=Path(python).resolve(strict=True)
        manifest,_=read_manifest(package,diagnostic=diagnostic,
                                 allow_experimental_fp16=allow_experimental_fp16,
                                 acoustic_arena_shrink=acoustic_arena_shrink,
                                 acoustic_chunk_frames=acoustic_chunk_frames,
                                 acoustic_session_policy=acoustic_session_policy)
        self.encoder=SimpleNamespace(manifest=manifest)
        self.sample_rate=manifest["config"]["sample_rate"]
        self.last_transfer=None
        self.diagnostic=diagnostic
        self.acoustic_arena_shrink=acoustic_arena_shrink
        self.acoustic_chunk_frames=acoustic_chunk_frames
        self.acoustic_session_policy=acoustic_session_policy
        command=[str(python),"-B",str(Path(__file__).with_name("ort_worker.py")),"--package",str(package)]
        if diagnostic:
            command.append("--diagnostic")
        if allow_experimental_fp16:
            command.append("--allow-experimental-fp16")
        if acoustic_arena_shrink:
            command.append("--acoustic-arena-shrink")
        if acoustic_chunk_frames is not None:
            command.extend(("--acoustic-chunk-frames",str(acoustic_chunk_frames)))
        command.extend(("--acoustic-session-policy",acoustic_session_policy))
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONUTF8="1")
        environment.pop("PYTHONPATH",None)
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            runtime,arrays=read_message(self.process.stdout)
            if runtime.get("status")!="ready":
                raise RuntimeError(runtime.get("error","Acoustic worker did not become ready"))
            if (runtime.get("worker_pid")!=self.process.pid or self.process.poll() is not None
                    or not runtime.get("providers") or runtime["providers"][0]!="CUDAExecutionProvider"):
                raise RuntimeError("Acoustic worker process or execution provider is unavailable")
            self.runtime,self.providers=runtime,runtime["providers"]
            self.provider_options=runtime["provider_options"]
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    validate_reference=ORTSoVITS.validate_reference
    _inputs=ORTSoVITS._inputs

    def decode(self,codes,phones,ge,ge512,noise,*,noise_scale=.5,speed=1.,capture=False):
        if self.process is None:
            raise RuntimeError("Acoustic model is unloaded")
        self.last_transfer=None
        if capture and not self.diagnostic:
            raise ValueError("Intermediate capture requires a diagnostic acoustic worker")
        feeds=self._inputs(codes,phones,ge,ge512,noise,noise_scale,speed)
        feeds.pop("noise_scale")
        arrays={}
        start=time.perf_counter()
        try:
            write_message(self.process.stdin,{"command":"decode","noise_scale":noise_scale,"speed":speed,"capture":capture},feeds)
            meta,arrays=read_message(self.process.stdout)
            if meta.get("status")!="ok":
                raise RuntimeError(meta.get("error","Acoustic worker failed"))
            total=(time.perf_counter()-start)*1000
            config=self.encoder.manifest["config"]
            expected_samples=(feeds["codes"].shape[-1]*config["semantic_upsample_factor"]
                              *math.prod(config["model"]["upsample_rates"]))
            waveform=arrays.get("waveform")
            if (not isinstance(waveform,np.ndarray) or waveform.dtype!=np.float32
                    or waveform.shape!=(1,1,expected_samples) or not np.isfinite(waveform).all()):
                raise RuntimeError("Acoustic worker returned an invalid complete waveform")
            self.last_transfer={"roundtrip_ms":total,"worker_compute_ms":meta["compute_ms"],
                "transport_and_scheduling_ms":total-meta["compute_ms"],
                "upload_bytes":sum(a.nbytes for a in feeds.values()),
                "download_bytes":sum(a.nbytes for a in arrays.values()),
                "worker_acoustic":meta.get("acoustic_transport"),
                "ipc_scope":"FP32/int64 inputs and complete FP32 waveform; split latent stays in worker"}
            return (waveform,dict(arrays)) if capture else waveform
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        finally:
            feeds.clear()
            arrays.clear()

    def release_request_state(self):
        self.last_transfer=None

    def close(self):
        process,self.process=getattr(self,"process",None),None
        self.release_request_state()
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    write_message(process.stdin,{"command":"close"})
                    exit_code=process.wait(timeout=30)
                    if exit_code!=0:
                        raise RuntimeError(f"Acoustic worker exited with code {exit_code} during graceful close")
                except (OSError,ValueError,subprocess.TimeoutExpired):
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
        finally:
            try:
                process.stdin.close()
            finally:
                process.stdout.close()

    unload=close
