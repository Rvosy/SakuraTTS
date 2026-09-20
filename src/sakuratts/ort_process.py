"""Use a private, persistent ORT process when its Python ABI differs."""

import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from .array_protocol import read_message, write_message
from .ort_sovits import ORTSoVITS, read_manifest


class ORTProcessSoVITS:
    def __init__(self,package,python,*,diagnostic=False,allow_experimental_fp16=False,acoustic_arena_shrink=False):
        if not isinstance(acoustic_arena_shrink, bool):
            raise ValueError("acoustic_arena_shrink must be a bool")
        package=Path(package).resolve(strict=True)
        python=Path(python).resolve(strict=True)
        manifest,_=read_manifest(package,diagnostic=diagnostic,
                                 allow_experimental_fp16=allow_experimental_fp16)
        self.encoder=SimpleNamespace(manifest=manifest)
        self.sample_rate=manifest["config"]["sample_rate"]
        self.last_transfer=None
        self.diagnostic=diagnostic
        self.acoustic_arena_shrink=acoustic_arena_shrink
        command=[str(python),"-B",str(Path(__file__).with_name("ort_worker.py")),"--package",str(package)]
        if diagnostic:
            command.append("--diagnostic")
        if allow_experimental_fp16:
            command.append("--allow-experimental-fp16")
        if acoustic_arena_shrink:
            command.append("--acoustic-arena-shrink")
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONUTF8="1")
        environment.pop("PYTHONPATH",None)
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            self.runtime,_=read_message(self.process.stdout)
            if self.runtime["status"]!="ready":
                raise RuntimeError(self.runtime.get("error","Acoustic worker did not become ready"))
        except BaseException:
            self.close()
            raise

    validate_reference=ORTSoVITS.validate_reference
    _inputs=ORTSoVITS._inputs

    def decode(self,codes,phones,ge,ge512,noise,*,noise_scale=.5,speed=1.,capture=False):
        if self.process is None:
            raise RuntimeError("Acoustic model is unloaded")
        if capture and not self.diagnostic:
            raise ValueError("Intermediate capture requires a diagnostic acoustic worker")
        feeds=self._inputs(codes,phones,ge,ge512,noise,noise_scale,speed)
        feeds.pop("noise_scale")
        start=time.perf_counter()
        try:
            write_message(self.process.stdin,{"command":"decode","noise_scale":noise_scale,"speed":speed,"capture":capture},feeds)
            meta,arrays=read_message(self.process.stdout)
            if meta["status"]!="ok":
                raise RuntimeError(meta.get("error","Acoustic worker failed"))
        except BaseException:
            self.close()
            raise
        total=(time.perf_counter()-start)*1000
        self.last_transfer={"roundtrip_ms":total,"worker_compute_ms":meta["compute_ms"],
            "transport_and_scheduling_ms":total-meta["compute_ms"],
            "upload_bytes":sum(a.nbytes for a in feeds.values()),
            "download_bytes":sum(a.nbytes for a in arrays.values())}
        return (arrays["waveform"],arrays) if capture else arrays["waveform"]

    def release_request_state(self):
        pass

    def close(self):
        process,self.process=getattr(self,"process",None),None
        if process is None:
            return
        if process.poll() is None:
            try:
                write_message(process.stdin,{"command":"close"})
                process.wait(timeout=30)
            except (OSError,subprocess.TimeoutExpired):
                process.terminate()
                process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()

    unload=close
