"""Verify Windows first use or relocated cache reuse from raw checkpoints.

Run with a development Python that has psutil. The service and every preparation
or inference child must use the selected bundle. This is a real GPU acceptance
run, not a unit test or a peak-memory benchmark. By default the model/reference
caches must be empty; --reuse-cache instead requires both existing caches.
It never deletes existing caches.
"""

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
from http.client import HTTPConnection
import io
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import threading
import time
import traceback
import wave


PROJECT = Path(__file__).resolve().parents[1]
TIMEOUT = 900
TEXT = "おはよう。今日もよろしくね。"
PREPARATION_MARKERS = ("首次转换 GPT / SoVITS 权重",
                       "正在提取参考音频特征", "运行准备命令:")
# The child cannot create descendants before it belongs to our Windows Job.
START_GATE = ("import sys,runpy; "
              "assert sys.stdin.buffer.read(1) == b'1', 'Missing ownership gate'; "
              "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_snapshot(bundle):
    """Include file metadata so an in-place rebuild cannot masquerade as a hit."""
    result = {}
    for name in ("models", "references"):
        root = bundle / "cache" / name
        rows = []
        keys = sorted(p.name for p in root.iterdir()) if root.exists() else []
        for path in sorted(root.rglob("*")) if root.exists() else []:
            if path.is_file():
                stat = path.stat()
                rows.append([path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns])
        result[name] = {"keys": keys, "files": len(rows), "bytes": sum(row[1] for row in rows),
                       "inventory_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest()}
    return result


def require_empty_cache(bundle):
    snapshot = cache_snapshot(bundle)
    if any(part["keys"] for part in snapshot.values()):
        raise ValueError("First-use verification requires empty cache/models and cache/references; "
                         "use a fresh bundle copy. Existing caches will not be removed.")
    return snapshot


def initial_cache(bundle, reuse):
    if not reuse:
        return require_empty_cache(bundle)
    snapshot = cache_snapshot(bundle)
    if not all(part["keys"] and part["files"] for part in snapshot.values()):
        raise ValueError("--reuse-cache requires existing model and reference caches from a completed first use")
    return snapshot


def external_environment(output):
    environment = dict(os.environ)
    for name in list(environment):
        if name.upper().startswith(("PYTHON", "CUDA_PATH", "CUDA_HOME", "SAKURATTS_")):
            environment.pop(name)
    system = Path(os.environ.get("SystemRoot", "C:/Windows"))
    invalid = str(output / "nonexistent-external-runtime")
    environment.update(PATH=os.pathsep.join(map(str, (system / "System32", system))),
        PYTHONHOME=invalid, PYTHONPATH=invalid, CUDA_PATH=invalid, CUDA_HOME=invalid,
        PYTHONIOENCODING="utf-8", PYTHONUTF8="1", NO_COLOR="1")
    return environment


def audio_info(data):
    with wave.open(io.BytesIO(data), "rb") as stream:
        rate, channels, width, frames = (stream.getframerate(), stream.getnchannels(),
                                        stream.getsampwidth(), stream.getnframes())
        pcm = stream.readframes(frames)
    if (rate, channels, width) != (32000, 1, 2) or not frames:
        raise AssertionError("Expected nonempty 32 kHz mono int16 WAV")
    if len(pcm) != frames * width or not any(pcm):
        raise AssertionError("WAV contains truncated or entirely silent PCM")
    return {"sample_rate": rate, "channels": channels, "frames": frames,
            "audio_seconds": frames / rate, "pcm_bytes": len(pcm),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "wav_sha256": hashlib.sha256(data).hexdigest()}


def compare_audio(paths):
    """Compare saved int16 WAVs against the first; keep the one-LSB bound fixed."""
    if len(paths) < 2:
        raise ValueError("At least two WAVs are required for an audio comparison")
    decoded = []
    for path in paths:
        data = Path(path).read_bytes()
        info = audio_info(data)
        with wave.open(io.BytesIO(data), "rb") as stream:
            pcm = array("h", stream.readframes(stream.getnframes()))
        if sys.byteorder != "little":
            pcm.byteswap()
        decoded.append((info, pcm))
    base_info, baseline = decoded[0]
    comparisons = []
    for path, (info, pcm) in zip(paths[1:], decoded[1:]):
        row = {"wav": str(path), "reference_pcm_sha256": base_info["pcm_sha256"],
               "pcm_sha256": info["pcm_sha256"], "reference_samples": len(baseline),
               "samples": len(pcm), "same_length": len(pcm) == len(baseline),
               "changed_samples": None, "max_abs_lsb": None, "rms_lsb": None,
               "bit_exact": False, "within_one_lsb": False}
        if row["same_length"]:
            changed = maximum = square_sum = 0
            for actual, expected in zip(pcm, baseline):
                error = abs(actual - expected)
                changed += error != 0
                maximum = max(maximum, error)
                square_sum += error * error
            row.update(changed_samples=changed, max_abs_lsb=maximum,
                       rms_lsb=(square_sum / len(pcm)) ** .5, bit_exact=changed == 0,
                       within_one_lsb=maximum <= 1)
        comparisons.append(row)
    return {"reference_wav": str(paths[0]), "allowed_max_abs_lsb": 1,
            "bit_exact": all(row["bit_exact"] for row in comparisons),
            "within_one_lsb": all(row["within_one_lsb"] for row in comparisons),
            "comparisons": comparisons}


def preparation_events(log):
    return {marker: log.count(marker) for marker in PREPARATION_MARKERS}


def assert_reused(before, after, log):
    if after != before:
        raise AssertionError("Model/reference cache changed during a reuse request")
    events = preparation_events(log)
    if any(events.values()):
        raise AssertionError("Reuse unexpectedly invoked preparation: " + repr(events))


def available_port(requested):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", requested or 0))
        return listener.getsockname()[1]


class Service:
    def __init__(self, bundle, config, output, name, mode, port, report):
        import psutil
        self.psutil = psutil
        self.bundle, self.output, self.mode, self.port = bundle, output, mode, port
        self.console_path = output / (name + "-console.log")
        self.log_path = output / (name + ".log")
        command = [str(bundle / "runtime/main/python.exe"), "-I", "-B", "-u", "-c", START_GATE,
                   str(bundle / "launcher.py"), "serve", "--tts-config", str(config),
                   "--host", "127.0.0.1", "--port", str(port), "--log-level", "info",
                   "--log-file", str(self.log_path)]
        if mode == "managed":
            command += ["--runtime-mode", "managed", "--wake-timeout-seconds", str(TIMEOUT),
                        "--operation-timeout-seconds", str(TIMEOUT), "--idle-sleep-seconds", "600"]
        self.command = command
        self.record = {"name": name, "mode": mode, "command": command,
                       "cwd": str(output / "external-cwd"), "processes": [], "checks": {}}
        report["services"].append(self.record)
        self.process = self.tree = self.console = self.sampler = None
        self.ready = False
        self.stop_sampling = threading.Event()
        self.processes = {}
        self.sampling_errors = []

    def __enter__(self):
        process_tree = runpy.run_path(str(PROJECT / "src/sakuratts/_internal/process_tree.py"))["ProcessTree"]
        self.tree = process_tree()
        try:
            self.console = self.console_path.open("xb")
            self.process = subprocess.Popen(self.command, cwd=self.record["cwd"],
                env=external_environment(self.output), stdin=subprocess.PIPE, stdout=self.console,
                stderr=subprocess.STDOUT, **self.tree.popen_options())
            self.tree.bind(self.process)
            self.record["pid"] = self.process.pid
            self.sampler = threading.Thread(target=self._sample, daemon=True)
            self.sampler.start()
            self.process.stdin.write(b"1")
            self.process.stdin.flush()
            self.process.stdin.close()
            return self
        except BaseException:
            self.close()
            raise

    def _sample(self):
        while not self.stop_sampling.is_set():
            try:
                parent = self.psutil.Process(self.process.pid)
                for process in [parent, *parent.children(recursive=True)]:
                    try:
                        identity = (process.pid, process.create_time())
                        if identity not in self.processes:
                            self.processes[identity] = {"pid": process.pid, "created": identity[1],
                                "executable": process.exe(), "command": process.cmdline()}
                    except self.psutil.NoSuchProcess:
                        continue
            except self.psutil.NoSuchProcess:
                return
            except Exception as error:
                self.sampling_errors.append(str(error))
                return
            self.stop_sampling.wait(.2)

    def http(self, path, payload=None, timeout=TIMEOUT + 30):
        if payload is not None or path.startswith("/control"):
            self.assert_listener_owned()
        connection = HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            connection.request("GET" if body is None else "POST", path, body=body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read()
            result = json.loads(data) if "json" in response.getheader("Content-Type", "") else data
            if response.status not in (200, 202):
                raise RuntimeError(f"HTTP {response.status} at {path}: {str(result)[:2000]}")
            return result
        finally:
            connection.close()

    def assert_listener_owned(self):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("The owned service is no longer running")
        connections = self.psutil.Process(self.process.pid).net_connections(kind="tcp")
        if not any(row.status == self.psutil.CONN_LISTEN and row.laddr.port == self.port
                   for row in connections):
            raise RuntimeError("HTTP listener does not belong to the launched service")

    def wait_ready(self):
        started = time.monotonic()
        deadline = started + TIMEOUT + 30
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Server exited before /health; see " + str(self.console_path))
            try:
                health = self.http("/health", timeout=2)
                if health.get("status") == "ready":
                    self.assert_listener_owned()
                    self.ready = True
                    self.record["startup_seconds"] = time.monotonic() - started
                    self.record["initial_health"] = health
                    return health
            except OSError:
                pass
            time.sleep(.2)
        raise TimeoutError("Server did not become ready; see " + str(self.console_path))

    def state(self, expected):
        deadline = time.monotonic() + TIMEOUT + 30
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Service exited while awaiting " + expected)
            state = self.http("/runtime", timeout=5)
            if state.get("state") == "failed":
                raise RuntimeError("Managed runtime failed: " + repr(state))
            if state.get("state") == expected:
                return state
            time.sleep(.2)
        raise TimeoutError("Managed runtime did not reach " + expected)

    def wake(self):
        started = time.monotonic()
        self.http("/runtime/wake", {"keep_alive_seconds": 600})
        result = self.state("awake")
        if not result.get("model_loaded") or result.get("worker_pid") is None:
            raise AssertionError("Default FP32 model was not loaded after wake")
        self.record["wake_seconds"] = time.monotonic() - started
        self.record["awake"] = result

    def sleep(self):
        awake = self.http("/runtime", timeout=5)
        worker_pid = awake.get("worker_pid")
        if awake.get("state") != "awake" or worker_pid is None or worker_pid == self.process.pid:
            raise AssertionError("Expected an awake inference worker before testing sleep")
        controller = self.psutil.Process(self.process.pid)
        worker = self.psutil.Process(worker_pid)
        if controller not in worker.parents():
            raise AssertionError("Reported inference worker is outside the owned service tree")
        # A console host belonging to the controller remains alive while the
        # service sleeps. Only the reported inference subtree must disappear.
        before = [worker, *worker.children(recursive=True)]
        self.record["sleep_inference_pids"] = [process.pid for process in before]
        self.http("/runtime/sleep", {})
        state = self.state("sleeping")
        if state.get("worker_pid") is not None or state.get("model_loaded"):
            raise AssertionError("Sleeping runtime retained its inference worker")
        _, alive = self.psutil.wait_procs(before, timeout=15)
        if alive:
            raise AssertionError("Inference descendants survived sleep: " + repr([p.pid for p in alive]))
        self.record["checks"]["sleep_descendants_exited"] = True
        self.record["sleeping"] = state

    def log(self):
        paths = sorted(self.log_path.parent.glob(self.log_path.name + ".*"), reverse=True)
        paths += [self.log_path]
        return "".join(path.read_text(encoding="utf-8", errors="replace") for path in paths if path.is_file())

    def synthesize(self, request, name):
        started = time.monotonic()
        data = self.http("/tts", request)
        if not isinstance(data, bytes):
            raise AssertionError("Expected WAV response, got JSON: " + repr(data))
        elapsed = time.monotonic() - started
        info = audio_info(data)
        path = self.output / (name + ".wav")
        with path.open("xb") as stream:
            stream.write(data)
        return dict(info, name=name, wav=str(path), http_seconds=elapsed)

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    if not self.ready:
                        raise RuntimeError("Service never acquired its HTTP listener")
                    self.http("/control?command=exit", timeout=5)
                    self.process.wait(timeout=20)
                    self.record["graceful_exit"] = True
                except Exception as error:
                    self.record["graceful_exit"] = False
                    self.record["shutdown_error"] = str(error)
            if self.tree is not None:
                self.tree.close(timeout=15)
        finally:
            self.stop_sampling.set()
            if self.sampler is not None:
                self.sampler.join(timeout=5)
            if self.console is not None:
                self.console.close()
            self.record["processes"] = list(self.processes.values())
            self.record["sampling_errors"] = self.sampling_errors
            if self.process is not None:
                self.record["exit_code"] = self.process.poll()

    def __exit__(self, kind, value, tb):
        try:
            self.close()
        except Exception:
            self.record["cleanup_error"] = traceback.format_exc()
            if kind is None:
                raise

    def assert_process_paths(self):
        system = (Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32").resolve()
        # Windows owns console hosts and helpers such as platform's `cmd /c ver`.
        # Only application executables need to come from this bundle.
        invalid = [row for row in self.record["processes"]
                   if not Path(row["executable"]).resolve().is_relative_to(self.bundle)
                   and not Path(row["executable"]).resolve().is_relative_to(system)]
        if invalid or self.sampling_errors:
            raise AssertionError("Process isolation failed: " + repr(invalid or self.sampling_errors))
        if not self.record.get("graceful_exit") or self.record.get("exit_code") != 0:
            raise AssertionError("Service did not exit normally: " + repr(self.record.get("shutdown_error")))
        self.record["checks"]["observed_executables_bundled_or_windows"] = True
        self.record["checks"]["owned_process_tree_reaped"] = True

    def assert_fp32(self):
        if "GPT FP32 / SoVITS FP32" not in self.log():
            raise AssertionError("Service log did not confirm default FP32 execution")
        self.record["checks"]["default_fp32"] = True


def check_inputs(args):
    args.bundle = args.bundle.resolve(strict=True)
    for name in ("gpt", "sovits", "reference"):
        path = getattr(args, name).resolve(strict=True)
        if not path.is_file():
            raise ValueError("Input must be a file: " + str(path))
        setattr(args, name, path)
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory: " + str(args.output))
    if args.output.is_relative_to(args.bundle):
        raise ValueError("Verification output must be outside the release bundle")
    for name in ("runtime/main/python.exe", "runtime/acoustic/python.exe", "launcher.py",
                 "runtime/portable.json", "bundle-manifest.json", "runtime/preparation/preparation.json",
                 "runtime/preparation/preparation-manifest.json", "runtime/preparation/python.exe"):
        if not (args.bundle / name).is_file():
            raise FileNotFoundError(args.bundle / name)
    if not args.prompt_text.strip():
        raise ValueError("Reference transcript must not be empty")
    if args.port is not None and not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    return initial_cache(args.bundle, args.reuse_cache)


def verify_first_use(args, report, config, request, port):
    print("Starting managed first use from raw checkpoints", flush=True)
    first = Service(args.bundle, config, args.output, "managed-first", "managed", port, report)
    with first:
        if first.wait_ready()["model_loaded"]:
            raise AssertionError("Managed service loaded a model before wake")
        first.state("sleeping")
        first.wake()
        after_wake = cache_snapshot(args.bundle)
        if len(after_wake["models"]["keys"]) != 1 or after_wake["references"]["keys"]:
            raise AssertionError("Wake must convert one model without preparing a reference")
        print("Model converted; preparing the first reference and synthesizing", flush=True)
        report["audio"].append(first.synthesize(request, "managed-first"))
        cached = cache_snapshot(args.bundle)
        if len(cached["models"]["keys"]) != 1 or len(cached["references"]["keys"]) != 1:
            raise AssertionError("First speech did not populate exactly one model and one reference cache")
        first_log = first.log()
        events = preparation_events(first_log)
        if not all(events[marker] for marker in PREPARATION_MARKERS):
            raise AssertionError("Logs do not demonstrate raw model and reference preparation: " + repr(events))
        first.record["preparation_events"] = events
        report["cache_after_first_speech"] = cached
        report["audio"].append(first.synthesize(request, "managed-repeat"))
        assert_reused(cached, cache_snapshot(args.bundle), first.log()[len(first_log):])
        first.record["checks"]["repeat_reused_cache"] = True
        first.sleep()
    first.assert_process_paths()
    if not any(Path(row["executable"]).resolve() == args.bundle / "runtime/preparation/python.exe"
               for row in first.record["processes"]):
        raise AssertionError("No bundled preparation interpreter was observed during first use")
    first.record["checks"]["preparation_interpreter_observed"] = True
    first.assert_fp32()
    report["checks"].update(raw_checkpoint_conversion=True, new_reference_preparation=True)
    return cached


def verify_reuse(args, report, config, request, port, cached):
    for mode in ("managed", "direct"):
        name = "managed-restart" if mode == "managed" and not args.reuse_cache else mode + "-reuse"
        print("Starting " + mode + " service to verify disk cache reuse", flush=True)
        service = Service(args.bundle, config, args.output, name, mode, port, report)
        with service:
            if bool(service.wait_ready()["model_loaded"]) != (mode == "direct"):
                raise AssertionError("Startup model_loaded differs from " + mode + " semantics")
            if mode == "managed":
                service.state("sleeping")
                service.wake()
            report["audio"].append(service.synthesize(request, name))
            if mode == "managed":
                service.sleep()
        service.assert_process_paths()
        assert_reused(cached, cache_snapshot(args.bundle), service.log())
        service.record["checks"]["disk_cache_reused"] = True
        service.assert_fp32()


def speech_request(reference, prompt_text):
    return {"text": TEXT, "text_lang": "ja", "ref_audio_path": str(reference),
        "prompt_text": prompt_text, "prompt_lang": "ja", "seed": 1234,
        "top_k": 15, "top_p": 1., "temperature": 1., "repetition_penalty": 1.35,
        "text_split_method": "cut0", "batch_size": 1, "split_bucket": False,
        "parallel_infer": False, "streaming_mode": False, "media_type": "wav"}


def verify(args, report):
    config = args.output / "tts-config.json"
    config.write_text(json.dumps({"custom": {"version": "v2ProPlus", "device": "cuda", "is_half": False,
        "t2s_weights_path": str(args.gpt), "vits_weights_path": str(args.sovits)}},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "external-cwd").mkdir()
    request = speech_request(args.reference, args.prompt_text)
    (args.output / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    port = available_port(args.port)
    report["port"] = port
    cached = (report["cache_before"] if args.reuse_cache else
              verify_first_use(args, report, config, request, port))
    verify_reuse(args, report, config, request, port, cached)
    comparison = report["pcm_comparison"] = compare_audio([audio["wav"] for audio in report["audio"]])
    if not comparison["within_one_lsb"]:
        raise AssertionError("Repeated/restarted/direct PCM differs in length or by more than 1 int16 LSB")
    report["checks"].update(default_fp32=True, repeated_pcm_within_one_lsb=True,
        managed_disk_cache=True, default_direct_disk_cache=True,
        sleeping_inference_tree_exited=True, observed_executables_bundled_or_windows=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "gpt", "sovits", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Verify existing caches, for example after copying the bundle to a new directory")
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error("This acceptance harness requires Windows and an NVIDIA GPU")
    import psutil  # Fail before creating any output or launching a service.
    initial = check_inputs(args)
    inputs = {name: {"path": str(getattr(args, name)), "sha256": sha256(getattr(args, name))}
              for name in ("gpt", "sovits", "reference")}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"format": "sakuratts-portable-first-use-v1", "status": "running",
        "verification_mode": "cache-reuse" if args.reuse_cache else "first-use",
        "started_utc": datetime.now(timezone.utc).isoformat(), "bundle": str(args.bundle),
        "bundle_manifest_sha256": sha256(args.bundle / "bundle-manifest.json"), "inputs": inputs,
        "harness_sha256": sha256(__file__), "psutil_version": psutil.__version__,
        "scope": "Local raw-checkpoint and reference acceptance (see verification_mode); no listening, cross-device, "
                 "network-isolation, file-read isolation, or peak-memory claim",
        "launch_environment": {"external_cwd": True, "initial_path_windows_only": True,
                               "poisoned_pythonhome_pythonpath_cuda": True},
        "cache_before": initial, "services": [], "audio": [], "checks": {}}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        verify(args, report)
        for name, spec in inputs.items():
            if sha256(spec["path"]) != spec["sha256"]:
                raise AssertionError("Input changed during verification: " + name)
        report["checks"]["raw_inputs_unchanged"] = True
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["cache_after"] = cache_snapshot(args.bundle)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Verification " + report["status"] + ": " + str(report_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
