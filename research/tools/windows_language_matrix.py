"""Measure managed CUDA HTTP synthesis and its full Windows service process tree.

Use the JSON config and request from verify_portable_first_use.py. Prepopulate
the portable model/reference cache for the desired precisions before comparing
startup times. Each language/profile starts a new service in the same bundle.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from datetime import datetime, timezone
from http.client import HTTPConnection
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import threading
import time
import wave

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "research/tools")]
from verify_portable_first_use import Service, audio_info, available_port, sha256
from windows_wddm_memory import WDDMMemorySampler


TEXTS = {
    "zh": {
        "short": "今天下午三点，我们在图书馆门口见。",
        "long": "今天下午三点，我们在图书馆门口见。天气预报说下午可能会下雨，你出门前记得带把伞。到了以后给我打个电话，我会在门口等你。",
    },
    "ja": {
        "short": "今日の午後三時に、図書館の前で会いましょう。",
        "long": "今日の午後三時に、図書館の前で会いましょう。天気予報では午後から雨が降るかもしれないので、出かけるときは傘を忘れずに。着いたら電話してください。入り口で待っています。",
    },
    "en": {
        "short": "Let's meet in front of the library at three this afternoon.",
        "long": "Let's meet in front of the library at three this afternoon. The forecast says it might rain, so remember to bring an umbrella when you leave. Call me when you get there. I'll be waiting by the entrance.",
    },
}
PROFILES = ("fp16", "low-memory", "minimum-memory", "fp32")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cuda_adapter():
    driver = ct.WinDLL("nvcuda")
    device, mask = ct.c_int(), ct.c_uint()
    name, luid = ct.create_string_buffer(256), ct.create_string_buffer(8)
    for function, arguments in (("cuInit", (0,)), ("cuDeviceGet", (ct.byref(device), 0)),
            ("cuDeviceGetName", (name, len(name), device)),
            ("cuDeviceGetLuid", (luid, ct.byref(mask), device))):
        status = getattr(driver, function)(*arguments)
        if status:
            raise OSError(f"{function} returned CUDA status {status}")
    low, high = struct.unpack("<II", luid.raw)
    return {"name": name.value.decode(), "luid": f"0x{high:08x}_0x{low:08x}", "node_mask": mask.value}


def host_settings(bundle):
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    packages = {}
    for role in ("main", "acoustic"):
        site = bundle / "runtime" / role / "Lib/site-packages"
        packages[role] = {pkg.name: pkg.version for pkg in distributions(path=[str(site)])
                          if pkg.name and pkg.name.startswith(("onnxruntime", "cupy", "numpy", "pypinyin",
                              "jieba", "opencc", "tokenizers"))}
    return {"cpu": cpu, "logical_cpus": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
            "physical_memory_bytes": psutil.virtual_memory().total, "python_packages": packages}


def reference_audio_info(path):
    with wave.open(str(path), "rb") as audio:
        return {"sample_rate": audio.getframerate(), "channels": audio.getnchannels(),
                "seconds": audio.getnframes() / audio.getframerate()}


class Monitor:
    def __init__(self, pid, gpu, directory, interval):
        self.parent = psutil.Process(pid)
        self.gpu, self.directory, self.interval = gpu, directory, interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect)
        self.error = None

    def __enter__(self):
        self.thread.start()
        return self

    def collect(self):
        try:
            with WDDMMemorySampler([self.parent.pid]) as pdh, (self.directory / "samples.jsonl").open(
                    "x", encoding="utf-8") as stream:
                write_json(self.directory / "pdh.json", pdh.metadata)
                while not self.stop.is_set():
                    started = time.perf_counter()
                    tree = [self.parent, *self.parent.children(recursive=True)]
                    private = {}
                    for process in tree:
                        try:
                            private[process.pid] = process.memory_info().private
                        except psutil.NoSuchProcess:
                            pass
                    sample = pdh.sample(pids=[p.pid for p in tree])
                    row = {"t": sample["monotonic_s"], "unix": sample["timestamp_unix_s"],
                           "pids": sample["pids"], "private": private,
                           "collect_status": sample["collect_status"]}
                    for counter in sample["counters"]:
                        if counter["metric"] != "dedicated_bytes":
                            continue
                        rows = [r for r in counter["rows"] if r["luid"] == self.gpu["luid"]
                                and self.gpu["node_mask"] & (1 << r["physical_adapter"])]
                        row[counter["scope"]] = [[r["pid"], r["physical_adapter"], r["bytes"],
                            r["valid"] and not r["duplicate_identity"]] for r in rows]
                        row[counter["scope"] + "_status"] = counter["status"]
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    self.stop.wait(max(0, started + self.interval - time.perf_counter()))
        except BaseException as error:
            self.error = error

    def __exit__(self, kind, value, tb):
        self.stop.set()
        self.thread.join()
        if self.error and kind is None:
            raise self.error


def request_audio(service, payload, path):
    service.assert_listener_owned()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    connection = HTTPConnection("127.0.0.1", service.port, timeout=300)
    try:
        start = time.perf_counter()
        connection.request("POST", "/tts", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        end = time.perf_counter()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {data[:2000]!r}")
    finally:
        connection.close()
    info = audio_info(data)
    path.write_bytes(data)
    return dict(info, start=start, end=end, http_seconds=end-start, rtf=(end-start)/info["audio_seconds"])


def run_cell(args, gpu, profile, language, config, request):
    cell = args.output / (profile + "-" + language)
    (cell / "external-cwd").mkdir(parents=True)
    config["sakuratts"].update(backend="cuda", profile=profile)
    write_json(cell / "config.json", config)
    result = {"profile": profile, "language": language, "texts": TEXTS[language],
              "services": [], "requests": []}
    service = Service(args.bundle, cell / "config.json", cell, "service", "managed",
                      available_port(None), result)
    with service, Monitor(service.process.pid, gpu, cell, args.interval_ms / 1000):
        service.wait_ready()
        service.wake(profile)
        time.sleep(1.3)
        for case, text in TEXTS[language].items():
            payload = dict(request, text=text, text_lang=language)
            for index in range(args.hot + 1):
                name = case + ("-warmup" if index == 0 else f"-hot-{index}")
                row = request_audio(service, payload, cell / (name + ".wav"))
                row.update(case=case, index=index, name=name)
                result["requests"].append(row)
                time.sleep(1.3)
            service.state("awake")
    service.assert_process_paths()
    service.assert_execution("cuda", profile)
    write_json(cell / "result.json", result)
    print(profile, language, "completed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES)
    parser.add_argument("--languages", nargs="+", choices=TEXTS, default=["ja", "en", "zh"])
    parser.add_argument("--hot", type=int, default=3)
    parser.add_argument("--interval-ms", type=float, default=20)
    args = parser.parse_args()
    if args.hot < 1 or args.interval_ms <= 0:
        parser.error("--hot and --interval-ms must be positive")
    args.output, args.bundle = args.output.resolve(), args.bundle.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    request = json.loads(args.request.read_text(encoding="utf-8"))
    gpu = cuda_adapter()
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    meta = {"utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu, "nvidia_smi": smi,
            "host": host_settings(args.bundle),
            "os": platform.platform(), "monitor_python": sys.version, "psutil": psutil.__version__,
            "profiles": args.profiles, "languages": args.languages, "hot": args.hot,
            "interval_ms": args.interval_ms, "texts": TEXTS,
            "bundle": json.loads((args.bundle / "runtime/portable.json").read_text("utf-8")),
            "manifest_sha256": sha256(args.bundle / "bundle-manifest.json"),
            "request": request, "inputs": {
                k: {"name": Path(v).name, "sha256": sha256(v)} for k, v in {
                    "gpt": config["custom"]["t2s_weights_path"],
                    "sovits": config["custom"]["vits_weights_path"],
                    "reference": request["ref_audio_path"]}.items()}}
    meta["inputs"]["reference"].update(reference_audio_info(request["ref_audio_path"]))
    write_json(args.output / "protocol.json", meta)
    for profile in args.profiles:
        for language in args.languages:
            run_cell(args, gpu, profile, language, config, request)


if __name__ == "__main__":
    main()
