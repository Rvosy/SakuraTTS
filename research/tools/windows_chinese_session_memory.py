"""Compare lossless CPU session layouts against the packaged Mandarin processor on Windows.

Run "screen --bundle RELEASE --resources CHINESE --output NEW_DIRECTORY" with a
Windows Python. The two packaged interpreters prepare local candidates and then
measure each in a fresh process. The input is the original inline FP32 export.
"""
import argparse
import ctypes as ct
import gc
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import time


TEXTS = ("今天下午三点，我们在图书馆门口见。", "银行行长在重庆开会，音乐老师正在教室里弹琴。",
         "明天出门前，请把门口的雨伞带上，到了以后给我打个电话。")
LAYOUT = {"baseline": None, "no-arena": None, "no-pattern": None, "sequential": None,
          "ort-plain": "ort", "ort-mmap": "ort", "inline-optimized": "inline",
          "external-default": "external", "externalized-prepacked": "prepacked",
          "portable-external": "portable", "aligned-external": "aligned"}


class ProcessMemory(ct.Structure):
    _fields_ = [("cb", ct.c_ulong), ("faults", ct.c_ulong)] + [(name, ct.c_size_t) for name in (
        "peak_wset", "wset", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile",
        "peak_pagefile", "private")]


def memory():
    metrics = ProcessMemory()
    metrics.cb = ct.sizeof(metrics)
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    if not kernel.K32GetProcessMemoryInfo(ct.c_void_p(-1), ct.byref(metrics), metrics.cb):
        raise ct.WinError(ct.get_last_error())
    return {name: getattr(metrics, name) for name in ("private", "wset", "peak_wset", "peak_pagefile")}


def prepare_optimized(args):
    import onnxruntime as ort
    for graph in ("g2pw", "bert"):
        for kind in ("ort", "external", "prepacked", "inline"):
            folder = args.output / kind
            folder.mkdir(parents=True, exist_ok=True)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads, opts.log_severity_level = 2, 3
            suffix = ".ort" if kind == "ort" else ".onnx"
            opts.optimized_model_filepath = str(folder / (graph + suffix))
            if kind == "ort":
                opts.add_session_config_entry("session.save_model_format", "ORT")
            elif kind in ("external", "prepacked"):
                opts.add_session_config_entry("session.optimized_model_external_initializers_file_name", graph + ".bin")
                opts.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "16384")
                if kind == "prepacked":
                    opts.add_session_config_entry("session.save_external_prepacked_constant_initializers", "1")
            session = ort.InferenceSession(str(args.resources / (graph + ".onnx")), opts,
                                           providers=["CPUExecutionProvider"])
            del session


def prepare_portable(args):
    import onnx
    target = args.output / "portable"
    target.mkdir(parents=True, exist_ok=False)
    for name in ("g2pw", "bert"):
        model = onnx.load(str(args.resources / (name + ".onnx")))
        onnx.save_model(model, str(target / (name + ".onnx")), save_as_external_data=True,
                        all_tensors_to_one_file=True, location=name + ".bin", size_threshold=16384)
        del model


def align(args):
    import onnx
    target = args.output / "aligned"
    target.mkdir(parents=True, exist_ok=False)
    for name in ("g2pw", "bert"):
        original = args.output / "portable" / (name + ".onnx")
        model = onnx.load(str(original), load_external_data=False)
        with original.with_suffix(".bin").open("rb") as source, (target / (name + ".bin")).open("xb") as dest:
            for tensor in model.graph.initializer:
                fields = {field.key: field for field in tensor.external_data}
                if not fields:
                    continue
                source.seek(int(fields["offset"].value))
                dest.write(bytes(-dest.tell() % 65536))
                fields["offset"].value = str(dest.tell())
                remaining = int(fields["length"].value)
                while remaining:
                    part = source.read(min(4*1024*1024, remaining))
                    if not part:
                        raise EOFError("Truncated external initializer: " + tensor.name)
                    dest.write(part)
                    remaining -= len(part)
        (target / original.name).write_bytes(model.SerializeToString())


def measure(args):
    import numpy as np
    import onnxruntime as ort
    from sakuratts.text.processors import ChineseProcessor
    from sakuratts.text.g2pw_session import G2PWSession
    folder = args.output / args.policy
    folder.mkdir(parents=True, exist_ok=False)
    original_session = ort.InferenceSession
    def create(path, sess_options, providers):
        layout = LAYOUT[args.policy]
        options = sess_options
        if layout:
            path = args.output / layout / (Path(path).stem + (".ort" if layout == "ort" else ".onnx"))
        if layout in ("ort", "inline", "external", "prepacked"):
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if args.policy == "no-arena":
            options.enable_cpu_mem_arena = False
        elif args.policy == "no-pattern":
            options.enable_mem_pattern = False
        elif args.policy == "ort-mmap":
            options.add_session_config_entry("session.use_memory_mapped_ort_model", "1")
            options.add_session_config_entry("session.use_ort_model_bytes_for_initializers", "1")
        return original_session(str(path), sess_options=options, providers=providers)

    ort.InferenceSession = create
    processor = ChineseProcessor(args.resources)
    report = {"policy": args.policy, "ort": ort.__version__, "texts": TEXTS, "before": memory(), "calls": []}
    arrays = {}
    probabilities = []
    predict = G2PWSession.run
    def capture(self, inputs):
        output = predict(self, inputs)
        if len(report["calls"]) < len(TEXTS):
            probabilities.append(output.copy())
        return output
    G2PWSession.run = capture
    for cycle in range(4):
        for i, text in enumerate(TEXTS):
            start = time.perf_counter()
            if args.policy == "sequential" and processor.bert is not None:
                processor.bert = None
            if args.policy == "sequential" and processor.phones is not None:
                g2pw = processor.phones.g2pw
                g2pw.session = G2PWSession(args.resources / "g2pw.onnx", g2pw.inputs.labels)
            phones, word2ph, normalized = processor.clean(text)
            if args.policy == "sequential":
                processor.phones.g2pw.close()
            g2p_end = time.perf_counter()
            after_clean = memory()
            features = processor.features(phones, word2ph, normalized)
            end = time.perf_counter()
            row = {"cycle": cycle, "case": i, "clean_ms": (g2p_end-start)*1000,
                   "feature_ms": (end-g2p_end)*1000, "total_ms": (end-start)*1000,
                   "after_clean": after_clean, "after_features": memory()}
            if cycle == 0:
                row.update(phones=phones, word2ph=word2ph, normalized=normalized)
                arrays[f"features_{i}"] = features.copy()
            report["calls"].append(row)
    report["hot_medians_ms"] = [median(r["total_ms"] for r in report["calls"] if r["cycle"] and r["case"] == i)
                                for i in range(len(TEXTS))]
    arrays.update({f"probabilities_{i}": a for i, a in enumerate(probabilities)})
    report["active"] = memory()
    processor.close()
    gc.collect()
    report["closed"] = memory()
    np.savez(folder / "outputs.npz", **arrays)
    (folder / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.policy, {k: round(v/2**20,1) for k,v in report["active"].items()}, report["hot_medians_ms"], flush=True)


def summarize(args):
    import numpy as np
    import onnxruntime as ort
    import winreg
    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    baseline = json.loads((args.output / "baseline/result.json").read_text("utf-8"))
    gold = np.load(args.output / "baseline/outputs.npz")
    candidates = {}
    for policy, layout in LAYOUT.items():
        folder = args.output / policy
        result = json.loads((folder / "result.json").read_text("utf-8"))
        arrays = np.load(folder / "outputs.npz")
        text_equal = all(r[k] == b[k] for r,b in zip(result["calls"][:len(TEXTS)], baseline["calls"])
                         for k in ("phones", "word2ph", "normalized"))
        exact = set(arrays.files) == set(gold.files) and all(np.array_equal(gold[k], arrays[k]) for k in gold.files)
        cold = result["calls"][0]
        hot = [r for r in result["calls"] if r["cycle"] > 0]
        models = args.resources if layout is None else args.output / layout
        files = [p for p in models.iterdir() if p.stem in ("g2pw", "bert") and p.suffix in (".bin", ".onnx", ".ort")]
        def mib(snapshot):
            return {k: round(v/2**20, 2) for k,v in snapshot.items()}
        candidates[policy] = {"two_models_bytes": sum(p.stat().st_size for p in files),
            "distribution": "original ONNX graph" if layout in (None, "portable", "aligned") else
                            "host-optimized ORT graph; CPU and runtime portability unverified",
            "before_mib": mib(result["before"]), "after_first_clean_mib": mib(cold["after_clean"]),
            "after_first_features_mib": mib(cold["after_features"]), "after_hot_requests_mib": mib(result["active"]),
            "after_close_mib": mib(result["closed"]), "first_request_ms": round(cold["total_ms"], 2),
            "hot_median_ms_by_text": [round(v, 2) for v in result["hot_medians_ms"]],
            "hot_calls_per_second": round(len(hot) * 1000 / sum(r["total_ms"] for r in hot), 3),
            "same_text_phones_word2ph": text_equal, "bit_exact_arrays": exact, "arrays_compared": len(gold.files)}
    summary = {"scope": "Mandarin processor in original unpacked Windows Python process, CPUExecutionProvider, 2 intra-op threads. Fresh process per candidate. First pass over each text saves final BERT features and actual G2PW probabilities, followed by three hot passes. Timings are exploratory under concurrent system load.",
        "units": "Windows GetProcessMemoryInfo in MiB = 1048576 bytes. private is PrivateUsage (private commit); wset is current working set; peak_wset and peak_pagefile are OS lifetime maxima at the stated boundary. File-backed working-set pages may be shared/reclaimed.",
        "os": platform.platform(), "cpu": cpu, "ort": ort.__version__, "numpy": np.__version__, "texts": list(TEXTS),
        "input_sha256": {g: sha(args.resources / (g + ".onnx")) for g in ("g2pw", "bert")},
        "layout": {"portable-external": "Original ONNX operators, FP32 parameters and runtime optimization. ONNX external_data size threshold 16384 bytes; one weights file per graph; contiguous offsets.",
                   "aligned-external": "Same as portable-external with each external tensor at a 64 KiB offset.",
                   "externalized-prepacked": "ORT_ENABLE_ALL saved graph and externally serialized prepacked tensors. Derived on this CPU. This layout also stores the source weights."},
        "candidates": candidates}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def screen(args):
    if args.bundle is None:
        raise ValueError("screen requires --bundle")
    args.output.mkdir(parents=True, exist_ok=False)
    for policy in ("prepare-portable", "align", "prepare", *LAYOUT, "summarize"):
        role = "preparation" if policy in ("prepare-portable", "align") else "main"
        python = args.bundle / "runtime" / role / "python.exe"
        command = [str(python.resolve()), "-I", "-B", "-X", "utf8", str(Path(__file__).resolve()), policy,
                   "--resources", str(args.resources), "--output", str(args.output)]
        with (args.output / (policy + ".log")).open("wb") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(policy, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=("screen", "summarize", "prepare", "prepare-portable", "align", *LAYOUT))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.resources, args.output = args.resources.resolve(), args.output.resolve()
    if args.policy == "screen":
        screen(args)
    elif args.policy == "summarize":
        summarize(args)
    elif args.policy == "prepare":
        prepare_optimized(args)
    elif args.policy == "prepare-portable":
        prepare_portable(args)
    elif args.policy == "align":
        align(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
