#!/usr/bin/env python3
"""Evaluate one-entry caching of five constant V2Pro reference projections.

The cache is experimental and confined to this Harness. All original weights
are retained. A is the official bundle reference; B changes ge/ge512 while
keeping A's semantic history, phones and noise, so B is an acoustic switching
probe, not a validated complete TTS request. Diagnostic workers precede a
single-model, alternating-order paired timing worker.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import weakref

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
CASES = ("ja-reported-intro", "ja-short", "ja-long", "ja-punctuation")
STAGES = ("quantized", "ssl_encoded", "text_encoded", "mrte", "encoder_hidden", "mean",
          "log_scale", "mask", "flow_input", "flow_output", "decoder_input", "waveform")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def spec(value):
    value = np.asarray(value)
    return dict(shape=list(value.shape), dtype=value.dtype.str, bytes=value.nbytes,
                sha256_raw_c_order=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def exact(actual, expected):
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and actual.tobytes(order="C") == expected.tobytes(order="C"))


def comparison(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(within_tolerance=False, shape_dtype_match=False)
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(within_tolerance=bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-5)),
                shape_dtype_match=True, max_abs=float(difference.max()),
                mean_abs=float(difference.mean()), atol=1e-4, rtol=1e-5)


def make_model(package, manifest_hash, softmax):
    """Return an instrumented model with unchanged weights and graph order."""
    import mlx.core as mx
    from sakuratts.backends.mlx.sovits import MLXSoVITS
    from sakuratts.backends.mlx.encoder import MLXSoVITSEncoder
    from sakuratts.backends.mlx.flow import MLXSoVITSFlow
    from sakuratts.backends.mlx.decoder import MLXSoVITSDecoder
    from sakuratts.backends.mlx.sovits_package import SoVITSPackage

    class Flow(MLXSoVITSFlow):
        active_projections = None

        def conv(self, x, prefix):
            if self.active_projections is not None and prefix in self.active_projections:
                return self.active_projections[prefix]
            return super().conv(x, prefix)

    class Decoder(MLXSoVITSDecoder):
        active_projections = None

        def conv(self, x, prefix):
            if self.active_projections is not None and prefix in self.active_projections:
                return self.active_projections[prefix]
            return super().conv(x, prefix)

    class Model(MLXSoVITS):
        def __init__(self, *args):
            super().__init__(*args)
            self.cache_key = None
            self.projections = {}
            self.last_event = None
            self.hits = self.misses = 0
            self.binding = dict(manifest_sha256=manifest_hash, model_id=id(self),
                flow_id=id(self.flow), decoder_id=id(self.decoder),
                flow_weights_id=id(self.flow.weights), decoder_weights_id=id(self.decoder.weights),
                device=str(self.device), fold_weight_norm=False)

        def prepare_projections(self, ge):
            started = time.perf_counter()
            value = np.ascontiguousarray(np.asarray(ge))
            if value.dtype != np.float32 or value.shape != (1, self.flow.gin_channels, 1):
                raise ValueError("Expected the original FP32 ge shape")
            content = spec(value)
            binding = dict(self.binding, model_id=id(self), flow_id=id(self.flow), decoder_id=id(self.decoder),
                flow_weights_id=id(self.flow.weights), decoder_weights_id=id(self.decoder.weights))
            key = (tuple(sorted(binding.items())), value.dtype.str, value.shape, content["sha256_raw_c_order"])
            key_seconds = time.perf_counter() - started
            hit = key == self.cache_key
            rebuild_seconds = 0.0
            if hit:
                self.hits += 1
            else:
                self.misses += 1
                # Drop the one old entry before allocating the replacement.
                self.cache_key = None
                self.projections.clear()
                started = time.perf_counter()
                with mx.stream(self.device):
                    incoming = mx.array(value.transpose(0, 2, 1))
                    self.projections = {f"flow.flows.{index}.enc.cond_layer":
                        MLXSoVITSFlow.conv(self.flow, incoming, f"flow.flows.{index}.enc.cond_layer")
                        for index in self.flow.couplings}
                    self.projections["dec.cond"] = MLXSoVITSDecoder.conv(self.decoder, incoming, "dec.cond")
                    mx.eval(*self.projections.values())
                rebuild_seconds = time.perf_counter() - started
                self.cache_key = key
            self.last_event = dict(hit=hit, ge=content, binding=binding, key_seconds=key_seconds,
                rebuild_seconds=rebuild_seconds, entries=int(self.cache_key is not None),
                projection_tensors=len(self.projections),
                projection_bytes=sum(value.nbytes for value in self.projections.values()))

        def decode(self, codes, phones, ge, ge512, noise, *, policy, **kwargs):
            if policy not in ("baseline", "cached"):
                raise ValueError("Unknown projection policy")
            self.last_event = None
            if policy == "cached":
                self.prepare_projections(ge)
                self.flow.active_projections = self.decoder.active_projections = self.projections
            try:
                return super().decode(codes, phones, ge, ge512, noise, **kwargs)
            finally:
                self.flow.active_projections = self.decoder.active_projections = None

    with SoVITSPackage.open(package) as source:
        with mx.stream(mx.cpu):
            encoder = MLXSoVITSEncoder.from_package(source, softmax=softmax)
        with mx.stream(mx.gpu):
            flow = Flow.from_package(source, fold_weight_norm=False)
            decoder = Decoder.from_package(source, fold_weight_norm=False)
    return Model(encoder, flow, decoder, mx.gpu, mx.cpu)


def inventory(model):
    return {name: dict(count=len(component.weights), bytes=sum(value.nbytes for value in component.weights.values()),
                      tensor_ids={key: id(value) for key, value in component.weights.items()})
            for name, component in (("encoder", model.encoder), ("flow", model.flow), ("decoder", model.decoder))}


def load_inputs(prepared):
    from portable_validation import load_bundle
    from sakuratts._internal.reference_condition import PreparedReference

    bundle = load_bundle(Path(prepared["bundle"]), sovits_package=Path(prepared["package"]))
    if bundle["manifest_sha256"] != prepared["bundle_manifest_sha256"]:
        raise ValueError("Bundle manifest changed")
    manifest = bundle["manifest"]
    if tuple(manifest["cases"]) != CASES:
        raise ValueError("Expected the four unchanged Japanese cases")
    other = PreparedReference.load(Path(prepared["reference_b"]),
        gpt_checkpoint_sha256=manifest["external_models"]["gpt"]["checkpoint_sha256"],
        sovits_checkpoint_sha256=manifest["external_models"]["sovits"]["checkpoint_sha256"],
        reference_language="ja", official_commit=manifest["official_commit"],
        manifest_sha256=prepared["reference_b_manifest_sha256"])
    first = bundle["arrays"][CASES[0]]
    if not all(exact(bundle["arrays"][name][key], first[key]) for name in CASES for key in ("ge", "ge512")):
        raise ValueError("A conditions differ across the four cases")
    if exact(first["ge"], other.ge) or exact(first["ge512"], other.ge512):
        raise ValueError("B must change both prepared acoustic conditions")
    return bundle, dict(A=(first["ge"], first["ge512"]), B=(other.ge, other.ge512)), other.manifest


def call(model, data, conditions, parameters, policy, capture):
    return model.decode(data["semantic"], data["acoustic_phones"], *conditions, data["noise"],
        noise_scale=parameters["noise_scale"], speed=parameters["speed"], capture=capture, policy=policy)


def memory(mx):
    mx.synchronize()
    return dict(mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
        mlx_allocator_peak_bytes=mx.get_peak_memory(),
        rss_at_boundary_bytes=int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024,
        process_lifetime_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def diagnostic(model, bundle, references, policy, output, report, mx):
    from sakuratts._internal.synthesis import single_fragment_pcm

    for name in CASES:
        data = bundle["arrays"][name]
        parameters = bundle["manifest"]["cases"][name]["parameters"]
        first = {}
        row = dict(parameters=parameters, calls=[], captures={})
        report["cases"][name] = row
        for index, reference in enumerate(("A", "A", "B", "B", "A", "A")):
            mx.reset_peak_memory()
            waveform, stages = call(model, data, references[reference], parameters, policy, True)
            if set(stages) != set(STAGES):
                raise ValueError("Expected all twelve actual computed acoustic stages")
            arrays = {key: np.ascontiguousarray(np.asarray(stages[key]).copy()) for key in STAGES}
            arrays["pcm"] = single_fragment_pcm(arrays["waveform"], parameters["sample_rate"], parameters["fragment_interval"])
            del waveform, stages
            measurement = memory(mx)
            record = dict(index=index, reference=reference, cache=model.last_event, memory=measurement,
                          arrays={key: spec(value) for key, value in arrays.items()})
            if reference not in first:
                first[reference] = arrays
                relative = f"{name}-{reference}.npz"
                with (output / relative).open("xb") as stream:
                    np.savez(stream, **arrays)
                row["captures"][reference] = dict(file=relative, sha256=sha256(output / relative))
                if reference == "A":
                    row["official_comparisons"] = {key: comparison(arrays[key], data["acoustic." + key]) for key in STAGES}
            record["repeat_bit_exact"] = {key: exact(value, first[reference][key]) for key, value in arrays.items()}
            row["calls"].append(record)
            del arrays
            write_json(output / "result.json", report)
        print(json.dumps(dict(case=name, policy=policy, status="diagnostic_completed")), flush=True)
    report["all_repeats_bit_exact"] = all(all(call["repeat_bit_exact"].values()) for row in report["cases"].values() for call in row["calls"])
    report["official_all_within_tolerance"] = all(check["within_tolerance"] for row in report["cases"].values() for check in row["official_comparisons"].values())


def paired(model, bundle, references, output, report, mx, warmup, repeat):
    from sakuratts._internal.synthesis import single_fragment_pcm

    for name in CASES:
        data = bundle["arrays"][name]
        parameters = bundle["manifest"]["cases"][name]["parameters"]
        gold = {}
        for reference in ("A", "B"):
            with np.load(output.parent / "diagnostic-baseline" / f"{name}-{reference}.npz", allow_pickle=False) as archive:
                gold[reference] = {key: archive[key] for key in ("waveform", "pcm")}
        rows = []
        report["cases"][name] = dict(parameters=parameters, pairs=rows)
        schedule = [("warmup", "A", index) for index in range(warmup)]
        schedule += [("steady", "A", index) for index in range(repeat)]
        schedule += [("switch_" + position, reference, index) for index in range(repeat)
                     for position, reference in (("A", "A"), ("B", "B"), ("return_A", "A"))]
        for pair_index, (phase, reference, cycle) in enumerate(schedule):
            order = ("baseline", "cached") if pair_index % 2 == 0 else ("cached", "baseline")
            pair = dict(phase=phase, reference=reference, cycle=cycle, order=list(order), runs={})
            outputs = {}
            for policy in order:
                mx.synchronize()
                start = time.perf_counter()
                waveform = call(model, data, references[reference], parameters, policy, False)
                mx.synchronize()
                seconds = time.perf_counter() - start
                # Output hashes, host copies, PCM and comparisons are outside timing.
                audio = np.ascontiguousarray(np.asarray(waveform).copy())
                del waveform
                pcm = single_fragment_pcm(audio, parameters["sample_rate"], parameters["fragment_interval"])
                pair["runs"][policy] = dict(seconds=seconds, cache=model.last_event,
                    waveform=spec(audio), pcm=spec(pcm),
                    diagnostic_waveform_bit_exact=exact(audio, gold[reference]["waveform"]),
                    diagnostic_pcm_bit_exact=exact(pcm, gold[reference]["pcm"]))
                outputs[policy] = (audio, pcm)
            pair["waveform_bit_exact"] = exact(outputs["baseline"][0], outputs["cached"][0])
            pair["pcm_bit_exact"] = exact(outputs["baseline"][1], outputs["cached"][1])
            pair["saved_seconds"] = pair["runs"]["baseline"]["seconds"] - pair["runs"]["cached"]["seconds"]
            pair["saved_fraction"] = pair["saved_seconds"] / pair["runs"]["baseline"]["seconds"]
            rows.append(pair)
            del outputs, audio, pcm
        summaries = {}
        for phase in ("steady", "switch_A", "switch_B", "switch_return_A"):
            selected = [row for row in rows if row["phase"] == phase]
            summaries[phase] = dict(pairs=len(selected),
                baseline_median_seconds=float(np.median([row["runs"]["baseline"]["seconds"] for row in selected])),
                cached_median_seconds=float(np.median([row["runs"]["cached"]["seconds"] for row in selected])),
                median_paired_saved_seconds=float(np.median([row["saved_seconds"] for row in selected])),
                median_paired_saved_fraction=float(np.median([row["saved_fraction"] for row in selected])),
                faster_pairs=sum(row["saved_seconds"] > 0 for row in selected))
        report["cases"][name]["summaries"] = summaries
        write_json(output / "result.json", report)
        print(json.dumps(dict(case=name, status="paired_completed", summaries=summaries)), flush=True)
    report["all_outputs_bit_exact"] = all(row["waveform_bit_exact"] and row["pcm_bit_exact"]
        and all(run["diagnostic_waveform_bit_exact"] and run["diagnostic_pcm_bit_exact"] for run in row["runs"].values())
        for case in report["cases"].values() for row in case["pairs"])


def worker(args):
    run, output = args.run.resolve(), args.run.resolve() / args.worker
    output.mkdir(exist_ok=False)
    prepared = read_json(run / "prepared.json")
    report = dict(status="running", worker=args.worker, cases={},
        command=[sys.executable, *sys.argv], quality=dict(asr="not_run", human_listening="not_run"),
        memory_scope="MLX Apple unified-memory allocator counters; boundary RSS is not a peak. Process maxrss covers all worker allocations, including NumPy diagnostic inputs/outputs.",
        timing_scope="Paired normal worker uses capture=False; policy selection, ge hash, miss rebuilding and completion synchronization are inside each timer. Host copies, PCM and comparisons are outside. Baseline retains the inactive 26 KiB candidate entry. No per-call memory reads in normal worker.",
        scope="Fixed semantics/phones/noise for four Japanese A cases. B ge/ge512 switching is an acoustic-only probe; B has no official twelve-stage oracle.")
    model = mx = None
    model_ref = None
    stage = "source_validation"
    try:
        for name, expected in prepared["source_sha256"].items():
            if sha256(run / "source" / name) != expected:
                raise ValueError("Frozen source changed: " + name)
        stage = "input_validation"
        bundle, references, reference_b = load_inputs(prepared)
        report.update(bundle_manifest_sha256=bundle["manifest_sha256"],
            model_identity=bundle["manifest"]["external_models"]["sovits"],
            reference_b_identity=reference_b["identity"],
            conditions={name: dict(ge=spec(value[0]), ge512=spec(value[1])) for name, value in references.items()},
            official_commit=bundle["manifest"]["official_commit"],
            dependencies={name: metadata.version(name) for name in ("numpy", "mlx", "mlx-metal")},
            platform=platform.platform(), python=sys.version)
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        mx.reset_peak_memory()
        report["before_load"] = memory(mx)
        stage = "model_load"
        started = time.perf_counter()
        model = make_model(Path(prepared["package"]), report["model_identity"]["manifest_sha256"], prepared["encoder_softmax"])
        mx.synchronize()
        report["load_seconds"] = time.perf_counter() - started
        model_ref = weakref.ref(model)
        report["after_load"] = memory(mx)
        weights = inventory(model)
        report["weights"] = {key: {name: value for name, value in item.items() if name != "tensor_ids"} for key, item in weights.items()}
        stage = args.worker
        if args.worker.startswith("diagnostic-"):
            diagnostic(model, bundle, references, args.worker.removeprefix("diagnostic-"), output, report, mx)
        else:
            paired(model, bundle, references, output, report, mx, prepared["warmup"], prepared["repeat"])
        report["weights_objects_unchanged"] = weights == inventory(model)
        report["cache_totals"] = dict(hits=model.hits, misses=model.misses,
            final_entries=int(model.cache_key is not None), final_projection_bytes=sum(v.nbytes for v in model.projections.values()))
        report["status"] = "completed"
    except Exception as error:
        report.update(status="error", error=dict(stage=stage, type=type(error).__name__,
            module=type(error).__module__, message=str(error), traceback=traceback.format_exc()))
    finally:
        # Store only error strings above; cleanup runs outside exception frames.
        model = None
        gc.collect()
        if mx is not None:
            mx.synchronize()
            mx.clear_cache()
            mx.synchronize()
            report["after_unload"] = memory(mx)
        report["model_destroyed"] = model_ref is None or model_ref() is None
        write_json(output / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def compare_diagnostics(run):
    baseline, candidate = (read_json(run / name / "result.json") for name in ("diagnostic-baseline", "diagnostic-cached"))
    report = dict(all_bit_exact=True, cases={})
    if any(row["status"] != "completed" or not row["all_repeats_bit_exact"] or not row["weights_objects_unchanged"]
           or not row["model_destroyed"] or row["after_unload"]["mlx_active_bytes"] or row["after_unload"]["mlx_cache_bytes"]
           for row in (baseline, candidate)):
        raise ValueError("Diagnostic execution, repetitions, weight ownership or unload failed")
    for name in CASES:
        report["cases"][name] = {}
        for reference in ("A", "B"):
            captures = []
            for policy, result in (("baseline", baseline), ("cached", candidate)):
                info = result["cases"][name]["captures"][reference]
                path = run / ("diagnostic-" + policy) / info["file"]
                if sha256(path) != info["sha256"]:
                    raise ValueError("Diagnostic capture changed")
                with np.load(path, allow_pickle=False) as archive:
                    captures.append({key: archive[key] for key in (*STAGES, "pcm")})
            checks = {key: exact(captures[0][key], captures[1][key]) for key in (*STAGES, "pcm")}
            report["cases"][name][reference] = checks
            report["all_bit_exact"] &= all(checks.values())
        hits = [row["cache"]["hit"] for row in candidate["cases"][name]["calls"]]
        expected = [name != CASES[0], True, False, True, False, True]
        if hits != expected:
            raise ValueError("Unexpected one-entry A/A/B/B/A/A cache transitions")
        if any(row["cache"]["entries"] != 1 or row["cache"]["projection_tensors"] != 5
               or row["cache"]["projection_bytes"] != 26624 for row in candidate["cases"][name]["calls"]):
            raise ValueError("Projection cache is not the expected bounded five-tensor entry")
    report["official_failures"] = {name: [stage for stage, check in row["official_comparisons"].items() if not check["within_tolerance"]]
                                   for name, row in baseline["cases"].items()}
    write_json(run / "diagnostic-comparison.json", report)
    return report["all_bit_exact"]


def run(args):
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    files = ["research/tools/sovits_reference_projection.py", "research/tools/portable_validation.py",
             *[f"src/sakuratts/{name}.py" for name in ("backends/mlx/sovits", "backends/mlx/encoder",
               "backends/mlx/flow", "backends/mlx/decoder", "backends/mlx/sovits_package", "_internal/weight_storage", "_internal/synthesis",
               "_internal/generation", "_internal/sampling", "_internal/reference_condition")]]
    for name in files:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, target)
    prepared = dict(command=[sys.executable, *sys.argv], created_at_utc=datetime.now(timezone.utc).isoformat(),
        bundle=str(args.bundle.resolve()), bundle_manifest_sha256=sha256(args.bundle / "manifest.json"),
        package=str(args.package.resolve()), reference_b=str(args.reference_b.resolve()),
        reference_b_manifest_sha256=sha256(args.reference_b / "manifest.json"),
        encoder_softmax=args.encoder_softmax, warmup=args.warmup, repeat=args.repeat,
        source_sha256={name: sha256(run / "source" / name) for name in files})
    write_json(run / "prepared.json", prepared)
    processes = []
    for worker_name in ("diagnostic-baseline", "diagnostic-cached", "paired"):
        if worker_name == "paired" and not compare_diagnostics(run):
            raise ValueError("Candidate changed diagnostic outputs; normal timing is forbidden")
        command = [sys.executable, str(run / "source/research/tools/sovits_reference_projection.py"),
                   "worker", "--run", str(run), "--worker", worker_name]
        started = time.perf_counter()
        with (run / (worker_name + ".stdout.log")).open("x", encoding="utf-8") as stdout, (run / (worker_name + ".stderr.log")).open("x", encoding="utf-8") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=run)
        processes.append(dict(worker=worker_name, command=command, exit_code=process.returncode,
                              wall_seconds=time.perf_counter() - started))
        write_json(run / "processes.json", processes)
        print(json.dumps(processes[-1]), flush=True)
        if process.returncode:
            return process.returncode
    paired_result = read_json(run / "paired/result.json")
    passed = (paired_result["all_outputs_bit_exact"] and paired_result["weights_objects_unchanged"]
              and paired_result["model_destroyed"] and paired_result["after_unload"]["mlx_active_bytes"] == 0
              and paired_result["after_unload"]["mlx_cache_bytes"] == 0)
    write_json(run / "result.json", dict(status="completed", candidate_equivalence_passed=passed,
        diagnostic_comparison_sha256=sha256(run / "diagnostic-comparison.json"),
        worker_result_sha256={name: sha256(run / name / "result.json") for name in ("diagnostic-baseline", "diagnostic-cached", "paired")}))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dispatch = sub.add_parser("run")
    for flag in ("bundle", "package", "reference-b", "output"):
        dispatch.add_argument("--" + flag, type=Path, required=True)
    dispatch.add_argument("--encoder-softmax", choices=("fp32", "fp64-accumulation"), default="fp32")
    dispatch.add_argument("--warmup", type=int, default=5)
    dispatch.add_argument("--repeat", type=int, default=7)
    child = sub.add_parser("worker")
    child.add_argument("--run", type=Path, required=True)
    child.add_argument("--worker", choices=("diagnostic-baseline", "diagnostic-cached", "paired"), required=True)
    args = parser.parse_args()
    if args.command == "run" and (args.warmup < 5 or args.repeat < 7):
        parser.error("Require five warmup pairs and at least seven measured pairs per phase and shape")
    return run(args) if args.command == "run" else worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
