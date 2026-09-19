#!/usr/bin/env python3
"""Validate the product reference-bound loader against complete-weight models.

This is a diagnostic acoustic test: supplied semantic histories, phones and
noise remain fixed. B is a ge/ge512 switching probe, not a complete B TTS
request. It records actual package reads, captures, rejection, cancellation and
partial-load recovery; it does not report normal inference speed.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import weakref

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sovits_reference_projection import (CASES, STAGES, comparison, exact, load_inputs,
    memory, read_json, sha256, spec, write_json)

VARIANTS = {
    "default": dict(encoder_softmax="fp32", fold_weight_norm=False, cases=list(CASES)),
    "folded": dict(encoder_softmax="fp32", fold_weight_norm=True, cases=["ja-short", "ja-punctuation"]),
    "fp64": dict(encoder_softmax="fp64-accumulation", fold_weight_norm=False, cases=["ja-punctuation"]),
    "fp64-folded": dict(encoder_softmax="fp64-accumulation", fold_weight_norm=True, cases=["ja-punctuation"]),
}
PREFIXES = tuple([f"flow.flows.{index}.enc.cond_layer" for index in (0, 2, 4, 6)] + ["dec.cond"])


def error_record(error):
    return dict(type=type(error).__name__, module=type(error).__module__, message=str(error), traceback=traceback.format_exc())


@contextmanager
def observe_reads():
    """Observe real read_fp32 calls, including tensors excluded before reading."""
    import sakuratts.sovits_package as package_module

    original_tensors = package_module.SoVITSPackage.tensors
    original_read = package_module.read_fp32
    report = dict(calls=[], reads=[])
    active = [None]

    def read(archive, manifest, name):
        report["reads"].append(dict(name=name, tensor_call=active[0]))
        return original_read(archive, manifest, name)

    def tensors(self, *prefixes, names=(), exclude=()):
        index = len(report["calls"])
        report["calls"].append(dict(prefixes=list(prefixes), names=sorted(names), exclude=sorted(exclude), yielded=[]))
        previous = active[0]
        active[0] = index
        try:
            for name, value in original_tensors(self, *prefixes, names=names, exclude=exclude):
                report["calls"][index]["yielded"].append(dict(name=name, bytes=value.nbytes))
                yield name, value
        finally:
            active[0] = previous

    package_module.read_fp32 = read
    package_module.SoVITSPackage.tensors = tensors
    try:
        yield report
    finally:
        package_module.read_fp32 = original_read
        package_module.SoVITSPackage.tensors = original_tensors


@contextmanager
def observe_encoder():
    from sakuratts.mlx_sovits_encoder import MLXSoVITSEncoder

    original = MLXSoVITSEncoder.encode
    counter = [0]

    def encode(self, *args, **kwargs):
        counter[0] += 1
        return original(self, *args, **kwargs)

    MLXSoVITSEncoder.encode = encode
    try:
        yield counter
    finally:
        MLXSoVITSEncoder.encode = original


def clear_model(holder, mx):
    mx.synchronize()
    reference = weakref.ref(holder[0]) if holder[0] is not None else None
    holder[0] = None
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    result = dict(destroyed=reference is None or reference() is None, memory=memory(mx))
    if not result["destroyed"] or result["memory"]["mlx_active_bytes"] or result["memory"]["mlx_cache_bytes"]:
        raise AssertionError("Acoustic model or request state remains after unload")
    return result


def verify_reads(trace, manifest, bound, model):
    names = set(manifest["tensor_sources"])
    excluded = {name for name in names if any(name.startswith(prefix + ".") for prefix in PREFIXES)}
    actual = [row["name"] for row in trace["reads"]]
    if len(actual) != len(names) or set(actual) != names:
        raise AssertionError("Every package tensor must be read exactly once per successful load")
    for index, call in enumerate(trace["calls"]):
        observed = [row["name"] for row in trace["reads"] if row["tensor_call"] == index]
        if observed != [row["name"] for row in call["yielded"]] or set(observed) & set(call["exclude"]):
            raise AssertionError("Excluded tensors reached read_fp32 or read/yield order differs")
    resident = set(model.encoder.weights) | set(model.flow.weights) | set(model.decoder.weights)
    if bound:
        preparation = trace["calls"][0]
        if preparation["prefixes"] or set(preparation["names"]) != excluded or len(excluded) != 14:
            raise AssertionError("Bound preparation did not select exactly fourteen condition weights")
        if sum(row["bytes"] for row in preparation["yielded"]) != 27314176:
            raise AssertionError("Condition FP32 byte inventory changed")
        for call in trace["calls"][1:]:
            if set(row["name"] for row in call["yielded"]) & excluded:
                raise AssertionError("Bound component reread excluded weights")
        if resident & excluded or any(prefix + ".weight" in resident for prefix in PREFIXES):
            raise AssertionError("Bound component retained original or folded condition weights")
        projections = dict(model.flow._reference_projections, **model.decoder._reference_projections)
        if set(projections) != set(PREFIXES) or sum(value.nbytes for value in projections.values()) != 26624:
            raise AssertionError("Unexpected resident projection inventory")
    elif trace["calls"][0]["prefixes"] != ["enc_p."] or model.bound_reference is not None:
        raise AssertionError("Default load unexpectedly prepared/bound reference conditions")
    return dict(bound=bound, read_tensors=len(actual), condition_tensors=len(excluded),
                original_condition_weights_absent=not bool(resident & excluded) if bound else None)


def load_model(holder, package, variant, reference, mx):
    from sakuratts.mlx_sovits import MLXSoVITS

    released = clear_model(holder, mx)
    with observe_reads() as trace:
        holder[0] = MLXSoVITS.load(package, encoder_device="cpu", encoder_softmax=variant["encoder_softmax"],
            fold_weight_norm=variant["fold_weight_norm"], reference=reference)
    checked = verify_reads(trace, holder[0].encoder.manifest, reference is not None, holder[0])
    return dict(previous_released=released, reads=trace, checks=checked, product_loaded_boundary=memory(mx))


def decode(model, data, reference, parameters, *, capture):
    return model.decode(data["semantic"], data["acoustic_phones"], reference.ge, reference.ge512, data["noise"],
                        noise_scale=parameters["noise_scale"], speed=parameters["speed"], capture=capture)


def capture(model, data, reference, parameters):
    from sakuratts.synthesis import single_fragment_pcm

    waveform, stages = decode(model, data, reference, parameters, capture=True)
    if set(stages) != set(STAGES):
        raise AssertionError("Missing one of the twelve actual acoustic capture stages")
    arrays = {name: np.ascontiguousarray(np.asarray(stages[name]).copy()) for name in STAGES}
    arrays["pcm"] = single_fragment_pcm(arrays["waveform"], parameters["sample_rate"], parameters["fragment_interval"])
    return arrays


def save_arrays(output, stem, arrays):
    path = output / (stem + ".npz")
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    return dict(file=path.name, sha256=sha256(path), arrays={key: spec(value) for key, value in arrays.items()})


def request_from_fixture(case, data, reference):
    """Acoustic cancellation input; no GPT inference is claimed in this Harness."""
    from sakuratts.generation import SemanticGeneration
    from sakuratts.sampling import StopResult
    from sakuratts.synthesis import PreparedSemantic, PreparedText

    generation = SemanticGeneration(data["tokens"].copy(), StopResult(data["history"].copy(), True,
        tuple(case["stop"]["reasons"]), case["stop"]["returned_index"]))
    if not exact(generation.semantic, data["semantic"]):
        raise AssertionError("Cancellation request reconstructed a different semantic suffix")
    text = PreparedText(case["text"], "ja", dict(phones=data["acoustic_phones"][0].tolist(), norm_text=case["normalized_text"]), 0.0)
    return PreparedSemantic(text, reference, deepcopy(reference.manifest["identity"]), data["acoustic_phones"][0].copy(),
        generation, np.random.default_rng(1234), dict(condition_seconds=0.0, semantic_seconds=0.0))


def rejection_and_cancellation(model, data, case, reference, other, expected):
    from sakuratts.generation import SynthesisCancelled
    from sakuratts.synthesis import synthesize_acoustic

    report = dict(rejections=[], cancellations=[])
    parameters = case["parameters"]
    request = request_from_fixture(case, data, reference)
    wrong_identity = deepcopy(reference.manifest)
    wrong_identity["identity"]["audio_sha256"] = "0" * 64
    wrong_reference = replace(reference, manifest=wrong_identity)
    with observe_encoder() as calls:
        for mode in ("reference", "ge", "ge512"):
            before = calls[0]
            failure = None
            try:
                if mode == "reference":
                    wrong_request = replace(request, reference=wrong_reference,
                                            reference_identity=deepcopy(wrong_identity["identity"]))
                    unexpected = synthesize_acoustic(wrong_request, sovits=model, acoustic_noise=data["noise"])
                else:
                    ge, ge512 = (other.ge, reference.ge512) if mode == "ge" else (reference.ge, other.ge512)
                    unexpected = model.decode(data["semantic"], data["acoustic_phones"], ge, ge512, data["noise"])
                del unexpected
            except ValueError as error:
                failure = error_record(error)
            if failure is None or calls[0] != before:
                raise AssertionError("Reference mismatch reached encoder execution")
            report["rejections"].append(dict(field=mode, before_encoder=True, error=failure))
            # A correct request must still work on this same instance.
            audio = decode(model, data, reference, parameters, capture=False)
            same = exact(np.asarray(audio), expected["waveform"])
            del audio
            if not same:
                raise AssertionError("Mismatch changed the original bound model")
        for target in ("before_acoustic", "after_acoustic"):
            polls = [0]

            def cancel():
                polls[0] += 1
                return target == "before_acoustic" or polls[0] > 1

            before = calls[0]
            failure = None
            try:
                unexpected = synthesize_acoustic(request, sovits=model, acoustic_noise=data["noise"], cancel_requested=cancel)
                del unexpected
            except SynthesisCancelled as error:
                failure = dict(stage=error.stage, **error_record(error))
            expected_calls = 0 if target == "before_acoustic" else 1
            if failure is None or failure["stage"] != target or calls[0] - before != expected_calls:
                raise AssertionError("Cancellation occurred at the wrong acoustic boundary")
            identity = id(model)
            speech = synthesize_acoustic(request, sovits=model, acoustic_noise=data["noise"],
                noise_scale=parameters["noise_scale"], fragment_interval=parameters["fragment_interval"])
            restored = exact(speech.waveform, expected["waveform"]) and exact(speech.pcm, expected["pcm"])
            del speech
            if not restored or id(model) != identity:
                raise AssertionError("Cancellation recovery did not use the same unchanged instance")
            report["cancellations"].append(dict(target=target, error=failure, same_instance_retry_bit_exact=restored))
    return report


def partial_load_failure(holder, package, reference, variant, mx):
    """Inject after the real decoder allocation, before MLXSoVITS publishes it."""
    from sakuratts.mlx_sovits import MLXSoVITS
    from sakuratts.mlx_sovits_decoder import MLXSoVITSDecoder

    released = clear_model(holder, mx)
    original = MLXSoVITSDecoder.__dict__["from_package"]
    allocated = []

    def fail(cls, *args, **kwargs):
        decoder = original.__func__(cls, *args, **kwargs)
        allocated.append(weakref.ref(decoder))
        raise RuntimeError("Injected failure after product decoder allocation, before model publication")

    MLXSoVITSDecoder.from_package = classmethod(fail)
    failure = None
    try:
        with observe_reads() as trace:
            holder[0] = MLXSoVITS.load(package, encoder_device="cpu", encoder_softmax=variant["encoder_softmax"],
                fold_weight_norm=variant["fold_weight_norm"], reference=reference)
    except RuntimeError as error:
        failure = error_record(error)
    finally:
        MLXSoVITSDecoder.from_package = original
    cleanup = clear_model(holder, mx)
    if failure is None or len(allocated) != 1 or allocated[0]() is not None or holder[0] is not None:
        raise AssertionError("Partial product load did not leave an empty slot")
    return dict(previous_released=released, error=failure, decoder_destroyed=True, current_is_none=True,
                reads=trace, after_failure=cleanup)


def worker(args):
    output = args.run.resolve() / args.variant
    output.mkdir(exist_ok=False)
    prepared = read_json(args.run / "prepared.json")
    variant = VARIANTS[args.variant]
    report = dict(status="running", variant=variant, command=[sys.executable, *sys.argv],
        loads=[], requests=[], edge_checks=[],
        scope="Product acoustic correctness with fixed semantic histories. B only changes prepared ge/ge512. This Harness executes no GPT/frontend/ASR/listening; cancellation reuses a validated saved semantic request.",
        resource_scope="Diagnostic cleanup counters only; no speed or normal peak claim. Apple MLX unified-memory counters are not NVIDIA VRAM.",
        quality=dict(asr="not_run", human_listening="not_run"))
    holder, mx = [None], None
    stage = "input_validation"
    try:
        for name, expected in prepared["source_sha256"].items():
            if sha256(args.run / "source" / name) != expected:
                raise ValueError("Frozen source changed: " + name)
        bundle, conditions, _ = load_inputs(prepared)
        from sakuratts.reference_condition import PreparedReference
        identities = bundle["manifest"]["external_models"]
        references = {name: PreparedReference.load(Path(prepared["reference_" + name.lower()]),
            gpt_checkpoint_sha256=identities["gpt"]["checkpoint_sha256"],
            sovits_checkpoint_sha256=identities["sovits"]["checkpoint_sha256"], reference_language="ja",
            official_commit=bundle["manifest"]["official_commit"],
            manifest_sha256=prepared["reference_" + name.lower() + "_manifest_sha256"])
            for name in ("A", "B")}
        if any(not exact(references[name].ge, conditions[name][0]) or not exact(references[name].ge512, conditions[name][1]) for name in references):
            raise AssertionError("Prepared reference packages differ from the acoustic fixtures")
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        report["dependencies"] = {name: metadata.version(name) for name in ("numpy", "mlx", "mlx-metal")}
        report["reference_identities"] = {name: reference.manifest["identity"] for name, reference in references.items()}
        package = Path(prepared["package"])
        old_root = Path(prepared["uncached_baseline_run"]) / "diagnostic-baseline"
        if sha256(old_root / "result.json") != prepared["uncached_baseline_result_sha256"]:
            raise ValueError("Original uncached baseline result changed")
        old_baseline = read_json(old_root / "result.json")
        stage = "complete_weight_baseline"
        report["loads"].append(dict(policy="complete", **load_model(holder, package, variant, None, mx)))
        gold = {}
        for reference_name in ("A", "B"):
            for name in variant["cases"]:
                data, case = bundle["arrays"][name], bundle["manifest"]["cases"][name]
                arrays = capture(holder[0], data, references[reference_name], case["parameters"])
                gold[(reference_name, name)] = arrays
                row = dict(policy="complete", reference=reference_name, case=name,
                    **save_arrays(output, f"complete-{reference_name}-{name}", arrays))
                if args.variant == "default":
                    info = old_baseline["cases"][name]["captures"][reference_name]
                    path = old_root / info["file"]
                    if sha256(path) != info["sha256"]:
                        raise ValueError("Original uncached acoustic capture changed")
                    with np.load(path, allow_pickle=False) as archive:
                        row["previous_default_baseline_bit_exact"] = {key: exact(value, archive[key]) for key, value in arrays.items()}
                    if not all(row["previous_default_baseline_bit_exact"].values()):
                        raise AssertionError("Default product path changed its original uncached output")
                report["requests"].append(row)
        for index, reference_name in enumerate(("A", "B", "A")):
            stage = "bound_reference_switch"
            reference = references[reference_name]
            mutable = replace(reference, manifest=deepcopy(reference.manifest), ge=reference.ge.copy(), ge512=reference.ge512.copy())
            report["loads"].append(dict(policy="bound", reference=reference_name, sequence_index=index,
                **load_model(holder, package, variant, mutable, mx)))
            binding = holder[0].bound_reference
            bound_hashes = dict(ge=spec(binding.ge), ge512=spec(binding.ge512), identity=binding.identity_json)
            mutable.ge.flat[0] += np.float32(1)
            mutable.ge512.flat[0] += np.float32(1)
            mutable.manifest["identity"]["audio_sha256"] = "f" * 64
            if dict(ge=spec(binding.ge), ge512=spec(binding.ge512), identity=binding.identity_json) != bound_hashes:
                raise AssertionError("Caller mutation changed the loaded binding snapshot")
            for value in (binding.ge, binding.ge512):
                try:
                    value.setflags(write=True)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Bound array is mutable")
            holder[0].validate_reference(reference)
            binding = value = mutable = None
            for name in variant["cases"]:
                data, case = bundle["arrays"][name], bundle["manifest"]["cases"][name]
                arrays = capture(holder[0], data, reference, case["parameters"])
                checks = {key: exact(value, gold[(reference_name, name)][key]) for key, value in arrays.items()}
                row = dict(policy="bound", reference=reference_name, case=name, sequence_index=index,
                    caller_mutation_did_not_change_binding=True, baseline_bit_exact=checks,
                    **save_arrays(output, f"bound-{index}-{reference_name}-{name}", arrays))
                if reference_name == "A":
                    row["official_comparisons"] = {key: comparison(arrays[key], data["acoustic." + key]) for key in STAGES}
                report["requests"].append(row)
                del arrays
                if not all(checks.values()):
                    raise AssertionError("Bound product output differs from the same-option complete-weight baseline")
            name = "ja-short" if "ja-short" in variant["cases"] else "ja-punctuation"
            report["edge_checks"].append(dict(reference=reference_name, sequence_index=index, case=name,
                **rejection_and_cancellation(holder[0], bundle["arrays"][name], bundle["manifest"]["cases"][name],
                    reference, references["B" if reference_name == "A" else "A"], gold[(reference_name, name)])))
            write_json(output / "result.json", report)
        stage = "partial_load_failure"
        report["load_failure"] = partial_load_failure(holder, package, references["B"], variant, mx)
        report["recovery_load"] = load_model(holder, package, variant, references["A"], mx)
        name = variant["cases"][0]
        arrays = capture(holder[0], bundle["arrays"][name], references["A"], bundle["manifest"]["cases"][name]["parameters"])
        report["post_failure_recovery"] = dict(case=name, bit_exact={key: exact(value, gold[("A", name)][key]) for key, value in arrays.items()})
        if not all(report["post_failure_recovery"]["bit_exact"].values()):
            raise AssertionError("Explicit recovery load did not restore product output")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="error", error=dict(stage=stage, **error_record(error)))
    if mx is not None:
        report["after_unload"] = clear_model(holder, mx)
    write_json(output / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prior = read_json(args.baseline_run / "prepared.json")
    files = sorted(set(prior["source_sha256"]) | {"harness/sovits_bound_reference_validation.py"})
    # Only the existing portable loader/helpers are reused. Product classes are
    # freshly frozen here; the previous experimental model is never imported.
    for name in files:
        path = output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, path)
    prepared = dict(prior, command=[sys.executable, *sys.argv], created_at_utc=datetime.now(timezone.utc).isoformat(),
        uncached_baseline_run=str(args.baseline_run.resolve()),
        uncached_baseline_result_sha256=sha256(args.baseline_run / "diagnostic-baseline/result.json"),
        reference_a=str(args.reference_a.resolve()), reference_a_manifest_sha256=sha256(args.reference_a / "manifest.json"),
        source_sha256={name: sha256(output / "source" / name) for name in files})
    write_json(output / "prepared.json", prepared)
    processes = []
    for variant in VARIANTS:
        command = [sys.executable, str(output / "source/harness/sovits_bound_reference_validation.py"),
                   "worker", "--run", str(output), "--variant", variant]
        started = time.perf_counter()
        with (output / (variant + ".stdout.log")).open("x", encoding="utf-8") as stdout, (output / (variant + ".stderr.log")).open("x", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=output, stdout=stdout, stderr=stderr)
        processes.append(dict(variant=variant, command=command, exit_code=process.returncode, wall_seconds=time.perf_counter() - started))
        write_json(output / "processes.json", processes)
        print(json.dumps(processes[-1]), flush=True)
        if process.returncode:
            return process.returncode
    write_json(output / "result.json", dict(status="completed", variants=list(VARIANTS),
        worker_result_sha256={name: sha256(output / name / "result.json") for name in VARIANTS}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dispatch = commands.add_parser("run")
    for name in ("baseline-run", "reference-a", "output"):
        dispatch.add_argument("--" + name, type=Path, required=True)
    child = commands.add_parser("worker")
    child.add_argument("--run", type=Path, required=True)
    child.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
