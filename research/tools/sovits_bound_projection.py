#!/usr/bin/env python3
"""Measure reference-bound acoustic instances that omit 14 condition weights.

The original verified package remains complete. A load prepares five projections
from the 14 weights, releases those weights, then loads only the remaining names
from the same package handle. This is a Harness experiment, not a runtime change.
Unknown conditions fail before encoder execution. Switching drops the old model
before preparation; failures leave no usable model and never fall back.
"""

import argparse
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from sovits_reference_projection import (CASES, STAGES, comparison, exact, inventory,
    load_inputs, memory, read_json, sha256, spec, write_json)

PREFIXES = tuple([f"flow.flows.{index}.enc.cond_layer" for index in (0, 2, 4, 6)] + ["dec.cond"])
WEIGHT_BYTES = 27314176
PROJECTION_BYTES = 26624


def exception_record(error, stage):
    return dict(stage=stage, type=type(error).__name__, module=type(error).__module__,
                message=str(error), traceback=traceback.format_exc())


def load_acoustic(package, policy, conditions, manifest_hash, *, fail_after_encoder=False):
    """One package verification/handle per load, including projection preparation."""
    import mlx.core as mx
    from sakuratts.backends.mlx.sovits import MLXSoVITS
    from sakuratts.backends.mlx.encoder import MLXSoVITSEncoder
    from sakuratts.backends.mlx.flow import MLXSoVITSFlow
    from sakuratts.backends.mlx.decoder import MLXSoVITSDecoder
    from sakuratts.backends.mlx.sovits_package import SoVITSPackage

    started = time.perf_counter()
    actual_hash = sha256(package / "manifest.json")
    if actual_hash != manifest_hash:
        raise ValueError("Requested package manifest identity differs from the actual package")
    binding = dict(package_manifest_sha256=actual_hash,
                   ge=spec(conditions[0]), ge512=spec(conditions[1]))
    if (conditions[0].dtype != np.float32 or conditions[0].shape != (1, 1024, 1)
            or conditions[1].dtype != np.float32 or conditions[1].shape != (1, 512, 1)):
        raise ValueError("Require exact V2Pro FP32 acoustic reference shapes")
    report = dict(policy=policy, binding=binding, prepare_read=[], remaining_read=[])

    class SelectedSource:
        def __init__(self, source, excluded):
            self.source, self.excluded, self.manifest = source, excluded, source.manifest

        def tensors(self, *prefixes, names=(), exclude=()):
            # Selection precedes read_fp32: passing prefixes through would OR
            # them with names and accidentally restore excluded weights.
            allowed = tuple(name for name in self.manifest["tensor_sources"]
                            if (name in names or name.startswith(prefixes))
                            and name not in self.excluded and name not in exclude)
            for name, value in self.source.tensors(names=allowed):
                report["remaining_read"].append(dict(name=name, bytes=value.nbytes))
                yield name, value

    class Flow(MLXSoVITSFlow):
        def conv(self, x, prefix):
            if prefix in PREFIXES:
                return self.projections[prefix]
            return super().conv(x, prefix)

    class Decoder(MLXSoVITSDecoder):
        def conv(self, x, prefix):
            if prefix in PREFIXES:
                return self.projections[prefix]
            return super().conv(x, prefix)

    class BoundModel(MLXSoVITS):
        def validate_binding(self, ge, ge512, package_manifest_sha256):
            if (package_manifest_sha256 != self.package_manifest_sha256
                    or self.binding["package_manifest_sha256"] != self.package_manifest_sha256):
                raise ValueError("Acoustic projection package binding mismatch")
            if spec(ge) != self.binding["ge"] or spec(ge512) != self.binding["ge512"]:
                raise ValueError("Acoustic reference binding mismatch; explicitly reload for this reference")

        def decode(self, codes, phones, ge, ge512, noise, *, package_manifest_sha256, **kwargs):
            self.validate_binding(ge, ge512, package_manifest_sha256)
            self.validated_decodes += 1
            return super().decode(codes, phones, ge, ge512, noise, **kwargs)

    def projections_only(source, excluded):
        weights = {}
        for name, value in source.tensors(names=tuple(excluded)):
            report["prepare_read"].append(dict(name=name, bytes=value.nbytes))
            weights[name] = mx.array(value)
        mx.eval(*weights.values())
        flow = MLXSoVITSFlow(source.manifest, {name: value for name, value in weights.items() if name.startswith("flow.")})
        decoder = MLXSoVITSDecoder(source.manifest, {name: value for name, value in weights.items() if name.startswith("dec.")})
        incoming = mx.array(conditions[0].transpose(0, 2, 1))
        values = {prefix: MLXSoVITSFlow.conv(flow, incoming, prefix) for prefix in PREFIXES[:-1]}
        values["dec.cond"] = MLXSoVITSDecoder.conv(decoder, incoming, "dec.cond")
        mx.eval(*values.values())
        return values, [weakref.ref(flow), weakref.ref(decoder)]

    with SoVITSPackage.open(package) as source:
        report["verification_seconds"] = time.perf_counter() - started
        all_names = set(source.manifest["tensor_sources"])
        excluded = {name for name in all_names if any(name.startswith(prefix + ".") for prefix in PREFIXES)}
        if len(excluded) != 14:
            raise ValueError("Unexpected V2Pro condition tensor inventory")
        phase_started = time.perf_counter()
        projections = {}
        if policy == "bound":
            with mx.stream(mx.gpu):
                projections, temporary_refs = projections_only(source, excluded)
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            mx.synchronize()
            if not all(reference() is None for reference in temporary_refs):
                raise AssertionError("Temporary projection components were retained")
            if sum(row["bytes"] for row in report["prepare_read"]) != WEIGHT_BYTES:
                raise ValueError("Projection FP32 weight byte inventory changed")
            if sum(value.nbytes for value in projections.values()) != PROJECTION_BYTES:
                raise ValueError("Unexpected projected output bytes")
        report["projection_prepare_release_seconds"] = time.perf_counter() - phase_started
        report["after_projection_release"] = dict(mlx_active_bytes=mx.get_active_memory(),
            mlx_cache_bytes=mx.get_cache_memory(), mlx_allocator_peak_bytes=mx.get_peak_memory())
        selected = SelectedSource(source, excluded if policy == "bound" else set())
        phase_started = time.perf_counter()
        with mx.stream(mx.cpu):
            encoder = MLXSoVITSEncoder.from_package(selected, softmax="fp32")
        if fail_after_encoder:
            raise RuntimeError("Injected load failure after acoustic encoder allocation")
        with mx.stream(mx.gpu):
            flow = (Flow if policy == "bound" else MLXSoVITSFlow).from_package(selected, fold_weight_norm=False)
            decoder = (Decoder if policy == "bound" else MLXSoVITSDecoder).from_package(selected, fold_weight_norm=False)
        mx.synchronize()
        report["remaining_model_load_seconds"] = time.perf_counter() - phase_started
        model = (BoundModel if policy == "bound" else MLXSoVITS)(encoder, flow, decoder, mx.gpu, mx.cpu)
        if policy == "bound":
            model.projections = flow.projections = decoder.projections = projections
            model.binding = binding
            model.package_manifest_sha256 = actual_hash
            model.validated_decodes = 0
        read_names = [row["name"] for row in report["remaining_read"]]
        expected_names = all_names - excluded if policy == "bound" else all_names
        if set(read_names) != expected_names or len(read_names) != len(expected_names):
            raise AssertionError("Remaining weights were missing, duplicated or excluded incorrectly")
        if policy == "bound" and set(read_names) & excluded:
            raise AssertionError("Bound model reread excluded condition weights")
        report["excluded_names"] = sorted(excluded) if policy == "bound" else []
        report["loaded_weight_bytes"] = sum(row["bytes"] for row in report["remaining_read"])
        report["projection_bytes"] = sum(value.nbytes for value in projections.values())
    mx.synchronize()
    report["prepare_load_total_seconds"] = time.perf_counter() - started
    report["after_load"] = memory(mx)
    return model, report


class ModelSlot:
    """No fallback ownership: every switch discards the prior instance first."""
    def __init__(self, mx):
        self.mx = mx
        self.current = None

    def clear(self):
        self.mx.synchronize()
        previous = weakref.ref(self.current) if self.current is not None else None
        self.current = None
        gc.collect()
        self.mx.clear_cache()
        self.mx.synchronize()
        result = dict(model_destroyed=previous is None or previous() is None, memory=memory(self.mx))
        if not result["model_destroyed"]:
            raise AssertionError("Previous acoustic model remains owned after switch/unload")
        return result

    def switch(self, package, policy, conditions, manifest_hash, *, fail_after_encoder=False):
        released = self.clear()
        self.mx.reset_peak_memory()
        report = dict(status="running", previous_released=released)
        try:
            self.current, loaded = load_acoustic(package, policy, conditions, manifest_hash,
                                               fail_after_encoder=fail_after_encoder)
            report.update(status="completed", load=loaded)
        except Exception as error:
            report.update(status="error", error=exception_record(error, "reference_prepare_and_load"))
        if report["status"] == "error":
            report["failed_state"] = self.clear()
            report["failed_state"]["current_is_none"] = self.current is None
        return report


def invoke(model, policy, data, conditions, parameters, manifest_hash, *, capture=False, semantic=None):
    kwargs = dict(noise_scale=parameters["noise_scale"], speed=parameters["speed"], capture=capture)
    if policy == "bound":
        kwargs["package_manifest_sha256"] = manifest_hash
    return model.decode(data["semantic"] if semantic is None else semantic, data["acoustic_phones"],
                        *conditions, data["noise"], **kwargs)


def gold_arrays(prepared, name, reference):
    root = Path(prepared["baseline_run"]) / "diagnostic-baseline"
    result = read_json(root / "result.json")
    info = result["cases"][name]["captures"][reference]
    if sha256(root / info["file"]) != info["sha256"]:
        raise ValueError("Uncached baseline capture hash changed")
    with np.load(root / info["file"], allow_pickle=False) as archive:
        return {key: archive[key] for key in (*STAGES, "pcm")}


def load_or_raise(slot, package, policy, conditions, manifest_hash):
    row = slot.switch(package, policy, conditions, manifest_hash)
    if row["status"] != "completed":
        raise RuntimeError(json.dumps(row["error"]))
    return row


def diagnostic(slot, bundle, references, prepared, output, report, package, manifest_hash):
    from sakuratts._internal.synthesis import single_fragment_pcm

    report["switches"] = []
    report["mismatch_rejections"] = []
    for sequence_index, reference in enumerate(("A", "B", "A")):
        switched = load_or_raise(slot, package, "bound", references[reference], manifest_hash)
        report["switches"].append(dict(reference=reference, **switched))
        weight_objects = inventory(slot.current)
        for name in CASES:
            data = bundle["arrays"][name]
            parameters = bundle["manifest"]["cases"][name]["parameters"]
            waveform, stages = invoke(slot.current, "bound", data, references[reference], parameters, manifest_hash, capture=True)
            if set(stages) != set(STAGES):
                raise ValueError("Require all twelve actual acoustic stages")
            arrays = {key: np.ascontiguousarray(np.asarray(stages[key]).copy()) for key in STAGES}
            arrays["pcm"] = single_fragment_pcm(arrays["waveform"], parameters["sample_rate"], parameters["fragment_interval"])
            del waveform, stages
            gold = gold_arrays(prepared, name, reference)
            checks = {key: exact(value, gold[key]) for key, value in arrays.items()}
            path = output / f"{sequence_index}-{reference}-{name}.npz"
            with path.open("xb") as stream:
                np.savez(stream, **arrays)
            row = dict(reference=reference, case=name, sequence_index=sequence_index, file=path.name,
                sha256=sha256(path), arrays={key: spec(value) for key, value in arrays.items()},
                baseline_bit_exact=checks)
            if reference == "A":
                row["official_comparisons"] = {key: comparison(arrays[key], data["acoustic." + key]) for key in STAGES}
            report["requests"].append(row)
            if not all(checks.values()):
                raise AssertionError("Bound projection changed an acoustic stage or PCM")
            del arrays, gold
        if inventory(slot.current) != weight_objects:
            raise AssertionError("Resident weight objects changed")
        # Rejection must precede the first encoder operation; the same valid
        # bound model remains usable until an explicit switch begins.
        data = bundle["arrays"]["ja-short"]
        parameters = bundle["manifest"]["cases"]["ja-short"]["parameters"]
        other = references["B" if reference == "A" else "A"]
        for mode in ("ge", "ge512", "package"):
            conditions = (other[0], references[reference][1]) if mode == "ge" else (
                (references[reference][0], other[1]) if mode == "ge512" else references[reference])
            before = slot.current.validated_decodes
            rejected = None
            try:
                unexpected = invoke(slot.current, "bound", data, conditions, parameters,
                                    "0" * 64 if mode == "package" else manifest_hash)
                del unexpected
            except ValueError as error:
                rejected = exception_record(error, "binding_check")
            if rejected is None or slot.current.validated_decodes != before:
                raise AssertionError("Unknown reference/package was not rejected before encode")
            report["mismatch_rejections"].append(dict(reference=reference, field=mode, error=rejected))
        write_json(output / "result.json", report)
    failed = slot.switch(package, "bound", references["B"], manifest_hash, fail_after_encoder=True)
    if (failed["status"] != "error" or not failed["failed_state"]["current_is_none"]
            or failed["failed_state"]["memory"]["mlx_active_bytes"] or failed["failed_state"]["memory"]["mlx_cache_bytes"]
            or failed["error"]["message"] != "Injected load failure after acoustic encoder allocation"):
        raise AssertionError("Partial-load failure did not leave an empty model slot")
    report["injected_load_failure"] = failed
    report["recovery_load"] = load_or_raise(slot, package, "bound", references["A"], manifest_hash)
    data = bundle["arrays"]["ja-short"]
    parameters = bundle["manifest"]["cases"]["ja-short"]["parameters"]
    waveform = invoke(slot.current, "bound", data, references["A"], parameters, manifest_hash)
    audio = np.asarray(waveform).copy()
    del waveform
    gold = gold_arrays(prepared, "ja-short", "A")
    report["recovery_waveform_bit_exact"] = exact(audio, gold["waveform"])
    report["recovery_pcm_bit_exact"] = exact(single_fragment_pcm(audio, parameters["sample_rate"], parameters["fragment_interval"]), gold["pcm"])
    if not report["recovery_waveform_bit_exact"] or not report["recovery_pcm_bit_exact"]:
        raise AssertionError("Explicit reload after failure did not restore the expected output")


def stage_profile(model, policy, data, conditions, parameters, manifest_hash, mx):
    """Separate diagnostic stage boundaries; not used for normal timings."""
    rows = {}

    def begin():
        mx.synchronize()
        mx.reset_peak_memory()
        return time.perf_counter()

    def end(name, started):
        mx.synchronize()
        rows[name] = dict(seconds=time.perf_counter() - started, memory=memory(mx))

    started = begin()
    if policy == "bound":
        model.validate_binding(*conditions, manifest_hash)
    end("binding", started)
    started = begin()
    with mx.stream(model.encoder_device):
        mean, log_scale, mask = model.encoder.encode(data["semantic"], data["acoustic_phones"], conditions[1],
                                                   speed=parameters["speed"], capture=False)
        mx.eval(mean, log_scale, mask)
    end("encoder", started)
    started = begin()
    with mx.stream(model.device):
        latent = mean + mx.array(data["noise"]) * mx.exp(log_scale) * parameters["noise_scale"]
        mx.eval(latent)
    end("flow_input", started)
    started = begin()
    with mx.stream(model.device):
        flowed = model.flow.reverse(latent, mask, conditions[0])
    end("flow", started)
    started = begin()
    with mx.stream(model.device):
        waveform = model.decoder.decode(flowed * mask, conditions[0])
    end("decoder", started)
    audio = np.ascontiguousarray(np.asarray(waveform).copy())
    return rows, audio


def resource_run(slot, bundle, references, prepared, output, report, package, manifest_hash, policy, mx):
    from sakuratts._internal.synthesis import single_fragment_pcm

    report["load"] = load_or_raise(slot, package, policy, references["A"], manifest_hash)
    report["idle_after_load"] = memory(mx)
    weights = inventory(slot.current)
    for name in CASES:
        data = bundle["arrays"][name]
        parameters = bundle["manifest"]["cases"][name]["parameters"]
        gold = gold_arrays(prepared, name, "A")
        row = dict(parameters=parameters, audio_body_seconds=gold["waveform"].shape[-1] / parameters["sample_rate"], calls=[])
        report["cases"][name] = row
        for index in range(prepared["warmup"] + prepared["repeat"]):
            mx.synchronize()
            mx.reset_peak_memory()
            started = time.perf_counter()
            waveform = invoke(slot.current, policy, data, references["A"], parameters, manifest_hash)
            mx.synchronize()
            seconds = time.perf_counter() - started
            measured = memory(mx)
            audio = np.ascontiguousarray(np.asarray(waveform).copy())
            del waveform
            pcm = single_fragment_pcm(audio, parameters["sample_rate"], parameters["fragment_interval"])
            checks = dict(waveform=exact(audio, gold["waveform"]), pcm=exact(pcm, gold["pcm"]))
            if not all(checks.values()):
                raise AssertionError("Uncaptured acoustic output changed")
            row["calls"].append(dict(index=index, warmup=index < prepared["warmup"], seconds=seconds,
                rtf=seconds / row["audio_body_seconds"], memory=measured, idle_after_output_release=memory(mx),
                waveform=spec(audio), pcm=spec(pcm), baseline_bit_exact=checks))
            del audio, pcm
        profile, audio = stage_profile(slot.current, policy, data, references["A"], parameters, manifest_hash, mx)
        row["instrumented_stage_profile"] = dict(stages=profile, waveform_bit_exact=exact(audio, gold["waveform"]))
        if not row["instrumented_stage_profile"]["waveform_bit_exact"]:
            raise AssertionError("Diagnostic acoustic stage decomposition changed waveform")
        del audio, gold
        write_json(output / "result.json", report)
        print(json.dumps(dict(worker=report["worker"], case=name, status="completed")), flush=True)
    report["weights_objects_unchanged"] = inventory(slot.current) == weights
    if not report["weights_objects_unchanged"]:
        raise AssertionError("Resident weight objects changed during resource measurement")


def prepared_request(slot, bundle, references, prepared, output, report, package, manifest_hash, policy, mx):
    from sakuratts.backends.mlx.gpt import MLXGPT
    from sakuratts._internal.generation import generate_semantic
    from sakuratts._internal.synthesis import single_fragment_pcm

    gpt = None
    gpt_ref = None
    try:
        mx.reset_peak_memory()
        started = time.perf_counter()
        gpt = MLXGPT.load(Path(prepared["gpt_package"]), capacity=1024, prefill_precision="fp64")
        gpt_ref = weakref.ref(gpt)
        mx.synchronize()
        report["gpt_load_seconds"] = time.perf_counter() - started
        report["after_gpt_load"] = memory(mx)
        report["load"] = load_or_raise(slot, package, policy, references["A"], manifest_hash)
        report["combined_model_idle"] = memory(mx)
        for name in CASES:
            data = bundle["arrays"][name]
            case = bundle["manifest"]["cases"][name]
            parameters = case["parameters"]
            draws = [data[f"draw.{index}"] for index in range(sum(key.startswith("draw.") for key in data))]
            consumed = []

            def draw(index, shape):
                if index >= len(draws) or draws[index].shape != shape:
                    raise ValueError("Own GPT history exceeded recorded official draws")
                consumed.append(index)
                return draws[index]

            mx.synchronize()
            mx.reset_peak_memory()
            started = time.perf_counter()
            try:
                generated = generate_semantic(gpt, data["phones"], data["prompt"], data["bert"],
                    **{key: parameters[key] for key in ("eos", "top_k", "top_p", "early_stop_num", "temperature", "repetition_penalty")},
                    random_draw=draw)
            finally:
                gpt.release_request_state()
            semantic_done = time.perf_counter()
            # The acoustic call receives this request's actual generated history.
            waveform = invoke(slot.current, policy, data, references["A"], parameters, manifest_hash,
                              semantic=generated.semantic)
            mx.synchronize()
            acoustic_done = time.perf_counter()
            audio = np.ascontiguousarray(np.asarray(waveform).copy())
            del waveform
            pcm = single_fragment_pcm(audio, parameters["sample_rate"], parameters["fragment_interval"])
            finished = time.perf_counter()
            measured = memory(mx)
            gold = gold_arrays(prepared, name, "A")
            checks = dict(tokens=exact(generated.sampled_tokens, data["tokens"]),
                history=exact(generated.stop.history, data["history"]), semantic=exact(generated.semantic, data["semantic"]),
                returned_index=generated.stop.returned_index == case["stop"]["returned_index"],
                reasons=set(generated.stop.reasons) == set(case["stop"]["reasons"]),
                waveform=exact(audio, gold["waveform"]), pcm=exact(pcm, gold["pcm"]),
                draws=consumed == list(range(generated.sampled_tokens.size)),
                gpt_state_released=gpt.length == 0 and len(gpt.keys) == 0 and len(gpt.values) == 0)
            arrays = dict(tokens=generated.sampled_tokens, history=generated.stop.history,
                          semantic=generated.semantic, waveform=audio, pcm=pcm)
            path = output / (name + ".npz")
            with path.open("xb") as stream:
                np.savez(stream, **arrays)
            row = dict(case=name, parameters=parameters, sampled_steps=generated.sampled_tokens.size,
                semantic_tokens=generated.semantic.shape[-1],
                actual_stop=dict(returned_index=generated.stop.returned_index, reasons=list(generated.stop.reasons)),
                expected_stop=case["stop"], checks=checks,
                arrays={key: spec(value) for key, value in arrays.items()}, file=path.name, sha256=sha256(path),
                semantic_seconds=semantic_done - started, acoustic_seconds=acoustic_done - semantic_done,
                output_seconds=finished - acoustic_done, request_seconds=finished - started,
                audio_body_seconds=audio.shape[-1] / parameters["sample_rate"], memory=measured)
            row["rtf"] = row["request_seconds"] / row["audio_body_seconds"]
            report["requests"].append(row)
            write_json(output / "result.json", report)
            if not all(checks.values()):
                raise AssertionError("Prepared own-history request differs from its validated original baseline")
            del arrays, audio, pcm, generated, gold
    finally:
        if gpt is not None:
            gpt.release_request_state()
        gpt = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        report["gpt_destroyed"] = gpt_ref is None or gpt_ref() is None


def worker(args):
    run, output = args.run.resolve(), args.run.resolve() / args.worker
    output.mkdir(exist_ok=False)
    prepared = read_json(run / "prepared.json")
    report = dict(status="running", worker=args.worker, requests=[], cases={},
        command=[sys.executable, *sys.argv], quality=dict(asr="not_run", human_listening="not_run"),
        precision="GPU FP32 flow/decoder, CPU FP32 acoustic encoder; no WeightNorm folding. Prepared own-history workers additionally use CPU FP64 GPT Prefill and GPU FP32 Decode.",
        package_scope="Original complete package remains unchanged. This Harness only skips loading 14 weights into a reference-bound instance after preparing five exact projections.",
        timing_scope="Resource workers use uncaptured, synchronized complete acoustic calls; binding checks are inside timers. Host output copy, PCM, hashes, memory snapshots and separate stage profiles are outside. ABBA independent worker batches, not per-request paired timings.",
        prepared_request_scope="Actual own GPT generation from original prepared text/reference features and recorded official real draws, then its own semantic history -> acoustic -> PCM. No frontend/reference-audio preparation, file I/O or model loading inside request timer. One diagnostic complete request per case/policy, not steady end-to-end speed evidence.",
        memory_scope="MLX allocator active/cache/peak on Apple unified memory, not NVIDIA VRAM. Per-call allocator reset measures the uncaptured acoustic/request envelope. Boundary RSS is not a peak; OS maxrss is worker lifetime.")
    mx = slot = None
    stage = "source_validation"
    try:
        for name, expected in prepared["source_sha256"].items():
            if sha256(run / "source" / name) != expected:
                raise ValueError("Frozen source changed: " + name)
        for name, expected in prepared["baseline_result_sha256"].items():
            if sha256(Path(prepared["baseline_run"]) / name) != expected:
                raise ValueError("Existing baseline result changed: " + name)
        bundle, references, reference_b = load_inputs(prepared)
        report.update(bundle_manifest_sha256=bundle["manifest_sha256"],
            model_identity=bundle["manifest"]["external_models"]["sovits"], reference_b_identity=reference_b["identity"],
            conditions={name: dict(ge=spec(pair[0]), ge512=spec(pair[1])) for name, pair in references.items()},
            dependencies={name: metadata.version(name) for name in ("numpy", "mlx", "mlx-metal")})
        package, manifest_hash = Path(prepared["package"]), report["model_identity"]["manifest_sha256"]
        if args.worker.startswith("prepared-"):
            from portable_validation import load_bundle
            load_bundle(Path(prepared["bundle"]), gpt_package=Path(prepared["gpt_package"]))
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        mx.reset_peak_memory()
        slot = ModelSlot(mx)
        report["before_load"] = memory(mx)
        stage = args.worker
        if args.worker == "diagnostic-bound":
            diagnostic(slot, bundle, references, prepared, output, report, package, manifest_hash)
        elif args.worker.startswith("resource-"):
            resource_run(slot, bundle, references, prepared, output, report, package, manifest_hash,
                         args.worker.split("-")[1], mx)
        else:
            prepared_request(slot, bundle, references, prepared, output, report, package, manifest_hash,
                             args.worker.removeprefix("prepared-"), mx)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="error", error=exception_record(error, stage))
    if slot is not None:
        report["after_unload"] = slot.clear()
        if (report["after_unload"]["memory"]["mlx_active_bytes"]
                or report["after_unload"]["memory"]["mlx_cache_bytes"]):
            report["status"] = "unload_failed"
    write_json(output / "result.json", report)
    return 0 if report["status"] == "completed" else 1


def summarize(run):
    prepared = read_json(run / "prepared.json")
    reports = {name: read_json(run / name / "result.json") for name in prepared["workers"]}
    if any(row["status"] != "completed" for row in reports.values()):
        raise ValueError("An experiment worker did not complete")
    package = Path(prepared["package"])
    identity = reports["diagnostic-bound"]["model_identity"]
    if (sha256(package / "manifest.json") != identity["manifest_sha256"]
            or sha256(package / identity["weights_file"]) != identity["weights_sha256"]):
        raise ValueError("Original complete acoustic package changed during experiment")
    summary = dict(status="completed", package_bytes_unchanged=True,
                   original_weights_file_bytes=(package / identity["weights_file"]).stat().st_size,
                   theory_net_bytes=WEIGHT_BYTES - PROJECTION_BYTES,
                   resources={}, prepared_requests={}, worker_sha256={name: sha256(run / name / "result.json") for name in reports})
    for name in CASES:
        row = {}
        for policy in ("baseline", "bound"):
            workers = [report for worker, report in reports.items() if worker.startswith("resource-" + policy)]
            calls = [call for report in workers for call in report["cases"][name]["calls"] if not call["warmup"]]
            row[policy] = dict(measured_calls=len(calls),
                median_seconds=float(np.median([call["seconds"] for call in calls])),
                allocator_peak_bytes=[call["memory"]["mlx_allocator_peak_bytes"] for call in calls],
                idle_active_bytes=[report["idle_after_load"]["mlx_active_bytes"] for report in workers],
                load_seconds=[report["load"]["load"]["prepare_load_total_seconds"] for report in workers],
                load_allocator_peak_bytes=[report["load"]["load"]["after_load"]["mlx_allocator_peak_bytes"] for report in workers])
        row["median_peak_saved_bytes"] = float(np.median(row["baseline"]["allocator_peak_bytes"]) - np.median(row["bound"]["allocator_peak_bytes"]))
        row["median_seconds_saved_independent_batches"] = row["baseline"]["median_seconds"] - row["bound"]["median_seconds"]
        summary["resources"][name] = row
        own = {policy: next(row for row in reports["prepared-" + policy]["requests"] if row["case"] == name)
               for policy in ("baseline", "bound")}
        if any(not all(row["checks"].values()) for row in own.values()):
            raise ValueError("Prepared own-history request failed")
        if own["baseline"]["arrays"] != own["bound"]["arrays"]:
            raise ValueError("Prepared candidate output differs from uncached prepared request")
        summary["prepared_requests"][name] = {policy: {key: row[key] for key in
            ("request_seconds", "semantic_seconds", "acoustic_seconds", "memory", "rtf", "sampled_steps", "semantic_tokens")}
            for policy, row in own.items()}
    write_json(run / "result.json", summary)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prior = read_json(args.baseline_run / "prepared.json")
    prior_result = read_json(args.baseline_run / "result.json")
    if prior_result["status"] != "completed" or not prior_result["candidate_equivalence_passed"]:
        raise ValueError("Require the completed projection-cache baseline experiment")
    files = sorted(set(prior["source_sha256"]) | {"research/tools/sovits_bound_projection.py", "src/sakuratts/backends/mlx/gpt.py", "src/sakuratts/backends/mlx/gpt_prefill.py"})
    for name in files:
        path = output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, path)
    workers = ["diagnostic-bound", "resource-baseline-1", "resource-bound-1", "resource-bound-2", "resource-baseline-2",
               "prepared-baseline", "prepared-bound"]
    prepared = dict(prior, command=[sys.executable, *sys.argv], created_at_utc=datetime.now(timezone.utc).isoformat(),
        baseline_run=str(args.baseline_run.resolve()), gpt_package=str(args.gpt_package.resolve()),
        warmup=args.warmup, repeat=args.repeat, workers=workers,
        baseline_result_sha256={name: sha256(args.baseline_run / name) for name in
            ("prepared.json", "result.json", "diagnostic-baseline/result.json", "diagnostic-comparison.json")},
        source_sha256={name: sha256(output / "source" / name) for name in files})
    write_json(output / "prepared.json", prepared)
    processes = []
    for name in workers:
        command = [sys.executable, str(output / "source/research/tools/sovits_bound_projection.py"),
                   "worker", "--run", str(output), "--worker", name]
        started = time.perf_counter()
        with (output / (name + ".stdout.log")).open("x", encoding="utf-8") as stdout, (output / (name + ".stderr.log")).open("x", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=output, stdout=stdout, stderr=stderr)
        processes.append(dict(worker=name, command=command, exit_code=process.returncode, wall_seconds=time.perf_counter() - started))
        write_json(output / "processes.json", processes)
        print(json.dumps(processes[-1]), flush=True)
        if process.returncode:
            return process.returncode
    summarize(output)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dispatch = commands.add_parser("run")
    for name in ("baseline-run", "gpt-package", "output"):
        dispatch.add_argument("--" + name, type=Path, required=True)
    dispatch.add_argument("--warmup", type=int, default=5)
    dispatch.add_argument("--repeat", type=int, default=7)
    child = commands.add_parser("worker")
    child.add_argument("--run", type=Path, required=True)
    child.add_argument("--worker", required=True, choices=("diagnostic-bound", "resource-baseline-1", "resource-baseline-2",
        "resource-bound-1", "resource-bound-2", "prepared-baseline", "prepared-bound"))
    args = parser.parse_args()
    if args.command == "run" and (args.warmup < 5 or args.repeat < 7):
        parser.error("Require five warmups and seven measured calls per case and resource worker")
    return run(args) if args.command == "run" else worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
