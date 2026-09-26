"""Decode instrumentation must preserve results and remain outside normal runs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/tools"))
import directml_gpt_static as probe


class Binding:
    def __init__(self):
        self.value = 0

    def bind_cpu_input(self, name, value):
        if name == "hidden":
            self.value = value

    def bind_ortvalue_input(self, name, value):
        pass

    def bind_ortvalue_output(self, name, value):
        pass

    def bind_output(self, name, device, ordinal):
        pass

    def synchronize_outputs(self):
        pass

    def get_outputs(self):
        return [SimpleNamespace(numpy=lambda: np.asarray([[self.value, self.value + 1]], np.float16))]


class Session:
    def __init__(self):
        self.fail = False

    def run_with_iobinding(self, binding):
        if self.fail:
            raise RuntimeError("decode failure")

    def end_profiling(self):
        return "unused-profile.json"


class Model:
    def __init__(self):
        self.session = Session()
        self.static_manifest = {"graph": {"file": "decode.onnx", "sha256": "graph"}}
        self.cache = {"key.0": SimpleNamespace(device_name=lambda: "DML")}
        self.closed = False

    def prefill(self, *inputs):
        self.bindings = [Binding(), Binding()]
        self.binding = self.bindings[0]
        self.index = 0
        return np.asarray([[1, 2]], np.float32)

    def decode(self, token):
        self.binding = self.bindings[self.index % 2]
        self.index += 1
        for name in ("hidden", "mask", "write_index"):
            self.binding.bind_cpu_input(name, token)
        for name in ("key.0", "value.0"):
            self.binding.bind_ortvalue_input(name, None)
            if self.index <= 2:
                self.binding.bind_output(name, "dml", 0)
            else:
                self.binding.bind_ortvalue_output(name, None)
        self.session.run_with_iobinding(self.binding)
        self.binding.synchronize_outputs()
        return self.binding.get_outputs()[0].numpy().astype(np.float32, copy=True)

    def close(self):
        self.closed = True


class DecodeBreakdownTests(unittest.TestCase):
    def test_counts_both_ping_pong_bindings_and_preserves_decode_outputs(self):
        model = Model()
        original = Binding.bind_cpu_input
        result, logits = probe.timing_breakdown(model, (), [3, 8, 5])
        np.testing.assert_array_equal(logits, [[1, 2], [3, 4], [8, 9], [5, 6]])
        self.assertEqual(result["decode_tokens"], 3)
        self.assertEqual(result["sections"]["bind_cpu_input"]["calls"], 9)
        self.assertEqual(result["sections"]["bind_kv_input"]["calls"], 6)
        self.assertEqual(result["sections"]["bind_kv_output"]["calls"], 2)
        self.assertEqual(result["sections"]["bind_allocate_output"]["calls"], 4)
        for name in ("run", "synchronize", "get_outputs"):
            self.assertEqual(result["sections"][name]["calls"], 3)
        for step in result["steps"]:
            measured = sum(value["wall_ms"] for value in step["sections"].values())
            self.assertAlmostEqual(step["decode_ms"], measured + step["host_and_logits_residual_ms"])
        self.assertIs(Binding.bind_cpu_input, original)

    def test_context_restores_methods_after_a_decode_error(self):
        model = Model()
        model.session.fail = True
        original_binding, original_run = Binding.get_outputs, model.session.run_with_iobinding
        with self.assertRaisesRegex(RuntimeError, "decode failure"):
            probe.timing_breakdown(model, (), [3])
        self.assertIs(Binding.get_outputs, original_binding)
        self.assertEqual(model.session.run_with_iobinding, original_run)
        model.session.fail = False
        np.testing.assert_array_equal(model.decode(4), [[4, 5]])


class ModelProbeTests(unittest.TestCase):
    def test_uses_prepared_package_and_keeps_first_hot_breakdown_and_profile_separate(self):
        models, options = [], []
        original = Binding.bind_cpu_input

        def load(package, **kwargs):
            self.assertEqual(package, Path("prepared-gpt"))
            self.assertIs(Binding.bind_cpu_input, original)
            options.append(kwargs)
            model = Model()
            models.append(model)
            return model

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe"
            args = SimpleNamespace(output=output, model=Path("model"), result=Path("saved.json"),
                case="long", precision="fp16", capacity=1280, device_id=3, threads=4,
                steps=0, warmups=1, repeats=2, sampling=False, profile=True, timing_breakdown=True,
                memory_boundaries=False)
            with patch("cpu_gpt_ort.load_inputs", return_value=(Path("prepared-gpt"), (),
                    np.asarray([3, 8], np.int64), {"case": "long"})), \
                    patch.object(probe.StaticDirectMLGPT, "load", side_effect=load), \
                    patch.object(probe.StaticDirectMLGPT, "decode", Model.decode), \
                    patch.object(probe, "profile_summary", return_value={"DmlExecutionProvider:DmlFusedNode": 2}), \
                    patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(__version__="test")}), \
                    patch.object(probe.time, "sleep", side_effect=AssertionError("No boundary wait when disabled")), \
                    patch("builtins.print"):
                probe.model_probe(args)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("memory_boundaries", result)
            self.assertEqual(result["first_replay"]["decode_tokens"], 2)
            self.assertEqual(len(result["warmups"]), 1)
            self.assertEqual(len(result["measurements"]), 2)
            self.assertEqual(result["runtime_sha256"], hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            for key in ("prefill_ms", "decode_ms", "decode_cpu_ms"):
                self.assertEqual(result["median"][key], sum(row[key] for row in result["measurements"]) / 2)
            self.assertTrue(result["timing_breakdown"]["matches_normal_logits"])
            self.assertEqual(result["profile_replay"]["logits_sha256"], result["logits_sha256"])
            self.assertEqual(len(models), 2)
            self.assertTrue(all(model.closed for model in models))
            self.assertEqual(options, [{"capacity": 1280, "precision": "fp16", "threads": 4, "device_id": 3}] * 2)
            self.assertFalse(any(output.glob("*.onnx")))

    def test_memory_boundaries_follow_lifecycle_outside_normal_replays(self):
        events = []
        raw_sample = {"pids_without_instances": [123], "collect_status": "0x800007d5", "counters": []}
        sampler = Mock(metadata={"source": "test_pdh"})
        sampler.sample.side_effect = lambda: (events.append("sample"), raw_sample)[1]
        sampler.close.side_effect = lambda: events.append("sampler_close")
        factory = Mock(return_value=sampler)
        model = Model()
        model.release_request_state = lambda: events.append("release")
        model.close = lambda: events.append("close")

        def load(*args, **kwargs):
            events.append("load")
            return model

        def replay(*args):
            events.append("replay")
            self.assertEqual(events.count("sample"), 1)
            return {"prefill_ms": 1., "decode_ms": 2., "decode_cpu_ms": 3.}, np.asarray([[1., 2.]])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe"
            args = SimpleNamespace(output=output, model=Path("model"), result=Path("saved.json"),
                case="short", precision="fp16", capacity=1280, device_id=0, threads=4,
                steps=0, warmups=1, repeats=2, sampling=False, profile=False, timing_breakdown=False,
                memory_boundaries=True)
            process = SimpleNamespace(pid=123, memory_info=lambda: SimpleNamespace(rss=100, private=80))
            with patch("cpu_gpt_ort.load_inputs", return_value=(Path("gpt"), (), [3], {})), \
                    patch("cpu_gpt_profile.replay", side_effect=replay), \
                    patch.object(probe.StaticDirectMLGPT, "load", side_effect=load), \
                    patch.object(probe.time, "sleep") as sleep, \
                    patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(__version__="test"),
                        "psutil": SimpleNamespace(Process=lambda: process),
                        "windows_wddm_memory": SimpleNamespace(WDDMMemorySampler=factory)}), \
                    patch("builtins.print"):
                probe.model_probe(args)
            factory.assert_called_once_with([123], include_adapters=False)
            self.assertEqual(events, ["sample", "load", *["replay"] * 4, "sample",
                "release", "sample", "close", "sample", "sampler_close"])
            self.assertEqual([call.args for call in sleep.call_args_list], [(.15,)] * 4)
            samples = json.loads((output / "result.json").read_text())["memory_boundaries"]["samples"]
            self.assertEqual([row["boundary"] for row in samples], ["before_load", "after_replays",
                "after_release_request_state", "after_close"])
            self.assertTrue(all(row["wddm"] == raw_sample for row in samples))
            self.assertTrue(all(row["rss_bytes"] == 100 and row["private_bytes"] == 80 for row in samples))


class SessionProbeTests(unittest.TestCase):
    def test_session_disables_constructor_fallback_and_keeps_requested_adapter(self):
        session = Mock()
        session.get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
        constructor = Mock(return_value=session)
        with patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(InferenceSession=constructor)}), \
                patch.object(probe, "session_options", return_value="options"):
            self.assertIs(probe.make_session(Path("graph.onnx"), 3), session)
            self.assertFalse(constructor.call_args.kwargs["enable_fallback"])
            self.assertEqual(constructor.call_args.kwargs["providers"][0],
                ("DmlExecutionProvider", {"device_id": "3"}))
            session.disable_fallback.assert_not_called()
            session.get_providers.return_value = ["CPUExecutionProvider"]
            with self.assertRaisesRegex(RuntimeError, "did not activate DmlExecutionProvider"):
                probe.make_session(Path("graph.onnx"), 3)

    def test_tiny_probe_reuses_selected_session_outputs_with_internal_ordinal_zero(self):
        bindings, outputs = [], []
        session = Mock()

        def make_binding():
            binding = Mock()
            binding.inputs = {}
            binding.bind_cpu_input.side_effect = lambda name, value: binding.inputs.__setitem__(name, value)
            binding.bind_ortvalue_input.side_effect = lambda name, value: binding.inputs.__setitem__(name, value)
            bindings.append(binding)
            return binding

        def run(binding):
            cache = binding.inputs["cache"]
            actual = (cache if isinstance(cache, np.ndarray) else cache.data).copy()
            actual[binding.inputs["index"].reshape(-1)] = binding.inputs["update"]
            value = SimpleNamespace(data=actual, device_name=lambda: "DML", shape=lambda: list(actual.shape))
            outputs.append(value)
            binding.get_outputs.return_value = [value]
            binding.copy_outputs_to_cpu.return_value = [actual]

        session.io_binding.side_effect = make_binding
        session.run_with_iobinding.side_effect = run
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe"
            with patch.dict(sys.modules, {"onnxruntime": SimpleNamespace(__version__="test")}), \
                    patch.object(probe, "make_session", return_value=session) as create, \
                    patch.object(probe, "profile_summary", return_value={}), patch("builtins.print"):
                probe.tiny_probe(output, 3)
            create.assert_called_once_with(output / "tiny.onnx", 3, profile_prefix=output / "profile")
            self.assertIsInstance(bindings[0].inputs["cache"], np.ndarray)
            for index in (1, 2):
                self.assertIs(bindings[index].inputs["cache"], outputs[index - 1])
            for binding in bindings:
                binding.bind_output.assert_called_once_with("next_cache", "dml", 0)
            for binding in bindings[:-1]:
                binding.copy_outputs_to_cpu.assert_not_called()
            bindings[-1].copy_outputs_to_cpu.assert_called_once_with()
            result = json.loads((output / "result.json").read_text())
            self.assertTrue(result["correct"])
            self.assertNotIn("direct_allocation_error", result)


if __name__ == "__main__":
    unittest.main()
