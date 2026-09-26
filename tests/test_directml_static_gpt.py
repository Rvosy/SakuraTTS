"""Static cache math on real small CPU graphs; device I/O is simulated.

The simulated binder copies host inputs at bind time, as DirectML may do. This
checks the first-request stale-input failure without requiring a GPU in CI.
"""
import importlib.util
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from test_cpu_gpt import model_data, save_package, full_prefix
from sakuratts.backends.cpu.onnx_gpt import ONNXCPUGPT
from sakuratts._internal.reference_condition import sha256_file
from sakuratts.backends.directml.static_gpt import StaticDirectMLGPT, read_static_sidecar, static_directory


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime"))
CREATE_SESSION = StaticDirectMLGPT._create_session


class DeviceValue:
    def __init__(self, value, adapter=None):
        self.value, self.adapter = value, adapter

    @staticmethod
    def ortvalue_from_numpy(value, device="cpu", *args):
        if device != "cpu":
            raise AssertionError("GPU buffers must be allocated by the selected session")
        return DeviceValue(value)

    @staticmethod
    def ortvalue_from_shape_and_type(shape, dtype, *args):
        raise AssertionError("GPU buffers must be allocated by the selected session")

    def numpy(self):
        return self.value


class Binding:
    def __init__(self, adapter):
        self.adapter = adapter
        self.inputs, self.outputs = {}, {}
        self.auto_outputs = set()
        self.get_outputs_calls = 0

    def bind_cpu_input(self, name, value):
        self.inputs[name] = value.copy()

    def bind_ortvalue_input(self, name, value):
        # An ORT partition may consume a device input through a host copy.
        if value.adapter is not None and value.adapter != self.adapter:
            raise AssertionError("KV input belongs to a different adapter")
        self.inputs[name] = value.value.copy()

    def bind_output(self, name, device, device_id):
        if (device, device_id) != ("dml", 0):
            raise AssertionError("Output must use the session-local DML ordinal")
        self.outputs[name] = None
        self.auto_outputs.add(name)

    def bind_ortvalue_output(self, name, value):
        if value.adapter is not None and value.adapter != self.adapter:
            raise AssertionError("KV output belongs to a different adapter")
        self.outputs[name] = value
        self.auto_outputs.discard(name)

    def synchronize_outputs(self):
        pass

    def get_outputs(self):
        self.get_outputs_calls += 1
        return list(self.outputs.values())

    def clear_binding_inputs(self):
        self.inputs.clear()

    def clear_binding_outputs(self):
        self.outputs.clear()
        self.auto_outputs.clear()


class DeviceSession:
    def __init__(self, graph, adapter):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
        self.adapter, self.bindings = adapter, []
        self.allocations = 0
        self.allocated_values = []

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def io_binding(self):
        binding = Binding(self.adapter)
        self.bindings.append(binding)
        return binding

    def run_with_iobinding(self, binding):
        names = list(binding.outputs)
        results = self.session.run(names, binding.inputs)
        for name, value in zip(names, results):
            if name in binding.auto_outputs:
                binding.outputs[name] = DeviceValue(value, self.adapter)
                self.allocations += 1
                self.allocated_values.append(weakref.ref(binding.outputs[name]))
            else:
                binding.outputs[name].value[...] = value


@unittest.skipUnless(AVAILABLE, "ONNX preparation and runtime dependencies are optional")
class StaticDirectMLGPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sakuratts._internal.conversion.export_gpt_onnx import export_sidecar
        from sakuratts._internal.conversion.export_gpt_directml import export_sidecar as export_static
        cls.folder = tempfile.TemporaryDirectory()
        cls.manifest, cls.weights = model_data()
        cls.root = save_package(cls.folder.name, cls.manifest, cls.weights)
        export_sidecar(cls.root, cls.root / "onnx")
        export_sidecar(cls.root, cls.root / "onnx-fp16", precision="fp16")
        for precision in ("fp32", "fp16"):
            export_static(cls.root, precision=precision, capacity=8)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        self.phones = np.array([[2, 5, 1]], np.int64)
        self.prompt = np.array([[3, 0]], np.int64)
        self.bert = np.random.default_rng(2).normal(size=(1, 3, 5)).astype(np.float32)
        self.patches = [patch("sakuratts.backends.directml.static_gpt.DirectMLGPT._from_sidecar", side_effect=ONNXCPUGPT._from_sidecar),
            patch.object(StaticDirectMLGPT, "_create_session", autospec=True,
                side_effect=lambda model, graph: DeviceSession(graph, model.device_id)),
            patch("onnxruntime.OrtValue", DeviceValue)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_first_request_and_reset_match_independent_full_prefix(self):
        for precision, tolerance in (("fp32", 1e-5), ("fp16", .01)):
            with self.subTest(precision=precision):
                model = StaticDirectMLGPT.load(self.root, capacity=8, precision=precision, threads=1)
                try:
                    first = model.prefill(self.phones, self.prompt, self.bert)
                    audio = self.prompt.copy()
                    np.testing.assert_allclose(first, full_prefix(self.manifest, self.weights, self.phones, audio, self.bert),
                        atol=tolerance, rtol=tolerance)
                    results = []
                    for token in (2, 7, 1):
                        audio = np.concatenate((audio, [[token]]), axis=1)
                        actual = model.decode(token)
                        self.assertEqual(actual.dtype, np.float32)
                        self.assertTrue(np.isfinite(actual).all())
                        np.testing.assert_allclose(actual, full_prefix(self.manifest, self.weights, self.phones, audio, self.bert),
                            atol=tolerance, rtol=tolerance)
                        results.append(actual)
                    np.testing.assert_array_equal(first, model.prefill(self.phones, self.prompt, self.bert))
                    for token, expected in zip((2, 7, 1), results):
                        np.testing.assert_array_equal(model.decode(token), expected)
                finally:
                    model.close()

    def test_reordered_graph_outputs_follow_binding_order(self):
        import onnx

        with tempfile.TemporaryDirectory() as folder:
            root = Path(shutil.copytree(self.root, Path(folder) / "package"))
            sidecar = static_directory(root, "fp32", 8)
            manifest_path = sidecar / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            graph_path = sidecar / manifest["graph"]["file"]
            graph = onnx.load(graph_path)
            reordered = list(reversed(graph.graph.output))
            del graph.graph.output[:]
            graph.graph.output.extend(reordered)
            onnx.save(graph, graph_path)
            manifest["graph"].update(sha256=sha256_file(graph_path), bytes=graph_path.stat().st_size)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            model = StaticDirectMLGPT.load(root, capacity=8, threads=1)
            try:
                self.assertNotEqual(model.session.get_outputs()[0].name, "logits")
                model.prefill(self.phones, self.prompt, self.bert)
                audio = self.prompt.copy()
                for token in (2, 7, 1):
                    audio = np.concatenate((audio, [[token]]), axis=1)
                    actual = model.decode(token)
                    expected = full_prefix(self.manifest, self.weights, self.phones, audio, self.bert)
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

                # Check the same insertion order against native ORT IOBinding,
                # in addition to the simulated device cache lifecycle above.
                executed = model.next_binding
                session = model.session.session
                binding = session.io_binding()
                for name, value in executed.inputs.items():
                    binding.bind_cpu_input(name, value)
                for name in executed.outputs:
                    binding.bind_output(name, "cpu")
                session.run_with_iobinding(binding)
                binding.synchronize_outputs()
                for actual, expected in zip(binding.get_outputs(), executed.outputs.values()):
                    np.testing.assert_array_equal(actual.numpy(), expected.value)
            finally:
                model.close()

    def test_capacity_state_release_and_kernel_failure(self):
        model = StaticDirectMLGPT.load(self.root, capacity=8, threads=1)
        try:
            with self.assertRaisesRegex(RuntimeError, "prefill"):
                model.decode(1)
            model.prefill(self.phones, self.prompt, self.bert)
            for token in (2, 7, 1):
                model.decode(token)
            with self.assertRaisesRegex(ValueError, "capacity"):
                model.decode(2)
            with self.assertRaisesRegex(ValueError, "integer"):
                model.decode(True)
            model.prefill(self.phones, self.prompt, self.bert)
            retained_binding = model.binding
            with patch.object(model.session, "run_with_iobinding", side_effect=RuntimeError("kernel failed")):
                with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                    model.decode(2)
            self.assertIsNone(model.cache)
            self.assertIsNone(model.binding)
            self.assertEqual(retained_binding.inputs, {})
            self.assertEqual(retained_binding.outputs, {})
            with self.assertRaisesRegex(RuntimeError, "prefill"):
                model.decode(1)
            self.assertTrue(np.isfinite(model.prefill(self.phones, self.prompt, self.bert)).all())
            model.release_request_state()
            self.assertIsNone(model.spare)
        finally:
            model.close()
        self.assertEqual(model.embedding, {})
        with self.assertRaisesRegex(RuntimeError, "unloaded"):
            model.prefill(self.phones, self.prompt, self.bert)

    def test_shorter_request_resets_mask_and_uses_selected_adapter_buffers(self):
        model = StaticDirectMLGPT.load(self.root, capacity=8, precision="fp16", threads=1, device_id=3)
        try:
            model.prefill(self.phones, self.prompt, self.bert)
            self.assertTrue(all(value.adapter is None for value in model.cache.values()))
            self.assertIs(model.cache["key.0"].value.base, model.cache["key.1"].value.base)
            self.assertTrue(all(np.count_nonzero(value.value[model.length + 1:]) == 0
                                for value in model.cache.values()))
            seed_values = [weakref.ref(value) for value in model.cache.values()]
            seed_binding = model.binding
            for index, token in enumerate((2, 7, 1), 1):
                model.decode(token)
                self.assertTrue(all(value.adapter == 3 for value in model.cache.values()))
                self.assertEqual(model.session.allocations, min(index, 2) * model.layers * 2)
                self.assertEqual(sum(binding.get_outputs_calls for binding in model.session.bindings), min(index, 2))
                if index == 1:
                    self.assertEqual(seed_binding.inputs, {})
                    self.assertTrue(all(reference() is None for reference in seed_values))
            phones, prompt, bert = self.phones[:, :2], self.prompt[:, :1], self.bert[:, :2]
            old_bindings = list(model.session.bindings)
            model.prefill(phones, prompt, bert)
            for binding in old_bindings:
                self.assertEqual(binding.inputs, {})
                self.assertEqual(binding.outputs, {})
            actual = model.decode(2)
            expected = full_prefix(self.manifest, self.weights, phones,
                np.concatenate((prompt, [[2]]), axis=1), bert)
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
        finally:
            model.close()

    def test_failed_seed_second_allocation_and_reuse_release_all_bindings(self):
        for completed_steps in (0, 1, 2):
            with self.subTest(completed_steps=completed_steps):
                model = StaticDirectMLGPT.load(self.root, capacity=8, threads=1, device_id=3)
                try:
                    model.prefill(self.phones, self.prompt, self.bert)
                    for token in (2, 7)[:completed_steps]:
                        model.decode(token)
                    references = [weakref.ref(value) for value in model.cache.values()]
                    run = model.session.run_with_iobinding

                    def fail_after_outputs(binding):
                        run(binding)
                        raise RuntimeError("kernel failed")

                    with patch.object(model.session, "run_with_iobinding", side_effect=fail_after_outputs):
                        try:
                            model.decode(1)
                        except RuntimeError as error:
                            retained_traceback = error.__traceback__
                            self.assertIn("kernel failed", str(error))
                        else:
                            self.fail("Kernel failure was not reported")
                    gc.collect()
                    self.assertIsNotNone(retained_traceback)
                    self.assertTrue(all(reference() is None for reference in references))
                    self.assertTrue(all(reference() is None for reference in model.session.allocated_values))
                    self.assertIsNone(model.cache)
                    self.assertIsNone(model.spare)
                    self.assertIsNone(model.binding)
                    self.assertIsNone(model.next_binding)
                    for binding in model.session.bindings:
                        self.assertEqual(binding.inputs, {})
                        self.assertEqual(binding.outputs, {})
                    model.prefill(self.phones, self.prompt, self.bert)
                    self.assertTrue(np.isfinite(model.decode(2)).all())
                finally:
                    model.close()

    def test_static_source_capacity_and_graph_hash_are_bound(self):
        root = static_directory(self.root, "fp16", 8)
        path = root / "manifest.json"
        original = path.read_text(encoding="utf-8")
        try:
            value = json.loads(original)
            value["source"]["graph_sha256"] = "different model"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source, precision or capacity"):
                read_static_sidecar(self.root, "fp16", 8)
            value = json.loads(original)
            value["capacity"] = 9
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "capacity"):
                read_static_sidecar(self.root, "fp16", 8)
            value = json.loads(original)
            value["graph"]["sha256"] = "corrupt"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_static_sidecar(self.root, "fp16", 8)
        finally:
            path.write_text(original, encoding="utf-8")

    def test_constructor_failure_releases_prefill_and_sessions(self):
        metadata, graph, prefill = read_static_sidecar(self.root, "fp32", 8)
        model = StaticDirectMLGPT.__new__(StaticDirectMLGPT)
        with patch.object(StaticDirectMLGPT, "_validate_contract", side_effect=ValueError("contract mismatch")):
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                model.__init__(prefill, metadata, graph, 8, "fp32", 1, 0)
        self.assertIsNone(model.session)
        self.assertIsNone(model.prefill_model)
        self.assertEqual(model.embedding, {})

    def test_each_load_checks_executed_resources_once_and_detects_later_changes(self):
        checked = Counter()

        def checksum(path):
            checked[Path(path)] += 1
            return sha256_file(path)

        metadata, graph, prefill = read_static_sidecar(self.root, "fp32", 8)
        resources = (graph, prefill[2], prefill[3])
        with patch("sakuratts.backends.cpu.onnx_gpt.sha256_file", side_effect=checksum), \
                patch("sakuratts.backends.directml.static_gpt.sha256_file", side_effect=checksum):
            for _ in range(2):
                checked.clear()
                model = StaticDirectMLGPT.load(self.root, capacity=8, threads=1)
                model.close()
                for path in resources:
                    self.assertEqual(checked[path], 1, str(path))
            original = graph.read_bytes()
            try:
                graph.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "checksum or size"):
                    StaticDirectMLGPT.load(self.root, capacity=8, threads=1)
            finally:
                graph.write_bytes(original)

    def test_provider_rejection_does_not_retain_unassigned_session(self):
        references = []

        class WrongProvider:
            def disable_fallback(self):
                pass

            def get_providers(self):
                return ["CPUExecutionProvider"]

        def create(*args, **kwargs):
            session = WrongProvider()
            references.append(weakref.ref(session))
            return session

        model = StaticDirectMLGPT.__new__(StaticDirectMLGPT)
        model.threads, model.device_id = 1, 0
        with patch("onnxruntime.InferenceSession", side_effect=create):
            try:
                CREATE_SESSION(model, "unused.onnx")
            except RuntimeError as error:
                retained_traceback = error.__traceback__
                self.assertIn("did not activate", str(error))
                gc.collect()
                self.assertIsNone(references[0]())
                self.assertIsNotNone(retained_traceback)
            else:
                self.fail("Wrong provider was accepted")


if __name__ == "__main__":
    unittest.main()
