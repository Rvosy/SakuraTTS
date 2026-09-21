"""Portable evidence integrity and comparison boundaries; no inference models."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HARNESS = Path(__file__).resolve().parents[2] / "research/tools"
sys.path.insert(0, str(HARNESS))
from portable_validation import (ACOUSTIC_STAGES, BUNDLE_FORMAT, CANDIDATE_FORMAT,
                                 array_spec, compare, load_bundle, load_candidate, sha256_file)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def case_arrays(steps=12):
    length = steps - 1
    tokens = np.ones(steps, dtype=np.int64)
    tokens[-1] = 1024
    prompt = np.asarray([[2, 3, 4]], dtype=np.int64)
    history = np.concatenate((prompt[0], tokens[:-1]))
    arrays = {
        "phones": np.asarray([[1, 2, 3, 4, 5]], dtype=np.int64),
        "prompt": prompt, "bert": np.zeros((1, 5, 1024), dtype=np.float32),
        "tokens": tokens, "history": history, "semantic": history[-length:][None, None, :].copy(),
        "logits": np.zeros((steps, 1025), dtype=np.float32),
        "acoustic_phones": np.asarray([[4, 5]], dtype=np.int64),
        "ge": np.zeros((1, 1024, 1), dtype=np.float32),
        "ge512": np.zeros((1, 512, 1), dtype=np.float32),
        "noise": np.zeros((1, 192, 2 * length), dtype=np.float32),
    }
    for index in range(steps):
        width = 1024 if index < 11 else 1025
        arrays[f"draw.{index}"] = np.ones((1, width), dtype=np.float32)
        probability = np.zeros((1, width), dtype=np.float32)
        if index == steps - 1:
            probability[0, 1024] = 1
        else:
            probability[0, 1:3] = [.6, .4]
        arrays[f"prob.{index}"] = probability
    for stage in ACOUSTIC_STAGES:
        shape = ((1, 768, length) if stage == "quantized" else
                 (1, 192, 2) if stage == "text_encoded" else
                 (1, 1, 2 * length) if stage == "mask" else
                 (1, 1, 1280 * length) if stage == "waveform" else (1, 192, 2 * length))
        arrays["acoustic." + stage] = np.zeros(shape, dtype=np.float32)
    return arrays


def candidate_arrays(oracle):
    mapping = {"fixed_logits": "logits", "own_logits": "logits", "own_tokens": "tokens",
               "own_history": "history", "own_semantic": "semantic", "own_waveform": "acoustic.waveform"}
    mapping.update({"fixed_acoustic." + stage: "acoustic." + stage for stage in ACOUSTIC_STAGES})
    mapping.update({"own_" + key: key for key in oracle if key.startswith("prob.")})
    return {key: oracle[source].copy() for key, source in mapping.items()}


class PortableValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle_dir, self.candidate_dir = self.root / "bundle", self.root / "candidate"
        self.oracle = case_arrays()
        self.actual = candidate_arrays(self.oracle)
        self.parameters = dict(eos=1024, top_k=15, top_p=1.0, early_stop_num=2700, temperature=1.0,
                               repetition_penalty=1.35, speed=1.0, noise_scale=.5, fragment_interval=.3, sample_rate=32000)
        self.models = {name: dict(manifest_sha256="a" * 64, weights_sha256="b" * 64,
                                 checkpoint_sha256="c" * 64, weights_file="weights.npz") for name in ("gpt", "sovits")}
        self.bundle_manifest = dict(format=BUNDLE_FORMAT, official_commit="d" * 40,
            scope="Synthetic Japanese shape and comparison fixtures", external_models=self.models,
            source_sha256={}, known_failures=[], cases={})
        self.candidate_manifest = dict(format=CANDIDATE_FORMAT, status="completed", external_models=copy.deepcopy(self.models),
            execution=dict(backend="test-double", device="cpu", precision="fp32"), source_sha256={}, cases={})
        for directory, manifest in ((self.bundle_dir, self.bundle_manifest), (self.candidate_dir, self.candidate_manifest)):
            (directory / "sources").mkdir(parents=True)
            source = directory / "sources/original.json"
            write_json(source, {"原文": "こんにちは。", "descriptive_path": "Z:/missing/history.json"})
            manifest["source_sha256"] = {"sources/original.json": sha256_file(source)}
        self.save()

    def row(self, directory, arrays):
        path = directory / "case.npz"
        np.savez(path, **arrays)
        return dict(file=path.name, sha256=sha256_file(path), arrays={key: array_spec(value) for key, value in arrays.items()},
                    parameters=dict(self.parameters), stop=dict(returned_index=self.oracle["tokens"].size - 1,
                    reasons=["sample_eos", "argmax_eos"]))

    def save(self):
        self.bundle_manifest["cases"] = {"ja": dict(self.row(self.bundle_dir, self.oracle), text="こんにちは。",
            language="ja", normalized_text="こんにちは。", provenance={"description": "Synthetic fixture, no model"})}
        write_json(self.bundle_dir / "manifest.json", self.bundle_manifest)
        self.candidate_manifest["bundle_manifest_sha256"] = sha256_file(self.bundle_dir / "manifest.json")
        self.candidate_manifest["cases"] = {"ja": dict(self.row(self.candidate_dir, self.actual), status="completed")}
        write_json(self.candidate_dir / "manifest.json", self.candidate_manifest)

    def compare_saved(self):
        bundle = load_bundle(self.bundle_dir)
        return compare(bundle, load_candidate(self.candidate_dir, bundle))

    def test_relocated_utf8_bundle_and_cli_do_not_need_original_paths_or_models(self):
        moved = self.root / "別の場所 with spaces"
        shutil.copytree(self.bundle_dir, moved)
        shutil.rmtree(self.bundle_dir)
        bundle = load_bundle(moved)
        self.assertEqual(bundle["manifest"]["cases"]["ja"]["text"], "こんにちは。")
        self.assertEqual(bundle["external_models_status"], dict(gpt="external_models_not_checked", sovits="external_models_not_checked"))
        report = compare(bundle, load_candidate(self.candidate_dir, bundle))
        self.assertEqual(report["status"], "passed")
        process = subprocess.run([sys.executable, str(HARNESS / "portable_validation.py"), "verify", "--bundle", str(moved)],
                                 text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "verified")

    def test_empty_case_sets_or_incomplete_execution_cannot_pass(self):
        for field in ("cases", "status", "case_status"):
            with self.subTest(field=field):
                current = copy.deepcopy(self.candidate_manifest)
                if field == "cases":
                    current["cases"] = {}
                elif field == "status":
                    del current["status"]
                else:
                    current["cases"]["ja"]["status"] = "partial"
                write_json(self.candidate_dir / "manifest.json", current)
                with self.assertRaises(ValueError):
                    load_candidate(self.candidate_dir, load_bundle(self.bundle_dir))
        current = copy.deepcopy(self.bundle_manifest)
        current["cases"] = {}
        write_json(self.bundle_dir / "manifest.json", current)
        with self.assertRaisesRegex(ValueError, "no cases"):
            load_bundle(self.bundle_dir)

    def test_missing_stage_or_probability_is_rejected_even_with_consistent_hashes(self):
        for key in ("fixed_acoustic.mrte", "own_prob.11", "own_waveform"):
            with self.subTest(key=key):
                original = self.actual.pop(key)
                self.save()
                with self.assertRaisesRegex(ValueError, "Missing required array"):
                    self.compare_saved()
                self.actual[key] = original

    def test_self_consistent_wrong_shape_dtype_or_nonfinite_data_is_rejected(self):
        original = self.actual["fixed_logits"]
        for replacement in (original[:, :-1], original.astype(np.int64), np.full(original.shape, np.nan, dtype=np.float32)):
            with self.subTest(shape=replacement.shape, dtype=replacement.dtype):
                self.actual["fixed_logits"] = replacement
                self.save()
                with self.assertRaisesRegex(ValueError, "shape|Nonfinite"):
                    self.compare_saved()

    def test_relative_paths_reject_traversal_and_absolute_drives(self):
        for name in ("../case.npz", "/tmp/case.npz", "C:/case.npz", "C:case.npz", "cases\\case.npz", "./case.npz"):
            with self.subTest(path=name):
                current = copy.deepcopy(self.bundle_manifest)
                current["cases"]["ja"]["file"] = name
                write_json(self.bundle_dir / "manifest.json", current)
                with self.assertRaisesRegex(ValueError, "relative path"):
                    load_bundle(self.bundle_dir)

    def test_relative_paths_reject_symlink_escape(self):
        link = self.bundle_dir / "escape.npz"
        try:
            link.symlink_to(self.candidate_dir / "case.npz")
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows requires Developer Mode or symlink privileges for this check")
            raise
        current = copy.deepcopy(self.bundle_manifest)
        current["cases"]["ja"]["file"] = link.name
        write_json(self.bundle_dir / "manifest.json", current)
        with self.assertRaisesRegex(ValueError, "escapes"):
            load_bundle(self.bundle_dir)

    def test_archive_raw_array_and_source_hashes_are_all_enforced(self):
        for target in ("archive", "array", "source"):
            with self.subTest(target=target):
                current = copy.deepcopy(self.bundle_manifest)
                if target == "archive":
                    current["cases"]["ja"]["sha256"] = "0" * 64
                elif target == "array":
                    current["cases"]["ja"]["arrays"]["logits"]["sha256_raw_c_order"] = "0" * 64
                else:
                    current["source_sha256"]["sources/original.json"] = "0" * 64
                write_json(self.bundle_dir / "manifest.json", current)
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    load_bundle(self.bundle_dir)

    def test_candidate_bundle_model_and_parameter_bindings_are_enforced(self):
        for field in ("bundle", "model", "parameters"):
            with self.subTest(field=field):
                current = copy.deepcopy(self.candidate_manifest)
                if field == "bundle":
                    current["bundle_manifest_sha256"] = "0" * 64
                elif field == "model":
                    current["external_models"]["gpt"]["weights_sha256"] = "0" * 64
                else:
                    current["cases"]["ja"]["parameters"]["temperature"] = 2
                write_json(self.candidate_dir / "manifest.json", current)
                with self.assertRaises(ValueError):
                    self.compare_saved()

    def test_first_divergent_token_step_is_compared_later_history_is_not(self):
        self.actual["own_tokens"][2] = 7
        self.actual["own_logits"][3:] = 100
        self.actual["own_prob.3"][0, 1:3] = [.1, .9]
        self.save()
        result = self.compare_saved()
        generation = result["cases"]["ja"]["own_generation"]
        self.assertEqual(result["status"], "numerical_mismatch")
        self.assertEqual(generation["first_token_divergence"], 2)
        self.assertEqual(generation["aligned_steps"], 3)
        self.assertEqual(generation["skipped_different_history_steps"], 9)
        self.assertTrue(generation["checks"]["aligned_logits_within_tolerance"])
        self.assertTrue(generation["checks"]["aligned_probabilities_within_tolerance"])
        self.actual["own_logits"][2, 10] = .001
        self.save()
        self.assertFalse(self.compare_saved()["cases"]["ja"]["own_generation"]["checks"]["aligned_logits_within_tolerance"])

    def test_known_probability_and_mrte_failures_are_not_waived(self):
        self.oracle = case_arrays(276)
        self.oracle["prob.274"][:] = 0
        self.oracle["prob.274"][0, 1] = .8546
        self.oracle["prob.274"][0, 857] = .1454
        self.oracle["acoustic.mrte"][0, 21, 125] = np.float32(2.363581895828247)
        self.actual = candidate_arrays(self.oracle)
        self.actual["own_prob.274"][0, 857] += np.float32(3.59118e-6)
        self.actual["own_prob.274"][0, 1] -= np.float32(3.59118e-6)
        self.actual["fixed_acoustic.mrte"][0, 21, 125] = np.float32(2.363433837890625)
        self.bundle_manifest["known_failures"] = [{"case": "ja", "scope": "own_generation"}, {"case": "ja", "scope": "fixed_acoustic"}]
        self.save()
        result = self.compare_saved()
        self.assertEqual(result["status"], "numerical_mismatch")
        row = result["cases"]["ja"]
        self.assertTrue(row["own_generation"]["checks"]["tokens_equal"])
        self.assertTrue(row["own_waveform"]["within_tolerance"])
        probability = row["own_generation"]["probability_steps"][274]["probabilities"]
        self.assertFalse(probability["within_tolerance"])
        self.assertIn([0, 857], probability["outside_tolerance_indices"])
        self.assertIn([0, 21, 125], row["fixed_acoustic"]["mrte"]["outside_tolerance_indices"])

    def test_changed_own_semantics_skip_waveform_numerics_but_keep_case_failed(self):
        self.actual["own_semantic"][0, 0, 0] = 7
        self.actual["own_waveform"][:] = 100
        self.save()
        result = self.compare_saved()
        row = result["cases"]["ja"]
        self.assertEqual(result["status"], "numerical_mismatch")
        self.assertFalse(row["own_generation"]["checks"]["semantic_equal"])
        self.assertEqual(row["own_waveform"], {"status": "skipped_different_semantic", "within_tolerance": None})
        self.assertTrue(all(stage["within_tolerance"] for stage in row["fixed_acoustic"].values()))

    def test_external_model_checks_hash_files_without_loading_weights(self):
        directories = {}
        for name in self.models:
            directory = self.root / (name + "-model")
            directory.mkdir()
            weights = directory / "weights.npz"
            weights.write_bytes(b"checksum only, no model or archive parsing")
            model = dict(weights=dict(file=weights.name, sha256=sha256_file(weights)),
                         source=dict(checkpoint_sha256="c" * 64, official_commit="d" * 40))
            write_json(directory / "manifest.json", model)
            self.models[name].update(manifest_sha256=sha256_file(directory / "manifest.json"), weights_sha256=sha256_file(weights))
            directories[name + "_package"] = directory
        self.save()
        verified = load_bundle(self.bundle_dir, **directories)
        self.assertEqual(verified["external_models_status"], dict(gpt="verified", sovits="verified"))
        (directories["gpt_package"] / "weights.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "weights hash mismatch"):
            load_bundle(self.bundle_dir, **directories)


if __name__ == "__main__":
    unittest.main()
