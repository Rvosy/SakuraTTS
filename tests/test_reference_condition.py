"""Reference binding identity and immutable-condition checks without a backend."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts._internal.reference_condition import BoundAcousticReference, PreparedReference


class BoundReferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = dict(source=dict(checkpoint_sha256="sovits", official_commit="official"),
                          config=dict(model=dict(version="v2Pro")))
        self.reference = PreparedReference(
            dict(model_family="v2Pro", identity=dict(reference_language="ja",
                 sovits_checkpoint_sha256="sovits", gpt_checkpoint_sha256="gpt",
                 official_commit="official", audio_sha256="audio", reference_text="こんにちは。")),
            np.array([1, 2], dtype=np.int64), np.array([3], dtype=np.int64),
            np.zeros((1024, 2), dtype=np.float32),
            np.ones((1, 1024, 1), dtype=np.float32), np.ones((1, 512, 1), dtype=np.float32),
        )

    def test_mutating_original_manifest_or_arrays_cannot_rebind_loaded_instance(self):
        bound = BoundAcousticReference.from_reference(self.reference, self.model)
        bound.validate_reference(self.reference)
        bound.validate_conditions(self.reference.ge.copy(), self.reference.ge512.copy())
        self.reference.ge[0, 0, 0] = 2
        with self.assertRaises(ValueError):
            bound.validate_reference(self.reference)
        self.assertEqual(bound.ge[0, 0, 0], 1)
        self.reference.ge[0, 0, 0] = 1
        self.reference.manifest["identity"]["audio_sha256"] = "changed"
        with self.assertRaises(ValueError):
            bound.validate_reference(self.reference)
        with self.assertRaises(ValueError):
            bound.ge.setflags(write=True)
        with self.assertRaises(ValueError):
            bound.ge512.setflags(write=True)

    def test_each_condition_requires_exact_dtype_shape_and_content(self):
        bound = BoundAcousticReference.from_reference(self.reference, self.model)
        for name in ("ge", "ge512"):
            original = getattr(self.reference, name)
            changed = original.copy()
            changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.float32(2))
            for invalid in (original.astype(np.float64), original.reshape(-1), changed):
                with self.subTest(name=name, shape=invalid.shape, dtype=invalid.dtype):
                    other = replace(self.reference, **{name: invalid})
                    with self.assertRaises(ValueError):
                        bound.validate_conditions(other.ge, other.ge512)

    def test_another_transcript_with_same_acoustic_arrays_is_not_same_reference(self):
        bound = BoundAcousticReference.from_reference(self.reference, self.model)
        other = replace(self.reference, manifest=dict(self.reference.manifest,
                        identity=dict(self.reference.manifest["identity"], reference_text="今日は晴れ。")))
        with self.assertRaises(ValueError):
            bound.validate_reference(other)

    def test_invalid_model_language_and_nonfinite_conditions_fail_at_binding(self):
        for field in ("sovits_checkpoint_sha256", "official_commit", "reference_language"):
            other = replace(self.reference, manifest=dict(self.reference.manifest,
                            identity=dict(self.reference.manifest["identity"], **{field: "other"})))
            with self.subTest(field=field), self.assertRaises(ValueError):
                BoundAcousticReference.from_reference(other, self.model)
        invalid = self.reference.ge.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            BoundAcousticReference.from_reference(replace(self.reference, ge=invalid), self.model)


if __name__ == "__main__":
    unittest.main()
