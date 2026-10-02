"""Reference binding identity and immutable-condition checks without a backend."""

from dataclasses import replace
from pathlib import Path
import sys
import json
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sakuratts.module.reference_condition import BoundAcousticReference, PreparedReference


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

    def test_local_archive_loads_with_stale_metadata_but_requires_aligned_arrays(self):
        from sakuratts.module.reference_condition import ARRAY_DTYPES, FORMAT
        arrays = {name: getattr(self.reference, name) for name in ARRAY_DTYPES}
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            manifest = dict(self.reference.manifest, format=FORMAT,
                            archive={"file": "custom.npz", "sha256": "stale", "bytes": 0})
            (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.savez(package / "custom.npz", **arrays, unused=np.zeros(1))
            restored = PreparedReference.load(package)
            np.testing.assert_array_equal(restored.ge, self.reference.ge)
            self.assertFalse(restored.ge.flags.writeable)
            arrays["reference_bert"] = arrays["reference_bert"][:, :1]
            np.savez(package / "custom.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "alignment"):
                PreparedReference.load(package)

    def test_bound_conditions_are_independent_of_the_callers_arrays(self):
        bound = BoundAcousticReference.from_reference(self.reference, self.model)
        bound.validate_reference(self.reference)
        bound.validate_conditions(self.reference.ge.copy(), self.reference.ge512.copy())
        self.reference.ge[0, 0, 0] = 2
        with self.assertRaises(ValueError):
            bound.validate_reference(self.reference)
        self.assertEqual(bound.ge[0, 0, 0], 1)
        self.reference.ge[0, 0, 0] = 1
        self.reference.manifest["identity"]["audio_sha256"] = "changed"
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

if __name__ == "__main__":
    unittest.main()
