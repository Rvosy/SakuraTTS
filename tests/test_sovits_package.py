"""Small archive-contract checks independent of MLX and real model resources."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sakuratts.backends.mlx.sovits_package import SoVITSPackage
from sakuratts.module.weight_storage import read_fp32


class SoVITSPackageTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)

    def package(self, arrays):
        np.savez(self.path / 'weights.npz', **arrays)
        manifest = dict(format='sakuratts-sovits-decode-fp32-v1', dtype='float32',
                        config={'model': {'version': 'v2Pro'}},
                        weights={'file': 'weights.npz'})
        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest):
        (self.path / 'manifest.json').write_text(json.dumps(manifest))

    def test_streams_selected_exact_fp32_and_closes_archive(self):
        self.package({'enc_p.a': np.array([1.5], dtype=np.float16),
                      'flow.a': np.array([2.25], dtype=np.float32)})
        with SoVITSPackage.open(self.path) as source:
            selected = dict(source.tensors('enc_p.'))
            archive = source._archive
        self.assertEqual(set(selected), {'enc_p.a'})
        self.assertEqual(selected['enc_p.a'].dtype, np.float32)
        np.testing.assert_array_equal(selected['enc_p.a'], np.array([1.5], dtype=np.float32))
        self.assertIsNone(archive.zip)

    def test_excluded_weights_are_not_read_even_when_named_and_prefix_selected(self):
        self.package({'flow.condition': np.array([1], dtype=np.float32),
                      'flow.other': np.array([2], dtype=np.float32),
                      'dec.condition': np.array([3], dtype=np.float32)})
        with SoVITSPackage.open(self.path) as source:
            with patch('sakuratts.backends.mlx.sovits_package.read_fp32', wraps=read_fp32) as read:
                selected = dict(source.tensors('flow.', names=('flow.condition', 'dec.condition'),
                                               exclude=('flow.condition', 'dec.condition')))
        self.assertEqual(set(selected), {'flow.other'})
        self.assertEqual([call.args[-1] for call in read.call_args_list], ['flow.other'])

if __name__ == '__main__':
    unittest.main()
