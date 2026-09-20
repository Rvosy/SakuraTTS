"""Small archive-contract checks independent of MLX and real model resources."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sakuratts.backends.mlx.sovits_package import SoVITSPackage, sha256
from sakuratts._internal.weight_storage import LOSSLESS_STORAGE, array_sha256, read_fp32


class SoVITSPackageTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)

    def package(self, arrays, *, compact=False):
        np.savez(self.path / 'weights.npz', **arrays)
        manifest = dict(format='sakuratts-sovits-decode-fp32-v1', dtype='float32',
                        config={'model': {'version': 'v2Pro'}},
                        weights={'file': 'weights.npz', 'sha256': sha256(self.path / 'weights.npz')},
                        tensor_sources={k: {'shape': list(v.shape)} for k, v in arrays.items()})
        if compact:
            manifest['weights']['storage'] = dict(format=LOSSLESS_STORAGE, runtime_dtype='float32', tensors={
                k: dict(storage_dtype=str(v.dtype), expanded_dtype='float32', shape=list(v.shape),
                        storage_sha256_raw_c_order=array_sha256(v),
                        expanded_fp32_sha256_raw_c_order=array_sha256(v.astype(np.float32)))
                for k, v in arrays.items()})
        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest):
        (self.path / 'manifest.json').write_text(json.dumps(manifest))

    def test_streams_selected_exact_fp32_and_closes_archive(self):
        self.package({'enc_p.a': np.array([1.5], dtype=np.float16),
                      'flow.a': np.array([2.25], dtype=np.float32)}, compact=True)
        with SoVITSPackage.open(self.path) as source:
            selected = dict(source.tensors('enc_p.'))
            archive = source._archive
        self.assertEqual(set(selected), {'enc_p.a'})
        self.assertEqual(selected['enc_p.a'].dtype, np.float32)
        np.testing.assert_array_equal(selected['enc_p.a'], np.array([1.5], dtype=np.float32))
        self.assertIsNone(archive.zip)

    def test_corrupt_archive_rejected_before_read(self):
        self.package({'flow.a': np.array([1], dtype=np.float32)})
        with (self.path / 'weights.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            with SoVITSPackage.open(self.path):
                self.fail('corrupt package was opened')

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

    def test_undeclared_tensor_rejected(self):
        manifest = self.package({'flow.a': np.array([1], dtype=np.float32)})
        manifest['tensor_sources'] = {}
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'tensor set'):
            with SoVITSPackage.open(self.path):
                self.fail('undeclared weight was accepted')

    def test_changed_expansion_identity_is_rejected(self):
        manifest = self.package({'flow.a': np.array([1], dtype=np.float16)}, compact=True)
        manifest['weights']['storage']['tensors']['flow.a']['expanded_fp32_sha256_raw_c_order'] = '0' * 64
        self.write_manifest(manifest)
        with SoVITSPackage.open(self.path) as source:
            with self.assertRaisesRegex(ValueError, 'Expanded FP32 tensor checksum'):
                dict(source.tensors('flow.'))


if __name__ == '__main__':
    unittest.main()
