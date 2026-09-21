"""CPU byte-addressing and synchronization checks; never load the CUDA DLL."""

import ctypes
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/tools"), str(ROOT / "tools"), str(ROOT / "src")]
from windows_device_latent import CudaCopies, copy_layout


class HostPointer:
    """The minimal data_ptr interface, backed by a retained CPU array."""

    def __init__(self, array):
        self.array = array

    def data_ptr(self):
        return self.array.ctypes.data


class DeferredCudaLibrary:
    """Model D2D host-asynchronous completion without touching a CUDA runtime."""

    def __init__(self, *, copy_error=0, sync_error=0):
        self.copy_error, self.sync_error = copy_error, sync_error
        self.events, self.pending = [], []

    def cudaMemcpy2D(self, destination, destination_pitch, source, source_pitch, width, height, kind):
        arguments = (destination, destination_pitch, source, source_pitch, width, height, kind)
        self.events.append(("copy", arguments))
        if self.copy_error:
            return self.copy_error

        def complete():
            for row in range(height):
                ctypes.memmove(destination + row * destination_pitch, source + row * source_pitch, width)

        self.pending.append(complete)
        return 0

    def cudaDeviceSynchronize(self):
        self.events.append(("synchronize",))
        if self.sync_error:
            return self.sync_error
        for operation in self.pending:
            operation()
        self.pending.clear()
        return 0

    def cudaGetErrorString(self, status):
        self.events.append(("error", status))
        return f"synthetic CUDA failure {status}".encode("utf-8")


def copies_with_fake_library(**kwargs):
    copies = CudaCopies.__new__(CudaCopies)
    copies.library = DeferredCudaLibrary(**kwargs)
    return copies


class DeviceLatentCopyTests(unittest.TestCase):
    def test_rectangular_pack_matches_independent_channel_time_slicing(self):
        for dtype in (np.uint16, np.uint32):
            itemsize = np.dtype(dtype).itemsize
            for total in (1, 2, 11, 33, 65, 129, 257, 511, 1350):
                source = ((np.arange(192 * total, dtype=np.uint64) * 31 + 17)
                          .astype(dtype).reshape(1, 192, total))
                source_before = source.tobytes()
                intervals = {(0, total), (0, 1), (total - 1, total)}
                for start in (1, 5, 11, 31, 64, 128, 245, 501, 757, 1013):
                    for length in (1, 2, 11, 22, 64, 128, 256, 267, 278):
                        if start + length <= total:
                            intervals.add((start, start + length))
                for start, end in sorted(intervals):
                    with self.subTest(itemsize=itemsize, total=total, start=start, end=end):
                        length = end - start
                        # Guard the destination on both sides. The fake transfer
                        # has no knowledge of NumPy slicing or the expected data.
                        guarded = np.full(192 * length + 16, 0xA55A, dtype)
                        destination = guarded[8:-8].reshape(1, 192, length)
                        copies = copies_with_fake_library()
                        layout = copy_layout(source.shape, start, end, itemsize)
                        copies.pack(HostPointer(destination), HostPointer(source), layout)
                        expected = source[..., start:end].copy(order="C")
                        self.assertEqual(destination.tobytes(), expected.tobytes())
                        self.assertEqual(source.tobytes(), source_before)
                        self.assertTrue(np.all(guarded[:8] == 0xA55A))
                        self.assertTrue(np.all(guarded[-8:] == 0xA55A))
                        events = copies.library.events
                        self.assertEqual([event[0] for event in events], ["copy", "synchronize"])
                        self.assertEqual(events[0][1], (
                            destination.ctypes.data, length * itemsize,
                            source.ctypes.data + start * itemsize, total * itemsize,
                            length * itemsize, 192, 3))
                        self.assertFalse(copies.library.pending)

    def test_interior_slice_is_not_a_contiguous_view_of_all_channels(self):
        source = np.arange(192 * 37, dtype=np.uint16).reshape(1, 192, 37)
        start, end = 5, 16
        expected = source[..., start:end].copy()
        incorrectly_bound = source.reshape(-1)[start:start + expected.size]
        self.assertNotEqual(incorrectly_bound.tobytes(), expected.tobytes())
        destination = np.empty_like(expected)
        copies = copies_with_fake_library()
        copies.pack(HostPointer(destination), HostPointer(source), copy_layout(source.shape, start, end, 2))
        np.testing.assert_array_equal(destination, expected)

    def test_layout_rejects_empty_outside_or_unsupported_tensor_slices(self):
        invalid = [
            ((192, 17), 0, 1, 2),
            ((1, 192, 17, 1), 0, 1, 2),
            ((2, 192, 17), 0, 1, 2),
            ((0, 192, 17), 0, 1, 2),
            ((1, 0, 17), 0, 1, 2),
            ((1, 192, 0), 0, 1, 2),
            ((1, -192, 17), 0, 1, 2),
            ((1, 192, 17), -1, 1, 2),
            ((1, 192, 17), 0, 0, 2),
            ((1, 192, 17), 5, 4, 2),
            ((1, 192, 17), 17, 18, 2),
            ((1, 192, 17), 0, 18, 2),
            ((1, 192, 17), 0, 1, 1),
            ((1, 192, 17), 0, 1, 8),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                copy_layout(*arguments)

    def test_failed_copy_is_reported_before_synchronization_and_does_not_write(self):
        source = np.arange(192 * 17, dtype=np.uint16).reshape(1, 192, 17)
        destination = np.full((1, 192, 11), 0xA55A, np.uint16)
        copies = copies_with_fake_library(copy_error=17)
        with self.assertRaisesRegex(RuntimeError, "synthetic CUDA failure 17"):
            copies.pack(HostPointer(destination), HostPointer(source), copy_layout(source.shape, 3, 14, 2))
        self.assertEqual([event[0] for event in copies.library.events], ["copy", "error"])
        self.assertTrue(np.all(destination == 0xA55A))

    def test_failed_completion_is_reported_instead_of_returning_unready_data(self):
        source = np.arange(192 * 17, dtype=np.uint32).reshape(1, 192, 17)
        destination = np.full((1, 192, 11), 0xA55A, np.uint32)
        copies = copies_with_fake_library(sync_error=719)
        with self.assertRaisesRegex(RuntimeError, "synthetic CUDA failure 719"):
            copies.pack(HostPointer(destination), HostPointer(source), copy_layout(source.shape, 3, 14, 4))
        self.assertEqual([event[0] for event in copies.library.events], ["copy", "synchronize", "error"])
        self.assertTrue(np.all(destination == 0xA55A))
        self.assertEqual(len(copies.library.pending), 1)


if __name__ == "__main__":
    unittest.main()
