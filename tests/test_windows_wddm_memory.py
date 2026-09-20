"""CPU checks for WDDM identity, validity, churn, and cleanup boundaries."""

import ctypes as ct
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("windows_wddm_memory", ROOT / "research/tools/windows_wddm_memory.py")
wddm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wddm)
NAME = "pid_123_luid_0x00000000_0x0001DF70_phys_0"


class WDDMMemoryTests(unittest.TestCase):
    def test_identity_preserves_adapter_and_duplicate_suffix(self):
        identity = wddm.parse_instance(NAME + "#2")
        self.assertEqual(identity, {"pid": 123, "luid": "0x00000000_0x0001df70",
                                    "physical_adapter": 0, "duplicate_index": 2})
        adapter = wddm.parse_instance("luid_0xffffffff_0x1234_phys_1")
        self.assertIsNone(adapter["pid"])
        self.assertEqual(adapter["luid"], "0xffffffff_0x00001234")
        self.assertEqual(adapter["physical_adapter"], 1)
        for invalid in ("_Total", NAME + "_eng_0", NAME + "junk", "pid_bad_luid_0x0_0x0_phys_0"):
            self.assertIsNone(wddm.parse_instance(invalid))

    def test_filter_keeps_multi_adapter_rows_and_flags_duplicate_identities(self):
        items = [(NAME, 0, 100), (NAME + "#1", 1, 100),
                 (NAME.replace("1DF70", "1F046"), 0, 25),
                 (NAME.replace("pid_123", "pid_124"), 0, 500), ("future_format", 0, 30)]
        rows, unparsed = wddm.normalize_counter(items, scope="process", pids={123})
        self.assertEqual([row["bytes"] for row in rows], [100, 100, 25])
        self.assertEqual([row["duplicate_identity"] for row in rows], [True, True, False])
        self.assertEqual(unparsed, ["future_format"])
        lone_suffix, _ = wddm.normalize_counter([(NAME + "#1", 0, 1)], scope="process", pids={123})
        self.assertTrue(lone_suffix[0]["duplicate_identity"])

    def test_absent_invalid_and_actual_zero_are_distinct(self):
        self.assertEqual(wddm.normalize_counter([], scope="process", pids={123}), ([], []))
        for status, value, valid, expected in ((0, 0, True, 0), (1, 42, True, 42),
                                               (0xC0000BC6, 999, False, None), (0, -1, False, None)):
            with self.subTest(status=status, value=value):
                rows, _ = wddm.normalize_counter([(NAME, status, value)], scope="process", pids={123})
                self.assertEqual(rows[0]["valid"], valid)
                self.assertEqual(rows[0]["bytes"], expected)
                self.assertEqual(rows[0]["status"], f"0x{status:08x}")

    def test_size_query_is_restarted_after_instance_churn(self):
        calls = []
        def read(handle, formatting, size, count, buffer):
            size_ptr, count_ptr = ct.cast(size, ct.POINTER(wddm.DWORD)), ct.cast(count, ct.POINTER(wddm.DWORD))
            calls.append((size_ptr.contents.value, buffer is None))
            if len(calls) == 1:
                size_ptr.contents.value = 32
                return wddm.PDH_MORE_DATA
            if len(calls) == 2:
                size_ptr.contents.value = 9999  # This size must not be reused.
                return wddm.PDH_MORE_DATA
            if len(calls) == 3:
                size_ptr.contents.value = 32
                return wddm.PDH_MORE_DATA
            count_ptr.contents.value = 0
            return 0
        sampler = object.__new__(wddm.WDDMMemorySampler)
        sampler._api = SimpleNamespace(PdhGetFormattedCounterArrayW=read)
        self.assertEqual(sampler._read(1), (0, []))
        self.assertEqual(calls, [(0, True), (32, False), (0, True), (32, False)])

    def test_invalid_counter_return_is_not_read_as_data(self):
        sampler = object.__new__(wddm.WDDMMemorySampler)
        sampler._api = SimpleNamespace(PdhGetFormattedCounterArrayW=Mock(return_value=0xC0000BC6))
        self.assertEqual(sampler._read(1), (0xC0000BC6, []))
        self.assertEqual(sampler._api.PdhGetFormattedCounterArrayW.call_count, 1)

    def test_failed_collection_does_not_reuse_old_values_and_close_is_idempotent(self):
        sampler = object.__new__(wddm.WDDMMemorySampler)
        sampler.pids = {123}
        sampler._query = wddm.HANDLE(1)
        sampler._counters = [{"handle": 2, "scope": "process", "metric": "dedicated_bytes"}]
        sampler._api = SimpleNamespace(PdhCollectQueryData=Mock(return_value=0x800007D5), PdhCloseQuery=Mock())
        sampler._read = Mock(side_effect=AssertionError("Must not read stale values"))
        report = sampler.sample(pids=[123, 456])
        self.assertEqual(report["pids_without_instances"], [123, 456])
        self.assertEqual(report["collect_status"], "0x800007d5")
        self.assertEqual(report["counters"][0]["rows"], [])
        sampler.close()
        sampler.close()
        self.assertEqual(sampler._api.PdhCloseQuery.call_count, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            sampler.sample()

    def test_pid_validation(self):
        self.assertEqual(wddm._pids([123, "456", 123]), {123, 456})
        for invalid in ([], [0], [-1], [123, 0]):
            with self.assertRaises(ValueError):
                wddm._pids(invalid)


if __name__ == "__main__":
    unittest.main()
