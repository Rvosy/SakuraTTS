"""Lifetime ownership for the optional, complete inference process tree."""

import logging
import os
import signal
import subprocess
import threading
import time


class ProcessTree:
    """A Windows Job Object also reaps descendants if the owner crashes."""

    def __init__(self):
        self.process = None
        self.handle = None
        self._lock = threading.Lock()
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Limits(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

            class IO(ctypes.Structure):
                _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", Limits), ("IoInfo", IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            class Accounting(ctypes.Structure):
                _fields_ = [(name, ctypes.c_longlong) for name in
                    ("TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")]
                _fields_ += [(name, wintypes.DWORD) for name in
                    ("TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]

            self._ctypes, self._accounting = ctypes, Accounting
            kernel = self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            signatures = {
                "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
                "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
                "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
                "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
                "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
                "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
            }
            for name, (args, result) in signatures.items():
                function = getattr(kernel, name)
                function.argtypes, function.restype = args, result
            self.handle = kernel.CreateJobObjectW(None, None)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            limits = Extended()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                error = ctypes.WinError(ctypes.get_last_error())
                kernel.CloseHandle(self.handle)
                self.handle = None
                raise error

    @staticmethod
    def popen_options():
        return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}

    def bind(self, process):
        self.process = process
        if self.handle and not self._kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        threading.Thread(target=self._reap_after_exit, args=(process,), daemon=True,
            name="sakuratts-inference-lifetime").start()

    def _reap_after_exit(self, process):
        try:
            process.wait()
            self.close()
        except Exception:
            # Keep ownership on failure; the next explicit cleanup can retry.
            logging.getLogger("sakuratts.engine").exception("Failed to reap the exited inference worker's descendants")

    def close(self, timeout=5):
        """Return only after all owned Windows processes have exited."""
        with self._lock:
            self._close(timeout)

    def _close(self, timeout):
        process = self.process
        if self.handle:
            ctypes = self._ctypes
            if not self._kernel.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + timeout
            while True:
                accounting = self._accounting()
                if not self._kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(accounting),
                        ctypes.sizeof(accounting), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not accounting.ActiveProcesses:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Inference process tree did not exit")
                time.sleep(.01)
            if not self._kernel.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None
        elif process is not None and os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None:
            # Assignment failure still leaves a private, uninitialized child.
            if process.poll() is None:
                process.kill()
            process.wait(timeout=timeout)
        self.process = None
