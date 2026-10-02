"""Event-loop ownership of the optional inference process lifecycle."""

import asyncio
import math
import time

from sakuratts.engine import BusyError
from .cancellation import SynthesisCancelled


def positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


class ManagedRuntime:
    """All state transitions run on the HTTP event loop; work uses one thread."""

    def __init__(self, inference, pool, *, idle_sleep_seconds):
        self.inference, self.pool = inference, pool
        self.idle_sleep_seconds = positive_seconds(idle_sleep_seconds, "idle_sleep_seconds")
        self.state = "sleeping"
        self.active = False
        self.closing = False
        self.last_error = None
        self.last_wake_ms = None
        self.generation = 0
        self._wake_task = self._sleep_task = self._timer = None
        self._pending_keep_alive = 0.
        self._keep_until = self._idle_since = 0.

    def _task(self, coroutine):
        task = asyncio.create_task(coroutine)
        # Background wake/sleep errors remain available through /runtime.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return task

    async def _execute(self, operation):
        return await asyncio.get_running_loop().run_in_executor(self.pool, operation)

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _refresh(self):
        if self.state == "awake" and not self.active and not self.inference.alive:
            self.state = "failed"
            self.last_error = "Inference worker exited; submit wake or tts to retry"
            self._cancel_timer()

    def snapshot(self):
        self._refresh()
        info = self.inference.info() if self.state == "awake" else None
        preparation = self.inference.preparation
        return {"mode": "managed", "state": self.state, "busy": self.active,
                "model_configured": self.inference.configured,
                "model_loaded": info is not None and preparation == "model_load",
                "model": info, "worker_pid": self.inference.pid,
                "generation": self.generation, "last_error": self.last_error,
                "last_wake_ms": self.last_wake_ms, "idle_sleep_seconds": self.idle_sleep_seconds,
                "keep_alive_remaining_seconds": max(0., self._keep_until - time.monotonic()),
                "preparation": preparation}

    def request_wake(self, keep_alive_seconds=0.):
        if self.closing:
            raise RuntimeError("Runtime is closing")
        if (isinstance(keep_alive_seconds, bool) or not isinstance(keep_alive_seconds, (int, float))
                or not math.isfinite(keep_alive_seconds) or not 0 <= keep_alive_seconds <= 3600):
            raise ValueError("keep_alive_seconds must be finite and between 0 and 3600")
        if not self.inference.configured:
            raise ValueError("Configure a model before waking the runtime")
        self._refresh()
        if self.state == "awake":
            self._keep_until = max(self._keep_until, time.monotonic() + keep_alive_seconds)
            self._schedule_idle()
            return None
        self._pending_keep_alive = max(self._pending_keep_alive, keep_alive_seconds)
        if self._wake_task is None or self._wake_task.done():
            self._cancel_timer()
            if self.state != "stopping":
                self.state = "waking"
            self._wake_task = self._task(self._wake())
        return self._wake_task

    async def _wake(self):
        started = time.monotonic()
        try:
            if self._sleep_task is not None and not self._sleep_task.done():
                await asyncio.shield(self._sleep_task)
            if self.closing:
                raise RuntimeError("Runtime is closing")
            self.state = "waking"
            self.generation += 1
            await self._execute(self.inference.wake)
            self.state = "awake"
            self.last_error = None
            self.last_wake_ms = (time.monotonic() - started) * 1000
            self._idle_since = time.monotonic()
            self._keep_until = self._idle_since + self._pending_keep_alive
        except Exception as error:
            self.state = "failed"
            self.last_error = str(error)
            raise
        finally:
            self._pending_keep_alive = 0.
            self._schedule_idle()

    async def ensure_awake(self, cancel_requested=None):
        task = self.request_wake()
        if task is not None:
            # Cancellation belongs to this request, never to the shared wake.
            while not task.done():
                if cancel_requested is not None and cancel_requested():
                    raise SynthesisCancelled("runtime_wake")
                await asyncio.wait({task}, timeout=.05)
            await asyncio.shield(task)
        if cancel_requested is not None and cancel_requested():
            raise SynthesisCancelled("runtime_wake")

    def begin_operation(self):
        if self.closing:
            raise RuntimeError("Runtime is closing")
        self._refresh()
        self.active = True
        self._cancel_timer()

    def end_operation(self, error=None):
        self.active = False
        self._idle_since = time.monotonic()
        if error is not None and not isinstance(error, SynthesisCancelled):
            self.last_error = str(error)
            if not self.inference.alive or self.inference.info() is None:
                self.state = "failed"
        self._refresh()
        self._schedule_idle()

    def _schedule_idle(self):
        self._cancel_timer()
        if self.closing or self.active or self.state != "awake":
            return
        delay = max(self._idle_since + self.idle_sleep_seconds, self._keep_until) - time.monotonic()
        self._timer = asyncio.get_running_loop().call_later(max(0., delay), self._idle_expired)

    def _idle_expired(self):
        self._timer = None
        if self.closing or self.active or self.state != "awake":
            return
        if time.monotonic() < max(self._idle_since + self.idle_sleep_seconds, self._keep_until):
            self._schedule_idle()
            return
        self._start_sleep()

    def _start_sleep(self):
        self._cancel_timer()
        self.state = "stopping"
        self._sleep_task = self._task(self._sleep())
        return self._sleep_task

    async def _sleep(self):
        try:
            await self._execute(self.inference.sleep)
            self.state = "sleeping"
            self._keep_until = 0.
        except Exception as error:
            self.state = "failed"
            self.last_error = str(error)
            raise

    async def sleep(self):
        if self.active or (self._wake_task is not None and not self._wake_task.done()):
            raise BusyError("Runtime is busy; retry after the current operation completes")
        if self.state == "sleeping":
            return
        task = self._sleep_task if self.state == "stopping" else self._start_sleep()
        await asyncio.shield(task)

    def begin_shutdown(self):
        self.closing = True
        self._cancel_timer()

    async def close(self):
        self.begin_shutdown()
        tasks = [task for task in (self._wake_task, self._sleep_task) if task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._execute(self.inference.close)
        self.state = "sleeping"
