"""Managed HTTP lifecycle without loading models or creating GPU processes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import threading
import time
import unittest
from unittest.mock import patch

from test_public_api import audio
from test_server import REQUEST


def process_double():
    """A controllable worker boundary; events represent unfinished native work."""
    class FakeProcess:
        instances = []
        wake_entered = threading.Event()
        wake_release = threading.Event()
        tts_entered = threading.Event()
        tts_release = threading.Event()
        sleep_entered = threading.Event()
        sleep_release = threading.Event()
        wake_failures = 0
        wake_calls = 0
        tts_calls = 0
        sleep_calls = 0
        thread_ids = []

        def __init__(self, model=None, *, tts_config=None, **kwargs):
            self.configured = model is not None or tts_config is not None
            self.preparation = ("runtime_init" if (kwargs.get("experimental") or {}).get("policy") == "staged"
                                else "model_load")
            self.alive = False
            self.pid = None
            self.instances.append(self)
            self.thread_ids.append(threading.get_ident())

        def wake(self):
            self.thread_ids.append(threading.get_ident())
            type(self).wake_calls += 1
            self.wake_entered.set()
            if not self.wake_release.wait(5):
                raise TimeoutError("Test did not release wake")
            if self.wake_failures:
                type(self).wake_failures -= 1
                raise RuntimeError("Model preparation failed")
            self.alive = True
            self.pid = 12345

        def info(self):
            return {"name": "managed-test"} if self.alive else None

        def tts(self, request, *, on_fragment=None, cancel_requested=None):
            self.thread_ids.append(threading.get_ident())
            type(self).tts_calls += 1
            self.tts_entered.set()
            if not self.tts_release.wait(5):
                raise TimeoutError("Test did not release synthesis")
            result = audio()
            if on_fragment:
                on_fragment(result.pcm, result.sample_rate)
            return result

        def set_weights(self, *args):
            self.thread_ids.append(threading.get_ident())

        def set_reference_audio(self, *args):
            self.thread_ids.append(threading.get_ident())

        def sleep(self):
            self.thread_ids.append(threading.get_ident())
            type(self).sleep_calls += 1
            self.sleep_entered.set()
            if not self.sleep_release.wait(5):
                raise TimeoutError("Test did not release sleep")
            self.alive = False
            self.pid = None

        def close(self):
            self.thread_ids.append(threading.get_ident())
            self.alive = False
            self.pid = None

    FakeProcess.wake_release.set()
    FakeProcess.tts_release.set()
    FakeProcess.sleep_release.set()
    return FakeProcess


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
                     "Install server and dev extras")
class ManagedServerTests(unittest.TestCase):
    def wait_state(self, client, expected, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = client.get("/runtime").json()
            if status["state"] == expected:
                return status
            time.sleep(.01)
        self.fail(f"Expected runtime {expected!r}, last status: {status}")

    def test_default_mode_keeps_its_original_routes(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        class DirectInference:
            def __init__(self, *args, **kwargs): pass
            def info(self): return {"name": "direct-test"}
            def close(self): pass
        with patch("sakuratts.server.Inference", DirectInference), TestClient(create_app("model")) as client:
            self.assertTrue(client.get("/health").json()["model_loaded"])
            self.assertEqual(client.get("/runtime").status_code, 404)
            self.assertEqual(client.post("/runtime/wake").status_code, 404)
            self.assertEqual(client.post("/runtime/sleep").status_code, 404)

    def test_sleeping_startup_and_health_checks_do_not_initialize_inference(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                patch("sakuratts.server.Inference", side_effect=AssertionError("Direct engine was constructed")), \
                TestClient(create_app("model", runtime_mode="managed")) as client:
            for _ in range(3):
                status = client.get("/runtime").json()
                self.assertEqual(status["state"], "sleeping")
                self.assertTrue(status["model_configured"])
                self.assertFalse(status["model_loaded"])
                self.assertIsNone(status["worker_pid"])
                self.assertFalse(client.get("/health").json()["model_loaded"])
                self.assertEqual(client.get("/models").json(), [])
            self.assertEqual(client.post("/runtime/sleep").status_code, 200)
            self.assertEqual(client.post("/runtime/sleep").status_code, 200)
            self.assertEqual(fake.wake_calls, 0)
            self.assertEqual(fake.tts_calls, 0)

    def test_empty_configuration_rejects_wake_without_starting_a_worker(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app(runtime_mode="managed")) as client:
            self.assertFalse(client.get("/runtime").json()["model_configured"])
            self.assertEqual(client.post("/runtime/wake").status_code, 400)
            self.assertEqual(fake.wake_calls, 0)

    def test_staged_ready_does_not_claim_gpu_weights_are_loaded(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app("model", runtime_mode="managed",
                                      experimental={"policy": "staged"})) as client:
            self.assertEqual(client.post("/runtime/wake").status_code, 202)
            ready = self.wait_state(client, "awake")
            self.assertEqual(ready["preparation"], "runtime_init")
            self.assertFalse(ready["model_loaded"])
            self.assertEqual(ready["model"], {"name": "managed-test"})
            self.assertFalse(client.get("/health").json()["model_loaded"])
            self.assertEqual(client.get("/models").json(), [{"name": "managed-test"}])
            self.assertEqual(client.post("/tts", json=REQUEST).status_code, 200)
            self.assertFalse(client.get("/runtime").json()["model_loaded"])
            self.assertEqual(client.post("/runtime/sleep").status_code, 200)
            self.assertEqual(client.get("/models").json(), [])
            self.assertEqual(client.post("/tts", json=REQUEST).status_code, 200)
            self.assertEqual(fake.wake_calls, 2)

    def test_repeated_wake_is_shared_and_waiting_tts_owns_the_single_slot(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        fake.wake_release.clear()
        fake.tts_release.clear()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app("model", runtime_mode="managed")) as client:
            with ThreadPoolExecutor(max_workers=1) as requests:
                try:
                    self.assertEqual(client.post("/runtime/wake").status_code, 202)
                    self.assertTrue(fake.wake_entered.wait(3))
                    self.assertEqual(client.post("/runtime/wake").status_code, 202)
                    self.assertEqual(client.post("/runtime/sleep").status_code, 409)
                    first = requests.submit(client.post, "/tts", json=REQUEST)
                    deadline = time.monotonic() + 3
                    while not client.get("/runtime").json()["busy"] and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(client.get("/runtime").json()["busy"])
                    self.assertEqual(client.post("/tts", json=REQUEST).status_code, 409)
                    self.assertEqual(client.get("/set_gpt_weights", params={"weights_path": "other"}).status_code, 409)
                    self.assertEqual(fake.wake_calls, 1)
                    self.assertEqual(fake.tts_calls, 0)
                    fake.wake_release.set()
                    self.assertTrue(fake.tts_entered.wait(3))
                    self.assertEqual(client.post("/runtime/sleep").status_code, 409)
                finally:
                    fake.wake_release.set()
                    fake.tts_release.set()
                self.assertEqual(first.result(timeout=3).status_code, 200)
            status = self.wait_state(client, "awake")
            self.assertTrue(status["model_loaded"])
            self.assertEqual(status["worker_pid"], 12345)
            self.assertEqual(client.post("/runtime/wake").status_code, 200)
            self.assertEqual(fake.wake_calls, 1)
            self.assertEqual(client.post("/runtime/sleep").status_code, 200)
            self.assertEqual(self.wait_state(client, "sleeping")["worker_pid"], None)
        self.assertEqual(len(set(fake.thread_ids)), 1)

    def test_failed_wake_can_be_retried(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        fake.wake_failures = 1
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app("model", runtime_mode="managed")) as client:
            self.assertEqual(client.post("/runtime/wake").status_code, 202)
            failed = self.wait_state(client, "failed")
            self.assertFalse(failed["busy"])
            self.assertFalse(failed["model_loaded"])
            self.assertTrue(failed["last_error"])
            self.assertEqual(client.post("/tts", json=REQUEST).status_code, 200)
            recovered = self.wait_state(client, "awake")
            self.assertIsNone(recovered["last_error"])
            self.assertEqual(fake.wake_calls, 2)

    def test_waiting_tts_releases_admission_after_wake_failure(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                fake = process_double()
                fake.wake_release.clear()
                fake.wake_failures = 1
                with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                        TestClient(create_app("model", runtime_mode="managed")) as client:
                    with ThreadPoolExecutor(max_workers=1) as requests:
                        first = requests.submit(client.post, "/tts", json=dict(REQUEST, streaming_mode=streaming))
                        try:
                            self.assertTrue(fake.wake_entered.wait(3))
                            self.assertEqual(client.post("/tts", json=REQUEST).status_code, 409)
                        finally:
                            fake.wake_release.set()
                        self.assertEqual(first.result(timeout=3).status_code, 400)
                    self.assertFalse(self.wait_state(client, "failed")["busy"])
                    self.assertEqual(fake.tts_calls, 0)
                    self.assertEqual(client.post("/tts", json=REQUEST).status_code, 200)
                    self.assertEqual(fake.wake_calls, 2)
                    self.assertEqual(fake.tts_calls, 1)

    def test_wake_and_tts_during_sleep_wait_for_the_old_worker_to_exit(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        fake.sleep_release.clear()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app("model", runtime_mode="managed")) as client:
            self.assertEqual(client.post("/runtime/wake").status_code, 202)
            self.wait_state(client, "awake")
            with ThreadPoolExecutor(max_workers=2) as requests:
                sleeping = requests.submit(client.post, "/runtime/sleep")
                try:
                    self.assertTrue(fake.sleep_entered.wait(3))
                    self.assertEqual(client.get("/runtime").json()["state"], "stopping")
                    self.assertEqual(client.post("/runtime/wake").status_code, 202)
                    first = requests.submit(client.post, "/tts", json=REQUEST)
                    deadline = time.monotonic() + 3
                    while not client.get("/runtime").json()["busy"] and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(client.get("/runtime").json()["busy"])
                    self.assertEqual(fake.wake_calls, 1)
                    self.assertEqual(fake.tts_calls, 0)
                    self.assertEqual(client.post("/tts", json=REQUEST).status_code, 409)
                    self.assertEqual(client.post("/runtime/sleep").status_code, 409)
                finally:
                    fake.sleep_release.set()
                self.assertEqual(sleeping.result(timeout=3).status_code, 200)
                self.assertEqual(first.result(timeout=3).status_code, 200)
            self.assertEqual(self.wait_state(client, "awake")["generation"], 2)
            self.assertEqual(fake.wake_calls, 2)
            self.assertEqual(fake.tts_calls, 1)
            self.assertEqual(fake.sleep_calls, 1)

    def test_an_old_idle_deadline_cannot_sleep_a_new_wake(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app("model", runtime_mode="managed", idle_sleep_seconds=.15)) as client:
            self.assertEqual(client.post("/runtime/wake", json={"keep_alive_seconds": 0}).status_code, 202)
            self.wait_state(client, "awake")
            self.assertEqual(client.post("/runtime/sleep").status_code, 200)
            self.assertEqual(client.post("/runtime/wake", json={"keep_alive_seconds": .4}).status_code, 202)
            self.wait_state(client, "awake")
            time.sleep(.2)
            self.assertEqual(client.get("/runtime").json()["state"], "awake")
            self.assertEqual(fake.wake_calls, 2)
            self.assertEqual(fake.sleep_calls, 1)
            self.wait_state(client, "sleeping")
            self.assertEqual(fake.sleep_calls, 2)

    def test_dead_worker_is_reported_and_next_tts_recovers(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app("model", runtime_mode="managed")) as client:
            self.assertEqual(client.post("/tts", json=REQUEST).status_code, 200)
            for inspect_status in (True, False):
                fake.instances[0].alive = False
                fake.instances[0].pid = None
                if inspect_status:
                    failed = client.get("/runtime").json()
                    self.assertEqual(failed["state"], "failed")
                    self.assertFalse(failed["model_loaded"])
                    self.assertTrue(failed["last_error"])
                self.assertEqual(client.post("/tts", json=REQUEST).status_code, 200)
                self.assertIsNone(self.wait_state(client, "awake")["last_error"])
            self.assertEqual(fake.wake_calls, 3)
            self.assertEqual(fake.tts_calls, 3)

    def test_stream_failure_is_recorded_and_does_not_leave_admission_busy(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        class FailingStream(fake):
            def tts(self, request, **kwargs):
                if request["text"] == "fail":
                    raise RuntimeError("Acoustic synthesis failed")
                return super().tts(request, **kwargs)

        with patch("sakuratts._internal.inference_process.ProcessInference", FailingStream), \
                TestClient(create_app("model", runtime_mode="managed")) as client:
            with self.assertLogs("sakuratts.server", level="ERROR"):
                response = client.post("/tts", json=dict(REQUEST, text="fail", streaming_mode=True))
            self.assertEqual(response.status_code, 400)
            status = client.get("/runtime").json()
            deadline = time.monotonic() + 3
            while status["busy"] and time.monotonic() < deadline:
                time.sleep(.01)
                status = client.get("/runtime").json()
            self.assertEqual(status["state"], "awake")
            self.assertFalse(status["busy"])
            self.assertIn("Acoustic synthesis failed", status["last_error"])
            self.assertEqual(client.post("/tts", json=REQUEST).status_code, 200)
            self.assertEqual(FailingStream.wake_calls, 1)

    def test_disconnected_tts_releases_slot_without_cancelling_shared_wake(self):
        import httpx
        from sakuratts.server import create_app

        async def run(fake, streaming):
            app = create_app("model", runtime_mode="managed")
            async with app.router.lifespan_context(app):
                initial = True
                disconnected = asyncio.Event()
                async def receive():
                    nonlocal initial
                    if initial:
                        initial = False
                        return {"type": "http.request", "body": json.dumps(dict(REQUEST, streaming_mode=streaming)).encode()}
                    await disconnected.wait()
                    return {"type": "http.disconnect"}
                async def send(message):
                    pass
                scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
                    "http_version": "1.1", "method": "POST", "path": "/tts", "raw_path": b"/tts",
                    "query_string": b"", "headers": [(b"content-type", b"application/json")],
                    "client": ("127.0.0.1", 1234), "server": ("test", 80), "scheme": "http"}
                request = asyncio.create_task(app(scope, receive, send))
                try:
                    self.assertTrue(await asyncio.to_thread(fake.wake_entered.wait, 3))
                    disconnected.set()
                    await asyncio.wait_for(request, 3)
                    await asyncio.wait_for(asyncio.gather(*app.state.jobs, return_exceptions=True), 1)
                    self.assertFalse(app.state.busy)
                    self.assertEqual(fake.tts_calls, 0)
                    self.assertEqual(fake.wake_calls, 1)
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                        self.assertEqual((await client.get("/runtime")).json()["state"], "waking")
                        fake.wake_release.set()
                        self.assertEqual((await client.post("/tts", json=REQUEST)).status_code, 200)
                    self.assertEqual(fake.wake_calls, 1)
                    self.assertEqual(fake.tts_calls, 1)
                finally:
                    disconnected.set()
                    fake.wake_release.set()
                    await asyncio.gather(request, return_exceptions=True)

        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                fake = process_double()
                fake.wake_release.clear()
                with patch("sakuratts._internal.inference_process.ProcessInference", fake):
                    asyncio.run(run(fake, streaming))

    def test_idle_sleep_waits_for_actual_synthesis_completion(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        fake.tts_release.clear()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app("model", runtime_mode="managed", idle_sleep_seconds=.05)) as client:
            with ThreadPoolExecutor(max_workers=1) as requests:
                first = requests.submit(client.post, "/tts", json=REQUEST)
                try:
                    self.assertTrue(fake.tts_entered.wait(3))
                    time.sleep(.15)
                    self.assertTrue(client.get("/runtime").json()["busy"])
                    self.assertEqual(fake.sleep_calls, 0)
                    self.assertTrue(fake.instances[0].alive)
                finally:
                    fake.tts_release.set()
                self.assertEqual(first.result(timeout=3).status_code, 200)
            sleeping = self.wait_state(client, "sleeping")
            self.assertFalse(sleeping["model_loaded"])
            self.assertFalse(sleeping["busy"])
            self.assertEqual(fake.sleep_calls, 1)
            self.assertEqual(client.post("/tts", json=REQUEST).status_code, 200)
            self.assertEqual(fake.wake_calls, 2)

    def test_keep_alive_extends_an_existing_wake_without_reloading(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app("model", runtime_mode="managed", idle_sleep_seconds=.05)) as client:
            self.assertEqual(client.post("/runtime/wake", json={"keep_alive_seconds": .3}).status_code, 202)
            self.wait_state(client, "awake")
            time.sleep(.1)
            self.assertEqual(client.post("/runtime/wake", json={"keep_alive_seconds": .3}).status_code, 200)
            time.sleep(.12)
            self.assertEqual(client.get("/runtime").json()["state"], "awake")
            self.assertEqual(fake.wake_calls, 1)
            self.assertEqual(fake.sleep_calls, 0)
            self.wait_state(client, "sleeping")
            self.assertEqual(fake.sleep_calls, 1)

    def test_invalid_keep_alive_is_rejected_before_waking(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        fake = process_double()
        with patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                TestClient(create_app("model", runtime_mode="managed")) as client:
            for keep_alive in (-1, 3601, "invalid"):
                response = client.post("/runtime/wake", json={"keep_alive_seconds": keep_alive})
                self.assertEqual(response.status_code, 422)
            self.assertEqual(fake.wake_calls, 0)

    def test_stream_disconnect_cannot_sleep_until_native_work_releases(self):
        from sakuratts.server import create_app
        fake = process_double()
        first_sent, finish = threading.Event(), threading.Event()
        class StreamingProcess(fake):
            def tts(self, request, *, on_fragment, cancel_requested):
                result = audio()
                on_fragment(result.pcm, result.sample_rate)
                first_sent.wait(3)
                finish.wait(3)
                return result

        async def run():
            app = create_app("model", runtime_mode="managed", idle_sleep_seconds=.05)
            async with app.router.lifespan_context(app):
                initial = True
                disconnected = asyncio.Event()
                async def receive():
                    nonlocal initial
                    if initial:
                        initial = False
                        return {"type": "http.request", "body": json.dumps(dict(REQUEST, streaming_mode=True)).encode()}
                    await disconnected.wait()
                    return {"type": "http.disconnect"}
                async def send(message):
                    if message["type"] == "http.response.body" and message.get("body"):
                        first_sent.set()
                        disconnected.set()
                scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
                    "http_version": "1.1", "method": "POST", "path": "/tts", "raw_path": b"/tts",
                    "query_string": b"", "headers": [(b"content-type", b"application/json")],
                    "client": ("127.0.0.1", 1234), "server": ("test", 80), "scheme": "http"}
                try:
                    await asyncio.wait_for(app(scope, receive, send), 4)
                    self.assertTrue(first_sent.is_set())
                    await asyncio.sleep(.15)
                    self.assertTrue(app.state.busy)
                    self.assertEqual(StreamingProcess.sleep_calls, 0)
                finally:
                    finish.set()
                await asyncio.gather(*app.state.jobs, return_exceptions=True)
                self.assertFalse(app.state.busy)
        with patch("sakuratts._internal.inference_process.ProcessInference", StreamingProcess):
            asyncio.run(run())

    def test_shutdown_finishes_stream_waiting_for_its_first_fragment(self):
        import httpx
        from sakuratts._internal.generation import SynthesisCancelled
        from sakuratts.server import create_app
        fake = process_double()
        class SlowFirstFragment(fake):
            def tts(self, request, *, on_fragment, cancel_requested):
                self.tts_entered.set()
                deadline = time.monotonic() + 3
                while not cancel_requested() and time.monotonic() < deadline:
                    time.sleep(.01)
                if cancel_requested():
                    raise SynthesisCancelled("before_first_fragment")
                raise TimeoutError("Shutdown did not cancel synthesis")

        async def run():
            app = create_app("model", runtime_mode="managed")
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                request = None
                try:
                    async with app.router.lifespan_context(app):
                        request = asyncio.create_task(client.post("/tts", json=dict(REQUEST, streaming_mode=True)))
                        self.assertTrue(await asyncio.to_thread(fake.tts_entered.wait, 3))
                    await asyncio.wait_for(request, 1)
                    self.assertFalse(app.state.busy)
                    self.assertFalse(fake.instances[0].alive)
                finally:
                    if request is not None and not request.done():
                        request.cancel()
                        await asyncio.gather(request, return_exceptions=True)

        with patch("sakuratts._internal.inference_process.ProcessInference", SlowFirstFragment):
            asyncio.run(run())

    def test_connected_slow_stream_consumer_cannot_hold_inference_past_timeout(self):
        from sakuratts.server import create_app
        fake = process_double()
        returned_fragments = []
        class ManyFragments(fake):
            def tts(self, request, *, on_fragment, cancel_requested):
                result = audio()
                for index in range(100):
                    on_fragment(result.pcm, result.sample_rate)
                    returned_fragments.append(index)
                return result

        async def run():
            app = create_app("model", runtime_mode="managed", operation_timeout_seconds=.15)
            async with app.router.lifespan_context(app):
                initial = True
                first_sent = asyncio.Event()
                release_send = asyncio.Event()
                disconnected = asyncio.Event()
                async def receive():
                    nonlocal initial
                    if initial:
                        initial = False
                        return {"type": "http.request", "body": json.dumps(dict(REQUEST, streaming_mode=True)).encode()}
                    await disconnected.wait()
                    return {"type": "http.disconnect"}
                async def send(message):
                    if message["type"] == "http.response.body" and message.get("body"):
                        first_sent.set()
                        await release_send.wait()
                scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
                    "http_version": "1.1", "method": "POST", "path": "/tts", "raw_path": b"/tts",
                    "query_string": b"", "headers": [(b"content-type", b"application/json")],
                    "client": ("127.0.0.1", 1234), "server": ("test", 80), "scheme": "http"}
                request = asyncio.create_task(app(scope, receive, send))
                try:
                    await asyncio.wait_for(first_sent.wait(), 3)
                    deadline = time.monotonic() + 3
                    while app.state.busy and time.monotonic() < deadline:
                        await asyncio.sleep(.01)
                    self.assertFalse(app.state.busy)
                    self.assertFalse(disconnected.is_set())
                    self.assertFalse(release_send.is_set())
                    self.assertGreater(len(returned_fragments), 0)
                    self.assertLess(len(returned_fragments), 100)
                    self.assertIn("timed out", app.state.runtime.snapshot()["last_error"])
                finally:
                    disconnected.set()
                    release_send.set()
                    await asyncio.wait_for(asyncio.gather(request, return_exceptions=True), 3)

        with patch("sakuratts._internal.inference_process.ProcessInference", ManyFragments):
            asyncio.run(run())
