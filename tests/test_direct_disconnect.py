"""Disconnect cancellation must not admit work before native resources return."""

import asyncio
import importlib.util
import json
import threading
import unittest
from unittest.mock import patch

from sakuratts._internal.cancellation import SynthesisCancelled
from test_public_api import audio
from test_server import REQUEST


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "Install server and dev extras")
class DirectDisconnectTests(unittest.TestCase):
    def test_disconnect_before_audio_keeps_slot_until_worker_finishes(self):
        import httpx
        from sakuratts.server import create_app
        async def run(streaming):
            entered, cancelled, release = (threading.Event() for _ in range(3))
            class FakeInference:
                def __init__(self, *args, **kwargs): pass
                def info(self): return None
                def close(self): pass
                def set_weights(self, *args):
                    raise AssertionError("Busy model switches must not execute")
                def tts(self, values, *, cancel_requested, on_fragment=None):
                    if values["text"] == "slow":
                        entered.set()
                        while not cancel_requested():
                            if release.wait(.01):
                                raise AssertionError("Cancellation was not delivered")
                        cancelled.set()
                        if not release.wait(3):
                            raise AssertionError("Test did not release native work")
                        raise SynthesisCancelled("after_prefill")
                    result = audio()
                    if on_fragment:
                        on_fragment(result.pcm, result.sample_rate)
                    return result

            with patch("sakuratts.server.Inference", FakeInference):
                app = create_app()
                async with app.router.lifespan_context(app):
                    initial = True
                    disconnect = asyncio.Event()
                    messages = []
                    async def receive():
                        nonlocal initial
                        if initial:
                            initial = False
                            return {"type": "http.request", "body": json.dumps(dict(
                                REQUEST, text="slow", streaming_mode=streaming)).encode()}
                        await disconnect.wait()
                        return {"type": "http.disconnect"}
                    async def send(message):
                        messages.append(message)
                    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
                        "http_version": "1.1", "method": "POST", "path": "/tts", "raw_path": b"/tts",
                        "query_string": b"", "headers": [(b"content-type", b"application/json")],
                        "client": ("127.0.0.1", 1234), "server": ("test", 80), "scheme": "http"}
                    request = asyncio.create_task(app(scope, receive, send))
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                        disconnect.set()
                        await asyncio.wait_for(request, 3)
                        self.assertTrue(await asyncio.to_thread(cancelled.wait, 3))
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                            self.assertEqual((await client.post("/tts", json=REQUEST)).status_code, 409)
                            self.assertEqual((await client.get("/set_gpt_weights", params={"weights_path": "next"})).status_code, 409)
                            release.set()
                            await asyncio.wait_for(asyncio.gather(*app.state.jobs, return_exceptions=True), 3)
                            response = await client.post("/tts", json=REQUEST)
                            self.assertEqual(response.status_code, 200)
                            self.assertTrue(response.content.startswith(b"RIFF"))
                        self.assertEqual(messages[0]["status"], 499)
                    finally:
                        disconnect.set()
                        release.set()
                        await asyncio.gather(request, return_exceptions=True)

        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                asyncio.run(run(streaming))


if __name__ == "__main__":
    unittest.main()
