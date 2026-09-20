"""Exercise HTTP validation, single-flight admission, and GPU thread ownership."""

import importlib.util
import shutil
import subprocess
import threading
import unittest
from unittest.mock import patch

from test_public_api import audio

REQUEST = {"text": "こんにちは。", "text_lang": "ja", "ref_audio_path": "reference.wav",
           "prompt_lang": "ja", "prompt_text": "参考音声です。"}


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "Install server and dev extras")
class ServerTests(unittest.TestCase):
    def test_failed_audio_encoding_has_one_error_and_no_completion_summary(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        class FakeInference:
            def __init__(self, *args, **kwargs): pass
            def info(self): return {"name": "test"}
            def close(self): pass
            def tts(self, *args, **kwargs): return audio()
        with patch("sakuratts.server.Inference", FakeInference), TestClient(create_app()) as client, \
                patch("sakuratts.server.pack_audio", side_effect=RuntimeError("encoder failed")), \
                self.assertLogs("sakuratts.server", level="INFO") as logs:
            response = client.post("/tts", json=REQUEST)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("音频编码失败", logs.output[0])
        self.assertIsNotNone(logs.records[0].exc_info)

    def test_startup_failure_is_not_reported_as_success(self):
        from sakuratts.server import start_server
        with patch("uvicorn.Server") as server:
            server.return_value.started = False
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                start_server()

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is optional")
    def test_compressed_output_can_be_decoded_by_ffmpeg(self):
        from sakuratts.server import pack_audio
        import numpy as np
        pcm = np.tile(audio().pcm, 1000)
        for media in ("ogg", "aac"):
            data = pack_audio(pcm, 32000, media)
            self.assertGreater(len(data), 0)
            decoded = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                "-f", "s16le", "pipe:1"], input=data, capture_output=True, check=True)
            self.assertGreater(len(decoded.stdout), 0)

    def test_shared_engine_routes_limits_busy_and_shutdown(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        threads = []
        entered, release = threading.Event(), threading.Event()
        class Model:
            name = "試験"
            references = ("通常",)
            default_reference = "通常"
            def info(self): return {"name": self.name}
        class FakeEngine:
            model = Model()
            def __init__(self, *args, **kwargs): threads.append(threading.get_ident())
            def info(self): return self.model.info()
            def set_weights(self, *args): raise AssertionError("Busy operations must not execute")
            def tts(self, request, **kwargs):
                text = request["text"]
                threads.append(threading.get_ident())
                if text == "slow":
                    entered.set()
                    release.wait(5)
                if text == "error": raise RuntimeError("private details")
                return audio("stopped_at_limit" if text == "limit" else "completed")
            def close(self): threads.append(threading.get_ident())
        with patch("sakuratts.server.Inference", FakeEngine):
            with TestClient(create_app("model")) as client:
                self.assertEqual(client.get("/health").json()["status"], "ready")
                self.assertEqual(client.get("/models").json(), [{"name": "試験"}])
                self.assertEqual(client.get("/").status_code, 404)
                for update in ({"text": " "}, {"ref_audio_path": ""}, {"seed": -2},
                               {"text_lang": "zh"}, {"speed_factor": 1.2}, {"top_p": .8},
                               {"streaming_mode": 2}, {"batch_size": 2}, {"aux_ref_audio_paths": ["a.wav"]}):
                    self.assertEqual(client.post("/tts", json=dict(REQUEST, **update)).status_code, 400)
                response = client.post("/tts", json=dict(REQUEST, text="limit"))
                self.assertEqual(response.headers["X-SakuraTTS-Status"], "stopped_at_limit")
                self.assertEqual(response.headers["content-type"], "audio/wav")
                self.assertTrue(response.content.startswith(b"RIFF"))
                thread = threading.Thread(target=lambda: client.post("/tts", json=dict(REQUEST, text="slow")))
                thread.start()
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(client.get("/health").json()["status"], "busy")
                    self.assertEqual(client.post("/tts", json=REQUEST).status_code, 409)
                    self.assertEqual(client.get("/set_gpt_weights", params={"weights_path": "b.ckpt"}).status_code, 409)
                finally:
                    release.set()
                    thread.join(5)
                with self.assertLogs("sakuratts.server", level="ERROR"):
                    response = client.post("/tts", json=dict(REQUEST, text="error"))
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["message"], "tts failed")
                self.assertEqual(client.get("/health").json()["status"], "ready")
        self.assertEqual(len(set(threads)), 1)

    def test_original_get_defaults_switches_and_stream(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        calls = []
        class FakeInference:
            def __init__(self, *a, **k): pass
            def info(self): return {"name": "test"}
            def close(self): pass
            def set_weights(self, kind, path): calls.append((kind, path))
            def set_reference_audio(self, path): calls.append(("reference", path))
            def tts(self, request, on_fragment=None, **kwargs):
                calls.append(request)
                result = audio()
                if on_fragment:
                    on_fragment(result.pcm, result.sample_rate)
                    on_fragment(result.pcm, result.sample_rate)
                return result
        with patch("sakuratts.server.Inference", FakeInference), TestClient(create_app()) as client:
            self.assertEqual(client.get("/tts", params=REQUEST).status_code, 200)
            self.assertEqual(calls[-1]["seed"], -1)
            self.assertEqual(calls[-1]["text_split_method"], "cut5")
            for kind in ("gpt", "sovits"):
                self.assertEqual(client.get("/set_" + kind + "_weights",
                    params={"weights_path": "weights"}).json(), {"message": "success"})
                self.assertEqual(calls[-1], (kind, "weights"))
            self.assertEqual(client.get("/set_refer_audio", params={"refer_audio_path": "a.wav"}).status_code, 200)
            self.assertEqual(calls[-1], ("reference", "a.wav"))
            raw = client.post("/tts", json=dict(REQUEST, media_type="raw"))
            self.assertEqual(raw.content, audio().pcm.astype("<i2").tobytes())
            streamed = client.get("/tts", params=dict(REQUEST, streaming_mode="1"))
            self.assertTrue(streamed.content.startswith(b"RIFF"))
            self.assertEqual(streamed.content[44:], raw.content * 2)
            self.assertEqual(client.get("/tts", params=dict(REQUEST, streaming_mode="2")).status_code, 400)
            self.assertEqual(client.get("/tts", params=dict(REQUEST, seed="invalid")).status_code, 422)

    def test_empty_startup_has_no_backend_and_no_default_reference(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        with patch("sakuratts.engine.Engine.load") as load, TestClient(create_app()) as client:
            self.assertFalse(client.get("/health").json()["model_loaded"])
            self.assertEqual(client.get("/models").json(), [])
            load.assert_not_called()
            self.assertEqual(client.post("/tts", json={"text": "a"}).status_code, 400)

    def test_stream_emits_before_completion_and_disconnect_keeps_worker_owned(self):
        import asyncio
        import json
        from sakuratts.server import create_app
        first_sent, finish = threading.Event(), threading.Event()
        class FakeInference:
            def __init__(self, *a, **k): pass
            def info(self): return None
            def close(self): pass
            def tts(self, request, on_fragment, cancel_requested):
                result = audio()
                on_fragment(result.pcm, result.sample_rate)
                self.assert_sent = first_sent.wait(3)
                finish.wait(3)
                return result
        async def run():
            app = create_app()
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
                    self.assertTrue(app.state.busy)
                finally:
                    finish.set()
                await asyncio.gather(*app.state.jobs)
                self.assertFalse(app.state.busy)
        with patch("sakuratts.server.Inference", FakeInference):
            asyncio.run(run())
