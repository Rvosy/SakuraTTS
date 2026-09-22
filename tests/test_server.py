"""Exercise HTTP validation, single-flight admission, and GPU thread ownership."""

import importlib.util
import shutil
import subprocess
import threading
import unittest
from unittest.mock import patch

from test_public_api import audio

REQUEST = {"text": "こんにちは。", "text_lang": "ja", "ref_audio_path": "reference.wav",
           "prompt_lang": "ja", "prompt_text": "参考音声です。", "parallel_infer": False}


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "Install server and dev extras")
class ServerTests(unittest.TestCase):
    def test_unimplemented_requests_fail_before_direct_or_managed_inference(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        from test_managed_server import process_double
        class DirectInference:
            def __init__(self, *args, **kwargs): pass
            def info(self): return None
            def close(self): pass
            def tts(self, *args, **kwargs):
                raise AssertionError("Unsupported requests must not reach inference")
        cases = [{key: value for key, value in REQUEST.items() if key != "parallel_infer"}]
        cases += [dict(REQUEST, **update) for update in (
            {"parallel_infer": True}, {"text_lang": "zh"}, {"prompt_lang": "ko"},
            {"top_p": .8}, {"speed_factor": 1.2}, {"batch_size": 2},
            {"streaming_mode": 2}, {"streaming_mode": 3}, {"prompt_text": ""},
            {"super_sampling": True}, {"unknown_option": "ignored before"})]
        for mode in ("direct", "managed"):
            fake = process_double()
            with patch("sakuratts.server.Inference", DirectInference), \
                    patch("sakuratts._internal.inference_process.ProcessInference", fake), \
                    TestClient(create_app("model", runtime_mode=mode)) as client:
                for request in cases:
                    for method in ("get", "post"):
                        with self.subTest(mode=mode, method=method, request=request):
                            response = (client.get("/tts", params=request) if method == "get"
                                        else client.post("/tts", json=request))
                            self.assertEqual(response.status_code, 400)
                            self.assertEqual(response.json()["error"], "unsupported_feature")
                if mode == "managed":
                    self.assertEqual(fake.wake_calls, 0)
                    self.assertEqual(client.get("/runtime").json()["state"], "sleeping")

    def test_runtime_unsupported_errors_keep_upstream_synthesis_envelope(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        class FakeInference:
            def __init__(self, *args, **kwargs): pass
            def info(self): return None
            def close(self): pass
            def tts(self, *args, **kwargs):
                raise NotImplementedError("English segment G2P is not implemented")
        with patch("sakuratts.server.Inference", FakeInference), TestClient(create_app()) as client:
            for mode in (0, 1):
                with self.subTest(streaming_mode=mode), self.assertLogs("sakuratts.server", level="ERROR"):
                    response = client.post("/tts", json=dict(REQUEST, streaming_mode=mode))
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"message": "tts failed",
                    "Exception": "English segment G2P is not implemented", "error": "unsupported_feature"})

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

    def test_get_and_post_share_defaults_types_and_validation(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import SpeechRequest, create_app
        calls = []
        class FakeInference:
            def __init__(self, *args, **kwargs): pass
            def info(self): return {"name": "test"}
            def close(self): pass
            def tts(self, request, on_fragment=None, **kwargs):
                calls.append(request)
                result = audio()
                if on_fragment:
                    on_fragment(result.pcm, result.sample_rate)
                return result
        with patch("sakuratts.server.Inference", FakeInference), TestClient(create_app()) as client:
            requests = [REQUEST, dict(REQUEST, text_lang="auto", prompt_lang="en"),
                        dict(REQUEST, text_lang="en", prompt_lang="auto")]
            for mode in (False, True, 0, 1):
                requests.append(dict(REQUEST, streaming_mode=mode, media_type="raw", seed=42,
                    top_k=20, temperature=0.65, repetition_penalty=1.15, fragment_interval=0.2,
                    parallel_infer=False, split_bucket=False, batch_threshold=0.5,
                    sample_steps=16, overlap_length=4, min_chunk_length=8))
            for request in requests:
                with self.subTest(request=request):
                    get_response = client.get("/tts", params=request)
                    post_response = client.post("/tts", json=request)
                    self.assertEqual(get_response.status_code, 200)
                    self.assertEqual(post_response.status_code, 200)
                    self.assertEqual(get_response.content, post_response.content)
                    self.assertEqual(calls[-2], calls[-1])
                    self.assertEqual(type(calls[-2]["streaming_mode"]), type(calls[-1]["streaming_mode"]))
            for update, status in (({"streaming_mode": 2}, 400), ({"streaming_mode": 3}, 400),
                                   ({"streaming_mode": 4}, 400), ({"streaming_mode": "invalid"}, 422),
                                   ({"seed": "invalid"}, 422), ({"text": ""}, 400)):
                with self.subTest(update=update):
                    request = dict(REQUEST, **update)
                    self.assertEqual(client.get("/tts", params=request).status_code, status)
                    self.assertEqual(client.post("/tts", json=request).status_code, status)
            with patch.object(SpeechRequest, "checked", autospec=True, side_effect=SpeechRequest.checked) as checked:
                response = client.get("/tts", params=[*REQUEST.items(),
                    ("aux_ref_audio_paths", "first.wav"), ("aux_ref_audio_paths", "second.wav")])
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"], "unsupported_feature")
                self.assertEqual(checked.call_args.args[0].aux_ref_audio_paths, ["first.wav", "second.wav"])

    def test_get_openapi_exposes_original_query_fields_and_defaults(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        with TestClient(create_app()) as client:
            schema = client.get("/openapi.json").json()
        operation = schema["paths"]["/tts"]["get"]
        self.assertNotIn("requestBody", operation)
        parameters = {parameter["name"]: parameter for parameter in operation["parameters"]}
        post_properties = schema["components"]["schemas"]["SpeechRequest"]["properties"]
        self.assertEqual(set(parameters), set(post_properties))
        for name, parameter in parameters.items():
            self.assertEqual(parameter["in"], "query")
            self.assertEqual(parameter["schema"], post_properties[name])
        for name, default in {"seed": -1, "text_split_method": "cut5", "top_k": 15,
                              "top_p": 1, "temperature": 1, "batch_size": 1,
                              "batch_threshold": 0.75, "split_bucket": True,
                              "streaming_mode": False, "parallel_infer": True,
                              "speed_factor": 1, "fragment_interval": 0.3,
                              "repetition_penalty": 1.35, "media_type": "wav"}.items():
            self.assertEqual(parameters[name]["schema"]["default"], default)

    def test_switch_failures_keep_original_error_envelopes(self):
        from fastapi.testclient import TestClient
        from sakuratts.server import create_app
        class FakeInference:
            def __init__(self, *args, **kwargs): pass
            def info(self): return {"name": "test"}
            def close(self): pass
            def set_weights(self, kind, path):
                if path == "unsupported":
                    raise NotImplementedError("Unsupported model version")
                raise ValueError("Cannot load " + path)
            def set_reference_audio(self, path): raise ValueError("Cannot load " + path)
        with patch("sakuratts.server.Inference", FakeInference), TestClient(create_app()) as client:
            for endpoint, field, message, required in (
                ("/set_gpt_weights", "weights_path", "change gpt weight failed", "gpt weight path is required"),
                ("/set_sovits_weights", "weights_path", "change sovits weight failed", "sovits weight path is required"),
                ("/set_refer_audio", "refer_audio_path", "set refer audio failed", "refer_audio_path is required"),
            ):
                with self.subTest(endpoint=endpoint):
                    with self.assertLogs("sakuratts.server", level="ERROR"):
                        response = client.get(endpoint, params={field: "missing.file"})
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json(), {"message": message, "Exception": "Cannot load missing.file"})
                    for params in ({}, {field: ""}):
                        response = client.get(endpoint, params=params)
                        self.assertEqual(response.status_code, 400)
                        self.assertEqual(response.json(), {"message": required})
            with self.assertLogs("sakuratts.server", level="ERROR"):
                response = client.get("/set_sovits_weights", params={"weights_path": "unsupported"})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json(), {"message": "change sovits weight failed",
                "Exception": "Unsupported model version", "error": "unsupported_feature"})

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
