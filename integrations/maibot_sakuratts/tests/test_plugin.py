"""Exercise the real SDK with local HTTP and Host RPC doubles, without sending messages."""

import asyncio
import base64
import importlib.util
import io
import json
import re
from pathlib import Path
import unittest
from unittest.mock import patch
import wave

if importlib.util.find_spec("maibot_sdk") is None or importlib.util.find_spec("httpx") is None:
    raise unittest.SkipTest("Install the MaiBot plugin test dependencies listed in its README")

import httpx
from maibot_sdk.context import PluginContext

spec = importlib.util.spec_from_file_location("sakura_maibot_plugin", Path(__file__).parents[1] / "plugin.py")
plugin_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin_module)


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(32000)
        wav.writeframes(b"\x00\x00" * 320)
    return output.getvalue()


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.rpc_calls = []
        self.sent = True
        self.response = httpx.Response(200, content=wav_bytes(), headers={"X-SakuraTTS-Status": "completed"})

        async def respond(request):
            self.requests.append(request)
            return self.response

        async def rpc(method, plugin_id, payload, **kwargs):
            self.rpc_calls.append((method, plugin_id, payload))
            return self.sent

        self.plugin = plugin_module.create_plugin()
        self.plugin.set_plugin_config({})
        config = self.plugin.get_plugin_config_data()
        config["voice"].update(ref_audio_path="/srv/voices/reference.wav", prompt_text="参考音声です。")
        self.plugin.set_plugin_config(config)
        self.plugin._set_context(PluginContext("rvosy.sakura-tts", rpc_call=rpc))
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.plugin._client = self.client

    async def speak(self, stream_id="", text=""):
        if stream_id:
            success, message, _ = await self.plugin._send_command_voice(stream_id, text, "用户")
        else:
            success, message, _ = await self.plugin.speak_command(matched_groups={"text": text})
        return {"success": success, "message": message}

    async def extension(self, voice="会長、お疲れさまです。", mode="auto"):
        text = json.dumps({"text": "会长，辛苦了。", "voice_ja": voice}, ensure_ascii=False)
        messages = [{"segments": [{"type": "text", "data": text}], "quote_previous": False}]
        return await self.plugin.reply_voice(phase="before_send", parameters={"mode": mode}, text=text,
                                             messages=messages, reply_id="reply-test")

    async def asyncTearDown(self):
        await self.plugin.on_unload()

    async def test_command_uses_api_v2_and_sends_standard_voice_via_sdk(self):
        result = await self.speak(stream_id="chat-1", text="こんにちは。")
        self.assertTrue(result["success"])
        request = self.requests[0]
        self.assertEqual(str(request.url), "http://127.0.0.1:9880/tts")
        self.assertEqual(json.loads(request.content), {
            "text": "こんにちは。", "text_lang": "all_ja", "ref_audio_path": "/srv/voices/reference.wav",
            "prompt_text": "参考音声です。", "prompt_lang": "ja", "streaming_mode": False, "media_type": "wav",
            "speed_factor": 1.0,
        })
        method, plugin_id, payload = self.rpc_calls[0]
        self.assertEqual(method, "cap.call")
        self.assertEqual(plugin_id, "rvosy.sakura-tts")
        self.assertEqual(payload["capability"], "send.hybrid")
        params = payload["args"]
        self.assertEqual(params["stream_id"], "chat-1")
        self.assertEqual(len(params["segments"]), 1)
        self.assertEqual(params["segments"][0]["type"], "voice")
        self.assertEqual(base64.b64decode(params["segments"][0]["binary_data_base64"]), wav_bytes())
        self.assertTrue(params["sync_to_maisaka_history"])
        self.assertEqual(params["maisaka_source_kind"], "guided_reply")

    async def test_command_history_attributes_requested_reading_without_extra_message(self):
        original = 'こんにちは。\n「Navi」です。'
        for user, fallback, expected in (
            ({"user_cardname": "群名片", "user_nickname": "昵称"}, "user-1", "群名片"),
            ({"user_nickname": "昵称"}, "user-1", "昵称"),
            ({}, "user-1", "user-1"),
            ({}, "", "用户"),
        ):
            with self.subTest(user=user, fallback=fallback):
                self.rpc_calls.clear()
                result = await self.plugin.speak_command(
                    stream_id="group-1", matched_groups={"text": original}, user_id=fallback,
                    message={"message_info": {"user_info": user}},
                )
                self.assertEqual(result, (True, "已加入语音队列。", 2))
                await self.plugin._command_queue.join()
                self.assertEqual(len(self.rpc_calls), 1)
                params = self.rpc_calls[0][2]["args"]
                record = params["segments"][0]["data"]
                self.assertEqual(record, params["processed_plain_text"])
                self.assertIn(f'应 "{expected}" 的 /tts 指令发送语音，台词由请求者提供。', record)
                self.assertEqual(json.loads(record.split("朗读原文：", 1)[1]), original)
                self.assertEqual(json.loads(self.requests[-1].content)["text"], original)

    async def test_tts_command_routes_text_and_rejects_old_or_prefixed_names(self):
        commands = [c for c in self.plugin.get_components() if c["type"] == "COMMAND"]
        self.assertEqual([c["name"] for c in commands], ["tts"])
        command = commands[0]
        pattern = re.compile(command["metadata"]["command_pattern"])
        match = pattern.fullmatch("/tts こんにちは。\n元気ですか。")
        self.assertIsNotNone(match)
        result = await getattr(self.plugin, command["metadata"]["handler_name"])(
            stream_id="chat-1", matched_groups=match.groupdict())
        self.assertTrue(result[0])
        await self.plugin._command_queue.join()
        self.assertEqual(json.loads(self.requests[0].content)["text"], "こんにちは。\n元気ですか。")
        self.assertIsNotNone(pattern.fullmatch("/tts"))
        for text in ("/sakura_tts こんにちは。", "/tts_extra こんにちは。"):
            self.assertIsNone(pattern.fullmatch(text))

    async def test_command_uses_captured_multiline_text(self):
        result = await self.plugin.speak_command(stream_id="chat-1", matched_groups={"text": "こんにちは。\n元気ですか。"})
        self.assertTrue(result[0])
        await self.plugin._command_queue.join()
        self.assertEqual(json.loads(self.requests[0].content)["text"], "こんにちは。\n元気ですか。")

    async def test_fixed_speed_applies_to_commands_and_reply_extensions(self):
        config = self.plugin.get_plugin_config_data()
        config["voice"]["speed_factor"] = 0.9
        self.plugin.set_plugin_config(config)
        self.assertTrue((await self.speak(stream_id="chat-1", text="こんにちは。"))["success"])
        await self.extension()
        self.assertEqual([json.loads(r.content)["speed_factor"] for r in self.requests], [0.9, 0.9])

    async def test_group_mention_and_command_case_preserve_original_text(self):
        command = next(c for c in self.plugin.get_components() if c["type"] == "COMMAND")
        pattern = re.compile(command["metadata"]["command_pattern"])
        original = "シム。音声合成モジュールのテストシーケンスを実行します。発話の明瞭さや音量バランスに違和感はありませんか？"
        for prefix in ("/TTS ", "/TtS ", "@N.A.V.I.  /tts ", "@N.A.V.I.  /TTS "):
            with self.subTest(prefix=prefix):
                match = pattern.search(prefix + original)
                self.assertIsNotNone(match)
                result = await getattr(self.plugin, command["metadata"]["handler_name"])(
                    stream_id="group-1", matched_groups=match.groupdict())
                self.assertTrue(result[0])
                await self.plugin._command_queue.join()
                self.assertEqual(json.loads(self.requests[-1].content)["text"], original)
        for text in ("解释一下 /tts 的用法", "@N.A.V.I. 请用 /tts 朗读", "@N.A.V.I. /tts_extra こんにちは。"):
            self.assertIsNone(pattern.search(text))

    async def test_failed_send_is_not_reported_as_success(self):
        self.sent = False
        result = await self.speak(stream_id="chat-1", text="こんにちは。")
        self.assertFalse(result["success"])
        self.assertIn("发送失败", result["message"])

    async def test_service_errors_and_bad_audio_never_send_voice_or_retry(self):
        responses = [
            httpx.Response(409),
            httpx.Response(400, json={"message": "private/path", "error": "unsupported_feature"}),
            httpx.Response(302, headers={"Location": "http://elsewhere/tts"}),
            httpx.Response(200, content=wav_bytes(), headers={"X-SakuraTTS-Status": "stopped_at_limit"}),
            httpx.Response(200, content=b'{"error":"bad"}'),
            httpx.Response(200, content=wav_bytes()[:-10]),
        ]
        for response in responses:
            with self.subTest(status=response.status_code, headers=dict(response.headers)):
                self.response = response
                before = len(self.requests)
                result = await self.speak(stream_id="chat-1", text="こんにちは。")
                self.assertFalse(result["success"])
                self.assertNotIn("private/path", result["message"])
                self.assertEqual(len(self.requests), before + 1)
                self.assertEqual([c for c in self.rpc_calls if c[2]["capability"] == "send.hybrid"], [])

    async def test_commands_queue_silently_and_send_in_order_across_chats(self):
        started = asyncio.Event()
        release = asyncio.Event()
        order = []

        async def delayed(request):
            order.append(json.loads(request.content)["text"])
            if len(order) == 1:
                started.set()
                await release.wait()
            return self.response

        async with httpx.AsyncClient(transport=httpx.MockTransport(delayed)) as client:
            self.plugin._client = client
            async with asyncio.timeout(2):
                await self.plugin.speak_command(stream_id="chat-1", matched_groups={"text":"一番。"})
                await started.wait()
                await self.plugin.speak_command(stream_id="chat-2", matched_groups={"text":"二番。"})
                await self.plugin.speak_command(stream_id="chat-1", matched_groups={"text":"三番。"})
                await asyncio.sleep(0)
                self.assertEqual(order, ["一番。"])
                self.assertEqual(self.rpc_calls, [])
                release.set()
                await self.plugin._command_queue.join()
            self.assertEqual(order, ["一番。", "二番。", "三番。"])
            self.assertEqual([c[2]["capability"] for c in self.rpc_calls], ["send.hybrid"] * 3)
            self.assertEqual([c[2]["args"]["stream_id"] for c in self.rpc_calls], ["chat-1", "chat-2", "chat-1"])
        self.plugin._client = self.client

    async def test_queue_continues_after_synthesis_failure(self):
        async def fail_first(request):
            text = json.loads(request.content)["text"]
            return httpx.Response(400) if text == "一番。" else self.response
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail_first)) as client:
            self.plugin._client = client
            await self.plugin.speak_command(stream_id="chat-1", matched_groups={"text":"一番。"})
            await self.plugin.speak_command(stream_id="chat-2", matched_groups={"text":"二番。"})
            async with asyncio.timeout(2):
                await self.plugin._command_queue.join()
            self.assertEqual([c[2]["capability"] for c in self.rpc_calls], ["send.text", "send.hybrid"])
            self.assertEqual(self.rpc_calls[-1][2]["args"]["stream_id"], "chat-2")
        self.plugin._client = self.client

    async def test_unload_cancels_current_and_pending_command_audio(self):
        started = asyncio.Event()
        async def delayed(request):
            started.set()
            await asyncio.Event().wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(delayed)) as client:
            self.plugin._client = client
            await self.plugin.speak_command(stream_id="chat-1", matched_groups={"text":"一番。"})
            await started.wait()
            await self.plugin.speak_command(stream_id="chat-2", matched_groups={"text":"二番。"})
            async with asyncio.timeout(2):
                await self.plugin.on_unload()
                await self.plugin._command_queue.join()
            self.assertFalse(self.plugin._synthesis_lock.locked())
            self.assertTrue(client.is_closed)
            self.assertEqual(self.rpc_calls, [])
        self.plugin._client = self.client

    async def test_total_timeout_closes_response_and_allows_next_request(self):
        class SlowAudio(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b"RIFF"
                await asyncio.Event().wait()

            async def aclose(self):
                self.closed = True

        stream = SlowAudio()
        self.response = httpx.Response(200, stream=stream)
        config = self.plugin.get_plugin_config_data()
        config["engine"]["timeout_seconds"] = 0.02
        self.plugin.set_plugin_config(config)
        result = await self.speak(stream_id="chat-1", text="こんにちは。")
        self.assertFalse(result["success"])
        self.assertIn("超时", result["message"])
        self.assertTrue(stream.closed)
        self.assertEqual([c for c in self.rpc_calls if c[2]["capability"] == "send.hybrid"], [])
        self.response = httpx.Response(200, content=wav_bytes())
        self.assertTrue((await self.speak(stream_id="chat-1", text="こんにちは。"))["success"])

    async def test_config_update_changes_engine_and_voice_without_reloading(self):
        await self.speak(stream_id="chat-1", text="こんにちは。")
        config = self.plugin.get_plugin_config_data()
        config["engine"]["base_url"] = "http://engine-v2:9880/"
        config["voice"].update(ref_audio_path="/srv/voices/new.wav", prompt_text="新しい音声です。")
        self.plugin.set_plugin_config(config)
        await self.plugin.on_config_update("self", config, "1.0.0")
        await self.speak(stream_id="chat-1", text="こんにちは。")
        self.assertEqual(str(self.requests[-1].url), "http://engine-v2:9880/tts")
        self.assertEqual(json.loads(self.requests[-1].content)["ref_audio_path"], "/srv/voices/new.wav")

    async def test_configuration_and_input_errors_do_not_call_engine(self):
        for text in ("", "a" * 301):
            self.assertFalse((await self.speak(stream_id="chat-1", text=text))["success"])
        self.assertFalse((await self.speak(text="こんにちは。"))["success"])
        self.plugin.set_plugin_config({})
        self.assertFalse((await self.speak(stream_id="chat-1", text="こんにちは。"))["success"])
        self.assertEqual(self.requests, [])

    async def test_audio_size_bound_prevents_ipc_copy(self):
        with patch.object(plugin_module, "MAX_AUDIO_BYTES", 10):
            result = await self.speak(stream_id="chat-1", text="こんにちは。")
        self.assertFalse(result["success"])
        self.assertEqual([c for c in self.rpc_calls if c[2]["capability"] == "send.hybrid"], [])

    async def test_sdk_config_schema_and_component_registration(self):
        self.assertTrue(self.plugin.build_config_schema(plugin_id="rvosy.sakura-tts"))
        components = self.plugin.get_components()
        self.assertEqual({entry["type"] for entry in components}, {"REPLY_EXTENSION", "COMMAND", "HOOK_HANDLER"})
        self.assertTrue(all(entry["metadata"]["timeout_ms"] == 330_000 for entry in components if entry["type"] != "HOOK_HANDLER"))

    async def test_unload_closes_http_connections(self):
        await self.plugin.on_unload()
        self.assertTrue(self.client.is_closed)
        await self.plugin.on_unload()

    async def test_extension_returns_chinese_and_japanese_voice_without_sending(self):
        result = await self.extension()
        self.assertEqual(result["messages"][0]["segments"], [{"type": "text", "data": "会长，辛苦了。"}])
        voice = result["messages"][1]["segments"][0]
        self.assertEqual(voice["data"], "会長、お疲れさまです。")
        self.assertEqual(base64.b64decode(voice["binary_data_base64"]), wav_bytes())
        payload = json.loads(self.requests[0].content)
        self.assertEqual(payload["text"], voice["data"])
        self.assertEqual(payload["text_lang"], "all_ja")
        self.assertEqual(payload["prompt_lang"], "ja")
        self.assertEqual(self.rpc_calls, [])

    async def test_null_voice_keeps_chinese_without_engine_or_send(self):
        result = await self.extension(None)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.rpc_calls, [])

    async def test_invalid_envelopes_are_rejected_before_synthesis(self):
        for raw in ['plain text', '```json\n{}\n```', '{}', '[]',
                    '{"text":"中文","voice_ja":3}', '{"text":"","voice_ja":null}',
                    '{"text":"中文","voice_ja":null,"other":1}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                await self.plugin.reply_voice(phase="before_send", parameters={"mode": "auto"},
                                              text=raw, messages=[])
        with self.assertRaises(ValueError):
            await self.extension(None, mode="requested")
        self.assertEqual(self.requests, [])
        self.assertEqual(self.rpc_calls, [])

    async def test_extension_preserves_attachments_and_quotes(self):
        raw = json.dumps({"text": "中文回复", "voice_ja": None})
        image = {"type": "image", "data": "image", "hash": "abc"}
        at = {"type": "at", "data": {"target_user_id": "friend"}}
        messages = [{"segments": [at, {"type": "text", "data": raw}], "quote_previous": False},
                    {"segments": [image], "quote_previous": True}]
        result = await self.plugin.reply_voice(phase="before_send", parameters={"mode": "auto"},
                                              text=raw, messages=messages)
        self.assertEqual(result["messages"][0]["segments"], [at, {"type": "text", "data": "中文回复"}])
        self.assertEqual(result["messages"][1], messages[1])
        self.assertEqual(messages[0]["segments"][1]["data"], raw)

    async def test_synthesis_failure_retains_chinese_and_reports_requested_voice(self):
        for mode in ["auto", "requested"]:
            self.response = httpx.Response(409)
            result = await self.extension(mode=mode)
            self.assertEqual(result["messages"][0]["segments"][0]["data"], "会长，辛苦了。")
            self.assertTrue(all(s["type"] == "text" for m in result["messages"] for s in m["segments"]))
            self.assertEqual(len(result["messages"]), 2 if mode == "requested" else 1)
            self.assertEqual(self.rpc_calls, [])

    async def test_prepare_and_postprocess_hook_only_affect_selected_reply(self):
        prepared = await self.plugin.reply_voice(phase="prepare", parameters={"mode": "auto"}, text="", messages=[])
        self.assertIn('"voice_ja":null', prepared["extra_prompt"])
        self.assertEqual(self.requests, [])
        args = {"reply_tool_args": {}, "response": "中文", "enable_chinese_typo": False, "skip_post_process": False}
        self.assertEqual((await self.plugin.preserve_envelope(**args))["modified_kwargs"], args)
        args["reply_tool_args"] = {"plugin_options": {plugin_module.EXTENSION_NAME: {"mode": "auto"}}}
        result = (await self.plugin.preserve_envelope(**args))["modified_kwargs"]
        self.assertTrue(result["skip_post_process"])
        self.assertFalse(result["enable_chinese_typo"])
        self.assertEqual(result["reply_tool_args"], args["reply_tool_args"])

    async def test_output_schema_only_applies_to_selected_reply_and_preserves_handoff(self):
        args = {"reply_tool_args": {"reply_reference": "交接原文"}, "extra_prompt": "角色资料", "attempt": 1}
        self.assertEqual((await self.plugin.set_output_schema(**args))["modified_kwargs"], args)
        for mode in ["auto", "requested"]:
            args["reply_tool_args"]["plugin_options"] = {plugin_module.EXTENSION_NAME: {"mode": mode}}
            result = (await self.plugin.set_output_schema(**args))["modified_kwargs"]
            self.assertEqual(result["reply_tool_args"], args["reply_tool_args"])
            self.assertEqual(result["extra_prompt"], args["extra_prompt"])
            self.assertNotIn("response_schema", args)
            expected = "string" if mode == "requested" else ["string", "null"]
            self.assertEqual(result["response_schema"]["schema"]["properties"]["voice_ja"]["type"], expected)


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("numpy") and importlib.util.find_spec("sakuratts"),
                     "Install server dependencies for the ASGI contract test")
class ServerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_accepts_current_server_wav_and_rejects_stopped_output(self):
        import numpy as np
        from sakuratts.engine import Audio
        from sakuratts.server import create_app

        class InferenceDouble:
            status = "completed"

            def __init__(self, *args, **kwargs):
                pass

            def info(self):
                return None

            def close(self):
                pass

            def tts(self, *args, **kwargs):
                return Audio(np.zeros(320, dtype=np.int16), 32000,
                             {"status": self.status, "request_ms": 1})

        config = plugin_module.SakuraConfig()
        config.voice.ref_audio_path = "/srv/voice.wav"
        config.voice.prompt_text = "参考音声です。"
        with patch("sakuratts.server.Inference", InferenceDouble):
            app = create_app()
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
                    audio = await plugin_module.synthesize(client, config, "こんにちは。")
                    self.assertTrue(audio.startswith(b"RIFF"))
                    InferenceDouble.status = "stopped_at_limit"
                    with self.assertRaisesRegex(plugin_module.SynthesisError, "不完整"):
                        await plugin_module.synthesize(client, config, "こんにちは。")


if __name__ == "__main__":
    unittest.main()
