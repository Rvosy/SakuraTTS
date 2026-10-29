"""Verify frequency limits and the image allowlist with the real SDK and a fake Host."""

import asyncio
import base64
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from maibot_sdk.context import PluginContext, PluginPaths
from maibot_sdk.types import ToolComponentInfo

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("navi_expressions", ROOT / "plugin.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExpressionsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.paths = PluginPaths(data_dir=Path(self.directory.name), runtime_dir=Path(self.directory.name) / "runtime")
        self.calls = []
        self.sent = True
        self.error = None
        self.clock = patch.object(module.time, "time", return_value=1000)
        self.now = self.clock.start()
        self.roll = patch.object(module.random, "random", return_value=0)
        self.random = self.roll.start()
        self.plugin = await self.make_plugin()

    async def make_plugin(self):
        async def rpc(method, plugin_id, payload, **kwargs):
            self.calls.append(payload)
            await asyncio.sleep(0)
            if self.error:
                raise self.error
            await plugin.record_sent_expression(sent=self.sent, message={
                "session_id": payload["args"]["stream_id"], "raw_message": payload["args"]["segments"]})
            return {"success": self.sent, "message_id": "test-picture" if self.sent else None}
        plugin = module.create_plugin()
        plugin.set_plugin_config({})
        plugin._set_context(PluginContext("rvosy.navi-expressions", rpc_call=rpc, paths=self.paths))
        await plugin.on_load()
        return plugin

    async def asyncTearDown(self):
        await self.plugin.on_unload()
        self.roll.stop()
        self.clock.stop()
        self.directory.cleanup()

    async def test_three_choices_send_exact_original_and_history_description(self):
        for emotion, (filename, _) in module.EXPRESSIONS.items():
            with self.subTest(emotion=emotion):
                result = await self.plugin.send_expression(emotion=emotion, stream_id=emotion)
                self.assertTrue(result["sent"])
                self.assertEqual(result["message_id"], "test-picture")
                call = self.calls[-1]
                self.assertEqual(call["capability"], "send.hybrid")
                args = call["args"]
                self.assertEqual(args["stream_id"], emotion)
                self.assertEqual(len(args["segments"]), 1)
                segment = args["segments"][0]
                self.assertEqual(segment["type"], "emoji")
                self.assertEqual(base64.b64decode(segment["binary_data_base64"]), (ROOT / "assets" / filename).read_bytes())
                self.assertTrue(args["sync_to_maisaka_history"])
                self.assertEqual(args["processed_plain_text"], segment["data"])

    async def test_invalid_selection_and_missing_chat_never_send(self):
        for emotion in ["", "angry", "../happy.png", "https://example.com/a.png"]:
            result = await self.plugin.send_expression(emotion=emotion, stream_id="group")
            self.assertFalse(result["success"])
        self.assertFalse((await self.plugin.send_expression(emotion="开心"))["success"])
        self.assertFalse((await self.plugin.send_expression(emotion="开心", mode="force", stream_id="group"))["success"])
        self.assertEqual(self.calls, [])

    async def test_probability_miss_is_silent_and_cannot_reroll_by_changing_emotion(self):
        self.random.return_value = 0.25
        result = await self.plugin.send_expression(emotion="开心", stream_id="group")
        self.assertEqual(result["reason"], "probability")
        self.random.return_value = 0
        self.assertEqual((await self.plugin.send_expression(emotion="无奈认真", stream_id="group"))["reason"], "interval")
        self.assertEqual(self.random.call_count, 1)
        self.assertEqual(self.calls, [])
        self.now.return_value += 30
        self.assertTrue((await self.plugin.send_expression(emotion="开心", stream_id="group"))["sent"])

    async def test_success_cooldown_survives_reload_and_is_per_chat(self):
        self.assertTrue((await self.plugin.send_expression(emotion="开心", stream_id="group"))["sent"])
        self.plugin = await self.make_plugin()
        self.now.return_value = 1029
        self.assertFalse((await self.plugin.send_expression(emotion="无奈认真", stream_id="group"))["sent"])
        self.assertTrue((await self.plugin.send_expression(emotion="意料之外", stream_id="private"))["sent"])
        self.now.return_value = 1060
        self.random.return_value = 0.04
        self.assertTrue((await self.plugin.send_expression(emotion="无奈认真", stream_id="group"))["sent"])

    async def test_concurrent_calls_send_at_most_one_image_per_chat(self):
        results = await asyncio.gather(*(self.plugin.send_expression(emotion=e, stream_id="group") for e in module.EXPRESSIONS))
        self.assertEqual(sum(r["sent"] for r in results), 1)
        self.assertEqual(len(self.calls), 1)

    async def test_platform_failure_is_not_reported_as_sent_and_does_not_retry(self):
        self.sent = False
        result = await self.plugin.send_expression(emotion="开心", stream_id="group")
        self.assertFalse(result["success"])
        self.assertFalse(result["sent"])
        self.sent = True
        self.now.return_value += 29
        self.assertFalse((await self.plugin.send_expression(emotion="开心", stream_id="group"))["sent"])
        self.now.return_value += 1
        self.assertTrue((await self.plugin.send_expression(emotion="开心", stream_id="group"))["sent"])

    async def test_uncertain_send_keeps_attempt_interval_without_claiming_success(self):
        self.error = TimeoutError("platform timed out")
        with self.assertRaises(TimeoutError):
            await self.plugin.send_expression(emotion="开心", stream_id="group")
        self.plugin = await self.make_plugin()
        self.error = None
        self.now.return_value += 29
        self.assertFalse((await self.plugin.send_expression(emotion="开心", stream_id="group"))["sent"])
        self.assertEqual(len(self.calls), 1)

    async def test_probability_config_boundaries(self):
        self.plugin.set_plugin_config({"plugin": {"config_version": "1.0.0"}, "max_probability": 0})
        self.assertFalse((await self.plugin.send_expression(emotion="开心", stream_id="zero"))["sent"])
        self.plugin.set_plugin_config({"plugin": {"config_version": "1.0.0"}, "max_probability": 1})
        self.random.return_value = 0.999
        self.assertTrue((await self.plugin.send_expression(emotion="开心", stream_id="one"))["sent"])

    async def test_explicit_request_bypasses_probability_and_all_cooldowns(self):
        self.random.return_value = 0.99
        self.assertFalse((await self.plugin.send_expression(emotion="开心", stream_id="group"))["sent"])
        for emotion in module.EXPRESSIONS:
            self.assertTrue((await self.plugin.send_expression(emotion=emotion, mode="requested", stream_id="group"))["sent"])
        self.assertEqual(self.random.call_count, 1)
        self.assertEqual(len(self.calls), 3)
        self.plugin = await self.make_plugin()
        self.assertFalse((await self.plugin.send_expression(emotion="开心", stream_id="group"))["sent"])
        self.assertTrue((await self.plugin.send_expression(emotion="开心", mode="requested", stream_id="group"))["sent"])

    async def test_time_weight_recovers_and_caps_without_reset_on_miss(self):
        config = self.plugin.config
        self.assertEqual(module.auto_probability(config, 1000, None), 0.25)
        for elapsed, expected in [(-1, 0), (0, 0), (29, 0), (60, 0.05), (180, 0.15), (300, 0.25), (600, 0.25)]:
            with self.subTest(elapsed=elapsed):
                self.assertAlmostEqual(module.auto_probability(config, 1000 + elapsed, 1000), expected)
        await self.plugin.happy(mode="requested", stream_id="group")
        self.random.return_value = 0.10
        self.now.return_value = 1060
        self.assertFalse((await self.plugin.send_expression("开心", stream_id="group"))["sent"])
        self.plugin = await self.make_plugin()
        self.now.return_value = 1180
        self.assertTrue((await self.plugin.send_expression("开心", stream_id="group"))["sent"])

    async def test_tools_are_discoverable_and_each_sends_its_specific_image(self):
        components = [c for c in self.plugin.get_components() if c["type"] == "TOOL"]
        self.assertEqual(len(components), 3)
        for tool in components:
            self.assertTrue(tool["metadata"]["core_tool"])
            schema = ToolComponentInfo(name=tool["name"], **tool["metadata"]).get_parameters_schema()
            self.assertEqual(set(schema["properties"]), {"mode"})
            self.assertEqual(schema["properties"]["mode"]["enum"], ["requested"])
        for handler, emotion in [(self.plugin.happy, "开心"), (self.plugin.serious, "无奈认真"), (self.plugin.unexpected, "意料之外")]:
            result = await handler(mode="requested", stream_id="group")
            self.assertTrue(result["sent"])
            self.assertEqual(result["emotion"], emotion)

    async def test_reply_context_preserves_character_handoff_and_voice_schema(self):
        original = {"extra_prompt": "Existing character context", "reply_tool_args": {"handoff": "original"},
                    "response_schema": {"type": "object"}, "session_id": "group"}
        result = await self.plugin.expression_context(**original)
        updated = result["modified_kwargs"]
        self.assertEqual(result["action"], "continue")
        self.assertEqual(updated["reply_tool_args"], original["reply_tool_args"])
        self.assertEqual(updated["response_schema"], original["response_schema"])
        self.assertEqual(original["extra_prompt"], "Existing character context")
        self.assertTrue(updated["extra_prompt"].startswith(original["extra_prompt"]))
        for emotion in module.EXPRESSIONS:
            self.assertIn(emotion, updated["extra_prompt"])
        self.assertEqual((await self.plugin.expression_context(**updated))["modified_kwargs"], updated)

    async def test_reply_attaches_after_text_and_voice_without_sending_early(self):
        messages = [{"segments": [{"type": "text", "data": "谢谢。"}], "quote_previous": True},
                    {"segments": [{"type": "voice", "data": "ありがとう", "binary_data_base64": "YQ=="}], "quote_previous": False}]
        result = await self.plugin.reply_expression(phase="before_send", parameters={"emotion": "开心"},
                                                   messages=messages, session_id="group")
        self.assertEqual(result["messages"][:-1], messages)
        self.assertEqual(len(messages), 2)
        self.assertEqual(result["messages"][-1]["segments"][0]["type"], "emoji")
        self.assertIn("开心", result["messages"][-1]["segments"][0]["data"])
        self.assertEqual(self.calls, [])
        self.assertNotIn("last_sent", self.plugin._state["group"])
        await self.plugin.record_sent_expression(sent=True, message={"session_id": "group", "raw_message": result["messages"][-1]["segments"]})
        self.assertEqual(self.plugin._state["group"]["last_sent"], 1000)
        self.plugin = await self.make_plugin()
        self.assertEqual(self.plugin._state["group"]["last_sent"], 1000)

    async def test_reply_probability_miss_keeps_normal_text(self):
        messages = [{"segments": [{"type": "text", "data": "谢谢。"}], "quote_previous": False}]
        self.random.return_value = 0.9
        result = await self.plugin.reply_expression(phase="before_send", parameters={"emotion": "开心"}, messages=messages, session_id="group")
        self.assertEqual(result, {"messages": messages})
        self.assertEqual(self.calls, [])
        result = await self.plugin.reply_expression(phase="before_send", parameters={"emotion": "开心", "mode": "requested"}, messages=messages, session_id="group")
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(self.random.call_count, 1)

    async def test_reply_without_dialogue_and_prepare_do_not_send_or_reserve(self):
        for emotion in module.EXPRESSIONS:
            prepared = await self.plugin.reply_expression(phase="prepare", parameters={"emotion": emotion}, messages=[], session_id="group")
            self.assertIn(emotion, prepared["extra_prompt"])
        self.assertEqual(await self.plugin.reply_expression(phase="before_send", parameters={"emotion": "开心"}, messages=[], session_id="group"), {"messages": []})
        self.assertEqual(self.plugin._state, {})
        self.assertEqual(self.calls, [])

    async def test_failed_reply_sticker_and_unrelated_messages_do_not_record_success(self):
        for sent, segment in [(False, self.plugin._segment("开心")), (True, {"type": "text", "data": "谢谢"}),
                              (True, {"type": "emoji", "binary_data_base64": "YQ=="})]:
            await self.plugin.record_sent_expression(sent=sent, message={"session_id": "group", "raw_message": [segment]})
        self.assertEqual(self.plugin._state, {})

    async def test_independent_tools_reject_autonomous_sending(self):
        for handler in [self.plugin.happy, self.plugin.serious, self.plugin.unexpected]:
            self.assertFalse((await handler(mode="auto", stream_id="group"))["sent"])
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
