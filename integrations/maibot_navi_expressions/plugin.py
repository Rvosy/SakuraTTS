"""让 N.A.V.I. 用三种固定表情包表达自己的情绪。"""

import asyncio
import base64
import json
from pathlib import Path
import random
import time
from typing import Any

from maibot_sdk import Field, HookHandler, MaiBotPlugin, PluginConfigBase, ReplyExtension, Tool
from maibot_sdk.types import HookMode, HookOrder


EXPRESSIONS = {
    "无奈认真": ("无奈认真.png", "无奈认真：睁眼、微微张嘴，带着无奈认真解释或提醒"),
    "意料之外": ("意料之外.png", "意料之外：眉眼下垂、嘴微张，对突然的转折感到意外或措手不及"),
    "开心": ("开心.png", "开心：闭眼微笑，表现愉快、满足或温柔的谢意"),
}
TOOL_RULES = (
    "这是已有的角色表情包，只能发送开心、无奈认真、意料之外三种，不能画出新表情。"
    "独立工具仅用于用户明确要求单独看表情或样貌，使用 requested。"
    "日常聊天要配表情时，在 reply 的 plugin_options 中选择 rvosy.navi-expressions.expression，"
    "先确定回应适合哪种神情，再随正文发送；不要先单独发表情再调用 reply。"
    "普通夸奖不算要求发表情，尊重用户不发表情的要求。只有 sent=true 才表示已发送。"
)
REPLY_CONTEXT = (
    "【N.A.V.I. 的表情】\n"
    "你可以在聊天中用自己的三张固定表情包表达神情：开心（闭眼微笑）、无奈认真、意料之外。"
    "问你能发哪些表情时，只说明这三种；这项能力是发送已有表情包，不能生成害羞、不满等新图片。"
    "用户说‘笑一个’可以用开心回应；有人夸你时也可以自然地开心，但不必每次都配表情。"
    "把表情当作自己的神情，用平常的角色口吻回应；无需播报工具、发送流程或解释实体面孔。"
    "日常聊天以回复正文为主，可以随这次回复配一张合适的表情；不为发表情额外凑一句话，"
    "也不逐字复述画面。只有用户明确要求单独看表情时才单独展示。"
    "画面里的房间是固定插画背景，不代表你此刻所在的位置，也不代表你与某人现实中待在一起。"
    "被问到背景时据此自然说明，不编造陪伴现场。只有工具结果 sent=true 才能说已发表情，"
    "未发送时正常继续聊天，不解释概率或冷却。"
)
MODE_SCHEMA = {
    "type": "object",
    "properties": {"mode": {
        "type": "string", "enum": ["requested"], "default": "requested",
        "description": "仅用于用户本轮明确要求单独看表情或样貌；普通夸奖应通过 reply 扩展配表情。",
    }},
    "additionalProperties": False,
}


class PluginSettings(PluginConfigBase):
    config_version: str = "1.0.0"


class ExpressionConfig(PluginConfigBase):
    plugin: PluginSettings = Field(default_factory=PluginSettings)
    max_probability: float = Field(default=0.25, ge=0, le=1, description="自主发送的最高概率")
    recovery_seconds: int = Field(default=300, ge=1, description="距上次发送多久后恢复到最高概率；此前按时间线性恢复")
    min_gap_seconds: int = Field(default=30, ge=1, description="同一会话自主发送的最短间隔；用户主动要求不受限制")
    retry_seconds: int = Field(default=30, ge=1, description="同一会话两次自主抽取的最短间隔，防止换表情反复抽取")


def auto_probability(config: ExpressionConfig, now: float, last_sent: float | None) -> float:
    if last_sent is None:
        return config.max_probability
    elapsed = max(0.0, now - last_sent)
    if elapsed < config.min_gap_seconds:
        return 0.0
    return config.max_probability * min(1.0, elapsed / config.recovery_seconds)


class NaviExpressionsPlugin(MaiBotPlugin):
    config_model = ExpressionConfig

    def __init__(self) -> None:
        super().__init__()
        self._lock = asyncio.Lock()
        self._state: dict[str, dict[str, float]] = {}
        self._images: dict[str, str] = {}

    async def on_load(self) -> None:
        self._state_path = self.ctx.paths.data_dir / "send_state.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        if self._state_path.exists():
            self._state = json.loads(self._state_path.read_text(encoding="utf-8"))
        for emotion, (filename, _) in EXPRESSIONS.items():
            self._images[emotion] = base64.b64encode((Path(__file__).parent / "assets" / filename).read_bytes()).decode("ascii")

    async def on_unload(self) -> None:
        pass

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        pass

    @HookHandler("maisaka.replyer.before_request", mode=HookMode.BLOCKING, order=HookOrder.LATE)
    async def expression_context(self, **kwargs: Any) -> dict[str, Any]:
        extra = kwargs.get("extra_prompt") or ""
        if REPLY_CONTEXT not in extra:
            extra = "\n\n".join(part for part in (extra, REPLY_CONTEXT) if part)
        return {"action": "continue", "modified_kwargs": {**kwargs, "extra_prompt": extra}}

    def _save_state(self, stream_id: str, value: dict[str, float]) -> None:
        state = {**self._state, stream_id: value}
        temporary = self._state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(self._state_path)
        self._state = state

    def _reserve_attempt(self, stream_id: str, mode: str) -> str | None:
        now = time.time()
        previous = self._state.get(stream_id, {})
        if mode == "auto" and now < previous.get("next_attempt", 0):
            return "interval"
        self._save_state(stream_id, {**previous, "next_attempt": now + self.config.retry_seconds})
        if mode == "auto":
            probability = auto_probability(self.config, now, previous.get("last_sent"))
            if probability == 0 or random.random() >= probability:
                return "probability"
        return None

    def _segment(self, emotion: str) -> dict[str, str]:
        return {"type": "emoji", "data": f"N.A.V.I. 的表情包：{emotion}",
                "binary_data_base64": self._images[emotion]}

    @ReplyExtension(
        "expression",
        description="本次说话时可配一张自己的表情包。先判断话题和准备说的内容：开心适合愉快感谢或微笑回应；无奈认真适合无奈纠正、认真提醒；意料之外适合没想到的转折。选定相符的 emotion 后随 reply 正文发送；不合适或不想配就省略此扩展。普通夸奖用 auto，用户明确要求配表情才用 requested。不要另外调用独立表情工具。",
        parameters={
            "emotion": {"type": "string", "enum": list(EXPRESSIONS), "required": True,
                        "description": "与本次回复内容一致的神情，调用前确定，不随机挑选。"},
            "mode": {"type": "string", "enum": ["auto", "requested"], "default": "auto",
                     "description": "auto 按距上次发送的时间加权；requested 仅用于用户明确要求配表情，跳过概率和间隔。"},
        },
        priority=100,
    )
    async def reply_expression(self, *, phase: str, parameters: dict, messages: list,
                               session_id: str = "", **kwargs: Any) -> dict:
        emotion = parameters["emotion"]
        mode = parameters.get("mode", "auto")
        if emotion not in EXPRESSIONS or mode not in {"auto", "requested"}:
            raise ValueError("表情或发送模式无效。")
        if phase == "prepare":
            return {"extra_prompt": f"这次回复选择配「{emotion}」表情：{EXPRESSIONS[emotion][1]}。"
                    "正文自然表达当前回应，表情随正文发送，无需口头介绍、预告或解释表情。"
                    "自主配图可能因频率控制省略，正文必须独立成立，不写‘看这个表情’之类依赖附件的话。"}
        if phase != "before_send":
            raise ValueError(f"未知回复扩展阶段：{phase}")
        # 表情跟随实际对白；普通回复的引用、分段、附件和语音全部保留。
        if not any(s.get("type") == "text" and s.get("data", "").strip()
                   for message in messages for s in message["segments"]):
            return {"messages": messages}
        if not session_id:
            raise ValueError("缺少当前会话，无法附带表情。")
        async with self._lock:
            if self._reserve_attempt(session_id, mode):
                return {"messages": messages}
            return {"messages": [*messages, {"segments": [self._segment(emotion)], "quote_previous": False}]}

    @HookHandler("send_service.after_send", mode=HookMode.BLOCKING)
    async def record_sent_expression(self, *, message: dict, sent: bool, **kwargs: Any) -> dict:
        if sent and message.get("session_id") and any(
            s.get("type") == "emoji" and s.get("binary_data_base64") in self._images.values()
            for s in message.get("raw_message", [])
        ):
            # 此处不等待发送锁；独立工具的发送回执也会在同一调用链触发本 Hook。
            now = time.time()
            self._save_state(message["session_id"], {"last_sent": now, "next_attempt": now + self.config.retry_seconds})
        return {"action": "continue"}

    async def _requested(self, emotion: str, mode: str, stream_id: str) -> dict:
        if mode != "requested":
            return {"success": False, "sent": False, "error": "自主表情请通过 reply 扩展随正文发送。"}
        return await self.send_expression(emotion, mode, stream_id)

    @Tool(
        "navi_expression_happy",
        description="用户明确要求‘笑一个’‘看看你的笑脸’时，单独展示 N.A.V.I. 的「开心」表情包：闭眼微笑。" + TOOL_RULES,
        parameters=MODE_SCHEMA,
        core_tool=True,
    )
    async def happy(self, mode: str = "requested", stream_id: str = "", **kwargs: Any) -> dict[str, Any]:
        return await self._requested("开心", mode, stream_id)

    @Tool(
        "navi_expression_serious",
        description="用户点名要看时，单独展示 N.A.V.I. 的「无奈认真」表情包：睁眼微微张嘴，带着无奈认真说明；不是愤怒或嘲讽。" + TOOL_RULES,
        parameters=MODE_SCHEMA,
        core_tool=True,
    )
    async def serious(self, mode: str = "requested", stream_id: str = "", **kwargs: Any) -> dict[str, Any]:
        return await self._requested("无奈认真", mode, stream_id)

    @Tool(
        "navi_expression_unexpected",
        description="用户点名要看时，单独展示 N.A.V.I. 的「意料之外」表情包：眉眼下垂、嘴微张，略显措手不及。" + TOOL_RULES,
        parameters=MODE_SCHEMA,
        core_tool=True,
    )
    async def unexpected(self, mode: str = "requested", stream_id: str = "", **kwargs: Any) -> dict[str, Any]:
        return await self._requested("意料之外", mode, stream_id)

    async def send_expression(self, emotion: str, mode: str = "auto", stream_id: str = "") -> dict[str, Any]:
        if emotion not in EXPRESSIONS:
            return {"success": False, "sent": False, "error": "只能选择无奈认真、意料之外或开心。"}
        if mode not in {"auto", "requested"}:
            return {"success": False, "sent": False, "error": "mode 只能选择 auto 或 requested。"}
        if not stream_id:
            return {"success": False, "sent": False, "error": "缺少当前会话，未发送表情。"}
        async with self._lock:
            reason = self._reserve_attempt(stream_id, mode)
            if reason:
                return self._skipped(reason)
            segment = self._segment(emotion)
            result = await self.ctx.send.hybrid(
                segments=[segment],
                stream_id=stream_id,
                processed_plain_text=segment["data"],
                sync_to_maisaka_history=True,
                maisaka_source_kind="guided_reply",
                return_details=True,
            )
            if not result["sent"]:
                return {"success": False, "sent": False, "error": "表情未发送成功，本轮不要重试。"}
            return {"success": True, "sent": True, "emotion": emotion, "message_id": result.get("message_id")}

    @staticmethod
    def _skipped(reason: str) -> dict[str, Any]:
        return {"success": True, "sent": False, "reason": reason,
                "instruction": "本次不发送表情，正常继续对话。不要重试，也不要解释跳过原因。"}


def create_plugin() -> NaviExpressionsPlugin:
    return NaviExpressionsPlugin()
