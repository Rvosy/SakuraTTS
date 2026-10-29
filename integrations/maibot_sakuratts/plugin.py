"""MaiBot 的 SakuraTTS HTTP 客户端；推理服务独立安装和升级。"""

import asyncio
import base64
import io
import json
import wave
from typing import Any

import httpx
from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, ReplyExtension
from maibot_sdk.types import HookMode


# CPU 冷启动需要时间；HTTP 总期限低于 Host 的组件调用期限。
COMPONENT_TIMEOUT_MS = 330_000
MAX_AUDIO_BYTES = 4 * 1024 * 1024
EXTENSION_NAME = "rvosy.sakura-tts.voice"
REPLY_FORMAT = '''本次回复启用了日语语音扩展。沿用当前人物卡、资料与交流目的，在同一次生成中组织中文回复和可选的日语台词。
只输出一个 JSON 对象，包含两个字段：
{"text":"发给对方的完整中文回复","voice_ja":null}
text 是完整、可独立理解的中文对白。voice_ja 为 null 时仅发送中文。
适合用声音表达时，voice_ja 写一小段自然的日语台词，对应中文回复中的一个意思；不必逐句翻译全文，也可以选择不附语音。
日语沿用同一个角色的自称、称呼、判断和表达习惯，语音本身不意味着改变性格或增强情感。
两种语言的事实、立场与行动状态一致。日语字段只包含实际朗读的台词，中文释义和动作描述留在字段之外。
使用合法 JSON 字符串转义，直接输出对象，不加代码围栏或对象之外的说明。'''


def parse_reply(text: str, requested: bool = False) -> tuple[str, str | None]:
    """发送边界严格解析，畸形输出不能作为聊天正文放行。"""
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"text", "voice_ja"}:
        raise ValueError("语音回复必须包含 text 和 voice_ja 两个字段。")
    chinese, japanese = data["text"], data["voice_ja"]
    if not isinstance(chinese, str) or not chinese.strip():
        raise ValueError("语音回复缺少中文正文。")
    if japanese is not None and (not isinstance(japanese, str) or not japanese.strip()):
        raise ValueError("voice_ja 必须为非空日语台词或 null。")
    if requested and japanese is None:
        raise ValueError("用户明确要求语音，本次生成没有提供日语台词。")
    return chinese.strip(), japanese.strip() if japanese is not None else None


def replace_reply_text(messages: list[dict], text: str) -> list[dict]:
    """替换结构化正文，保留附件、引用和消息顺序。"""
    result = []
    inserted = False
    for message in messages:
        segments = []
        for segment in message["segments"]:
            if segment["type"] == "text":
                if not inserted:
                    segments.append({"type": "text", "data": text})
                    inserted = True
            else:
                segments.append(segment)
        if segments:
            result.append({"segments": segments, "quote_previous": message["quote_previous"]})
    if not inserted:
        result.insert(0, {"segments": [{"type": "text", "data": text}], "quote_previous": False})
    return result


class PluginSettings(PluginConfigBase):
    config_version: str = "1.0.0"


class EngineSettings(PluginConfigBase):
    __ui_label__ = "SakuraTTS 服务"
    base_url: str = Field(default="http://127.0.0.1:9880", description="服务根地址；Docker 中填写容器可访问的地址")
    timeout_seconds: float = Field(default=120, gt=0, le=300, description="一次合成的总超时秒数，包含唤醒等待")


class VoiceSettings(PluginConfigBase):
    __ui_label__ = "音色与语言"
    ref_audio_path: str = Field(default="", description="引擎所在环境中参考音频的绝对路径")
    prompt_text: str = Field(default="", description="参考音频的真实转写")
    speed_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="固定语速倍率；0.9 轻微放慢，1.0 原速。需要支持调速的引擎")


class SakuraConfig(PluginConfigBase):
    plugin: PluginSettings = Field(default_factory=PluginSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    max_text_length: int = Field(default=300, gt=0, description="单次文本字符上限，用于限制聊天中的 CPU 合成任务")


class SynthesisError(Exception):
    """可直接反馈给调用方的合成失败原因。"""


async def synthesize(client: httpx.AsyncClient, config: SakuraConfig, text: str) -> bytes:
    voice = config.voice
    if not voice.ref_audio_path.strip() or not voice.prompt_text.strip():
        raise SynthesisError("请先配置参考音频路径和转写。")
    payload = {
        "text": text,
        "text_lang": "all_ja",
        "ref_audio_path": voice.ref_audio_path,
        "prompt_text": voice.prompt_text,
        "prompt_lang": "ja",
        "streaming_mode": False,
        "media_type": "wav",
        "speed_factor": voice.speed_factor,
    }
    url = config.engine.base_url.rstrip("/") + "/tts"
    async with asyncio.timeout(config.engine.timeout_seconds):
        async with client.stream("POST", url, json=payload, timeout=config.engine.timeout_seconds) as response:
            if response.status_code == 409:
                raise SynthesisError("语音引擎正在处理其他请求，请稍后再试。")
            if response.is_error:
                # 保留服务端原因供日志排错，不把本地路径或参考转写发到聊天中。
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk[:2048 - len(body)])
                    if len(body) >= 2048:
                        break
                raise SynthesisError(f"语音引擎返回 HTTP {response.status_code}，请查看插件日志。") from RuntimeError(
                    body.decode("utf-8", errors="replace")
                )
            if response.status_code != 200:
                raise SynthesisError(f"语音接口返回 HTTP {response.status_code}，请检查服务地址。")
            status = response.headers.get("X-SakuraTTS-Status")
            if status == "stopped_at_limit":
                raise SynthesisError("合成触发生成上限，音频不完整；请缩短文本后重试。")
            if status not in (None, "completed"):
                raise SynthesisError(f"语音引擎返回未识别的完成状态：{status}")
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > MAX_AUDIO_BYTES:
                    raise SynthesisError("音频超过 4 MiB，请缩短文本。")
                data.extend(chunk)
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            if audio.getnframes() <= 0:
                raise SynthesisError("语音引擎返回了空音频。")
            frames = audio.readframes(audio.getnframes())
            if len(frames) != audio.getnframes() * audio.getnchannels() * audio.getsampwidth():
                raise SynthesisError("语音引擎返回的 WAV 数据不完整。")
    except (wave.Error, EOFError) as exc:
        raise SynthesisError("语音引擎没有返回有效的 WAV 音频。") from exc
    return bytes(data)


class SakuraTTSPlugin(MaiBotPlugin):
    config_model = SakuraConfig

    def __init__(self) -> None:
        super().__init__()
        self._client: httpx.AsyncClient | None = None
        self._synthesis_lock = asyncio.Lock()
        self._command_queue: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue()
        self._command_worker: asyncio.Task | None = None

    async def on_load(self) -> None:
        self._client = httpx.AsyncClient(trust_env=False)

    async def on_unload(self) -> None:
        if self._command_worker is not None:
            self._command_worker.cancel()
            await asyncio.gather(self._command_worker, return_exceptions=True)
            self._command_worker = None
        while not self._command_queue.empty():
            self._command_queue.get_nowait()
            self._command_queue.task_done()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        # Runner 已注入新配置；每次请求读取快照，不保留旧地址或旧音色。
        if scope == "self":
            self.ctx.logger.info("SakuraTTS 配置已更新，版本 %s", version)

    async def _audio(self, text: str) -> bytes:
        config = self.config
        text = text.strip()
        if not text:
            raise SynthesisError("请输入要朗读的日语台词。")
        if len(text) > config.max_text_length:
            raise SynthesisError(f"日语台词超过 {config.max_text_length} 字符，请缩短后重试。")
        if self._client is None:
            raise SynthesisError("语音插件尚未加载。")
        try:
            async with self._synthesis_lock:
                return await synthesize(self._client, config, text)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise SynthesisError("语音合成超时，请检查引擎负载或缩短文本。") from exc
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise SynthesisError("无法连接语音引擎，请检查服务状态和地址。") from exc

    @ReplyExtension(
        "voice",
        description="本次回复可以附日语语音时选择。用户明确要求语音选 requested；其他适合口头表达的回应选 auto，由 Replyer 决定是否附语音及具体台词。普通文字回复可不选。",
        parameters={
            "mode": {"type": "string", "enum": ["auto", "requested"], "default": "auto",
                     "description": "auto 允许 Replyer 选择不附语音；requested 表示用户明确要求语音。台词由 Replyer 生成。"},
        },
        timeout_ms=COMPONENT_TIMEOUT_MS,
    )
    async def reply_voice(self, *, phase: str, parameters: dict, text: str, messages: list,
                          reply_id: str = "", **kwargs: Any) -> dict:
        requested = parameters["mode"] == "requested"
        if phase == "prepare":
            requirement = "本次用户明确要求语音，请提供非空的日语台词。" if requested else "本次是否附语音由你结合交流决定，voice_ja 可以为 null。"
            return {"extra_prompt": REPLY_FORMAT + "\n" + requirement}
        if phase != "before_send":
            raise ValueError(f"未知回复扩展阶段：{phase}")
        chinese, japanese = parse_reply(text, requested)
        result = replace_reply_text(messages, chinese)
        if japanese is None:
            return {"messages": result}
        try:
            audio = await self._audio(japanese)
        except SynthesisError as exc:
            self.ctx.logger.warning("SakuraTTS 语音未生成 reply_id=%s：%s；原因：%s", reply_id, exc, exc.__cause__ or "无")
            if requested:
                result.append({"segments": [{"type": "text", "data": "本次日语语音未能生成：" + str(exc)}],
                               "quote_previous": False})
            return {"messages": result}
        result.append({"segments": [{"type": "voice", "data": japanese,
                                      "binary_data_base64": base64.b64encode(audio).decode("ascii")}],
                       "quote_previous": False})
        return {"messages": result}

    @HookHandler("maisaka.replyer.before_request", mode=HookMode.BLOCKING)
    async def set_output_schema(self, **kwargs: Any) -> dict:
        options = kwargs["reply_tool_args"].get("plugin_options", {})
        if EXTENSION_NAME not in options:
            return {"action": "continue", "modified_kwargs": kwargs}
        requested = options[EXTENSION_NAME]["mode"] == "requested"
        schema = {
            "name": "sakura_reply", "strict": True,
            "schema": {
                "type": "object",
                "properties": {"text": {"type": "string"},
                               "voice_ja": {"type": "string" if requested else ["string", "null"]}},
                "required": ["text", "voice_ja"], "additionalProperties": False,
            },
        }
        return {"action": "continue", "modified_kwargs": {**kwargs, "response_schema": schema}}

    @HookHandler("maisaka.reply.before_post_process", mode=HookMode.BLOCKING)
    async def preserve_envelope(self, **kwargs: Any) -> dict:
        # 结构化正文在 before_send 才解析；此处不缓存台词或直接发送。
        if EXTENSION_NAME in kwargs["reply_tool_args"].get("plugin_options", {}):
            kwargs = {**kwargs, "skip_post_process": True}
        return {"action": "continue", "modified_kwargs": kwargs}

    @Command(
        "tts",
        description="朗读指定文本：/tts こんにちは。",
        pattern=r"^\s*(?:@\S+\s+)?/(?i:tts)(?:\s+(?P<text>[\s\S]+))?$",
        timeout_ms=COMPONENT_TIMEOUT_MS,
    )
    async def speak_command(
        self, stream_id: str = "", matched_groups: dict[str, Any] | None = None, **kwargs: Any
    ) -> tuple[bool, str, int]:
        if not stream_id:
            return False, "缺少当前聊天流，无法发送语音。", 2
        text = (matched_groups or {}).get("text") or ""
        user = (kwargs.get("message") or {}).get("message_info", {}).get("user_info", {})
        requester = user.get("user_cardname") or user.get("user_nickname") or kwargs.get("user_id") or "用户"
        self._command_queue.put_nowait((stream_id, text, requester))
        if self._command_worker is None or self._command_worker.done():
            self._command_worker = asyncio.create_task(self._run_command_queue())
        return True, "已加入语音队列。", 2

    async def _run_command_queue(self) -> None:
        # 命令入队即返回，排队时间不占用 Host 的命令调用期限。
        while True:
            stream_id, text, requester = await self._command_queue.get()
            try:
                await self._send_command_voice(stream_id, text, requester)
            except Exception:
                self.ctx.logger.exception("SakuraTTS 语音任务失败，继续处理队列中的下一条。")
            finally:
                self._command_queue.task_done()

    async def _send_command_voice(self, stream_id: str, text: str, requester: str) -> tuple[bool, str, int]:
        try:
            audio = await self._audio(text)
        except SynthesisError as exc:
            self.ctx.logger.warning("SakuraTTS 手动合成失败：%s；原因：%s", exc, exc.__cause__ or "无")
            success, message = False, str(exc)
        else:
            record = (
                f"应 {json.dumps(requester, ensure_ascii=False)} 的 /tts 指令发送语音，台词由请求者提供。\n"
                f"朗读原文：{json.dumps(text.strip(), ensure_ascii=False)}"
            )
            # 台词随语音保存；Host 仅在发送成功后同步当前会话历史。
            success = await self.ctx.send.hybrid(
                [{"type": "voice", "data": record, "binary_data_base64": base64.b64encode(audio).decode("ascii")}],
                stream_id,
                processed_plain_text=record,
                sync_to_maisaka_history=True,
                maisaka_source_kind="guided_reply",
            )
            message = "语音已发送。" if success else "语音已合成，但发送失败；请检查消息适配器。"
        if not success:
            await self.ctx.send.text(message, stream_id)
        return success, message, 2


def create_plugin() -> SakuraTTSPlugin:
    return SakuraTTSPlugin()
