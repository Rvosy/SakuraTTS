"""Exercise the installed plugin against a real managed engine, without sending messages."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import tomllib
import wave

import httpx
from maibot_sdk.context import PluginContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugin import create_plugin


async def check(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plugin = create_plugin()
    plugin.set_plugin_config(tomllib.loads(args.config.read_text(encoding="utf-8")))

    async def no_send(*args, **kwargs):
        raise RuntimeError("The integration check must not send messages")

    plugin._set_context(PluginContext("rvosy.sakura-tts", rpc_call=no_send))
    await plugin.on_load()
    report = {"status": "running", "requests": [], "runtime": []}
    responses = []

    async def record(response):
        responses.append({"http_status": response.status_code,
                          "headers": {k: v for k, v in response.headers.items()
                                      if k.startswith("x-sakuratts-")}})

    plugin._client.event_hooks["response"].append(record)
    base_url = plugin.config.engine.base_url.rstrip("/")

    async def transform(voice):
        text = json.dumps({"text": "这是一条语音联调消息。", "voice_ja": voice}, ensure_ascii=False)
        return await plugin.reply_voice(
            phase="before_send", parameters={"mode": "auto" if voice is None else "requested"},
            text=text, messages=[{"segments": [{"type": "text", "data": text}], "quote_previous": False}],
            reply_id="engine-check")

    async def synthesize(label):
        start = time.perf_counter()
        messages = (await transform(args.text))["messages"]
        voices = [s for m in messages for s in m["segments"] if s["type"] == "voice"]
        if len(voices) != 1 or voices[0]["data"] != args.text:
            raise RuntimeError(f"Expected one voice message: {messages}")
        audio = base64.b64decode(voices[0]["binary_data_base64"], validate=True)
        elapsed = time.perf_counter() - start
        with wave.open(io.BytesIO(audio), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            duration = wav.getnframes() / wav.getframerate()
        response = responses[-1]
        if response["headers"].get("x-sakuratts-status") != "completed":
            raise RuntimeError(f"Incomplete synthesis: {response}")
        (output / f"{label}.wav").write_bytes(audio)
        report["requests"].append({"label": label, "seconds": elapsed, "audio_seconds": duration,
                                   "pcm_sha256": hashlib.sha256(frames).hexdigest(), **response})
        print(f"{label}: {elapsed:.2f}s, audio {duration:.2f}s", flush=True)

    try:
        async with httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=10) as client:
            async def state():
                response = await client.get("/runtime")
                response.raise_for_status()
                return response.json()

            initial = await state()
            if initial["state"] != "sleeping":
                raise RuntimeError("Start a fresh isolated managed service in sleeping state")
            report["runtime"].append(initial)
            messages = (await transform(None))["messages"]
            if responses or any(s["type"] == "voice" for m in messages for s in m["segments"]):
                raise RuntimeError("Text-only reply unexpectedly called the engine")
            if (await state())["state"] != "sleeping":
                raise RuntimeError("Text-only reply woke the engine")
            await synthesize("cold")
            awake = await state()
            report["runtime"].append(awake)
            await synthesize("warm")
            if args.wait_for_sleep:
                deadline = time.monotonic() + initial["idle_sleep_seconds"] + 45
                while True:
                    current = await state()
                    if current["state"] == "sleeping":
                        break
                    if current["state"] == "failed" or time.monotonic() >= deadline:
                        raise RuntimeError(f"Idle worker did not sleep: {current}")
                    await asyncio.sleep(1)
                report["runtime"].append(current)
                await synthesize("rewake")
                resumed = await state()
                if resumed["generation"] <= awake["generation"]:
                    raise RuntimeError("Worker generation did not change after sleep")
                report["runtime"].append(resumed)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        await plugin.on_unload()
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Plugin config.toml for the isolated engine")
    parser.add_argument("--output", type=Path, required=True, help="New directory for WAV files and result.json")
    parser.add_argument("--text", default="こんにちは。Hello, world!")
    parser.add_argument("--wait-for-sleep", action="store_true", help="Verify automatic idle sleep and another cold request")
    asyncio.run(check(parser.parse_args()))


if __name__ == "__main__":
    main()
