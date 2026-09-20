"""GPT-SoVITS api_v2 request contract over the native inference engine."""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from functools import partial
import io
import logging
import math
import shutil
import subprocess
import threading
from typing import Optional, Union
import wave

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from .engine import Audio, Inference
from ._internal.generation import SynthesisCancelled
from ._internal.logging import log_result, request_id, request_scope, service_logging, set_stage, stage

logger = logging.getLogger("sakuratts.server")


class SpeechRequest(BaseModel):
    """Field names and defaults follow the pinned upstream api_v2.py."""
    model_config = ConfigDict(allow_inf_nan=False)
    text: Optional[str] = None
    text_lang: Optional[str] = None
    ref_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[list[str]] = None
    prompt_lang: Optional[str] = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16

    def checked(self):
        request = self.model_dump()
        for field in ("ref_audio_path", "text", "text_lang", "prompt_lang"):
            if not request[field] or not request[field].strip():
                raise ValueError(field + " is required")
        for field in ("text_lang", "prompt_lang"):
            request[field] = request[field].lower()
            if request[field] not in ("ja", "all_ja"):
                raise NotImplementedError(field + ": native inference currently supports ja and all_ja")
        if request["text_split_method"] not in {"cut" + str(i) for i in range(6)}:
            raise ValueError("text_split_method: " + request["text_split_method"] + " is not supported")
        if request["media_type"] not in ("wav", "raw", "ogg", "aac"):
            raise ValueError("media_type: " + request["media_type"] + " is not supported")
        if request["streaming_mode"] not in (0, 1, 2, 3):
            raise ValueError("streaming_mode must be 0, 1, 2, 3 or true/false")
        unsupported = []
        if request["top_p"] != 1:
            unsupported.append("top_p != 1 (sampling parity is not verified)")
        if request["speed_factor"] != 1:
            unsupported.append("speed_factor != 1")
        if request["batch_size"] != 1:
            unsupported.append("batch_size != 1")
        if request["streaming_mode"] in (2, 3):
            unsupported.append("streaming_mode 2/3 (semantic-token streaming)")
        if request["aux_ref_audio_paths"]:
            unsupported.append("aux_ref_audio_paths")
        if not request["prompt_text"].strip():
            unsupported.append("empty prompt_text")
        if request["super_sampling"]:
            unsupported.append("super_sampling")
        if unsupported:
            raise NotImplementedError("Not implemented by the native backend: " + ", ".join(unsupported))
        if request["seed"] < -1 or request["top_k"] < 1:
            raise ValueError("Require seed >= -1 and top_k >= 1")
        for field in ("temperature", "repetition_penalty"):
            if not math.isfinite(request[field]) or request[field] <= 0:
                raise ValueError(field + " must be finite and positive")
        if request["fragment_interval"] < 0:
            raise ValueError("fragment_interval must be nonnegative")
        if request["media_type"] in ("ogg", "aac") and not shutil.which("ffmpeg"):
            raise ValueError("ffmpeg is required for " + request["media_type"] + " output")
        return request


def pack_audio(pcm, rate, media_type):
    if media_type == "wav":
        return Audio(pcm, rate, {}).wav_bytes()
    raw = pcm.astype("<i2", copy=False).tobytes()
    if media_type == "raw":
        return raw
    codec = ["-c:a", "aac", "-b:a", "192k", "-f", "adts"] if media_type == "aac" else ["-c:a", "libvorbis", "-f", "ogg"]
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le",
        "-ar", str(rate), "-ac", "1", "-i", "pipe:0", *codec, "pipe:1"],
        input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout


def wave_header(rate):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"")
    return stream.getvalue()


def error_response(error, *, synthesis=False):
    if isinstance(error, NotImplementedError):
        return JSONResponse({"message": str(error), "error": "unsupported_feature"}, status_code=400)
    if synthesis:
        return JSONResponse({"message": "tts failed", "Exception": str(error)}, status_code=400)
    return JSONResponse({"message": str(error)}, status_code=400)


def create_app(model=None, *, tts_config=None, experimental=None, control=None):
    @asynccontextmanager
    async def lifespan(app):
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sakuratts")
        app.state.pool = pool
        app.state.busy = False
        app.state.jobs = set()
        app.state.streams = set()
        app.state.inference = None
        app.state.model_info = None
        try:
            inference = await asyncio.get_running_loop().run_in_executor(pool,
                partial(Inference, model, tts_config=tts_config, experimental=experimental))
            app.state.inference = inference
            app.state.model_info = inference.info()
            if inference.info() is None:
                logger.warning("尚未加载模型，请通过 --tts-config 指定配置")
            logger.debug("No default reference audio. Specify ref_audio_path, prompt_text and prompt_lang in /tts.")
            yield
        finally:
            for stop in app.state.streams:
                stop.set()
            if app.state.jobs:
                await asyncio.gather(*app.state.jobs, return_exceptions=True)
            if app.state.inference is not None:
                await asyncio.get_running_loop().run_in_executor(pool, app.state.inference.close)
            pool.shutdown(wait=True)
            logger.info("模型与工作进程已关闭")

    app = FastAPI(title="SakuraTTS", version="0.1.0a1", lifespan=lifespan)

    def start_job(operation):
        if app.state.busy:
            return None
        app.state.busy = True
        async def run():
            try:
                return await asyncio.get_running_loop().run_in_executor(app.state.pool, operation)
            finally:
                app.state.model_info = app.state.inference.info()
                app.state.busy = False
        job = asyncio.create_task(run())
        app.state.jobs.add(job)
        def finished(task):
            app.state.jobs.discard(task)
            if not task.cancelled():
                task.exception()
        job.add_done_callback(finished)
        return job

    def busy_response():
        return JSONResponse({"message": "Inference is busy; retry after the current operation completes"}, status_code=409)

    @app.get("/health")
    async def health():
        return {"status": "busy" if app.state.busy else "ready", "model_loaded": app.state.model_info is not None,
                "model": app.state.model_info, "compatibility": "api_v2 native subset"}

    @app.get("/models")
    async def models():
        return [app.state.model_info] if app.state.model_info is not None else []

    async def run_control(operation):
        job = start_job(operation)
        if job is None:
            return busy_response()
        try:
            await asyncio.shield(job)
            return JSONResponse({"message": "success"})
        except Exception as error:
            logger.exception("模型或参考音频操作失败")
            return error_response(error)

    @app.get("/set_gpt_weights")
    async def set_gpt_weights(weights_path: Optional[str] = None):
        if not weights_path:
            return error_response(ValueError("gpt weight path is required"))
        return await run_control(partial(app.state.inference.set_weights, "gpt", weights_path))

    @app.get("/set_sovits_weights")
    async def set_sovits_weights(weights_path: Optional[str] = None):
        if not weights_path:
            return error_response(ValueError("sovits weight path is required"))
        return await run_control(partial(app.state.inference.set_weights, "sovits", weights_path))

    @app.get("/set_refer_audio")
    async def set_refer_audio(refer_audio_path: Optional[str] = None):
        if not refer_audio_path:
            return error_response(ValueError("refer_audio_path is required"))
        return await run_control(partial(app.state.inference.set_reference_audio, refer_audio_path))

    @app.get("/control")
    async def control_endpoint(command: Optional[str] = None):
        if command not in ("exit", "restart"):
            return error_response(ValueError("command must be exit or restart"))
        if control is None:
            return error_response(ValueError("Process control is unavailable for an embedded ASGI app"))
        control(command)
        return {"message": "success"}

    def run_synthesis(values, *, on_fragment=None, cancel_requested=None):
        with request_scope():
            try:
                audio = app.state.inference.tts(values, on_fragment=on_fragment, cancel_requested=cancel_requested)
                if cancel_requested is not None and cancel_requested():
                    raise SynthesisCancelled("stream_output")
                set_stage("音频编码")
                data = None if on_fragment is not None else pack_audio(audio.pcm, audio.sample_rate, values["media_type"])
                log_result(audio)
                return audio, data
            except SynthesisCancelled:
                logger.warning("取消 #%s · %s", request_id.get(), stage.get(), extra={"block": "complete"})
                logger.debug("推理已取消", exc_info=True)
                raise
            except Exception:
                logger.exception("请求 #%s · %s失败", request_id.get(), stage.get(), extra={"block": "complete"})
                raise

    async def tts(request):
        try:
            values = request.checked()
        except Exception as error:
            return error_response(error)
        if values["streaming_mode"]:
            return await stream_tts(values)
        job = start_job(partial(run_synthesis, values))
        if job is None:
            return busy_response()
        try:
            audio, data = await asyncio.shield(job)
        except Exception as error:
            return error_response(error, synthesis=True)
        return Response(data, media_type="audio/" + values["media_type"], headers={
            "X-SakuraTTS-Status": audio.report["status"],
            "X-SakuraTTS-Request-Ms": str(round(audio.report["request_ms"], 2)), "Cache-Control": "no-store"})

    async def stream_tts(values):
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=2)
        stop = threading.Event()
        def put(value):
            if stop.is_set():
                return
            future = asyncio.run_coroutine_threadsafe(queue.put(value), loop)
            while True:
                try:
                    future.result(timeout=0.1)
                    return
                except FutureTimeout:
                    if stop.is_set():
                        future.cancel()
                        return
        def fragment(pcm, rate):
            if stop.is_set():
                from ._internal.generation import SynthesisCancelled
                raise SynthesisCancelled("stream_output")
            media = "raw" if values["media_type"] == "wav" else values["media_type"]
            previous = stage.get()
            set_stage("音频编码")
            encoded = pack_audio(pcm, rate, media)
            set_stage("音频发送")
            put((rate, encoded))
            set_stage(previous)
        def synthesize():
            try:
                run_synthesis(values, on_fragment=fragment, cancel_requested=stop.is_set)
            except Exception as error:
                if not stop.is_set():
                    put(error)
            finally:
                put(None)
        job = start_job(synthesize)
        if job is None:
            return busy_response()
        app.state.streams.add(stop)
        try:
            first = await queue.get()
        except BaseException:
            stop.set()
            app.state.streams.discard(stop)
            raise
        if first is None or isinstance(first, Exception):
            stop.set()
            app.state.streams.discard(stop)
            return error_response(first or ValueError("No audio generated"), synthesis=True)
        async def body():
            try:
                if values["media_type"] == "wav":
                    yield wave_header(first[0])
                yield first[1]
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item[1]
            finally:
                stop.set()
                app.state.streams.discard(stop)
        return StreamingResponse(body(), media_type="audio/" + values["media_type"])

    @app.post("/tts")
    async def tts_post(request: SpeechRequest):
        return await tts(request)

    @app.get("/tts")
    async def tts_get(request: Request):
        values = dict(request.query_params)
        if "aux_ref_audio_paths" in values:
            values["aux_ref_audio_paths"] = request.query_params.getlist("aux_ref_audio_paths")
        if values.get("streaming_mode") in ("0", "1", "2", "3"):
            values["streaming_mode"] = int(values["streaming_mode"])
        try:
            parsed = SpeechRequest.model_validate(values)
        except ValidationError as error:
            return JSONResponse({"detail": error.errors(include_context=False)}, status_code=422)
        return await tts(parsed)

    return app


def start_server(model=None, *, host="127.0.0.1", port=9880, tts_config=None, experimental=None,
                 log_file="logs/sakuratts.log", log_level="info"):
    import uvicorn
    with service_logging(log_file, log_level) as path:
        while True:
            action = []
            def control(command):
                action.append(command)
                server.should_exit = True
            logger.info("SakuraTTS · 推理服务", extra={"block": "startup"})
            logger.info("地址  http://%s:%d", host, port)
            logger.info("日志  %s", path)
            app = create_app(model, tts_config=tts_config, experimental=experimental, control=control)
            server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, workers=1, log_config=None))
            server.run()
            if not server.started:
                raise RuntimeError("Server startup failed; see the startup error above")
            if not action or action[-1] != "restart":
                return
