"""Terminal presentation and rotating diagnostics for the standalone service."""

from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
import ast
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata

request_id = ContextVar("sakuratts_request_id", default="-")
stage = ContextVar("sakuratts_stage", default="推理")
_requests = count(1)


@contextmanager
def request_scope():
    token = request_id.set(f"{next(_requests):04d}")
    phase = stage.set("参考准备")
    try:
        yield request_id.get()
    finally:
        stage.reset(phase)
        request_id.reset(token)


def set_stage(value):
    if request_id.get() != "-":
        stage.set(value)


def compact(value, limit=160):
    value = str(value).replace("\r", "\\r").replace("\n", "\\n")
    value = "".join(char if char.isprintable() else " " for char in value)
    return value if len(value) <= limit else value[:limit - 1] + "…"


def _display_width(value):
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in value)


def _wrap_columns(value, columns):
    """Wrap CJK text by terminal cells, keeping explicit line breaks."""
    value = str(value).replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    value = "".join(char for char in value if char.isprintable() or char == "\n")
    for paragraph in value.split("\n"):
        line, width = "", 0
        for char in paragraph:
            size = _display_width(char)
            if line and width + size > columns:
                yield line
                line, width = "", 0
            line += char
            width += size
        yield line


def terminal_progress_enabled(logger):
    current = logger
    while current is not None:
        for handler in current.handlers:
            if isinstance(handler, ConsoleHandler):
                return handler.level <= logging.INFO and handler.stream.isatty()
        if not current.propagate:
            break
        current = current.parent
    return sys.stderr.isatty()


class RequestFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id.get()
        return True


class ConsoleFilter(logging.Filter):
    def filter(self, record):
        if record.name == "uvicorn.access":
            return isinstance(record.args, tuple) and len(record.args) == 5 and int(record.args[4]) >= 400
        if record.name.startswith("uvicorn") and record.levelno < logging.WARNING:
            return str(record.msg).startswith("Uvicorn running on")
        return True


class ConsoleFormatter(logging.Formatter):
    def __init__(self, stream):
        super().__init__(datefmt="%H:%M:%S")
        self.stream = stream

    def _columns(self):
        try:
            return os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            return shutil.get_terminal_size((96, 24)).columns

    def _text_block(self, value):
        width = max(2, min(76, self._columns() - 5))
        return "  文本\n" + "\n".join("    " + line for line in _wrap_columns(value, width))

    def format(self, record):
        level = record.levelno
        message = record.getMessage()
        block = getattr(record, "block", None)
        if block == "text":
            return "\n" + self._text_block(record.text) + "\n"
        if record.name == "uvicorn.access":
            _, method, path, _, status = record.args
            message = f"HTTP {status} · {method} {str(path).split('?')[0]}"
            level = logging.ERROR if int(status) >= 500 else logging.WARNING
        elif record.name.startswith("uvicorn") and str(record.msg).startswith("Uvicorn running on"):
            message = "服务就绪，按 Ctrl+C 停止。"
        elif record.exc_info and record.exc_info[1] is not None:
            message += f" | {type(record.exc_info[1]).__name__}: {record.exc_info[1]}"
        message = compact(message, 500)
        label = "错误 " if level >= logging.ERROR else "警告 " if level >= logging.WARNING else ""
        heading = block in ("startup", "request")
        if heading or label:
            prefix = self.formatTime(record, self.datefmt) + "  "
        else:
            prefix = "    " if block == "progress" else "  "
        text = prefix + label + message
        if self.stream.isatty() and "NO_COLOR" not in os.environ:
            if level >= logging.ERROR:
                color = "31"
            elif level >= logging.WARNING:
                color = "33"
            elif block == "complete":
                color = "32"
            elif heading:
                color = "36"
            else:
                color = ""
            if color:
                text = f"\033[{color}m{text}\033[0m"
        if block == "request":
            return "\n" + "-" * max(1, min(80, self._columns() - 1)) + "\n" + text
        if block in ("stage", "complete"):
            return "\n" + text
        return text


class ConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            message = self.format(record)
            module = sys.modules.get("tqdm")
            if module is not None:
                module.tqdm.write(message, file=self.stream)
            else:
                self.stream.write(message + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


@contextmanager
def service_logging(log_file, level="info"):
    """Keep host logging untouched outside the standalone server lifetime."""
    path = Path(log_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    console = ConsoleHandler(sys.stderr)
    console.setLevel(level.upper())
    console.setFormatter(ConsoleFormatter(console.stream))
    console.addFilter(ConsoleFilter())
    file = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.addFilter(RequestFilter())
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] [请求 %(request_id)s] %(message)s"))
    saved = []
    for name in ("sakuratts", "uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        saved.append((logger, logger.level, logger.handlers[:], logger.propagate, logger.disabled))
        logger.handlers = [console, file]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.disabled = False
    try:
        yield path
    finally:
        for logger, previous_level, handlers, propagate, disabled in saved:
            logger.handlers = handlers
            logger.setLevel(previous_level)
            logger.propagate, logger.disabled = propagate, disabled
        console.close()
        file.close()


def log_result(audio):
    logger = logging.getLogger("sakuratts.server")
    report = audio.report
    duration = report.get("pcm_seconds", len(audio.pcm) / audio.sample_rate)
    seconds = report["request_ms"] / 1000
    limited = report["status"] == "stopped_at_limit"
    label = "达到长度上限" if limited else "完成"
    logger.log(logging.WARNING if limited else logging.INFO,
        "%s #%s  音频 %.2f s · 总耗时 %.3f s · RTF %.3f", label, request_id.get(),
        duration, seconds, seconds / duration if duration else 0, extra={"block": "complete"})
    if "fragments" in report:
        semantic = sum(part["timings"]["semantic_seconds"] for part in report["fragments"])
        acoustic = sum(part["timings"]["acoustic_seconds"] for part in report["fragments"])
        logger.debug("参考 %.3f s · 文本 %.3f s · GPT %.3f s · SoVITS %.3f s",
            report.get("reference_ms", 0) / 1000, report["frontend_ms"] / 1000, semantic, acoustic)
    logger.debug("合成报告: %s", report)


def run_conversion(command, *, env):
    """Stream preparation output to diagnostics without retaining it in RAM."""
    logger = logging.getLogger("sakuratts.converter")
    if not logger.isEnabledFor(logging.DEBUG):
        return subprocess.run(command, check=True, env=env)
    logger.debug("运行准备命令: %r", command)
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
        try:
            for line in process.stdout:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                logger.debug("准备进程: %s", line)
                match = re.search(r"_IncompatibleKeys\(missing_keys=(\[.*?\]), unexpected_keys=(\[.*?\])\)", line)
                if match:
                    try:
                        missing, unexpected = (ast.literal_eval(value) for value in match.groups())
                    except (SyntaxError, ValueError):
                        logger.warning("无法解析权重加载结果，详见日志文件")
                    else:
                        if unexpected or any(not str(key).startswith("enc_q.") for key in missing):
                            logger.warning("权重存在未匹配项  %s", compact(str(missing) + " / " + str(unexpected)))
                elif "Warning:" in line or "WARNING" in line:
                    logger.warning("准备进程  %s", compact(line))
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    return subprocess.CompletedProcess(command, code)
