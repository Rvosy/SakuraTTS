"""Keep terminal output readable without losing failure diagnostics."""

import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import unicodedata
from unittest.mock import patch

from sakuratts._internal.logging import request_id, request_scope, run_conversion, service_logging, terminal_progress_enabled


class LoggingTests(unittest.TestCase):
    def test_text_block_wraps_cjk_without_truncation_and_keeps_file_plain(self):
        # Combining dakuten must stay with its kana even at the right edge.
        source = "そういうことを、不意に言わないで。か\u3099" * 15 + "\n\nGPT / SoVITS"
        logger = logging.getLogger("sakuratts.inference")
        for columns in (24, 40, 120):
            console = io.StringIO()
            with self.subTest(columns=columns), tempfile.TemporaryDirectory() as folder, \
                    patch("sys.stderr", console), \
                    patch("shutil.get_terminal_size", return_value=os.terminal_size((columns, 24))):
                file = Path(folder) / "server.log"
                with service_logging(file):
                    logger.info("文本: %s", source, extra={"block": "text", "text": source})
                details = file.read_text(encoding="utf-8")
            lines = console.getvalue().splitlines()
            content = [line for line in lines if line.startswith("    ")]
            widths = {sum(0 if unicodedata.combining(char) else
                          2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in line)
                      for line in content}
            self.assertLess(max(widths), columns)
            text_lines = [line[4:] for line in content]
            self.assertEqual("".join(text_lines), source.replace("\n", ""))
            self.assertIn("", text_lines, "Keep explicit blank lines")
            self.assertTrue(all(not line or not unicodedata.combining(line[0]) for line in text_lines))
            self.assertIn(source, details)
            self.assertNotIn("+--", console.getvalue())
            self.assertNotIn("###", console.getvalue())
            self.assertNotIn("\x1b", console.getvalue())

    def test_console_keeps_errors_and_file_keeps_details_and_request_identity(self):
        logger = logging.getLogger("sakuratts.server")
        parent = logging.getLogger("sakuratts")
        previous = parent.handlers[:], parent.level, parent.propagate
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, patch("sys.stderr", console):
            file = Path(folder) / "server.log"
            with service_logging(file):
                with request_scope() as identifier:
                    logger.info("请求 #%s  日语", identifier, extra={"block": "request"})
                    logger.debug("完整路径 D:/voice/model.ckpt SHA256=abcdef")
                    logger.info("参考 reference.wav · 已准备")
                    try:
                        raise RuntimeError("声学设备加载失败")
                    except RuntimeError:
                        logger.exception("请求 #%s · 声学合成失败", identifier)
                self.assertEqual(request_id.get(), "-")
                access = logging.getLogger("uvicorn.access")
                access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:123", "POST", "/tts", "1.1", 200)
                access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:123", "GET", "/missing", "1.1", 404)
            details = file.read_text(encoding="utf-8")
        self.assertEqual((parent.handlers, parent.level, parent.propagate), previous)
        visible = console.getvalue()
        self.assertNotIn("INFO:", visible)
        self.assertNotIn("SHA256", visible)
        self.assertNotIn("Traceback", visible)
        self.assertNotIn("200", visible)
        self.assertIn("警告 HTTP 404", visible)
        self.assertIn("错误", visible)
        self.assertIn("声学设备加载失败", visible)
        self.assertIn("SHA256=abcdef", details)
        self.assertIn("Traceback", details)
        self.assertIn(f"[请求 {identifier}]", details)
        self.assertIn('POST /tts HTTP/1.1" 200', details)

    def test_warning_level_disables_live_progress_even_on_a_terminal(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        with tempfile.TemporaryDirectory() as folder, patch("sys.stderr", Terminal()):
            with service_logging(Path(folder) / "server.log", "warning"):
                self.assertFalse(terminal_progress_enabled(logging.getLogger("sakuratts.inference")))

    def test_preparation_output_is_saved_and_unknown_missing_weights_remain_visible(self):
        console = io.StringIO()
        known = "_IncompatibleKeys(missing_keys=['enc_q.pre.weight'], unexpected_keys=[])"
        unknown = "_IncompatibleKeys(missing_keys=['dec.weight'], unexpected_keys=[])"
        code = f"print({known!r}); print({unknown!r}); print('RuntimeWarning: genuine warning')"
        with tempfile.TemporaryDirectory() as folder, patch("sys.stderr", console):
            file = Path(folder) / "server.log"
            with service_logging(file):
                run_conversion([sys.executable, "-c", code], env=dict(os.environ, PYTHONUTF8="1"))
                with self.assertRaises(subprocess.CalledProcessError):
                    run_conversion([sys.executable, "-c", "print('conversion failed'); raise SystemExit(7)"],
                                   env=dict(os.environ, PYTHONUTF8="1"))
            details = file.read_text(encoding="utf-8")
        self.assertNotIn("enc_q.pre.weight", console.getvalue())
        self.assertIn("dec.weight", console.getvalue())
        self.assertIn("genuine warning", console.getvalue())
        self.assertIn(known, details)
        self.assertIn("conversion failed", details)


if __name__ == "__main__":
    unittest.main()
