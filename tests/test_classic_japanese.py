"""The classic frontend profile preserves complete segment text and errors."""

import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sakuratts.array_protocol import read_message, write_message
from sakuratts.classic_japanese import ClassicJapaneseG2P


def frontend(response):
    incoming = io.BytesIO()
    write_message(incoming, response)
    incoming.seek(0)
    model = object.__new__(ClassicJapaneseG2P)
    model.process = SimpleNamespace(stdin=io.BytesIO(), stdout=incoming)
    model._closed = False
    return model


class ClassicJapaneseTests(unittest.TestCase):
    def test_complete_segment_is_one_rpc_and_raw_prosody_is_preserved(self):
        phones = ["y", "o", "]", "N", "d", "e", "#", "i", "[", "t", "a", "]", "y", "o", "."]
        model = frontend({"status": "ok", "phones": phones})
        text = "本を読んでいたよ。「また明日」って、ちゃんと言ってね。"
        self.assertEqual(model.g2p(text), phones)
        sent = model.process.stdin
        sent.seek(0)
        metadata, arrays = read_message(sent)
        self.assertEqual(metadata, {"command": "g2p", "text": text})
        self.assertEqual(arrays, {})
        self.assertEqual(sent.read(), b"")

    def test_worker_error_drops_failed_process(self):
        model = frontend({"status": "error", "error": "diagnostic failure"})
        def stop():
            model.process = None
        model._stop = Mock(side_effect=stop)
        with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
            model.g2p("こんにちは。")
        model._stop.assert_called_once()
        self.assertIsNone(model.process)

    def test_invalid_phone_response_cannot_reach_symbol_mapping(self):
        model = frontend({"status": "ok", "phones": ["a", 3]})
        model._stop = Mock()
        with self.assertRaisesRegex(ValueError, "invalid phone"):
            model.g2p("こんにちは。")
        model._stop.assert_called_once()

    def test_explicit_close_does_not_restart_worker(self):
        model = frontend({"status": "ok", "phones": []})
        model._stop = Mock()
        model._start = Mock()
        model.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            model.g2p("こんにちは。")
        model._start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
