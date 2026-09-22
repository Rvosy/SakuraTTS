"""English normalization and phone rules against fixed upstream examples."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from sakuratts.frontend.english import EnglishG2P


class EnglishFrontendTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("inflect"), "Install the english extra")
    def test_normalization_matches_official_examples(self):
        from sakuratts.frontend._vendor.english_normalization import normalize
        frontend = EnglishG2P.__new__(EnglishG2P)
        frontend.normalize_numbers = normalize
        fixture = json.loads((Path(__file__).parent / "fixtures/gpt_sovits_english.json").read_text(encoding="utf-8"))
        for probe in fixture["probes"]:
            with self.subTest(text=probe["text"]):
                self.assertEqual(frontend.normalize(probe["text"]), probe["normalized"])

    def make_frontend(self):
        frontend = EnglishG2P.__new__(EnglishG2P)
        frontend.cmu = {"cat": [["K", "AE1", "T"]], "dog": [["D", "AO1", "G"]],
            "james": [["JH", "EY1", "M", "Z"]], "a": [["AH0"]], "i": [["AY1"]],
            "book": [["B", "UH1", "K"]], "case": [["K", "EY1", "S"]]}
        frontend.namedict = {"sakura": [["S", "AA1", "K", "UH0", "R", "AH0"]]}
        frontend.homograph2features = {"read": (["R", "IY1", "D"], ["R", "EH1", "D"], "VBP")}
        frontend.splitter = SimpleNamespace(segment=lambda word: ["book", "case"] if word == "bookcase" else [word])
        frontend.predict = Mock(return_value=["UNK"])
        return frontend

    def test_dictionary_name_compound_and_oov_routes(self):
        frontend = self.make_frontend()
        self.assertEqual(frontend.qryword("cat"), ["K", "AE1", "T"])
        self.assertEqual(frontend.qryword("Sakura"), ["S", "AA1", "K", "UH0", "R", "AH0"])
        self.assertEqual(frontend.qryword("bookcase"), ["B", "UH1", "K", "K", "EY1", "S"])
        self.assertEqual(frontend.qryword("AI"), ["EY1", "AY1"])
        self.assertEqual(frontend.qryword("qzxyplugh"), ["UNK"])
        frontend.predict.assert_called_once_with("qzxyplugh")

    def test_possessives_do_not_mutate_cached_dictionary(self):
        frontend = self.make_frontend()
        for word, suffix in (("cat", ["S"]), ("dog", ["Z"]), ("james", ["AH0", "Z"])):
            original = frontend.cmu[word][0][:]
            self.assertEqual(frontend.qryword(word + "'s"), original + suffix)
            self.assertEqual(frontend.cmu[word][0], original)

    def test_homographs_and_single_letters_keep_case_and_pos(self):
        frontend = self.make_frontend()
        frontend.tokenize = str.split
        for tag, expected in (("VBP", "IY1"), ("VB", "IY1"), ("VBD", "EH1")):
            frontend.tag = lambda words: [(word, tag) for word in words]
            self.assertEqual(frontend("read A a"), ["R", expected, "D", " ", "EY1", " ", "AH0"])


if __name__ == "__main__":
    unittest.main()
