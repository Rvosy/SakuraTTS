"""Build text resources independently of semantic and acoustic backends."""

from contextlib import ExitStack
from dataclasses import dataclass
from importlib import metadata
import json
import os
from pathlib import Path

from ..text.profiles import JAPANESE


@dataclass
class FrontendRuntime:
    text: object
    components: ExitStack
    profile: dict

    def close(self):
        self.components.close()


def load_frontend(config_path, config, package, manifest):
    """Construct only the implementation declared by this resource package."""
    from ..text.LangSegmenter import LanguageSegmenter
    from .TextPreprocessor import TextFrontend
    from ..text.processors import ChineseProcessor, EnglishProcessor, JapaneseProcessor

    if manifest["format"] != JAPANESE.resource_format:
        raise ValueError("Unsupported frontend resource package")
    languages = {JAPANESE.code}
    if "english_g2p" in manifest:
        languages.add("en")
    if "chinese_g2p" in manifest:
        languages.add("zh")
    if not set(config.get("languages", (JAPANESE.code,))).issubset(languages):
        raise ValueError("This frontend resource package provides " + ", ".join(sorted(languages)) + " only")
    root = Path(config_path).parent
    package = Path(package).resolve()
    profile = manifest.get("japanese_g2p", {"implementation": "pyopenjtalk-plus"})
    components = ExitStack()
    try:
        if profile["implementation"] == "pyopenjtalk-classic":
            from ..text.classic_japanese import ClassicJapaneseG2P
            python = config.get("frontend_python", config.get("acoustic_python"))
            if not python:
                raise ValueError("The classic frontend requires its prepared Python runtime")
            japanese = ClassicJapaneseG2P(root / python, package / profile["module_directory"],
                package / profile["main_dictionary"], package / "user.dict")
        elif profile["implementation"] == "pyopenjtalk-plus":
            from ..text.japanese import JapaneseG2P
            dictionary = config.get("main_dictionary")
            dictionary = (root / dictionary).resolve(strict=True) if dictionary else Path(
                metadata.distribution("pyopenjtalk-plus").locate_file("pyopenjtalk/dictionary"))
            os.environ["OPEN_JTALK_DICT_DIR"] = str(dictionary)
            japanese = JapaneseG2P(dictionary, package / "user.dict")
        else:
            raise ValueError("Unsupported Japanese frontend implementation")
        components.callback(japanese.close)
        segmenter = LanguageSegmenter(package)
        components.callback(segmenter.close)
        symbols = json.loads((package / "symbols-v2.json").read_text(encoding="utf-8"))
        processors = {JAPANESE.code: JapaneseProcessor(japanese)}
        if "english_g2p" in manifest:
            from ..text.english import EnglishG2P
            english = manifest["english_g2p"]
            if english["implementation"] != "gpt-sovits-english-v1":
                raise ValueError("Unsupported English frontend profile")
            processors["en"] = EnglishProcessor(EnglishG2P(package / english["directory"], symbols))
        if "chinese_g2p" in manifest:
            spec = manifest["chinese_g2p"]
            if spec["implementation"] != "gpt-sovits-chinese-v2-onnx":
                raise ValueError("Unsupported Chinese frontend profile")
            chinese = ChineseProcessor(package / spec["directory"])
            components.callback(chinese.close)
            processors["zh"] = chinese
        text = TextFrontend(processors, symbols, segmenter)
        return FrontendRuntime(text, components, profile)
    except BaseException as error:
        try:
            components.close()
        except BaseException as cleanup_error:
            error.add_note(f"Frontend construction cleanup failed: {cleanup_error!r}")
        raise
