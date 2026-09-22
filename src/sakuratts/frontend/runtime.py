"""Build text resources independently of semantic and acoustic backends."""

from dataclasses import dataclass
from importlib import metadata
import json
import os
from pathlib import Path

from .profiles import JAPANESE


@dataclass
class FrontendRuntime:
    text: object
    components: tuple
    profile: dict

    def close(self):
        for component in self.components:
            component.close()


def load_frontend(config_path, config, package, manifest):
    """Construct only the implementation declared by this resource package."""
    from .text_frontend import LanguageSegmenter, TextFrontend
    from .processors import JapaneseProcessor, EnglishProcessor
    from sakuratts._internal.reference_condition import sha256_file

    if manifest["format"] != JAPANESE.resource_format:
        raise ValueError("Unsupported frontend resource package")
    languages = {JAPANESE.code}
    if "english_g2p" in manifest:
        languages.add("en")
    if not set(config.get("languages", (JAPANESE.code,))).issubset(languages):
        raise ValueError("This frontend resource package provides " + ", ".join(sorted(languages)) + " only")
    root = Path(config_path).parent
    package = Path(package).resolve()
    if not {"symbols-v2.json", "user.dict", "lid.176.bin"}.issubset(manifest["files"]):
        raise ValueError("Incomplete frontend resource package")
    for name, spec in manifest["files"].items():
        path = (package / name).resolve(strict=True)
        if package not in path.parents:
            raise ValueError("Frontend resource must remain inside its package")
        if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"Frontend resource checksum mismatch: {name}")
    profile = manifest.get("japanese_g2p", {"implementation": "pyopenjtalk-plus"})
    japanese = segmenter = None
    try:
        if profile["implementation"] == "pyopenjtalk-classic":
            from .classic_japanese import ClassicJapaneseG2P
            python = config.get("frontend_python", config.get("acoustic_python"))
            if profile.get("version") != "0.3.4" or not python:
                raise ValueError("This classic frontend package requires version 0.3.4 and its prepared Python runtime")
            directories = {}
            for key in ("module_directory", "main_dictionary"):
                path = (package / profile[key]).resolve(strict=True)
                if package not in path.parents or not path.is_dir():
                    raise ValueError("Classic frontend directories must remain inside their package")
                directories[key] = path
            japanese = ClassicJapaneseG2P(root / python, directories["module_directory"],
                directories["main_dictionary"], package / "user.dict")
        elif profile["implementation"] == "pyopenjtalk-plus":
            from .japanese import JapaneseG2P
            dictionary = config.get("main_dictionary")
            dictionary = (root / dictionary).resolve(strict=True) if dictionary else Path(
                metadata.distribution("pyopenjtalk-plus").locate_file("pyopenjtalk/dictionary"))
            os.environ["OPEN_JTALK_DICT_DIR"] = str(dictionary)
            japanese = JapaneseG2P(dictionary, package / "user.dict")
        else:
            raise ValueError("Unsupported Japanese frontend implementation")
        segmenter = LanguageSegmenter(package)
        symbols = json.loads((package / "symbols-v2.json").read_text(encoding="utf-8"))
        processors = {JAPANESE.code: JapaneseProcessor(japanese)}
        if "english_g2p" in manifest:
            from .english import EnglishG2P
            english = manifest["english_g2p"]
            if english != {"implementation": "gpt-sovits-english-v1", "directory": "english"}:
                raise ValueError("Unsupported English frontend profile")
            if not {"english/g2p.json", "english/checkpoint.npz"}.issubset(manifest["files"]):
                raise ValueError("Incomplete English frontend resources")
            processors["en"] = EnglishProcessor(EnglishG2P(package / "english", symbols))
        text = TextFrontend(processors, symbols, segmenter)
        return FrontendRuntime(text, (japanese, segmenter), profile)
    except BaseException as error:
        for component in (segmenter, japanese):
            if component is not None:
                try:
                    component.close()
                except BaseException as cleanup_error:
                    error.add_note(f"Frontend construction cleanup failed: {cleanup_error!r}")
        raise
