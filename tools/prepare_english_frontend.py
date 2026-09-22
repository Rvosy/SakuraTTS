"""Add offline English resources to a new copy of a Japanese frontend package.

Export with an explicitly selected, trusted GPT-SoVITS preparation interpreter.
The original source, frontend and interpreter remain unchanged. Ordinary
inference reads JSON/NPZ, never the upstream dictionary/tagger pickle files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SOURCE_HASHES = {
    "GPT_SoVITS/text/english.py": "77837d1dfe2a21664d7cc3162e21f623e09cae93f1c96f0fa91ae97718fc0afb",
    "GPT_SoVITS/text/en_normalization/expend.py": "a42d670da8d10665283c53cd95d18c5b5b840b1912075d12d41a213587fffe27",
}
PROBES = ["Please check the audio.", "Hello, world!", "I read a complex book.",
          "AI and GPU", "OpenAI's SakuraTTS", "At 12:30, it costs $3.50.",
          "A cat's toy and James's book.", "supercalifragilisticexpialidocious",
          "qzxyplugh", "Wait... really?!", "Version 2.5 at 10km/h."]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(root, output):
    # Runs in the existing preparation environment, which may be Python 3.9.
    def offline(event, args):
        if event == "socket.connect":
            raise RuntimeError("English preparation is offline; install the missing local resource first")
    sys.addaudithook(offline)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "GPT_SoVITS"))
    # Upstream would create its dictionary cache on import if it were absent.
    if not (root / "GPT_SoVITS/text/engdict_cache.pickle").is_file():
        raise FileNotFoundError("Prepare the upstream English dictionary cache before exporting")
    import nltk
    def no_download(*args, **kwargs):
        raise RuntimeError("Missing local NLTK resource; automatic download is disabled")
    nltk.download = no_download
    from text import english
    import numpy as np
    from importlib.metadata import version
    if version("g2p-en") != "2.1.0":
        raise ValueError("English export requires g2p-en 2.1.0")
    g2p = english._g2p
    tagger = nltk.tag._get_tagger()
    output.mkdir()
    write_json(output / "g2p.json", {
        "cmu": g2p.cmu, "namedict": g2p.namedict, "homographs": g2p.homograph2features,
        "graphemes": g2p.graphemes, "phonemes": g2p.phonemes,
        "tagger": {"weights": tagger.model.weights, "tagdict": tagger.tagdict,
                   "classes": sorted(tagger.classes)},
    })
    np.savez(output / "checkpoint.npz", **dict(g2p.variables))
    write_json(output / "probes.json", [dict(text=text, normalized=english.text_normalize(text),
        phones=english.g2p(english.text_normalize(text))) for text in PROBES])
    write_json(output / "source.json", {"sources": SOURCE_HASHES,
        "versions": {name: version(name) for name in ("g2p-en", "nltk", "wordsegment", "inflect")}})
    # Keep the dictionary attribution with the exported data.
    cmu = Path(nltk.data.find("corpora/cmudict")) / "README"
    shutil.copy2(cmu, output / "CMUdict-README.txt")


def prepare(frontend, output, source, python):
    frontend, source = frontend.resolve(strict=True), source.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_relative_to(frontend) or output.is_relative_to(source):
        raise FileExistsError("Choose a new output directory outside the source frontend and official source")
    for name, expected in SOURCE_HASHES.items():
        if digest(source / name) != expected:
            raise ValueError("Unsupported upstream English implementation: " + name)
    manifest = json.loads((frontend / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "sakuratts-japanese-frontend-resources-v1" or "english_g2p" in manifest:
        raise ValueError("Expected a Japanese frontend without English resources")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="english-", dir=output.parent) as folder:
        candidate = Path(folder) / "frontend"
        candidate.mkdir()
        for name, spec in manifest["files"].items():
            original = (frontend / name).resolve(strict=True)
            if not original.is_relative_to(frontend) or original.stat().st_size != spec["bytes"] or digest(original) != spec["sha256"]:
                raise ValueError("Invalid source frontend resource: " + name)
            target = candidate / original.relative_to(frontend)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                           NLTK_DATA=str(python.resolve().parent / "nltk_data"))
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        subprocess.run([str(python.resolve(strict=True)), "-B", str(Path(__file__).resolve()),
            "--export", "--official-source", str(source), "--output", str(candidate / "english")],
            check=True, env=environment, cwd=source, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        notices = Path(__file__).resolve().parents[1] / "docs/third-party"
        for name in ("english.md", "GPT-SoVITS-LICENSE.txt", "Apache-2.0.txt"):
            shutil.copy2(notices / name, candidate / "english" / name)
        verify(candidate)
        for path in (candidate / "english").iterdir():
            manifest["files"][path.relative_to(candidate).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        manifest["english_g2p"] = {"implementation": "gpt-sovits-english-v1", "directory": "english"}
        write_json(candidate / "manifest.json", manifest)
        candidate.rename(output)
    return output


def verify(frontend):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from sakuratts.frontend.english import EnglishG2P
    symbols = json.loads((frontend / "symbols-v2.json").read_text(encoding="utf-8"))
    g2p = EnglishG2P(frontend / "english", symbols)
    probes = json.loads((frontend / "english/probes.json").read_text(encoding="utf-8"))
    for probe in probes:
        normalized = g2p.normalize(probe["text"])
        if normalized != probe["normalized"] or g2p.g2p(normalized) != probe["phones"]:
            raise ValueError("English frontend differs from export environment: " + probe["text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--export", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.export:
        export(args.official_source, args.output)
    elif args.frontend is None or args.python is None:
        parser.error("--frontend and --python are required")
    else:
        print(prepare(args.frontend, args.output, args.official_source, args.python))
