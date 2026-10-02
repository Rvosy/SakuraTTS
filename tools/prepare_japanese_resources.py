#!/usr/bin/env python3
"""Export Japanese frontend resources without loading models.

Copies an existing OpenJTalk user dictionary and the complete lid.176.bin.
Never rebuilds dictionaries, downloads resources or overwrites an output path.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


SYMBOL_SOURCE = "GPT_SoVITS/text/symbols2.py"
FORMAT = "sakuratts-japanese-frontend-resources-v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(official_source, user_dictionary, language_model, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output path already exists: " + str(output))
    official_source = official_source.resolve(strict=True)
    user_dictionary = user_dictionary.resolve(strict=True)
    language_model = language_model.resolve(strict=True)
    source_path = official_source / SYMBOL_SOURCE
    source = source_path.read_bytes()
    sources = {"user_dictionary": file_identity(user_dictionary), "language_model": file_identity(language_model)}
    if sources["user_dictionary"]["bytes"] == 0:
        raise ValueError("The existing user dictionary must not be empty")
    namespace = {"__name__": "official_symbols2"}
    exec(compile(source, str(source_path), "exec"), namespace)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "symbols-v2.json", namespace["symbols"])
    for key, filename in (("user_dictionary", "user.dict"), ("language_model", "lid.176.bin")):
        original = Path(sources[key]["path"])
        shutil.copy2(original, output / filename)
    write_json(output / "manifest.json", {
        "format": FORMAT, "official_commit": "source-sha256:" + hashlib.sha256(source).hexdigest(),
        "symbol_source_sha256": hashlib.sha256(source).hexdigest(), "sources": sources,
        "files": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                  for name in ("symbols-v2.json", "lid.176.bin", "user.dict")},
    })
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--user-dictionary", type=Path, required=True)
    parser.add_argument("--language-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.official_source, args.user_dictionary, args.language_model, args.output))
