"""Prepare official English frontend data from pinned NLTK sources, then probe CPU G2P.

Existing files are verified and left unchanged. Resources stay under References,
and the validation process restricts NLTK lookup to that directory. No packages
are installed and no upstream Python or NLTK security settings are modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import traceback
from urllib.request import urlopen
import xml.etree.ElementTree as ET
from zipfile import ZipFile


NLTK_COMMIT = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
BASE_URL = f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_COMMIT}"
PACKAGES = {
    "cmudict": ("corpora", "d07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6"),
    "averaged_perceptron_tagger": ("taggers", "e1f13cf2532daadfd6f3bc481a49859f0b8ea6432ccdcd83e6a49a5f19008de9"),
}
REUSED_RESOURCES = ("taggers/averaged_perceptron_tagger_eng", "tokenizers/punkt_tab")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def install_missing(path, data):
    """Never replace an existing file, even when its name matches a resource."""
    if path.is_symlink():
        raise ValueError(f"Refusing resource symlink: {path}")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing resource has different contents: {path}")
        return "verified_existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return "installed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--before-evidence", type=Path, help="Optional directory containing the original failure logs")
    args = parser.parse_args()
    root = args.references.resolve()
    target = root / "models/nltk_data"
    target.mkdir(parents=True, exist_ok=True)
    output = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-english-resource-prepare")
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / Path(__file__).name)
    manifest = {"status": "running", "command": [sys.executable, *sys.argv],
                "nltk_repository": "https://github.com/nltk/nltk_data", "nltk_commit": NLTK_COMMIT,
                "NLTK_DATA": str(target), "resources": [], "reused_files": [],
                "scope": "English CPU frontend resource preparation; no GPU inference"}
    if args.before_evidence:
        source = args.before_evidence.resolve()
        manifest["before_evidence"] = {"directory": str(source), "files": [
            {"file": str(path), "sha256": digest(path.read_bytes())}
            for path in sorted(source.iterdir()) if path.is_file()
        ]}
    repo = root / "GPT-SoVITS"
    manifest["upstream_status_before"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
    write_json(output / "result.json", manifest)
    print(f"RUN_DIRECTORY={output}", flush=True)
    try:
        with urlopen(BASE_URL + "/index.xml", timeout=30) as response:
            index = response.read()
        (output / "nltk-index.xml").write_bytes(index)
        manifest["index"] = {"url": BASE_URL + "/index.xml", "sha256": digest(index)}
        index_packages = {node.attrib["id"]: node.attrib for node in ET.fromstring(index).findall(".//package")}
        for package, (subdir, expected) in PACKAGES.items():
            if index_packages[package]["sha256_checksum"] != expected:
                raise ValueError(f"Pinned NLTK index checksum differs for {package}")
            archive_path = target / subdir / f"{package}.zip"
            url = f"{BASE_URL}/packages/{subdir}/{package}.zip"
            if archive_path.exists():
                data = archive_path.read_bytes()
            else:
                with urlopen(url, timeout=30) as response:
                    data = response.read()
            if digest(data) != expected:
                raise ValueError(f"Resource SHA-256 mismatch: {package}")
            archive_action = install_missing(archive_path, data)
            entry = {"package": package, "url": url, "archive": str(archive_path),
                     "archive_sha256": expected, "archive_bytes": len(data),
                     "archive_action": archive_action, "members": []}
            with ZipFile(archive_path) as archive:
                members = archive.infolist()
                for member in members:
                    relative = PurePosixPath(member.filename)
                    mode = member.external_attr >> 16
                    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != package or stat.S_ISLNK(mode):
                        raise ValueError(f"Unexpected ZIP member: {member.filename}")
                bad_member = archive.testzip()
                if bad_member:
                    raise ValueError(f"ZIP CRC failure: {bad_member}")
                for member in members:
                    if member.is_dir():
                        continue
                    content = archive.read(member)
                    destination = target / subdir / member.filename
                    action = install_missing(destination, content)
                    entry["members"].append({"path": str(destination), "bytes": len(content),
                                              "sha256": digest(content), "action": action})
            manifest["resources"].append(entry)
        local_nltk = root / "models/shared/g2p/en/nltk"
        for relative in REUSED_RESOURCES:
            source = local_nltk / relative
            if not source.is_dir():
                raise FileNotFoundError(source)
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Refusing source symlink: {path}")
                if not path.is_file():
                    continue
                content = path.read_bytes()
                destination = target / path.relative_to(local_nltk)
                action = install_missing(destination, content)
                manifest["reused_files"].append({"source": str(path), "destination": str(destination),
                                                 "bytes": len(content), "sha256": digest(content), "action": action})
        code = r'''
import importlib.metadata, json, os, sys
import nltk
nltk.data.path[:] = [os.environ["NLTK_DATA"]]
required = ["corpora/cmudict.zip", "taggers/averaged_perceptron_tagger.zip",
            "taggers/averaged_perceptron_tagger_eng", "tokenizers/punkt_tab/english"]
resources = {name: str(nltk.data.find(name)) for name in required}
sys.path.insert(0, "GPT_SoVITS")
from text.english import g2p
text = "Please check the audio."
phones = g2p(text)
if not phones or any(not isinstance(phone, str) for phone in phones):
    raise RuntimeError("English G2P returned no valid phone sequence")
print(json.dumps({"text": text, "phones": phones, "nltk_paths": nltk.data.path,
                  "resources": resources, "dependencies": {name: importlib.metadata.version(name)
                    for name in ["nltk", "g2p-en", "wordsegment", "numpy"]}}, ensure_ascii=False))
'''
        command = [str(root / ".venv-official-macos/bin/python"), "-c", code]
        completed = subprocess.run(command, cwd=repo, env=dict(os.environ, NLTK_DATA=str(target)),
                                   capture_output=True, text=True, timeout=60)
        (output / "probe.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / "probe.stderr.log").write_text(completed.stderr, encoding="utf-8")
        manifest["probe"] = {"command": command, "cwd": str(repo), "returncode": completed.returncode}
        if completed.returncode:
            raise RuntimeError(f"English CPU probe failed; see {output / 'probe.stderr.log'}")
        manifest["probe"]["result"] = json.loads(completed.stdout.strip().splitlines()[-1])
        manifest["upstream_status_after"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
        if manifest["upstream_status_after"] != manifest["upstream_status_before"]:
            raise RuntimeError("Upstream working-tree status changed during resource preparation")
        for item in manifest["reused_files"]:
            if digest(Path(item["source"]).read_bytes()) != item["sha256"] or digest(Path(item["destination"]).read_bytes()) != item["sha256"]:
                raise RuntimeError(f"Reused resource verification failed: {item['source']}")
        manifest["status"] = "completed"
        write_json(output / "result.json", manifest)
        print(json.dumps(manifest["probe"]["result"], ensure_ascii=False), flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output / "result.json", manifest)
        raise


if __name__ == "__main__":
    main()
