"""Benchmark local 7z settings or archive only a verified bundle manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time

PROFILES = {"balanced": ["-mx=5", "-md=32m"],
            "compact": ["-mx=7", "-md=64m"],
            "maximum": ["-mx=9", "-md=128m"],
            "extreme": ["-mx=9", "-md=512m", "-mfb=273", "-mqs=on"]}


def find_sevenzip():
    for name in ("7zz", "7z"):
        if executable := shutil.which(name):
            return Path(executable)
    if directory := os.environ.get("ProgramFiles"):
        executable = Path(directory) / "7-Zip/7z.exe"
        if executable.is_file():
            return executable
    return None


def sample_ranges(size, chunk=8 * 1024 * 1024):
    """Read small files once; take disjoint beginning/middle/end windows otherwise."""
    if size <= 3 * chunk:
        return [(0, size)]
    return [(0, chunk), ((size - chunk) // 2, chunk), (size - chunk, chunk)]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(sevenzip, arguments, cwd):
    start = time.perf_counter()
    result = subprocess.run([str(sevenzip), "-sccUTF-8", *arguments], cwd=cwd,
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return round(time.perf_counter() - start, 3)


def compress(sevenzip, source, listing, archive, profile):
    if archive.exists():
        raise FileExistsError(archive)
    seconds = run(sevenzip, ["a", "-t7z", "-m0=LZMA2", *PROFILES[profile],
                           "-ms=on", "-mmt=2", "-mtc=off", "-mta=off", "-scsUTF-8",
                           str(archive), "@" + str(listing)], source)
    tested = run(sevenzip, ["t", str(archive)], source)
    return {"profile": profile, "bytes": archive.stat().st_size, "compression_seconds": seconds,
            "test_seconds": tested, "parameters": ["-m0=LZMA2", *PROFILES[profile], "-ms=on", "-mmt=2"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sevenzip", type=Path, default=find_sevenzip())
    parser.add_argument("--format", choices=("7z", "tar.gz"), default="7z")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile", choices=PROFILES, default="maximum")
    args = parser.parse_args()
    if (args.format == "7z" or args.benchmark) and args.sevenzip is None:
        parser.error("7z compression requires --sevenzip")
    if args.format == "tar.gz" and args.benchmark:
        parser.error("--benchmark is only available for 7z")
    root, output = args.bundle.resolve(), args.output.resolve()
    manifest = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    for name, spec in manifest["files"].items():
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
            raise ValueError("Unsafe bundle file name: " + name)
        path = (root / name).resolve(strict=True)
        if root not in path.parents or not path.is_file() or digest(path) != spec["sha256"]:
            raise ValueError("Bundle file changed or escaped its directory: " + name)
    output.mkdir(parents=True)
    results = []
    if args.benchmark:
        sample = output / "sample"
        sample.mkdir()
        # Keep windows disjoint so repeated sample bytes cannot inflate compression gains.
        largest = sorted(manifest["files"], key=lambda n: -manifest["files"][n]["bytes"])[:12]
        for index, name in enumerate(largest):
            path = root / name
            with path.open("rb") as src, (sample / (str(index) + path.suffix)).open("wb") as dst:
                for offset, length in sample_ranges(path.stat().st_size):
                    src.seek(offset)
                    dst.write(src.read(length))
        listing = output / "sample-files.txt"
        listing.write_text("\n".join(p.name for p in sample.iterdir()), encoding="utf-8")
        for profile in PROFILES:
            archive = output / (profile + ".7z")
            result = compress(args.sevenzip, sample, listing, archive, profile)
            result["extraction_seconds"] = run(args.sevenzip, ["x", "-y", str(archive), "-o" + str(output / profile)], sample)
            results.append(result)
            print(json.dumps(result), flush=True)
        report = {"sample_bytes": sum(p.stat().st_size for p in sample.iterdir()), "results": results}
    else:
        listing = output / "archive-files.txt"
        names = [root.name + "/" + name for name in manifest["files"]] + [root.name + "/bundle-manifest.json"]
        listing.write_text("\n".join(names) + "\n", encoding="utf-8")
        archive = output / (root.name + "." + args.format)
        if args.format == "tar.gz":
            started = time.perf_counter()
            with tarfile.open(archive, "w:gz", dereference=True) as compressed:
                for name in names:
                    compressed.add(root.parent / name, arcname=name, recursive=False)
            result = {"format": "tar.gz", "bytes": archive.stat().st_size,
                      "compression_seconds": round(time.perf_counter() - started, 3)}
            with tarfile.open(archive, "r:gz") as compressed:
                if compressed.getnames() != names:
                    raise ValueError("Archive inventory differs from the bundle")
                for member in compressed:
                    name = Path(member.name).relative_to(root.name).as_posix()
                    expected = manifest["files"][name]["sha256"] if name in manifest["files"] else digest(root / name)
                    checksum = hashlib.file_digest(compressed.extractfile(member), "sha256").hexdigest()
                    if checksum != expected:
                        raise ValueError("Archive checksum mismatch: " + name)
        else:
            result = compress(args.sevenzip, root.parent, listing, archive, args.profile)
        result["sha256"] = digest(archive)
        (output / (archive.name + ".sha256")).write_text(result["sha256"] + "  " + archive.name + "\n", encoding="utf-8")
        report = dict(result, unpacked_bytes=manifest["bytes"], files=len(names))
        print(json.dumps(report), flush=True)
    (output / "compression-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
