"""Benchmark local 7z settings or archive only a verified bundle manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

PROFILES = {"balanced": ["-mx=5", "-md=32m"],
            "compact": ["-mx=7", "-md=64m"],
            "maximum": ["-mx=9", "-md=128m"]}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(sevenzip, arguments, cwd):
    start = time.perf_counter()
    result = subprocess.run([str(sevenzip), *arguments], cwd=cwd, capture_output=True, text=True)
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
    parser.add_argument("--sevenzip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile", choices=PROFILES, default="maximum")
    args = parser.parse_args()
    root, output = args.bundle.resolve(), args.output.resolve()
    manifest = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for name, spec in manifest["files"].items():
        path = (root / name).resolve(strict=True)
        if root not in path.parents or not path.is_file() or digest(path) != spec["sha256"]:
            raise ValueError("Bundle file changed or escaped its directory: " + name)
    results = []
    if args.benchmark:
        sample = output / "sample"
        sample.mkdir()
        # Sample DLL/code and dictionary data across each large payload, not just its header.
        largest = sorted(manifest["files"], key=lambda n: -manifest["files"][n]["bytes"])[:12]
        for index, name in enumerate(largest):
            path = root / name
            with path.open("rb") as src, (sample / (str(index) + path.suffix)).open("wb") as dst:
                for fraction in (0, .33, .66):
                    src.seek(int(max(0, path.stat().st_size - 8 * 1024 * 1024) * fraction))
                    dst.write(src.read(8 * 1024 * 1024))
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
        archive = output / (root.name + ".7z")
        result = compress(args.sevenzip, root.parent, listing, archive, args.profile)
        result["sha256"] = digest(archive)
        (output / (archive.name + ".sha256")).write_text(result["sha256"] + "  " + archive.name + "\n", encoding="ascii")
        report = dict(result, unpacked_bytes=manifest["bytes"], files=len(names))
        print(json.dumps(report), flush=True)
    (output / "compression-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
