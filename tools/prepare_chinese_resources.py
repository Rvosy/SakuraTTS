"""Export fixed official Chinese V2 tables; never modify the original files."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess


COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(upstream, output):
    sources = {}
    relative_paths = ["opencpop-strict.txt", "symbols2.py", "g2pw/polyphonic.rep",
                      "g2pw/polyphonic-fix.rep"]
    for relative in relative_paths:
        relative = "GPT_SoVITS/text/" + relative
        source = (upstream / relative).read_bytes()
        pinned = subprocess.check_output(["git", "-C", str(upstream), "show", COMMIT + ":" + relative])
        if source != pinned:
            raise ValueError("Official resource changed: " + relative)
        sources[relative] = dict(bytes=len(source), sha256=hashlib.sha256(source).hexdigest())

    corrections = {}
    overrides = []
    rows = 0
    for name in ("polyphonic.rep", "polyphonic-fix.rep"):
        for line_number, line in enumerate((upstream / "GPT_SoVITS/text/g2pw" / name).read_text().splitlines(), 1):
            key, literal = line.split(":")
            key, values = key.strip(), ast.literal_eval(literal.strip())
            if not isinstance(values, list) or len(values) != len(key) or not all(isinstance(v, str) for v in values):
                raise ValueError(f"Malformed correction {name}:{line_number}")
            if key in corrections:
                overrides.append(dict(file=name, line=line_number, key=key,
                                      previous=corrections[key], replacement=values))
            corrections[key] = values
            rows += 1
    # This converter is a development tool: only the hash-verified symbols
    # module executes here. Native runtime receives a plain JSON symbol list.
    symbols = runpy.run_path(str(upstream / "GPT_SoVITS/text/symbols2.py"))["symbols"]
    mapping = {line.split("\t")[0]: line.strip().split("\t")[1]
               for line in (upstream / "GPT_SoVITS/text/opencpop-strict.txt").read_text().splitlines()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "corrections.json", corrections)
    write_json(output / "pinyin-symbols.json", mapping)
    write_json(output / "symbols-v2.json", symbols)
    write_json(output / "manifest.json", dict(format="sakuratts-chinese-v2-resources-v1",
        official_commit=COMMIT, sources=sources, correction_rows=rows,
        correction_words=len(corrections), ordered_overrides=overrides,
        files={path.name: dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
               for path in output.iterdir()},
        license_scope="Official code MIT; correction/opencpop data source rights remain pending review before distribution"))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.upstream, args.output))
