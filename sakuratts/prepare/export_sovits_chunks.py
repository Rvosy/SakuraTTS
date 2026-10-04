"""Prepare CUDA FP16 partitions from an exported FP32 acoustic package."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile

if not __package__:
    from runpy import run_path
    run_path(str(Path(__file__).resolve().parents[1] / "runtime/worker.py"))["load_package"](Path(__file__).resolve().parents[1])
from sakuratts.prepare.export_sovits_fp16 import convert, file_spec
from sakuratts.prepare.split_sovits_vocoder import split_package
from sakuratts.prepare.vocoder_receptive_field import VocoderReceptiveField
from sakuratts.module.chunked_package import FORMAT, read_chunked_manifest
from sakuratts.module.sovits import read_manifest


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError("Choose a new output directory outside the source package")
    _, graph = read_manifest(source, diagnostic=True)
    planner = VocoderReceptiveField.from_onnx(graph)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cuda-fp16-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        fp16, staged = temporary / "fp16", temporary / "chunked"
        convert(source, fp16, lower_transpose="polyphase", deterministic_compute=True)
        split = split_package(fp16, staged)
        converted = json.loads((fp16 / "manifest.json").read_text(encoding="utf-8"))
        (staged / "vocoder-rf.json").write_text(json.dumps(planner.to_dict(), indent=2) + "\n", encoding="utf-8")
        result = {"format": FORMAT,
            **{key: deepcopy(converted[key]) for key in ("source", "dtype", "config", "inputs", "precision")},
            **{key: deepcopy(split[key]) for key in ("graphs", "weights", "interfaces", "cut", "settings")},
            "rf": file_spec(staged / "vocoder-rf.json"),
            "conversion": {"fp16": converted["conversion"], "partition": split["conversion"]},
            "validation": {"onnx_checker_passed": True, "gpu_tested": False, "quality_accepted": False}}
        # Preserve the numerical lowering checks without claiming CUDA screening.
        shutil.copy2(fp16 / "conversion.json", staged / "conversion.json")
        if (source / "GPT-SoVITS-LICENSE").is_file():
            shutil.copy2(source / "GPT-SoVITS-LICENSE", staged / "GPT-SoVITS-LICENSE")
        (staged / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        read_chunked_manifest(staged, allow_experimental_fp16=True,
                              acoustic_chunk_frames=256, acoustic_arena_shrink=True)
        staged.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
