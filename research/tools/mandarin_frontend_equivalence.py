"""Compare the packaged FP32 Mandarin frontend with a candidate in separate processes."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "research/experiments/configs/mandarin-pronunciation-corpus.txt"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture(resources, corpus, work, source=None):
    if source:
        sys.path.insert(0, str(source))
    import numpy as np
    import onnxruntime as ort
    from sakuratts.text.g2pw_session import G2PWSession
    from sakuratts.text.processors import ChineseProcessor

    probabilities, chosen, confidences = [], [], []
    run, predict = G2PWSession.run, G2PWSession.predict

    def observe(self, inputs):
        output = run(self, inputs)
        probabilities.append(output)
        return output

    def observe_predictions(self, inputs, texts):
        labels, scores = predict(self, inputs, texts)
        chosen.extend(labels)
        confidences.extend(scores)
        return labels, scores

    G2PWSession.run = observe
    G2PWSession.predict = observe_predictions
    processor = ChineseProcessor(resources)
    rows, arrays = [], {}
    try:
        for index, text in enumerate(corpus.read_text("utf-8").splitlines()):
            probabilities.clear()
            chosen.clear()
            confidences.clear()
            phones, counts, normalized = processor.clean(text)
            features = processor.features(phones, counts, normalized)
            rows.append({"text": text, "normalized": normalized, "phones": phones, "word2ph": counts,
                         "predicted_labels": list(chosen)})
            arrays[f"bert_{index}"] = features
            if probabilities:
                arrays[f"g2pw_{index}"] = np.concatenate(probabilities, axis=0)
                arrays[f"g2pw_confidence_{index}"] = np.asarray(confidences, dtype=np.float64)
    finally:
        processor.close()
        G2PWSession.run = run
        G2PWSession.predict = predict
    work.mkdir(parents=True, exist_ok=True)
    np.savez(work / "arrays.npz", **arrays)
    write_json(work / "rows.json", {"onnxruntime": ort.__version__, "sentences": rows})


def candidate_resources(original, graphs, work):
    if (graphs / "manifest.json").is_file() and (graphs / "corrections.json").is_file():
        return graphs
    folder = work / "candidate-resources"
    folder.mkdir(parents=True, exist_ok=True)
    entries = {p.name: p for p in original.iterdir() if p.is_file()}
    entries.update({p.name: p for p in graphs.iterdir() if p.suffix in (".onnx", ".bin", ".data")})
    for name, source in entries.items():
        target = folder / name
        if target.exists():
            if os.path.samefile(target, source):
                continue
            target.unlink()
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copyfile(source, target)
    return folder


def compare(work):
    import numpy as np

    base, candidate = [json.loads((work / kind / "rows.json").read_text("utf-8"))
                       for kind in ("baseline", "candidate")]
    refs, actuals = base["sentences"], candidate["sentences"]
    if len(refs) != len(actuals):
        raise ValueError("Frontend returned different numbers of sentences")
    summary = {"corpus": CORPUS.name, "sentences": len(refs), "onnxruntime": base["onnxruntime"],
               "provider": "CPUExecutionProvider", "source_phone_count": sum(len(row["phones"]) for row in refs),
               "normalized_mismatches": 0, "word2ph_mismatches": 0, "phone_mismatches": [],
               "g2pw_top1_differences": 0, "g2pw_label_sequence_mismatches": 0, "g2pw_queries": 0}
    norms = {"bert": (1e-5, 1e-4), "g2pw": (1e-6, 1e-5), "g2pw_confidence": (1e-6, 1e-5)}
    stats = {name: {"atol": a, "rtol": r, "arrays": 0, "arrays_byte_identical": 0, "shape_mismatches": 0, "values": 0,
                    "values_changed": 0, "outside_tolerance": 0, "max_absolute_difference": 0.}
             for name, (a, r) in norms.items()}
    with np.load(work / "baseline/arrays.npz") as before, np.load(work / "candidate/arrays.npz") as after:
        for index, (left, right) in enumerate(zip(refs, actuals)):
            if left["text"] != right["text"]:
                raise ValueError("Different source text at index " + str(index))
            for key in ("normalized", "word2ph"):
                summary[key + "_mismatches"] += left[key] != right[key]
            summary["g2pw_label_sequence_mismatches"] += left["predicted_labels"] != right["predicted_labels"]
            if left["phones"] != right["phones"]:
                summary["phone_mismatches"].append({"text": left["text"], "baseline": left["phones"], "candidate": right["phones"]})
            for name, (atol, rtol) in norms.items():
                key, totals = f"{name}_{index}", stats[name]
                if key not in before and key not in after:
                    continue
                totals["arrays"] += 1
                if key not in before or key not in after:
                    totals["shape_mismatches"] += 1
                    continue
                expected, actual = before[key], after[key]
                if expected.shape != actual.shape:
                    totals["shape_mismatches"] += 1
                    continue
                if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
                    raise ValueError("Non-finite " + key)
                totals["arrays_byte_identical"] += expected.dtype == actual.dtype and expected.tobytes() == actual.tobytes()
                diff = np.abs(actual - expected)
                totals["values"] += diff.size
                totals["values_changed"] += int(np.count_nonzero(diff))
                totals["outside_tolerance"] += int((diff > atol + rtol * np.abs(expected)).sum())
                totals["max_absolute_difference"] = max(totals["max_absolute_difference"], float(diff.max()))
                if name == "g2pw":
                    summary["g2pw_queries"] += len(expected)
                    summary["g2pw_top1_differences"] += int((expected.argmax(1) != actual.argmax(1)).sum())
    return {**summary, "arrays": stats}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--resources", type=Path, required=True, help="Original Chinese frontend tables and ONNX models")
    cli.add_argument("--output", type=Path, required=True, help="Local raw arrays and comparison directory")
    cli.add_argument("--corpus", type=Path, default=CORPUS)
    cli.add_argument("--candidate-graphs", type=Path, help="Complete exported Chinese resources, or replacement ONNX graphs and sidecars")
    cli.add_argument("--candidate-source", type=Path, help="Source checkout to use in the candidate process")
    cli.add_argument("--python", type=Path, default=Path(sys.executable), help="Python from the baseline runtime")
    cli.add_argument("--capture", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.capture:
        capture(args.resources, args.corpus, args.output, args.candidate_source)
        return
    if args.candidate_graphs is None:
        cli.error("--candidate-graphs is required for a comparison")
    resources = args.resources.resolve(strict=True)
    work = args.output.resolve()
    folders = (resources, candidate_resources(resources, args.candidate_graphs.resolve(strict=True), work))
    for kind, folder in zip(("baseline", "candidate"), folders):
        command = [str(args.python.resolve()), "-I", "-X", "utf8", "-B", str(Path(__file__).resolve()), "--capture",
                   "--resources", str(folder), "--corpus", str(args.corpus.resolve()), "--output", str(work / kind)]
        if kind == "candidate" and args.candidate_source:
            command.extend(("--candidate-source", str(args.candidate_source.resolve())))
        subprocess.run(command, check=True)
    summary = compare(work)
    summary["corpus"] = args.corpus.name
    write_json(work / "comparison.json", summary)
    print(work / "comparison.json")


if __name__ == "__main__":
    main()
