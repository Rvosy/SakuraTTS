"""Read the offline Mandarin frontend inventory shared by model and bundle preparation."""
import json


def read_chinese_resources(source, files):
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "sakuratts-chinese-v2-resources-v1":
        raise ValueError("Expected a prepared Chinese frontend resource directory")
    required = {"g2pw.onnx", "bert.onnx", "tokenizer.json", "MONOPHONIC_CHARS.txt",
                "POLYPHONIC_CHARS.txt", "bopomofo_to_pinyin_wo_tune_dict.json",
                "traditional_to_simplified.json", "corrections.json", "pinyin-symbols.json", "symbols-v2.json"}
    declared = manifest["files"].keys()
    available = {p.relative_to(source).as_posix(): p for p in files}
    missing = (required - declared) | (declared - available.keys())
    if missing:
        raise ValueError("Incomplete Chinese frontend resources: " + ", ".join(sorted(missing)))
    return manifest, [source / "manifest.json", *(available[name] for name in declared)]
