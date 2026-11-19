"""Prepare offline Mandarin G2PW, pronunciation tables and feature BERT."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_chinese_resources import prepare, write_json
from sakuratts.module.reference_condition import sha256_file
from sakuratts.text._vendor.zh_normalization.char_convert import t2s_dict


def export_bert(source, graph):
    import torch
    from sakuratts.text.bert_features import BertFeatures

    weights = next((source / name for name in ("model.safetensors", "pytorch_model.bin")
                    if (source / name).is_file()), None)
    if weights is None:
        raise FileNotFoundError(f"Expected model.safetensors or pytorch_model.bin in {source}")

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = BertFeatures.from_pretrained(source, use_safetensors=weights.suffix == ".safetensors")

        def forward(self, input_ids, token_type_ids, attention_mask):
            return self.encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask)

    model = Encoder().eval()
    ids = torch.tensor([[2, 3, 4, 1]], dtype=torch.int64)
    inputs = ["input_ids", "token_type_ids", "attention_mask"]
    with torch.inference_mode():
        torch.onnx.export(
            model, (ids, torch.zeros_like(ids), torch.ones_like(ids)), str(graph),
            input_names=inputs, output_names=["hidden"],
            dynamic_axes={name: {1: "tokens"} for name in [*inputs, "hidden"]},
            opset_version=17, dynamo=False,
        )
    return weights


def save_external_weights(source, graph):
    """Keep the original graph and store its large initializers beside it."""
    import onnx

    model = onnx.load(str(source))
    onnx.save_model(model, str(graph), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=graph.with_suffix(".bin").name, size_threshold=16 * 1024)


def record_sources(source, resources, bert_weights):
    manifest_path = resources / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notices = resources / "notices"
    notices.mkdir(exist_ok=True)
    for name in ("mandarin-resources.md", "chinese-source-manifest.json", "GPT-SoVITS-model-card.md",
                 "GPT-SoVITS-LICENSE.txt", "Apache-2.0.txt", "OpenCC-LICENSE.txt", "OpenCC-NOTICE.txt",
                 "PyPinyin-LICENSE.txt", "jieba-fast-LICENSE.txt"):
        shutil.copyfile(ROOT / "docs/third-party" / name, notices / name)
    files = {p.relative_to(resources).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(resources.rglob("*")) if p.is_file() and p != manifest_path}
    g2pw = "GPT_SoVITS/text/G2PWModel/"
    bert = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/"
    for name in ("MONOPHONIC_CHARS.txt", "POLYPHONIC_CHARS.txt", "bopomofo_to_pinyin_wo_tune_dict.json"):
        manifest["sources"][g2pw + name] = files[name]
    manifest["sources"][bert + "tokenizer.json"] = files["tokenizer.json"]
    for name in (g2pw + "g2pW.onnx", bert + "config.json", bert_weights.relative_to(source).as_posix()):
        p = source / name
        manifest["sources"][name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    manifest["files"] = files
    manifest["upstream"] = {
        "frontend": "https://github.com/RVC-Boss/GPT-SoVITS",
        "g2pw": "https://github.com/GitYCC/g2pW",
        "bert": "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large",
    }
    write_json(manifest_path, manifest)


def export(source, output):
    if output.exists():
        raise FileExistsError(output)
    text = source / "GPT_SoVITS/text"
    model_dir = text / "G2PWModel"
    bert_dir = source / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        resources = Path(staging) / "chinese"
        prepare(source, resources, verify_git=False)
        for name in ("MONOPHONIC_CHARS.txt", "POLYPHONIC_CHARS.txt", "bopomofo_to_pinyin_wo_tune_dict.json"):
            shutil.copyfile(model_dir / name, resources / name)
        write_json(resources / "traditional_to_simplified.json", t2s_dict)
        shutil.copyfile(bert_dir / "tokenizer.json", resources / "tokenizer.json")
        save_external_weights(model_dir / "g2pW.onnx", resources / "g2pw.onnx")
        print("Exporting Chinese BERT to CPU ONNX", flush=True)
        bert_graph = Path(staging) / "bert.onnx"
        bert_weights = export_bert(bert_dir, bert_graph)
        save_external_weights(bert_graph, resources / "bert.onnx")
        record_sources(source, resources, bert_weights)
        resources.rename(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.official_source, args.output))
