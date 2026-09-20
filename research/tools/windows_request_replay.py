"""Raw-text complete request replay with official draws, never target tokens."""

import argparse
import json
from pathlib import Path
import sys
import wave

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from sakuratts.backends.cuda.engine import NVIDIAEngine,write_wav


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--capture",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--repeats",type=int,default=3)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    metadata=json.loads(args.capture.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(args.capture) as source:
        captured={k:source[k] for k in source.files}
    with wave.open(str(args.capture.with_suffix(".wav")),"rb") as audio:
        original=np.frombuffer(audio.readframes(audio.getnframes()),dtype="<i2")
    engine=NVIDIAEngine(args.config)
    engine.load()
    results=[]
    try:
        for index in range(args.repeats):
            pcm,report=engine.synthesize(metadata["request"]["inputs"]["text"],reference="中性",
                random_inputs=[{"draws":captured["exponential_draws"],"noise":captured["acoustic_noise_00"]}])
            item=report["fragments"][0]
            checks={"phones_equal":np.array_equal(item["phones"],captured["enc_p_target_phones"].reshape(-1)),
                "tokens_equal":np.array_equal(item["sampled_tokens"],captured["sampled_tokens"].reshape(-1)),
                "semantic_equal":np.array_equal(item["semantic_tokens"],captured["semantic_generated_00"]),
                "pcm_length_equal":pcm.shape==original.shape}
            if checks["pcm_length_equal"]:
                difference=pcm.astype(np.int32)-original.astype(np.int32)
                checks.update(pcm_max_abs_lsb=int(np.abs(difference).max()),
                    pcm_different_samples=int(np.count_nonzero(difference)),
                    pcm_within_fp32_tolerance=bool(np.all(np.abs(difference/32768.)<=1e-4+1e-5*np.abs(original/32768.))))
            report["checks"]={k:bool(v) if isinstance(v,np.bool_) else v for k,v in checks.items()}
            results.append(report)
            write_wav(args.output/f"replay-{index}.wav",pcm,report["sample_rate"])
    finally:
        engine.close()
    (args.output/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps([{"request_ms":r["request_ms"],"checks":r["checks"],
                      "stages":r["fragments"][0]["timings"]} for r in results],indent=2))
    return 0 if all(all(r["checks"].get(k,False) for k in
        ("phones_equal","tokens_equal","semantic_equal","pcm_length_equal","pcm_within_fp32_tolerance")) for r in results) else 1


if __name__=="__main__":
    raise SystemExit(main())
