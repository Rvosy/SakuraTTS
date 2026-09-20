"""Replay actual official CUDA histories and draws without changing tolerances."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from sakuratts.backends.cuda.gpt import CUDAGPT
from sakuratts._internal.generation import generate_semantic
from sakuratts._internal.reference_condition import sha256_file


def metric(actual,expected):
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.shape!=expected.shape:
        return {"passed":False,"actual_shape":list(actual.shape),"expected_shape":list(expected.shape)}
    difference=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    allowed=1e-4+1e-5*np.abs(expected.astype(np.float64))
    return {"passed":bool(np.all(difference<=allowed)),"max_abs":float(difference.max()),
            "rms":float(np.sqrt(np.mean(difference**2))),"outside_count":int(np.count_nonzero(difference>allowed)),
            "atol":1e-4,"rtol":1e-5}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--gpt",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--no-cuda-graph",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.capture) as archive:
        a={k:archive[k] for k in archive.files}
    with np.load(args.reference/"conditions.npz") as archive:
        prompt=archive["prompt_semantic"][None]
    model=CUDAGPT.load(args.gpt,use_graph=not args.no_cuda_graph)
    phones=a["gpt_all_phones"][None]
    bert=a["gpt_all_bert"].T[None]
    started=time.perf_counter()
    logits=[model.prefill(phones,prompt,bert)[0]]
    for token in a["sampled_tokens"][:-1]:
        logits.append(model.decode(int(token.reshape(-1)[0]))[0])
    logits=np.stack(logits)
    fixed=metric(logits,a["raw_logits"])
    rows=[]
    def observer(index,raw,token,prob,stop):
        rows.append({"index":index,"token":token,"reasons":list(stop.reasons),
                     "logits":metric(raw.reshape(-1),a["raw_logits"][index])})
    generation=generate_semantic(model,phones,prompt,bert,eos=model.config["eos"],
        early_stop_num=2700,random_draw=lambda i,shape: a["exponential_draws"][i,:shape[-1]][None],
        observer=observer)
    candidate=generation.sampled_tokens
    expected=a["sampled_tokens"].reshape(-1)
    report={"capture_sha256":sha256_file(args.capture),"graph":not args.no_cuda_graph,
        "fixed_history_logits":fixed,"tokens_equal":bool(np.array_equal(candidate,expected)),
        "semantic_equal":bool(np.array_equal(generation.semantic.reshape(-1),a["semantic_generated_00"])),
        "stop_reasons":list(generation.stop.reasons),"returned_index":generation.stop.returned_index,
        "steps":rows,"diagnostic_ms":(time.perf_counter()-started)*1000,
        "passed":fixed["passed"] and np.array_equal(candidate,expected)
                 and np.array_equal(generation.semantic.reshape(-1),a["semantic_generated_00"])}
    report["passed"]=bool(report["passed"])
    np.savez(args.output/"candidate.npz",fixed_logits=logits,tokens=candidate,semantic=generation.semantic)
    (args.output/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    model.close()
    print(json.dumps({k:v for k,v in report.items() if k!="steps"},indent=2))
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
