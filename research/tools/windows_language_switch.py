"""Rotate zh, ja and en on one managed CUDA service per public FP16 profile."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median

from windows_language_matrix import (PROFILES, ROOT, TEXTS, Service, available_port, cuda_adapter,
                                     request_audio, sha256, write_json)


def run(bundle, output, config, request, profile, expected_identity):
    folder = output / profile
    (folder / "external-cwd").mkdir(parents=True)
    config["sakuratts"].update(backend="cuda", profile=profile)
    write_json(folder / "config.json", config)
    raw = {"services": [], "requests": []}
    service = Service(bundle, folder / "config.json", folder, "service", "managed", available_port(None), raw)
    with service:
        service.wait_ready()
        service.wake(profile)
        original = service.record["awake"]
        for index in range(4):
            for language in ("zh", "ja", "en"):
                name = language + ("-warmup" if index == 0 else f"-hot-{index}")
                payload = dict(request, text=TEXTS[language]["short"], text_lang=language, seed=7)
                result = request_audio(service, payload, folder / (name + ".wav"))
                state = service.state("awake")
                if any(state[name] != original[name] for name in ("model", "worker_pid", "generation")):
                    raise AssertionError("Model or worker changed during the language switch")
                raw["requests"].append(dict(result, language=language, index=index, state=state))
    service.assert_process_paths()
    service.assert_execution("cuda", profile)
    write_json(folder / "result.json", raw)
    reports = [ast.literal_eval(line.split("合成报告: ", 1)[1])
               for line in service.log().splitlines() if "合成报告: " in line]
    rows = []
    for result, report in zip(raw["requests"], reports, strict=True):
        language = result["language"]
        if (report["profile"] != profile or report["parameters"]["language"] != language
                or report["parameters"]["seed"] != 7
                or any(report["reference_identity"][key] != value for key, value in expected_identity.items())
                or report["text"] != TEXTS[language]["short"]):
            raise AssertionError("Backend report differs from submitted profile, reference or text")
        rows.append({"language": language, "index": result["index"], "wav_seconds": result["audio_seconds"],
                     "pcm_sha256": result["pcm_sha256"], "http_ms": round(1000 * result["http_seconds"], 3),
                     "rtf": round(result["rtf"], 6)})
    hot = {}
    for language in ("zh", "ja", "en"):
        samples = [r for r in rows if r["language"] == language and r["index"]]
        hot[language] = {k: median(r[k] for r in samples) for k in ("wav_seconds", "http_ms", "rtf")}
    return {"profile": profile, "model": original["model"], "generation": original["generation"],
            "checks": {"one_worker_and_model": True, "one_reference_and_profile": True,
                       **service.record["checks"]}, "hot_medians": hot, "requests": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    bundle, output = args.bundle.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config, request = (json.loads(p.read_text("utf-8")) for p in (args.config, args.request))
    public_request = {k: v for k, v in request.items() if k not in ("text", "text_lang", "ref_audio_path")}
    public_request["seed"] = 7
    summary = {"utc": datetime.now(timezone.utc).isoformat(), "gpu": cuda_adapter(),
               "manifest_sha256": sha256(bundle / "bundle-manifest.json"),
               "gpt_sha256": sha256(config["custom"]["t2s_weights_path"]),
               "sovits_sha256": sha256(config["custom"]["vits_weights_path"]),
               "reference_sha256": sha256(request["ref_audio_path"]),
               "raw_directory": output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else output.name,
               "parameters": public_request, "order": ["zh", "ja", "en"],
               "texts": {k: v["short"] for k, v in TEXTS.items()}, "warmup_cycles": 1, "hot_cycles": 3,
               "metric": "Full HTTP POST to received WAV body; no WDDM sampling in this run.", "profiles": []}
    identity = {"gpt_checkpoint_sha256": summary["gpt_sha256"], "sovits_checkpoint_sha256": summary["sovits_sha256"],
                "audio_sha256": summary["reference_sha256"], "reference_language": request["prompt_lang"],
                "reference_text": request["prompt_text"]}
    for profile in PROFILES:
        if profile != "fp32":
            summary["profiles"].append(run(bundle, output, config, request, profile, identity))
            write_json(output / "summary.json", summary)
            print(profile, summary["profiles"][-1]["hot_medians"], flush=True)
    summary["distinct_hot_pcm_per_language"] = {
        language: len({r["pcm_sha256"] for p in summary["profiles"] for r in p["requests"]
                       if r["language"] == language and r["index"] > 0}) for language in summary["order"]}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
