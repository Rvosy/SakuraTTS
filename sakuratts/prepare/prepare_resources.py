#!/usr/bin/env python3
"""Prepare portable Japanese/English frontend and V2Pro/V2ProPlus references.

Preparation alone needs PyTorch and an explicit GPT-SoVITS source distribution.
The resulting packages do not need that distribution for ordinary inference.
The default worker interpreter is the current development Python; --python can
select the distribution's existing interpreter without modifying its environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PACKAGE = Path(__file__).resolve().parents[1]
from runpy import run_path
run_path(str(PACKAGE / "runtime/worker.py"))["enable_windows_long_import_paths"]()
SOURCE_FILE = "GPT_SoVITS/TTS_infer_pack/TTS.py"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def character_inputs(character):
    character = Path(character).resolve(strict=True)
    voice = read_json(character / "character.json")["voice"]
    if voice.get("ref_lang", "ja").lower() != "ja":
        raise ValueError("Reference preparation currently supports Japanese only")
    result = {key: (character / voice[field]).resolve(strict=True) for key, field in
              (("gpt", "gpt_model"), ("sovits", "sovits_model"), ("refs", "tone_refs"))}
    references, tones = [], set()
    for number, line in enumerate(result["refs"].read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ValueError(f"Reference line {number} must contain audio|language|text|tone")
        audio, language, text, tone = fields
        if language.lower() != "ja" or not text.strip():
            raise ValueError(f"Reference line {number} must contain Japanese text")
        if not tone or tone in (".", "..") or any(c in tone for c in '/\\:*?"<>|') or tone in tones:
            raise ValueError(f"Invalid or duplicate reference tone on line {number}: {tone}")
        tones.add(tone)
        references.append({"audio": str((character / audio).resolve(strict=True)),
                           "language": "ja", "text": text, "tone": tone})
    if not references:
        raise ValueError("The character contains no reference audio entries")
    return result, references


def frontend_preflight():
    """Inspect the selected interpreter without initializing models or G2P."""
    disable_network()
    import importlib.metadata
    import pyopenjtalk

    module = Path(pyopenjtalk.__file__).resolve(strict=True).parent
    dictionary = Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR)).resolve(strict=True)
    if not (dictionary / "sys.dic").is_file():
        raise FileNotFoundError("The preparation interpreter's OpenJTalk dictionary is missing; automatic download is disabled")
    for distribution_name, implementation in (("pyopenjtalk-plus", "pyopenjtalk-plus"),
                                              ("pyopenjtalk", "pyopenjtalk-classic")):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        # A stale distribution record must not identify a different imported module.
        if Path(distribution.locate_file("pyopenjtalk")).resolve() != module:
            continue
        metadata_path = Path(distribution._path).resolve(strict=True)
        return {"implementation": implementation, "version": distribution.version,
                "module_version": pyopenjtalk.__version__, "module_directory": str(module),
                "main_dictionary": str(dictionary), "distribution_directory": str(metadata_path),
                "python": sys.version, "executable": sys.executable}
    raise ValueError("Cannot identify the preparation interpreter's pyopenjtalk distribution")


def inspect_frontend(python):
    python = Path(os.path.abspath(python))
    python.stat()
    command = [str(python), "-B", str(Path(__file__).resolve()), "--frontend-preflight"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                                     HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
    if result.returncode:
        raise RuntimeError("Frontend preflight failed in the selected preparation interpreter:\n" + result.stderr)
    return json.loads(result.stdout)


def frontend_profile(preflight):
    if preflight is None:
        return None
    implementation, version = preflight["implementation"], preflight["version"]
    if implementation == "pyopenjtalk-classic":
        return {"implementation": implementation, "version": version, "module_directory": "classic-python",
                "main_dictionary": "classic-python/pyopenjtalk/open_jtalk_dic_utf_8-1.11"}
    if implementation == "pyopenjtalk-plus":
        return {"implementation": implementation, "version": version}
    raise ValueError("Unsupported Japanese frontend implementation: " + implementation)


def classic_frontend_files(preflight):
    """Map the selected classic module and its notices to package-relative paths."""
    if preflight is None or preflight["implementation"] != "pyopenjtalk-classic":
        return {}
    module = Path(preflight["module_directory"]).resolve(strict=True)
    metadata = Path(preflight["distribution_directory"]).resolve(strict=True)
    dictionary = Path(preflight["main_dictionary"]).resolve(strict=True)
    if dictionary != module / "open_jtalk_dic_utf_8-1.11":
        raise ValueError("Classic preparation requires the original dictionary inside its pyopenjtalk package")
    if not any(p.is_file() for p in metadata.glob("LICENSE*")) or not (dictionary / "COPYING").is_file():
        raise FileNotFoundError("Classic frontend package or dictionary license is missing")
    files = {}
    for source_root, destination in ((module, "classic-python/pyopenjtalk"),
                                     (metadata, "classic-python/" + metadata.name)):
        for path in sorted(source_root.rglob("*")):
            relative = path.relative_to(source_root)
            if any("pycache" in part.lower() for part in relative.parts) or path.suffix in (".pyc", ".pyo"):
                continue
            if path.is_symlink():
                raise ValueError("Classic frontend resources must be regular files: " + str(path))
            if path.is_file():
                files[destination + "/" + relative.as_posix()] = path
    if not any(name.endswith(".pyd") for name in files):
        raise FileNotFoundError("Classic Windows pyopenjtalk native module is missing")
    return files


def prepare_frontend(root, output, language_model=None, preflight=None):
    output = Path(output).resolve()
    if output.exists():
        return read_json(output / "manifest.json")
    root = Path(root).resolve(strict=True)
    source = root / "GPT_SoVITS/text/symbols2.py"
    user = root / "GPT_SoVITS/text/ja_userdic"
    language = (Path(language_model).resolve(strict=True) if language_model is not None else
                root / "GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin")
    source_id = "source-sha256:" + digest(root / SOURCE_FILE)
    profile = frontend_profile(preflight)
    classic_files = classic_frontend_files(preflight)
    namespace = {"__name__": "sakuratts_prepared_symbols"}
    exec(compile(source.read_bytes(), str(source), "exec"), namespace)
    symbols = namespace["symbols"]
    if not isinstance(symbols, list) or not symbols or any(not isinstance(s, str) for s in symbols):
        raise ValueError("Official symbol source did not provide a nonempty symbol list")
    output.mkdir(parents=True)
    write_json(output / "symbols-v2.json", symbols)
    shutil.copy2(user / "user.dict", output / "user.dict")
    shutil.copy2(language, output / "lid.176.bin")
    shutil.copytree(root / "english", output / "english")
    classic_identities = {name: identity(path) for name, path in classic_files.items()}
    for name, source_path in classic_files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    manifest = {"format": "sakuratts-japanese-frontend-resources-v1", "official_commit": source_id,
                "symbol_source_sha256": digest(source),
                "sources": {"user_dictionary": identity(user / "user.dict"), "language_model": identity(language)},
                "files": {name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
                          for name in ("symbols-v2.json", "user.dict", "lid.176.bin", *classic_files)}}
    if profile is not None:
        manifest["japanese_g2p"] = profile
        manifest["sources"]["japanese_g2p_preflight"] = preflight
    if classic_files:
        manifest["sources"]["classic_files"] = classic_identities
    manifest["english_g2p"] = {"implementation": "gpt-sovits-english-v1", "directory": "english"}
    manifest["files"].update({"english/" + path.name: {"bytes": path.stat().st_size}
                              for path in (output / "english").iterdir() if path.is_file()})
    write_json(output / "manifest.json", manifest)
    return manifest


def write_runtime_config(output, references):
    config = {"format": "sakuratts-windows-config-v1", "gpt": "gpt", "sovits": "sovits", "frontend": "frontend",
              "references": {ref["tone"]: "references/" + ref["tone"] for ref in references},
              "default_reference": "中性" if any(ref["tone"] == "中性" for ref in references) else references[0]["tone"]}
    path = Path(output) / "runtime.json"
    if path.exists() and read_json(path) != config:
        raise FileExistsError("Refusing to replace an existing different runtime configuration: " + str(path))
    if not path.exists():
        write_json(path, config)
    return path


def disable_network():
    """Make missing local resources fail instead of triggering a download."""
    def check(event, _args):
        if event == "socket.connect":
            raise RuntimeError("Reference preparation is offline; provide the missing resource locally")
    sys.addaudithook(check)


def worker(job_file, session=None):
    job = read_json(job_file)
    root, run = Path(job["official_source"]), Path(job_file).parent
    cache = Path(job.get("cache_dir", run))
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      NUMBA_CACHE_DIR=str(cache / "numba-cache"), MPLCONFIGDIR=str(cache / "mpl-cache"),
                      PYTHONIOENCODING="utf-8")
    from runpy import run_path
    run_path(str(PACKAGE / "runtime/worker.py"))["load_package"](PACKAGE)
    run_path(str(PACKAGE / "runtime/worker.py"))["enable_windows_long_import_paths"]()
    sys.path[:0] = [str(root), str(root / "GPT_SoVITS")]
    os.chdir(root)
    result = {"status": "running", "prepare_only": True, "target_synthesis_calls": 0, "references": []}
    started = time.perf_counter()
    try:
        if session is None:
            disable_network()
        try:
            import numpy as np
            import torch
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
            from module import commons
            from process_ckpt import get_sovits_version_from_path_fast
            from TTS_infer_pack.text_segmentation_method import splits
            from sakuratts.module.reference_condition import validate_arrays
            import fast_langdetect
        except ImportError as error:
            raise RuntimeError("Preparation interpreter lacks an official source dependency: " + str(error)
                               + ". Install the development dependencies or pass --python with the existing official interpreter.") from error
        if job["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for reference preparation but is unavailable")
        torch.set_num_threads(job["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
            fast_langdetect.infer.LangDetectConfig(cache_dir=Path(job.get("frontend", str(Path(job["output"]) / "frontend")))))

        class ReferenceOnlyTTS(TTS):
            def _init_models(self):
                self.init_vits_weights(self.configs.vits_weights_path)
                self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)

        _, family, lora = get_sovits_version_from_path_fast(job["sovits"])
        if family not in ("v2Pro", "v2ProPlus") or lora:
            raise ValueError("Reference preparation supports non-LoRA V2Pro and V2ProPlus checkpoints")
        config_data = {"custom": {"version": family, "device": job["device"], "is_half": job["precision"] == "fp16",
                       "t2s_weights_path": job["gpt"], "vits_weights_path": job["sovits"],
                       "bert_base_path": str(root / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
                       "cnhuhbert_base_path": job.get("cnhubert", str(root / "GPT_SoVITS/pretrained_models/chinese-hubert-base"))}}
        # TTS_Config creates a relative configs directory even for an explicit
        # dictionary. Both that directory and later save_configs stay in run.
        os.chdir(run)
        config = TTS_Config(config_data)
        config.configs_path = str(run / "tts-prepare.yaml")
        os.chdir(root)
        engine = session.get("engine") if session is not None else None
        if engine is None:
            engine = ReferenceOnlyTTS(config)
            if session is not None:
                session["engine"] = engine
        if (engine.configs.version != family or str(engine.configs.device) != job["device"]
                or engine.t2s_model is not None or engine.bert_model is not None):
            raise RuntimeError("Reference-only initialization differs from the requested model family or device")
        for ref in job["references"]:
            phase = time.perf_counter()
            engine.set_ref_audio(ref["audio"])
            prompt = ref["text"].strip("\n")
            if prompt[-1] not in splits:
                prompt += "。"
            phones, bert, normalized = engine.text_preprocessor.segment_and_extract_feature_for_text(prompt, "ja", family)
            with torch.no_grad():
                spec, audio = engine.prompt_cache["refer_spec"][0]
                spec = spec.to(dtype=engine.precision, device=job["device"])
                mask = commons.sequence_mask(torch.LongTensor([spec.size(2)]).to(spec.device), spec.size(2)).unsqueeze(1).to(spec.dtype)
                ge = engine.vits_model.ref_enc(spec[:, :704] * mask, mask)
                sv = engine.sv_model.compute_embedding3(audio)
                ge += engine.vits_model.sv_emb(sv).unsqueeze(-1)
                ge = engine.vits_model.prelu(ge)
                ge = torch.stack([ge], 0).mean(0)
                ge512 = engine.vits_model.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
            arrays = {"reference_phones": np.asarray(phones, dtype=np.int64),
                      "prompt_semantic": engine.prompt_cache["prompt_semantic"].detach().cpu().numpy().astype(np.int64).reshape(-1),
                      "reference_bert": bert.detach().float().cpu().numpy(),
                      "ge": ge.detach().float().cpu().numpy(), "ge512": ge512.detach().float().cpu().numpy()}
            validate_arrays(arrays)
            package = Path(job["output"]) / "references" / ref["tone"]
            package.mkdir(parents=True, exist_ok=False)
            archive = package / "conditions.npz"
            np.savez(archive, **arrays)
            manifest = {"format": "sakuratts-prepared-reference-v1", "model_family": family,
                        "identity": {"gpt_checkpoint_sha256": job["source_hashes"][job["gpt"]],
                            "sovits_checkpoint_sha256": job["source_hashes"][job["sovits"]],
                            "audio_sha256": job["source_hashes"][ref["audio"]], "official_commit": job["source_id"],
                            "reference_text": ref["text"], "reference_language": "ja"},
                        "reference": {"prompt_text": prompt, "normalized_text": normalized, "tone": ref["tone"]},
                        "preparation": {"precision": job["precision"], "device": job["device"], "torch_version": torch.__version__,
                            "japanese_g2p": frontend_profile(job["frontend_preflight"]),
                            "tf32": False, "python": sys.version, "executable": sys.executable,
                            "target_synthesis_calls": 0, "single_reference_list_mean_preserved": True,
                            "stored_dtype": "float32; fp16 computations are promoted without recomputation when selected"},
                        "archive": {"file": archive.name, "bytes": archive.stat().st_size, "sha256": digest(archive)},
                        "arrays": {name: {"dtype": str(a.dtype), "shape": list(a.shape), "bytes": a.nbytes,
                            "sha256_raw_c_order": hashlib.sha256(a.tobytes()).hexdigest()} for name, a in arrays.items()},
                        "provenance": {"preparation_job": str(job_file), "source_files_sha256": job["source_hashes"]}}
            write_json(package / "manifest.json", manifest)
            result["references"].append({"tone": ref["tone"], "package": str(package),
                                         "elapsed_seconds": time.perf_counter() - phase})
            print("PREPARED_REFERENCE", package, flush=True)
        result.update(status="completed")
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    result["elapsed_seconds"] = time.perf_counter() - started
    write_json(run / "result.json", result)
    return 0 if result["status"] == "completed" else 1


def serve():
    # Third-party Python/native output must not enter the protocol stream.
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    run_path(str(PACKAGE / 'runtime/worker.py'))['load_package'](PACKAGE)
    from sakuratts.runtime.protocol import read_message, write_message
    disable_network()
    # The owning inference session fixes the model and preparation interpreter.
    session = {}
    try:
        while True:
            request, _ = read_message(sys.stdin.buffer)
            try:
                main(request['arguments'], session)
                write_message(output, {'status': 'ok'})
            except Exception:
                write_message(output, {'status': 'error', 'error': traceback.format_exc()})
    except EOFError:
        return 0


def main(argv=None, session=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--character", type=Path)
    parser.add_argument("--inputs", type=Path, help="JSON containing gpt, sovits and references, without a character directory")
    parser.add_argument("--frontend", type=Path, help="Reuse and verify an existing frontend package")
    parser.add_argument("--cnhubert", type=Path, help="Explicit upstream cnhuhbert_base_path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--language-model", type=Path, help="Existing complete lid.176.bin; defaults to the official pretrained_models directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, help="Reuse audio preprocessing compilation caches across references")
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--frontend-preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--serve", action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.serve:
        return serve()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if args.worker:
        return worker(args.worker.resolve(strict=True))
    if args.frontend_preflight:
        print(json.dumps(frontend_preflight(), ensure_ascii=False))
        return 0
    if not all((args.official_source, args.output)) or bool(args.character) == bool(args.inputs):
        parser.error("--official-source, --output and exactly one of --character/--inputs are required")
    if args.cpu_threads < 1 or (args.device == "cpu" and args.precision != "fp32"):
        parser.error("Use positive --cpu-threads; CPU preparation requires --precision fp32")
    root, output = args.official_source.resolve(strict=True), args.output.resolve()
    character = args.character.resolve(strict=True) if args.character else None
    if output == root or root in output.parents or character and (output == character or character in output.parents):
        raise ValueError("Output must be outside the official source and character directories")
    if character:
        inputs, references = character_inputs(character)
    else:
        request = read_json(args.inputs)
        inputs = {key: Path(request[key]).resolve(strict=True) for key in ("gpt", "sovits")}
        references = request.get("references", [])
        if not references and not args.frontend_only:
            raise ValueError("At least one reference is required for preparation")
        for ref in references:
            if ref.get("language") != "ja" or not ref.get("text", "").strip():
                raise ValueError("Reference preparation requires Japanese prompt text")
            if ref.get("tone") != "reference":
                raise ValueError("Direct reference preparation uses the internal name 'reference'")
            ref["audio"] = str(Path(ref["audio"]).resolve(strict=True))
    if session is None:
        preflight = inspect_frontend(args.python)
    else:
        if "preflight" not in session:
            session["preflight"] = inspect_frontend(args.python)
        preflight = session["preflight"]
    frontend_path = args.frontend.resolve(strict=True) if args.frontend else output / "frontend"
    frontend = prepare_frontend(root, frontend_path, args.language_model, preflight)
    print("PREPARED_FRONTEND", frontend_path, flush=True)
    if args.frontend_only:
        return 0
    for ref in references:
        if (output / "references" / ref["tone"]).exists():
            raise FileExistsError("Reference output already exists; choose a new output directory: " + ref["tone"])
    if session is None:
        source_hashes = {str(p): digest(p) for p in inputs.values()}
    else:
        if "model_hashes" not in session:
            session["model_hashes"] = {kind: digest(p) for kind, p in inputs.items()}
        # The owner fixes model contents; equivalent checkpoints may move between requests.
        source_hashes = {str(p): session["model_hashes"][kind] for kind, p in inputs.items()}
    source_hashes.update({ref["audio"]: digest(ref["audio"]) for ref in references})
    cnhubert = args.cnhubert.resolve(strict=True) if args.cnhubert else root / "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    job = {"official_source": str(root), "output": str(output), "frontend": str(frontend_path), "gpt": str(inputs["gpt"]), "sovits": str(inputs["sovits"]),
           "source_id": frontend["official_commit"], "cnhubert": str(cnhubert), "device": args.device, "precision": args.precision,
           "frontend_preflight": preflight,
           "references": references, "cpu_threads": args.cpu_threads,
           "cache_dir": str((args.cache_dir or output / "cache").resolve()),
           "source_hashes": source_hashes}
    run = output / ("prepare-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    run.mkdir(parents=True, exist_ok=False)
    job_file = run / "job.json"
    write_json(job_file, job)
    if session is not None:
        if worker(job_file, session):
            raise RuntimeError(read_json(run / 'result.json')['error'])
        return 0
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    command = [os.path.abspath(args.python), "-B", str(Path(__file__).resolve()), "--worker", str(job_file)]
    with (run / "worker.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", env=env,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = child.wait()
    write_json(run / "process.json", {"command": command, "returncode": returncode})
    if returncode:
        print("Reference preparation failed; details: " + str(run / "worker.log"), file=sys.stderr)
    elif character:
        config_path = write_runtime_config(output, references)
        print("RUNTIME_CONFIG", config_path)
        print("Next, convert the original models with the development interpreter (the prepare worker interpreter is independent):")
        print(subprocess.list2cmdline([sys.executable, "sakuratts/prepare/convert_gpt.py", "--checkpoint", str(inputs["gpt"]),
              "--official-source", str(root), "--output", str(output / "gpt")]))
        print(subprocess.list2cmdline([sys.executable, "sakuratts/prepare/export_sovits_onnx.py", "--checkpoint", str(inputs["sovits"]),
              "--official-source", str(root), "--output", str(output / "sovits")]))
        print("The runtime config becomes usable once both model conversions are complete.")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
