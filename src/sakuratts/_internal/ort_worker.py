"""Private persistent acoustic worker for a separately packaged ORT ABI."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback

if not __package__:
    from runpy import run_path
    run_path(str(Path(__file__).with_name("worker.py")))["load_package"](Path(__file__).resolve().parents[1])
from sakuratts._internal.protocol import read_message, write_message
from sakuratts.backends.onnx.sovits import INPUT_NAMES, ORTSoVITS
from sakuratts._internal.reference_condition import sha256_file


def send_error(output_stream, error):
    try:
        write_message(output_stream,{"status":"error","error":error})
    except BaseException:
        print("Could not report acoustic worker failure:\n"+error+"\n"+traceback.format_exc(),file=sys.stderr)


def serve(model, ready, input_stream, output_stream):
    """Return one complete waveform per request, including diagnostic captures."""
    arrays,stages,waveform,failure,status={}, {}, None, None, 0
    try:
        write_message(output_stream,ready)
        while True:
            meta,arrays=read_message(input_stream)
            if meta.get("command")=="close":
                if arrays:
                    raise ValueError("Close does not accept acoustic tensors")
                break
            if meta.get("command")!="decode":
                raise ValueError("Unknown acoustic command")
            if set(arrays)!=set(INPUT_NAMES[:-1]) or "noise_scale" not in meta or "speed" not in meta:
                raise ValueError("Decode requires the original five arrays and scalar noise_scale input")
            started=time.perf_counter()
            capture=meta.get("capture",False)
            result=model.decode(*(arrays[name] for name in INPUT_NAMES[:-1]),
                noise_scale=meta["noise_scale"],speed=meta["speed"],capture=capture)
            waveform,stages=result if capture else (result,{})
            transport=getattr(model,"last_transfer",None)
            write_message(output_stream,{"status":"ok","compute_ms":(time.perf_counter()-started)*1000,
                "acoustic_transport":deepcopy(transport)},dict(stages,waveform=waveform))
            arrays.clear()
            stages={}
            waveform=result=None
            model.release_request_state()
    except EOFError:
        pass
    except BaseException:
        failure,status=traceback.format_exc(),1
        send_error(output_stream,failure)
    finally:
        arrays.clear()
        stages={}
        waveform=None
        try:
            model.close()
        except BaseException:
            status=1
            if failure is None:
                send_error(output_stream,traceback.format_exc())
            else:
                print("Acoustic model cleanup failed after the original worker failure:\n"+traceback.format_exc(),
                      file=sys.stderr)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",required=True)
    parser.add_argument("--diagnostic",action="store_true")
    parser.add_argument("--allow-experimental-fp16",action="store_true")
    parser.add_argument("--acoustic-arena-shrink",action="store_true")
    parser.add_argument("--acoustic-chunk-frames",type=int)
    parser.add_argument("--acoustic-session-policy",choices=("resident","staged"),default="resident")
    args=parser.parse_args()
    model,failure,status=None,None,1
    try:
        model=ORTSoVITS.load(args.package,diagnostic=args.diagnostic,
                            allow_experimental_fp16=args.allow_experimental_fp16,
                            acoustic_arena_shrink=args.acoustic_arena_shrink,
                            acoustic_chunk_frames=args.acoustic_chunk_frames,
                            acoustic_session_policy=args.acoustic_session_policy)
        import onnxruntime
        runtime=getattr(model,"runtime",None)
        runtime=runtime if isinstance(runtime,dict) else {}
        runtime.update(shared_cuda_process=False,private_acoustic_process=True)
        ready={**deepcopy(runtime),"status":"ready","providers":model.providers,
            "provider_options":model.provider_options,"python":sys.version,
            "worker_pid":os.getpid(),"executable":str(Path(sys.executable).resolve()),
            "package_manifest_sha256":sha256_file(Path(args.package)/"manifest.json"),
            "diagnostic":args.diagnostic,"chunk_frames":runtime.get("chunk_frames",args.acoustic_chunk_frames),
            "acoustic_dtype":model.encoder.manifest["dtype"],
            "acoustic_arena_shrink":model.acoustic_arena_shrink,
            "acoustic_session_policy":args.acoustic_session_policy,
            "session_initialization":runtime.get("session_initialization","eager"),
            "onnxruntime":onnxruntime.__version__,"torch_imported":"torch" in sys.modules,
            "onnx_imported":"onnx" in sys.modules}
        if isinstance(runtime.get("providers"),dict):
            ready["session_provider_options"]=deepcopy(runtime["providers"])
        status=serve(model,ready,sys.stdin.buffer,sys.stdout.buffer)
        model=None
    except BaseException:
        failure=traceback.format_exc()
        send_error(sys.stdout.buffer,failure)
    finally:
        if model is not None:
            try:
                model.close()
            except BaseException:
                status=1
                if failure is None:
                    send_error(sys.stdout.buffer,traceback.format_exc())
                else:
                    print("Acoustic model cleanup failed:\n"+traceback.format_exc(),file=sys.stderr)
    return status


if __name__=="__main__":
    raise SystemExit(main())
