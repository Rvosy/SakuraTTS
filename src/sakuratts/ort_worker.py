"""Private persistent acoustic worker for a separately packaged ORT ABI."""

import argparse
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sakuratts.array_protocol import read_message, write_message
from sakuratts.ort_sovits import ORTSoVITS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",required=True)
    parser.add_argument("--diagnostic",action="store_true")
    parser.add_argument("--allow-experimental-fp16",action="store_true")
    args=parser.parse_args()
    model=None
    try:
        model=ORTSoVITS.load(args.package,diagnostic=args.diagnostic,
                            allow_experimental_fp16=args.allow_experimental_fp16)
        import onnxruntime
        write_message(sys.stdout.buffer,{"status":"ready","providers":model.providers,
            "provider_options":model.provider_options,"python":sys.version,
            "acoustic_dtype":model.encoder.manifest["dtype"],
            "onnxruntime":onnxruntime.__version__,"torch_imported":"torch" in sys.modules})
        while True:
            meta,arrays=read_message(sys.stdin.buffer)
            if meta["command"]=="close":
                break
            if meta["command"]!="decode":
                raise ValueError("Unknown acoustic command")
            started=time.perf_counter()
            result=model.decode(arrays["codes"],arrays["phones"],arrays["ge"],arrays["ge512"],
                arrays["noise"],noise_scale=meta["noise_scale"],speed=meta["speed"],capture=meta.get("capture",False))
            waveform,stages=result if meta.get("capture",False) else (result,{})
            write_message(sys.stdout.buffer,{"status":"ok","compute_ms":(time.perf_counter()-started)*1000},
                          dict(stages,waveform=waveform))
    except EOFError:
        pass
    except BaseException:
        write_message(sys.stdout.buffer,{"status":"error","error":traceback.format_exc()})
        return 1
    finally:
        if model is not None:
            model.close()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
