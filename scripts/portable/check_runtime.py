"""Exercise the selected execution provider with an actual matrix product."""

import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parent
ORT_PROBE = r'''
import base64, json, os
from pathlib import Path
import numpy as np
root = Path(__import__('sys').executable).parent
handles = [os.add_dll_directory(str(p)) for p in
           [root / 'cuda', *(root.parent / 'main/Lib/site-packages/nvidia').glob('*/bin')]]
import onnxruntime as ort
options = ort.SessionOptions()
options.enable_profiling = True
options.profile_file_prefix = str(Path.cwd() / 'cache/ort-smoke')
model = base64.b64decode('CAk6XQoRCgF4CgF5EgF6IgZNYXRNdWwSCWdwdS1zbW9rZVoTCgF4Eg4KDAgBEggKAggCCgIIAloTCgF5Eg4KDAgBEggKAggCCgIIAmITCgF6Eg4KDAgBEggKAggCCgIIAkIECgAQEQ==')
session = ort.InferenceSession(model, sess_options=options, providers=['CUDAExecutionProvider'], enable_fallback=False)
if session.get_providers()[0] != 'CUDAExecutionProvider':
    raise RuntimeError('ORT silently fell back to CPU')
a = np.array([[1,2],[3,4]], dtype=np.float32)
result = session.run(None, {'x': a, 'y': a})[0]
np.testing.assert_array_equal(result, a @ a)
profile = Path(session.end_profiling())
events = json.loads(profile.read_text(encoding='utf-8'))
if not any(e.get('args', {}).get('provider') == 'CUDAExecutionProvider' for e in events):
    raise RuntimeError('No ORT operation executed on CUDA')
print(json.dumps({'version': ort.__version__, 'cuda_execution': True}))
'''


def check_cuda():
    from sakuratts.backends.cuda.runtime import import_cupy, validate_gpt_cuda_include_paths
    validate_gpt_cuda_include_paths()
    cp = import_cupy()
    result = {"torch_imported": "torch" in sys.modules,
              "driver_version": cp.cuda.runtime.driverGetVersion(), "cupy": cp.__version__}
    props = cp.cuda.runtime.getDeviceProperties(0)
    result.update(gpu=props['name'].decode(), compute_capability=[props['major'], props['minor']],
                  total_memory=props['totalGlobalMem'])
    kernel = cp.RawKernel('extern "C" __global__ void twice(float* a) { a[threadIdx.x] *= 2; }', 'twice')
    data = cp.ones(32, dtype=cp.float32)
    kernel((1,), (32,), (data,))
    cp.testing.assert_array_equal(data, cp.full(32, 2, dtype=cp.float32))
    for dtype in (cp.float32, cp.float16):
        a = cp.ones((16, 16), dtype=dtype)
        cp.testing.assert_array_equal(a @ a, cp.full((16, 16), 16, dtype=dtype))
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        kernel((1,), (32,), (data,))
        graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    cp.testing.assert_array_equal(data, cp.full(32, 4, dtype=cp.float32))
    result.update(nvrtc=True, gemm_fp32=True, gemm_fp16=True, cuda_graph=True)
    child = subprocess.run([str(ROOT / 'runtime/acoustic/python.exe'), '-I', '-B', '-c', ORT_PROBE],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
    result['ort'] = json.loads(child.stdout)
    return result


def check_ort(backend, device_id=0):
    import numpy as np
    import onnxruntime as ort
    provider = "DmlExecutionProvider" if backend == "directml" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        raise RuntimeError("Required provider is unavailable: " + provider)
    adapters = []
    selected = None
    if backend == "directml":
        from sakuratts.backends.directml.devices import list_adapters
        adapters = list_adapters()
        selected = next((adapter for adapter in adapters if adapter["device_id"] == device_id), None)
        if selected is None:
            raise ValueError("Unknown DXGI device_id " + str(device_id) + "; available adapters: " + json.dumps(adapters))
        if selected["software"]:
            raise ValueError("DXGI adapter " + str(device_id) + " (" + selected["description"]
                             + ") is a software renderer, not a hardware GPU. Select a hardware device_id or use --backend cpu.")
    # Embedded ONNX MatMul graphs avoid adding the ONNX conversion dependency.
    graphs = {
        "fp32": 'CAk6XQoRCgF4CgF5EgF6IgZNYXRNdWwSCWdwdS1zbW9rZVoTCgF4Eg4KDAgBEggKAggCCgIIAloTCgF5Eg4KDAgBEggKAggCCgIIAmITCgF6Eg4KDAgBEggKAggCCgIIAkIECgAQEQ==',
        "fp16": 'CAk6XQoRCgF4CgF5EgF6IgZNYXRNdWwSCWdwdS1zbW9rZVoTCgF4Eg4KDAgKEggKAggCCgIIAloTCgF5Eg4KDAgKEggKAggCCgIIAmITCgF6Eg4KDAgKEggKAggCCgIIAkIECgAQEQ=='
    }
    precisions = ("fp32", "fp16") if backend == "directml" else ("fp32",)
    executed = {}
    for precision in precisions:
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_profiling = True
        options.profile_file_prefix = str(ROOT / "cache" / ("ort-" + backend + "-" + precision))
        providers = [(provider, {"device_id": str(device_id)})] if backend == "directml" else [provider]
        session = ort.InferenceSession(base64.b64decode(graphs[precision]), sess_options=options,
                                      providers=providers, enable_fallback=False)
        if session.get_providers()[0] != provider:
            raise RuntimeError("ORT did not initialize the requested provider: " + provider)
        dtype = np.float16 if precision == "fp16" else np.float32
        data = np.array([[1, 2], [3, 4]], dtype=dtype)
        actual = session.run(None, {"x": data, "y": data})[0]
        np.testing.assert_array_equal(actual, (data.astype(np.float32) @ data.astype(np.float32)).astype(dtype))
        if actual.dtype != dtype:
            raise RuntimeError("Unexpected " + precision + " output dtype: " + str(actual.dtype))
        events = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        operations = [event["args"] for event in events if event.get("args", {}).get("provider")]
        if not operations or any(operation["provider"] != provider for operation in operations):
            raise RuntimeError(precision + " probe did not execute entirely on " + provider)
        executed[precision] = {"matmul": True, "provider": provider,
            "operations": [{key: operation.get(key) for key in ("op_name", "input_type_shape", "output_type_shape")}
                           for operation in operations]}
    return {"backend": backend, "provider": provider, "ort": ort.__version__,
            "device_id": device_id if backend == "directml" else None,
            "adapter": selected, "adapters": adapters, "precisions": executed,
            "matmul": True, "torch_imported": "torch" in sys.modules}


def check_mlx():
    import mlx.core as mx
    from importlib.metadata import version
    if not mx.metal.is_available():
        raise RuntimeError("Apple Metal is unavailable")
    with mx.stream(mx.gpu):
        data = mx.array([[1., 2.], [3., 4.]], dtype=mx.float32)
        result = data @ data
        mx.eval(result)
        if result.tolist() != [[7., 10.], [15., 22.]]:
            raise RuntimeError("Metal matrix product differs from the expected result")
    return {"backend": "mlx", "mlx": version("mlx"), "metal_execution": True,
            "matmul": True, "torch_imported": "torch" in sys.modules}


if __name__ == '__main__':
    release = json.loads((ROOT / "runtime/portable.json").read_text(encoding="utf-8"))["release"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "directml", "cuda", "mlx"),
                        default="cpu" if release["backend"] == "directml" else release["backend"])
    parser.add_argument("--device-id", type=int, default=0, help="DXGI adapter index for DirectML")
    args = parser.parse_args()
    output = ROOT / 'cache/runtime-check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = (check_cuda() if args.backend == "cuda" else check_mlx() if args.backend == "mlx"
                  else check_ort(args.backend, args.device_id))
        report = dict(result, passed=True, synthesis_tested=False, quality_validated=False)
    except Exception as error:
        report = {'passed': False, 'error': str(error), 'error_type': type(error).__name__,
                  'traceback': traceback.format_exc()}
        if isinstance(error, subprocess.CalledProcessError):
            report['worker_stderr'] = error.stderr
    report.update(backend=args.backend, device_id=args.device_id if args.backend == "directml" else None,
                  python=sys.version, executable=sys.executable, temporary_directory=tempfile.gettempdir())
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
