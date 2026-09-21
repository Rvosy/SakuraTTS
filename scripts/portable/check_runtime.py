"""Exercise CUDA compilation, GEMM, graphs and the independent ORT worker."""

import json
from pathlib import Path
import subprocess
import sys

from sakuratts.backends.cuda.runtime import import_cupy, validate_gpt_cuda_include_paths

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
session = ort.InferenceSession(model, sess_options=options, providers=['CUDAExecutionProvider'])
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


def check():
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


if __name__ == '__main__':
    output = ROOT / 'cache/runtime-check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = dict(check(), passed=True, synthesis_tested=False, quality_validated=False)
    except Exception as error:
        report = {'passed': False, 'error': str(error)}
        if isinstance(error, subprocess.CalledProcessError):
            report['worker_stderr'] = error.stderr
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
