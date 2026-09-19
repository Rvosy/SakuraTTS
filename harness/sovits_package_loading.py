"""Compare one shared acoustic archive with the previous three-loader path.

Run diagnostics and normal load timings in separate fresh processes. The parent
saves source snapshots and real exits; verification copies weights after timing.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = 'a67074a'
COMPONENTS = ('encoder', 'flow', 'decoder')
sys.path.insert(0, str(PROJECT / 'src'))


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def worker(args):
    import mlx.core as mx
    import numpy as np
    from sakuratts import sovits_package as package_module
    from sakuratts.mlx_sovits import MLXSoVITS
    from sakuratts.weight_storage import array_sha256
    from g2pw_session_equivalence import memory as rss_memory
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024 * 1024)
    result = dict(status='running', policy=args.policy, mode=args.mode,
                  command=[sys.executable, *sys.argv], arrays={},
                  package=str(args.package), package_manifest_sha256=package_module.sha256(args.package / 'manifest.json'),
                  scope='V2Pro weight loading only. CPU encoder, GPU flow/decoder. Normal timings exclude observers and CPU weight copies. RSS lifetime max is not phase peak or NVIDIA VRAM.')
    modules = []
    if args.policy == 'previous':
        for component in COMPONENTS:
            name = f'sakuratts._previous_sovits_{component}'
            spec = importlib.util.spec_from_file_location(name, args.run / 'source' / 'previous' / f'mlx_sovits_{component}.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
    counters = dict(hash=0, archive_open=0, storage_validation=0, tensor_reads=0)
    if args.mode == 'diagnostic':
        def counted(fn, key):
            def invoke(*a, **kw):
                counters[key] += 1
                return fn(*a, **kw)
            return invoke
        # np.load is one shared module function; install exactly one wrapper.
        np.load = counted(np.load, 'archive_open')
        for module in modules or [package_module]:
            module.sha256 = counted(module.sha256, 'hash')
            module.validate_storage = counted(module.validate_storage, 'storage_validation')
            module.read_fp32 = counted(module.read_fp32, 'tensor_reads')
    def memory():
        return dict(**rss_memory(), mlx_active_bytes=mx.get_active_memory(),
                    mlx_cache_bytes=mx.get_cache_memory(), mlx_peak_bytes=mx.get_peak_memory())
    model = None
    try:
        result['before'] = memory()
        mx.reset_peak_memory()
        start = time.perf_counter()
        if modules:
            with mx.stream(mx.cpu):
                encoder = modules[0].MLXSoVITSEncoder.load(args.package, softmax='fp64-accumulation')
            flow = modules[1].MLXSoVITSFlow.load(args.package)
            decoder = modules[2].MLXSoVITSDecoder.load(args.package)
            model = MLXSoVITS(encoder, flow, decoder, mx.gpu, mx.cpu)
            del encoder, flow, decoder
        else:
            model = MLXSoVITS.load(args.package, encoder_device='cpu', encoder_softmax='fp64-accumulation')
        mx.synchronize()
        result['load_seconds'] = time.perf_counter() - start
        result['loaded'] = memory()
        if args.mode == 'diagnostic':
            result['counts'] = counters
            for component in COMPONENTS:
                for name, value in getattr(model, component).weights.items():
                    result['arrays'][name] = array_sha256(np.asarray(value))
            # Avoid keeping a loop reference after unloading.
            del value
        model = None
        gc.collect()
        mx.clear_cache()
        result['released'] = memory()
        result['torch_imported'] = 'torch' in sys.modules
        result['status'] = 'completed' if not result['torch_imported'] else 'error'
    except Exception:
        result.update(status='error', error=traceback.format_exc())
    dump(args.output, result)
    return 0 if result['status'] == 'completed' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--references', type=Path, default=PROJECT.parent / 'SakuraTTS-References')
    p.add_argument('--run', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--policy', choices=('previous', 'shared'))
    p.add_argument('--mode', choices=('normal', 'diagnostic'))
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if args.policy:
        return worker(args)
    run = args.references / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-sovits-package-loading')
    run.mkdir(parents=True)
    previous = run / 'source' / 'previous'
    previous.mkdir(parents=True)
    for component in COMPONENTS:
        filename = f'mlx_sovits_{component}.py'
        source = subprocess.check_output(['git', 'show', f'{BASELINE}:src/sakuratts/{filename}'], cwd=PROJECT)
        (previous / filename).write_bytes(source)
    for name in ['harness/sovits_package_loading.py', 'harness/g2pw_session_equivalence.py',
                 *['src/sakuratts/' + v + '.py' for v in ['mlx_sovits', 'sovits_package', 'weight_storage', *['mlx_sovits_' + c for c in COMPONENTS]]]]:
        target = run / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, target)
    from sakuratts.sovits_package import sha256
    dump(run / 'provenance.json', dict(command=[sys.executable, *sys.argv], previous_commit=BASELINE,
         source_sha256={str(f.relative_to(run)): sha256(f) for f in (run / 'source').rglob('*.py')}))
    print('RUN_DIRECTORY=' + str(run), flush=True)
    executions = []
    for mode, repeats in [('diagnostic', 1), ('normal', args.repeats)]:
        for index in range(repeats):
            for policy in (('previous', 'shared') if index % 2 == 0 else ('shared', 'previous')):
                output = run / f'{mode}-{policy}-{index}.json'
                command = [sys.executable, str(Path(__file__).resolve()), '--package', str(args.package),
                           '--run', str(run), '--output', str(output), '--policy', policy, '--mode', mode]
                with output.with_suffix('.log').open('w') as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                executions.append(dict(command=command, exit_code=process.returncode))
                dump(run / 'execution.json', executions)
                print(f'{mode} {policy} {index}: {process.returncode}', flush=True)
                if process.returncode:
                    return 1
    old = json.loads((run / 'diagnostic-previous-0.json').read_text())
    new = json.loads((run / 'diagnostic-shared-0.json').read_text())
    same = old['arrays'] == new['arrays'] and len(new['arrays']) == 650
    dump(run / 'comparison.json', dict(status='passed' if same else 'failed',
          all_650_weights_bit_exact=same, previous_counts=old['counts'], shared_counts=new['counts']))
    return 0 if same else 1


if __name__ == '__main__':
    raise SystemExit(main())
