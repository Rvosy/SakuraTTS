"""Exercise native MLX lifecycle policies on one prepared model/reference pair."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--text', default='こんにちは。今日はいい天気ですね。\n午後は図書館へ行って、新しい本を探そうと思っています。')
    args = parser.parse_args()
    from sakuratts import Engine
    from sakuratts.module.reference_condition import PreparedReference
    from sakuratts.runtime.cancellation import SynthesisCancelled
    import numpy as np
    import mlx.core as mx
    args.output.mkdir(parents=True, exist_ok=False)
    reference = PreparedReference.load(args.reference)
    report = {'status': 'running', 'platform': platform.platform(), 'text': args.text, 'policies': [],
              'quality': {'human_listening': 'not_run', 'asr': 'not_run'}}
    try:
        for profile in ('fp32', 'low-memory', 'minimum-memory'):
            print('Checking ' + profile, flush=True)
            with Engine.load(args.model, backend='mlx', profile=profile, load_references=False) as engine:
                row = {'profile': profile, 'requests': []}
                report['policies'].append(row)
                baseline = None
                for index in range(2):
                    started = time.perf_counter()
                    audio = engine.synthesize(args.text, reference=reference, seed=1234)
                    audio.save(args.output / (profile + '-' + str(index) + '.wav'))
                    row['requests'].append({'seconds': time.perf_counter() - started, 'report': audio.report,
                        'pcm_sha256': hashlib.sha256(audio.pcm.tobytes()).hexdigest()})
                    if len(audio.report['fragments']) < 2:
                        raise AssertionError('Lifecycle verification requires at least two synthesized fragments')
                    if baseline is None:
                        baseline = audio.pcm.copy()
                    else:
                        np.testing.assert_array_equal(audio.pcm, baseline)
                calls = 0
                def cancel():
                    nonlocal calls
                    calls += 1
                    return calls >= 10
                try:
                    engine.synthesize(args.text, reference=reference, seed=1234,
                                      cancel_requested=cancel)
                except SynthesisCancelled:
                    row['cancelled_during_request'] = calls >= 10
                else:
                    raise AssertionError('The request did not cancel')
                recovered = engine.synthesize(args.text, reference=reference, seed=1234)
                np.testing.assert_array_equal(recovered.pcm, baseline)
                row['recovered_pcm_equal'] = True
            mx.synchronize()
            row['active_bytes_after_close'] = mx.get_active_memory()
            row['cache_bytes_after_close'] = mx.get_cache_memory()
        report.update(status='passed', torch_imported='torch' in sys.modules)
        if report['torch_imported']:
            raise AssertionError('Ordinary synthesis imported Torch')
    except BaseException:
        import traceback
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Verified MLX lifecycle: ' + str(args.output / 'report.json'))


if __name__ == '__main__':
    main()
