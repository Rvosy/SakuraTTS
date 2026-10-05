"""Reference worker reuse and ownership without loading production models."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from sakuratts.prepare.reference_process import ReferencePreparer


class ReferenceProcessTests(unittest.TestCase):
    def test_session_reuses_model_identity_and_refreshes_new_audio(self):
        from sakuratts.prepare import prepare_resources as resources

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / 'official'
            source.mkdir()
            frontend = root / 'frontend'
            frontend.mkdir()
            (frontend / 'manifest.json').write_text('{"official_commit":"fixture"}')
            gpt, sovits, audio = (root / name for name in ('gpt.ckpt', 'sovits.pth', 'audio.wav'))
            for path in (gpt, sovits, audio):
                path.write_bytes(path.name.encode())
            inputs = root / 'inputs.json'
            inputs.write_text(json.dumps({'gpt': str(gpt), 'sovits': str(sovits),
                'references': [{'audio': str(audio), 'text': 'reference', 'language': 'ja', 'tone': 'reference'}]}))
            jobs = []
            def worker(job_file, session):
                jobs.append(resources.read_json(job_file))
                return 0

            session = {}
            with patch.object(resources, 'inspect_frontend', return_value={}) as preflight, \
                 patch.object(resources, 'digest', wraps=resources.digest) as digest, \
                 patch.object(resources, 'worker', side_effect=worker):
                for index, current in enumerate((session, session, {})):
                    audio.write_bytes(f'audio {index}'.encode())
                    if index == 2:
                        gpt.write_bytes(b'updated model')
                    resources.main(['--official-source', str(source), '--inputs', str(inputs),
                        '--frontend', str(frontend), '--output', str(root / f'output-{index}')], current)
                hashed = [Path(call.args[0]) for call in digest.call_args_list]
                self.assertEqual((hashed.count(gpt), hashed.count(sovits), hashed.count(audio)), (2, 2, 3))
                self.assertEqual(preflight.call_count, 2)
            self.assertEqual(jobs[0]['source_hashes'][str(gpt)], jobs[1]['source_hashes'][str(gpt)])
            self.assertNotEqual(jobs[1]['source_hashes'][str(gpt)], jobs[2]['source_hashes'][str(gpt)])
            self.assertEqual(len({job['source_hashes'][str(audio)] for job in jobs}), 3)

    def test_multiple_references_share_worker_and_failure_reaps_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = Path(__file__).resolve().parents[1] / 'sakuratts'
            (root / 'prepare_resources.py').write_text(f'''
import os, sys, runpy
from pathlib import Path
runpy.run_path({str(package / 'runtime/worker.py')!r})['load_package']({str(package)!r})
from sakuratts.runtime.protocol import read_message, write_message
count = 0
while True:
    request, _ = read_message(sys.stdin.buffer)
    count += 1
    target = request['arguments'][0]
    if target == 'fail':
        write_message(sys.stdout.buffer, {{'status': 'error', 'error': 'reference preparation failed'}})
    else:
        Path(target).write_text(str(os.getpid()) + ':' + str(count))
        write_message(sys.stdout.buffer, {{'status': 'ok'}})
''', encoding='utf-8')
            preparer = ReferencePreparer()
            with patch('sakuratts.prepare.reference_process.__file__', str(root / 'reference_process.py')):
                try:
                    for name in ('one', 'two'):
                        preparer.run([sys.executable, '-B', 'unused', str(root / name)], env=None)
                    first = (root / 'one').read_text().split(':')
                    second = (root / 'two').read_text().split(':')
                    self.assertEqual(first[0], second[0])
                    self.assertEqual(second[1], '2')
                    process = preparer.process
                    with self.assertRaisesRegex(RuntimeError, 'reference preparation failed'):
                        preparer.run([sys.executable, '-B', 'unused', 'fail'], env=None)
                    self.assertIsNotNone(process.poll())
                    self.assertIsNone(preparer.process)
                finally:
                    preparer.close()
