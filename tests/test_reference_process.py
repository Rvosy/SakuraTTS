"""Reference worker reuse and ownership without loading production models."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from sakuratts.prepare.reference_process import ReferencePreparer


class ReferenceProcessTests(unittest.TestCase):
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
