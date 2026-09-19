"""Compare complete G2PW pinyin calls without importing the full TTS frontend."""

import argparse
import ast
import copy
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from g2pw_dedup_diagnostic import COMMIT, extract_functions, literal_attribute, write_json
from g2pw_session_equivalence import memory

PROJECT = Path(__file__).resolve().parents[1]
MODULES = ('g2pw', 'g2pw_text', 'g2pw_inputs', 'g2pw_session', 'tokenizer')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    run = args.references / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-g2pw-pinyin')
    (run / 'source/sakuratts').mkdir(parents=True)
    print('RUN_DIRECTORY=' + str(run), flush=True)
    text_prepared = json.loads((args.text_run / 'prepared.json').read_text())
    resources = Path(text_prepared['resources'])
    cases = json.loads((args.text_run / 'cases.json').read_text())
    cases += [dict(id='repeated-batch', sentences=['重庆银行。', '天', '重庆银行。', ''], context=16, opencc=True),
              dict(id='tokenizer-error-empty-input', sentences='你好。', context=16, opencc=True, tokenizer_error=True)]
    write_json(run / 'cases.json', cases)
    for name in MODULES:
        shutil.copy2(PROJECT / 'src/sakuratts' / (name + '.py'), run / 'source/sakuratts' / (name + '.py'))
    for name in ('onnx_api.py', 'char_convert.py', 'char_bopomofo_dict.json'):
        shutil.copy2(args.text_run / 'source' / name, run / 'source' / name)
    official = args.references / 'GPT-SoVITS'
    for name in ('utils.py', 'dataset.py'):
        path = official / 'GPT_SoVITS/text/g2pw' / name
        if path.read_bytes() != subprocess.check_output(['git', '-C', str(official), 'show', COMMIT + ':GPT_SoVITS/text/g2pw/' + name]):
            raise ValueError('Official source changed: ' + name)
        shutil.copy2(path, run / 'source' / name)
    for name in ('g2pw_equivalence.py', 'g2pw_dedup_diagnostic.py', 'g2pw_session_equivalence.py'):
        shutil.copy2(PROJECT / 'harness' / name, run / 'source' / name)
    model = official / 'GPT_SoVITS/text/G2PWModel/g2pW.onnx'
    tokenizer = args.references / 'models/shared/chinese-roberta-wwm-ext-large'
    write_json(run / 'prepared.json', dict(command=[sys.executable, *sys.argv], text_run=str(args.text_run),
        text_prepared_sha256=sha256(args.text_run / 'prepared.json'), resources=str(resources),
        resource_manifest_sha256=sha256(resources / 'manifest.json'), model=str(model),
        model_sha256='2eb3c71fd95117b2e1abef8d2d0cd78aae894bbe7f0fac105ddc9c32ce63cbd0',
        tokenizer=str(tokenizer), tokenizer_sha256=sha256(tokenizer / 'tokenizer.json'),
        ort_package=str(args.ort_package.resolve()) if args.ort_package else None,
        ort_manifest_sha256=sha256(args.ort_package / 'manifest.json') if args.ort_package else None,
        official_commit=COMMIT, source_sha256={str(path.relative_to(run)): sha256(path) for path in (run / 'source').rglob('*') if path.is_file()},
        scope='Normalized Chinese segments through full G2PW pinyin; no Chinese tone sandhi, final phones, feature BERT or audio'))


def worker(args):
    os.environ['ORT_DISABLE_TELEMETRY'] = '1'
    import onnxruntime as ort
    from opencc import OpenCC
    from pypinyin import Style, pinyin
    run = args.run
    output = run / args.backend
    output.mkdir()
    prepared = json.loads((run / 'prepared.json').read_text())
    resources = Path(prepared['resources'])
    if sha256(Path(prepared['model'])) != prepared['model_sha256']:
        raise ValueError('Model hash differs from the verified G2PW model')
    captured = {}
    calls = []
    def capture_prepare(result):
        captured['prepared'] = copy.deepcopy(result)
        return result
    def capture_inputs(result):
        captured['inputs'] = {key: value.copy() for key, value in result.items()}
        return result
    def capture_predictions(result):
        captured['labels'], captured['confidences'] = result
        return result
    initial_memory = memory()
    started = time.perf_counter()
    if args.backend == 'official':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(prepared['tokenizer'], local_files_only=True)
        namespace = dict(np=np, re=re, Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple, Style=Style, pinyin=pinyin)
        exec(compile((run / 'source/char_convert.py').read_text(), 'official-char-convert', 'exec'), namespace)
        extract_functions(run / 'source/utils.py', {'wordize_and_map', 'tokenize_and_map'}, namespace)
        extract_functions(run / 'source/dataset.py', {'prepare_onnx_input', '_truncate_texts', '_truncate', 'get_phoneme_labels'}, namespace)
        extract_functions(run / 'source/onnx_api.py', {'predict'}, namespace)
        tree = extract_functions(run / 'source/onnx_api.py', {'__call__', '_prepare_data', '_convert_bopomofo_to_pinyin', '_predict_with_sentence_dedup'}, namespace, '_G2PWBaseOnnxConverter')
        polyphonic = [line.split('\t') for line in (resources / 'POLYPHONIC_CHARS.txt').read_text().strip().splitlines()]
        labels, char2phonemes = namespace['get_phoneme_labels'](polyphonic)
        chars = sorted(char2phonemes)
        shell = SimpleNamespace(labels=labels, chars=chars, char2phonemes=char2phonemes,
            char2id={char: index for index, char in enumerate(chars)},
            char_phoneme_masks={char: [1 if i in ids else 0 for i in range(len(labels))] for char, ids in char2phonemes.items()},
            polyphonic_chars_new=set(chars) - literal_attribute(tree, 'non_polyphonic'),
            monophonic_chars_dict=dict(line.split('\t') for line in (resources / 'MONOPHONIC_CHARS.txt').read_text().strip().splitlines()),
            bopomofo_convert_dict=json.loads((resources / 'bopomofo_to_pinyin_wo_tune_dict.json').read_text()),
            char_bopomofo_dict=json.loads((run / 'source/char_bopomofo_dict.json').read_text()),
            config=SimpleNamespace(use_mask=True, use_char_phoneme=False), tokenizer=tokenizer,
            cc=OpenCC('s2tw'), enable_opencc=True, enable_sentence_dedup=True, polyphonic_context_chars=16)
        for char in literal_attribute(tree, 'non_monophonic'):
            shell.monophonic_chars_dict.pop(char, None)
        shell.style_convert_func = lambda value: namespace['_convert_bopomofo_to_pinyin'](shell, value)
        shell._prepare_data = lambda sentences: capture_prepare(namespace['_prepare_data'](shell, sentences))
        pack = namespace['prepare_onnx_input']
        namespace['prepare_onnx_input'] = lambda *a, **kw: capture_inputs(pack(*a, **kw))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(prepared['model'], sess_options=options, providers=['CPUExecutionProvider'])
        def session_run(names, inputs):
            result = session.run(names, inputs)
            calls.append(dict(inputs={key: value.copy() for key, value in inputs.items()}, probabilities=result[0].copy()))
            return result
        shell._predict = lambda model_input: namespace['predict'](SimpleNamespace(run=session_run), model_input, labels)
        shell._predict_with_sentence_dedup = lambda model_input, texts: capture_predictions(namespace['_predict_with_sentence_dedup'](shell, model_input, texts))
        invoke = lambda sentences: namespace['__call__'](shell, sentences)
        convert = shell.style_convert_func
        def configure(case):
            shell.polyphonic_context_chars = case['context']
            shell.enable_opencc = case['opencc']
    else:
        sys.path.insert(0, str(run / 'source'))
        from sakuratts.g2pw import G2PW
        engine = G2PW(resources, Path(prepared['tokenizer']) / 'tokenizer.json',
                      None if prepared.get('ort_package') else prepared['model'], ort_package=prepared.get('ort_package'))
        session, tokenizer = engine.session.session, engine.inputs.tokenizer
        labels, chars = engine.inputs.labels, engine.inputs.chars
        text_prepare = engine.text.prepare
        engine.text.prepare = lambda sentences: capture_prepare(text_prepare(sentences))
        pack = engine.inputs.prepare
        engine.inputs.prepare = lambda *a, **kw: capture_inputs(pack(*a, **kw))
        predict = engine.session.predict
        engine.session.predict = lambda inputs, texts: capture_predictions(predict(inputs, texts))
        original_run = engine.session.run
        def session_run(inputs):
            probabilities = original_run(inputs)
            mapped = dict(zip(('input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids'),
                              (inputs[key] for key in ('input_ids', 'token_type_ids', 'attention_masks', 'phoneme_masks', 'char_ids', 'position_ids'))))
            calls.append(dict(inputs={key: value.copy() for key, value in mapped.items()}, probabilities=probabilities.copy()))
            return probabilities
        engine.session.run = session_run
        invoke, convert = engine, engine.text.convert_bopomofo
        cc = engine.text.cc
        def configure(case):
            engine.text.context_chars = case['context']
            engine.text.cc = cc if case['opencc'] else None
    load_seconds = time.perf_counter() - started
    report = dict(command=[sys.executable, *sys.argv], backend=args.backend, versions={name: metadata.version(name) for name in ('numpy', 'onnxruntime', 'tokenizers', 'pypinyin', 'opencc-python-reimplemented')},
        providers=session.get_providers(), ort_build=ort.get_build_info(), labels=labels, chars=chars,
        load_seconds=load_seconds, memory=dict(initial=initial_memory, loaded=memory()), cases=[],
        timing_scope='Diagnosis includes preparation/input/probability copies; no normal request speed claim')
    normal_tokenize = tokenizer.tokenize
    def broken_tokenize(_text):
        raise RuntimeError('Intentional tokenizer failure to exercise the official empty-input return')
    for index, case in enumerate(json.loads((run / 'cases.json').read_text())):
        captured.clear()
        calls.clear()
        configure(case)
        tokenizer.tokenize = broken_tokenize if case.get('tokenizer_error') else normal_tokenize
        started = time.perf_counter()
        try:
            result = invoke(case['sentences'])
            row = dict(id=case['id'], result=result)
        except Exception as error:
            row = dict(id=case['id'], error_type=type(error).__name__, error_message=str(error))
        row['diagnostic_seconds'] = time.perf_counter() - started
        tokenizer.tokenize = normal_tokenize
        row['prepared'] = captured.get('prepared')
        row['labels'], row['confidences'] = captured.get('labels', []), captured.get('confidences', [])
        arrays = {'packed.' + key: value for key, value in captured.get('inputs', {}).items()}
        for call_index, call in enumerate(calls):
            arrays.update({f'call{call_index}.{key}': value for key, value in call['inputs'].items()})
            arrays[f'call{call_index}.probabilities'] = call['probabilities']
        filename = f'case-{index}.npz'
        np.savez(output / filename, **arrays)
        row.update(arrays=filename, arrays_sha256=sha256(output / filename), session_calls=len(calls))
        report['cases'].append(row)
        print('CASE_COMPLETED=' + case['id'], flush=True)
    report['label_conversion'] = {label: convert(label) for label in labels}
    report['memory']['after_calls'] = memory()
    if args.backend == 'native':
        engine.close()
    session = None
    gc.collect()
    report['memory']['after_close'] = memory()
    report.update(status='completed', imported={name: name in sys.modules for name in ('torch', 'transformers', 'mlx')})
    write_json(output / 'result.json', report)


def execute(args):
    run = args.run
    if (run / 'comparison.json').exists():
        raise FileExistsError('Create new evidence instead of replacing this run')
    prepared = json.loads((run / 'prepared.json').read_text())
    if prepared.get('ort_package'):
        package = Path(prepared['ort_package'])
        if sha256(package / 'manifest.json') != prepared['ort_manifest_sha256']:
            raise ValueError('ORT package manifest changed')
        if json.loads((package / 'manifest.json').read_text())['source_sha256'] != prepared['model_sha256']:
            raise ValueError('ORT package was not converted from the official G2PW model')
    if sha256(Path(__file__)) != prepared['source_sha256']['source/g2pw_equivalence.py']:
        raise ValueError('Harness changed after preparation')
    for relative, expected in prepared['source_sha256'].items():
        if sha256(run / relative) != expected:
            raise ValueError('Source snapshot changed: ' + relative)
    resources = Path(prepared['resources'])
    if sha256(Path(prepared['tokenizer']) / 'tokenizer.json') != prepared['tokenizer_sha256']:
        raise ValueError('Tokenizer changed after preparation')
    if sha256(resources / 'manifest.json') != prepared['resource_manifest_sha256']:
        raise ValueError('Resource manifest changed')
    for name, expected in json.loads((resources / 'manifest.json').read_text())['files'].items():
        if sha256(resources / name) != expected:
            raise ValueError('Resource changed: ' + name)
    env = os.environ.copy()
    env.update(ORT_DISABLE_TELEMETRY='1', USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2', MKL_NUM_THREADS='2')
    processes = []
    for backend, venv in (('official', '.venv-official-macos'), ('native', '.venv-mlx-macos')):
        command = [str(args.references / venv / 'bin/python'), str(Path(__file__).resolve()), 'worker', '--run', str(run), '--backend', backend]
        with (run / (backend + '.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        processes.append(dict(command=command, exit_code=result.returncode))
        write_json(run / 'processes.json', processes)
        if result.returncode:
            raise SystemExit(result.returncode)
    gold, actual = [json.loads((run / name / 'result.json').read_text()) for name in ('official', 'native')]
    checks = []
    for before, after in zip(gold['cases'], actual['cases'], strict=True):
        fields = ('id', 'result', 'error_type', 'error_message', 'prepared', 'labels', 'confidences', 'session_calls')
        row = dict(id=before['id'], outputs_equal=all(before.get(key) == after.get(key) for key in fields),
                   exception=before.get('error_type'))
        with np.load(run / 'official' / before['arrays'], allow_pickle=False) as a, np.load(run / 'native' / after['arrays'], allow_pickle=False) as b:
            row['array_keys_equal'] = a.files == b.files
            row['bit_exact_arrays'] = all(key in b and a[key].dtype == b[key].dtype and a[key].shape == b[key].shape and
                                         a[key].tobytes() == b[key].tobytes() for key in a.files)
            row['array_count'] = len(a.files)
        checks.append(row)
    same_mapping = gold['label_conversion'] == actual['label_conversion']
    same_tables = gold['labels'] == actual['labels'] and gold['chars'] == actual['chars']
    passed = same_mapping and same_tables and not any(actual['imported'].values()) and all(
        row['outputs_equal'] and row['array_keys_equal'] and row['bit_exact_arrays'] for row in checks)
    write_json(run / 'comparison.json', dict(status='passed' if passed else 'mismatch', cases=checks,
        label_conversion_equal=same_mapping, model_tables_equal=same_tables, processes=processes))
    print(json.dumps(dict(run=str(run), passed=passed, cases=len(checks), array_count=sum(row['array_count'] for row in checks))))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=PROJECT.parent / 'SakuraTTS-References')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--text-run', type=Path, required=True)
    prep.add_argument('--ort-package', type=Path)
    for name in ('run', 'worker'):
        child = commands.add_parser(name)
        child.add_argument('--run', type=Path, required=True)
        if name == 'worker':
            child.add_argument('--backend', choices=('official', 'native'), required=True)
    args = parser.parse_args()
    return {'prepare': prepare, 'run': execute, 'worker': worker}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
