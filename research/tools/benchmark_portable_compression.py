"""Compare lossless 7z settings on disjoint windows from the largest bundle files."""

import argparse
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--sevenzip', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('archive', root / 'scripts/archive_portable.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
sevenzip = args.sevenzip or a.find_sevenzip()
if sevenzip is None:
    parser.error('7-Zip was not found; specify --sevenzip')
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
sample = out / 'sample'
sample.mkdir()
bundle = args.bundle.resolve()
manifest = json.loads((bundle / 'bundle-manifest.json').read_text(encoding='utf-8'))
selected = sorted(manifest['files'], key=lambda n: -manifest['files'][n]['bytes'])[:16]
inventory = []
for i, name in enumerate(selected):
    path = (bundle / name).resolve(strict=True)
    if bundle not in path.parents or a.digest(path) != manifest['files'][name]['sha256']:
        raise ValueError('Bundle file changed or escaped its directory: ' + name)
    ranges = a.sample_ranges(path.stat().st_size)
    target = sample / (f'{i:02}' + path.suffix)
    with path.open('rb') as src, target.open('wb') as dst:
        for offset, length in ranges:
            src.seek(offset)
            dst.write(src.read(length))
    inventory.append(dict(source=name, ranges=ranges, sample=target.name, sha256=a.digest(target)))
listing = out / 'files.txt'
listing.write_text('\n'.join(p.name for p in sample.iterdir()), encoding='utf-8')
a.PROFILES = {
    'baseline128': ['-mx=9', '-md=128m'],
    'sorted128': ['-mx=9', '-md=128m', '-mqs=on'],
    'dense256': ['-mx=9', '-md=256m', '-mqs=on', '-mfb=273'],
    'dense512': ['-mx=9', '-md=512m', '-mqs=on', '-mfb=273'],
    'deep256': ['-mx=9', '-md=256m', '-mqs=on', '-mfb=273', '-mmc=1000'],
}
report = dict(input_bytes=sum(p.stat().st_size for p in sample.iterdir()), inventory=inventory, results=[])
for name in a.PROFILES:
    archive = out / (name + '.7z')
    result = a.compress(sevenzip, sample, listing, archive, name)
    result['extraction_seconds'] = a.run(sevenzip, ['x', '-y', str(archive), '-o'+str(out/name)], sample)
    for item in inventory:
        if a.digest(out/name/item['sample']) != item['sha256']:
            raise ValueError('Extracted sample checksum mismatch: ' + item['sample'])
    result['sha256_verified'] = True
    report['results'].append(result)
    (out/'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
