"""Select relocatable CPython files and audit macOS binary requirements."""

from pathlib import Path, PurePosixPath
import posixpath
import re
import struct

from packaging.tags import parse_tag


def add_interpreter(plan, base, target):
    base = base.resolve(strict=True)
    candidates = [p for p in (base / 'lib').glob('python3.*') if p.is_dir() and (p / 'os.py').is_file()]
    if len(candidates) != 1 or (base / 'pyvenv.cfg').exists():
        raise ValueError('Provide a standalone relocatable CPython base, not a venv or framework installation')
    stdlib = candidates[0]
    version = stdlib.name.removeprefix('python')
    executable = (base / 'bin' / ('python' + version)).resolve(strict=True)
    if not executable.is_relative_to(base):
        raise ValueError('Python executable must belong to its standalone base')
    prefix = target.rstrip('/') + '/' if target else ''
    plan.add(executable, prefix + 'bin/python3', 'cpython')
    for source in sorted(stdlib.rglob('*')):
        relative = source.relative_to(base)
        if set(relative.parts) & {'site-packages', '__pycache__', 'test', 'tests', 'ensurepip', 'idlelib', 'tkinter'}:
            continue
        if source.is_dir() or source.suffix in ('.pyc', '.pyo', '.a') or source.name.startswith('_tkinter.'):
            continue
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(base):
            raise ValueError('Python resource escapes its base: ' + str(source))
        plan.add(resolved, prefix + relative.as_posix(), 'cpython')
    for source in sorted((base / 'lib').glob('libpython*.dylib')):
        plan.add(source.resolve(strict=True), prefix + 'lib/' + source.name, 'cpython')
    plan.add(stdlib / 'LICENSE.txt', prefix + 'LICENSE.txt', 'cpython')
    return version, prefix + 'lib/python' + version + '/site-packages'


def wheel_compatible(directory, minimum_version, python_version):
    """Metal libraries also have a wheel OS requirement, without Mach-O headers."""
    wheel = directory / 'WHEEL'
    if not wheel.is_file():
        raise ValueError('macOS runtime input requires wheel metadata: ' + str(directory))
    tags = {tag for line in wheel.read_text().splitlines() if line.startswith('Tag: ')
            for tag in parse_tag(line[5:])}
    target = tuple(map(int, minimum_version.split('.')))[:2]
    interpreter = 'cp' + python_version.replace('.', '')
    for tag in tags:
        if not (tag.interpreter in ('py3', 'py' + python_version.replace('.', ''), interpreter)
                or (tag.abi == 'abi3' and tag.interpreter.startswith('cp3')
                    and int(tag.interpreter[3:]) <= int(python_version.split('.')[1]))):
            continue
        if tag.abi not in ('none', 'abi3', interpreter):
            continue
        if tag.platform == 'any':
            return
        match = re.fullmatch(r'macosx_(\d+)_(\d+)_(arm64|universal2)', tag.platform)
        if match and (int(match[1]), int(match[2])) <= target:
            return
    raise ValueError(f'{directory.name} has no Apple silicon wheel compatible with Python {python_version} / macOS {minimum_version}')


def macho(path):
    """Read the arm64 slice's deployment target and dynamic loader commands."""
    with Path(path).open('rb') as stream:
        magic = stream.read(4)
        offset = 0
        if magic in (b'\xca\xfe\xba\xbe', b'\xca\xfe\xba\xbf'):
            count = struct.unpack('>I', stream.read(4))[0]
            wide = magic == b'\xca\xfe\xba\xbf'
            fmt = '>iiQQII' if wide else '>iiIII'
            slices = [struct.unpack(fmt, stream.read(struct.calcsize(fmt))) for _ in range(count)]
            arm = next((row for row in slices if row[0] == 0x100000c), None)
            if arm is None:
                raise ValueError('Mach-O has no arm64 slice: ' + str(path))
            offset = arm[2]
            stream.seek(offset)
            magic = stream.read(4)
        if magic not in (b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf'):
            return None
        if magic != b'\xcf\xfa\xed\xfe':
            raise ValueError('Expected a 64-bit arm64 Mach-O: ' + str(path))
        stream.seek(offset)
        header = struct.unpack('<IiiIIIII', stream.read(32))
        if header[1] != 0x100000c:
            raise ValueError('Mach-O has no arm64 slice: ' + str(path))
        result = {'minimum_macos': (0, 0, 0), 'dependencies': [], 'rpaths': [], 'install_name': None}
        for _ in range(header[4]):
            command, size = struct.unpack('<II', stream.read(8))
            payload = stream.read(size - 8)
            kind = command & 0x7fffffff
            if kind in (0x24, 0x32):
                version = struct.unpack_from('<I', payload, 0 if kind == 0x24 else 4)[0]
                result['minimum_macos'] = (version >> 16, (version >> 8) & 255, version & 255)
            elif kind in (0xc, 0xd, 0x18, 0x1f, 0x20, 0x23, 0x1c):
                start = struct.unpack_from('<I', payload)[0] - 8
                name = payload[start:].split(b'\0')[0].decode()
                if kind == 0xd:
                    result['install_name'] = name
                else:
                    result['rpaths' if kind == 0x1c else 'dependencies'].append(name)
        return result


def audit(plan, minimum_version, executable, *, preload_directories=()):
    """Reject external dylibs and binaries newer than the declared release OS."""
    target = tuple(map(int, minimum_version.split('.')))
    target = (*target, *(0 for _ in range(3 - len(target))))
    binaries = {name: info for name, row in plan.files.items() if (info := macho(row['source'])) is not None}
    executable_dir = str(PurePosixPath(executable).parent)
    preloaded = {info['install_name'] for name, info in binaries.items()
                 if str(PurePosixPath(name).parent) in preload_directories}

    def expand(value, loader):
        value = value.replace('@loader_path', str(PurePosixPath(loader).parent))
        value = value.replace('@executable_path', executable_dir)
        return posixpath.normpath(value)

    errors = []
    for name, info in binaries.items():
        if info['minimum_macos'] > target:
            errors.append(f'{name} requires macOS {".".join(map(str, info["minimum_macos"]))}, newer than {minimum_version}')
        rpaths = [expand(p, name) for p in info['rpaths']]
        rpaths += [expand(p, executable) for p in binaries.get(executable, {}).get('rpaths', [])]
        for dependency in info['dependencies']:
            if dependency.startswith(('/usr/lib/', '/System/Library/')):
                continue
            if dependency.startswith('@rpath/') and dependency in preloaded:
                continue
            candidates = ([posixpath.normpath(p + '/' + dependency[7:]) for p in rpaths]
                          if dependency.startswith('@rpath/') else [expand(dependency, name)])
            if not any(path in binaries for path in candidates):
                errors.append(f'{name} depends on a library outside the bundle: {dependency}')
    if errors:
        raise ValueError('\n'.join(errors))
    return {'minimum_macos': minimum_version, 'architecture': 'arm64', 'mach_o_files': len(binaries)}
