"""使用包内解释器检查导入、HTTP 生命周期；Windows 另做 CPU 实际合成。"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import socket
import shutil
import struct
import subprocess
import tempfile
import time
from urllib.request import Request, build_opener, ProxyHandler
import wave


def check_windows_long_paths(bundle, output):
    """Exercise each bundled NumPy ABI and its BLAS controller beyond MAX_PATH."""
    package = bundle / 'runtime/main/Lib/site-packages/sakuratts'
    code = '''import json,runpy,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
runpy.run_path(str(Path(sys.argv[2])/'runtime/worker.py'))['load_package'](sys.argv[2])
import numpy as np
from threadpoolctl import ThreadpoolController
controller=ThreadpoolController()
blas=[row for row in controller.info() if row['user_api']=='blas']
assert blas, controller.info()
assert all(len(row['filepath'])>260 and Path(row['filepath']).is_file() for row in blas), blas
before=[row['num_threads'] for row in controller.info()]
with controller.limit(limits=1,user_api='blas'):
    assert all(row['num_threads']==1 for row in controller.info() if row['user_api']=='blas')
    np.testing.assert_array_equal(np.ones((2,2))@np.ones((2,2)),np.full((2,2),2.))
assert [row['num_threads'] for row in controller.info()]==before
print(json.dumps({'numpy':np.__version__,'blas':blas}))
'''
    reports = {}
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        for name in ('main', 'preparation'):
            runtime = bundle / 'runtime' / name
            site = runtime / 'Lib/site-packages'
            destination = Path(temporary) / ('long-installation-' * 7) / ('中文 path-' * 12) / name
            destination.mkdir(parents=True)
            for folder in ('numpy', 'numpy.libs'):
                source = site / folder
                if source.exists():
                    shutil.copytree(source, destination / folder, ignore=shutil.ignore_patterns('__pycache__'))
            result = subprocess.run([str(runtime / 'python.exe'), '-I', '-X', 'utf8', '-c', code,
                                     str(destination), str(package)], capture_output=True,
                                    text=True, encoding='utf-8', timeout=60)
            (output / (name + '-long-path.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
            result.check_returncode()
            reports[name] = json.loads(result.stdout)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    release = json.loads((bundle / 'runtime/portable.json').read_text())['release']
    apple = release['target'] == 'macos-arm64'
    python = bundle / release.get('python_executable', 'runtime/main/python.exe')
    command = [str(python), '-I', str(bundle / 'launcher.py')]
    opener = build_opener(ProxyHandler({}))
    def fetch(url, destination):
        with opener.open(url, timeout=120) as response, destination.open('wb') as stream:
            while block := response.read(1024*1024):
                stream.write(block)
    report = {'platform': release['target'], 'gpuSynthesis': False, 'listeningQuality': False}
    if not apple:
        report['longPaths'] = check_windows_long_paths(bundle, output)
    ffmpeg = bundle / ('runtime/bin/ffmpeg' if apple else 'runtime/bin/ffmpeg.exe')
    pcm = b''.join(struct.pack('<h', int(6000 * math.sin(i*2*math.pi*220/32000))) for i in range(32000))
    for codec, container in [('libvorbis','ogg'),('aac','adts')]:
        encoded = subprocess.run([str(ffmpeg),'-v','error','-f','s16le','-ar','32000','-ac','1',
                                  '-i','pipe:0','-c:a',codec,'-f',container,'pipe:1'],
                                  input=pcm,capture_output=True,check=True).stdout
        decoded = subprocess.run([str(ffmpeg),'-v','error','-i','pipe:0','-f','f32le','-ac','1','-ar','32000','pipe:1'],
                                  input=encoded,capture_output=True,check=True).stdout
        assert len(decoded)>0
    report['audioFormats']=['wav','raw','ogg','aac']
    if apple:
        code = 'import mlx.core as mx; import onnxruntime; mx.set_default_device(mx.cpu); assert (mx.ones((2,2)) @ mx.ones((2,2))).tolist()==[[2.,2.],[2.,2.]]'
        # 启动器设置隔离环境，MLX 的 CPU 运算不声称验收 Metal。
        subprocess.run([str(python), '-I', '-c', code], check=True)
    else:
        subprocess.run([*command, 'check-runtime', '--backend', 'cpu'], check=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    def request(path, payload=None):
        request = Request(f'http://127.0.0.1:{port}' + path, data=None if payload is None else json.dumps(payload).encode(),
                          headers={'Content-Type':'application/json'})
        with opener.open(request, timeout=300) as response:
            return response.read()
    launch = [*command, 'serve', '--runtime-mode', 'managed', '--idle-sleep-seconds', '2', '-p', str(port)]
    if not apple:
        model_url = 'https://huggingface.co/lj1995/GPT-SoVITS/resolve/336b2ec4e8d4ac74740798dd40af44e74659ecaf/'
        for remote, local in [('s1v3.ckpt', 'gpt.ckpt'), ('v2Pro/s2Gv2ProPlus.pth', 'sovits.pth')]:
            fetch(model_url + remote, output / local)
        reference = output / 'reference.wav'
        with wave.open(str(reference), 'wb') as audio:
            audio.setparams((1,2,32000,0,'NONE','not compressed'))
            audio.writeframes(b''.join(struct.pack('<h', int(6000 * math.sin(i*2*math.pi*(170+(i//8000)%5*35)/32000))) for i in range(4*32000)))
        config = output / 'config.json'
        config.write_text(json.dumps({'custom': {'t2s_weights_path': str(output/'gpt.ckpt'),
                         'vits_weights_path':str(output/'sovits.pth'), 'is_half':False}, 'sakuratts':{'backend':'cpu'}}))
        launch.extend(['--backend', 'cpu', '-c', str(config)])
    with (output / 'server.log').open('wb') as log:
        process = subprocess.Popen(launch, cwd=bundle, env=dict(os.environ, SAKURATTS_CACHE_DIR=str(output/'cache')),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=apple)
        try:
            deadline = time.monotonic()+120
            while True:
                if process.poll() is not None:
                    raise RuntimeError('服务提前退出，见 server.log')
                try:
                    state = json.loads(request('/runtime'));break
                except OSError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.1)
            assert state['state']=='sleeping' and state['worker_pid'] is None, state
            report['initial'] = state
            if not apple:
                request('/runtime/wake', {'keep_alive_seconds':0})
                audio = request('/tts', {'text':'こんにちは。','text_lang':'ja', 'prompt_lang':'ja',
                    'prompt_text':'おはよう。', 'ref_audio_path':str(reference), 'parallel_infer':False,
                    'streaming_mode':False, 'media_type':'wav'})
                with wave.open(io.BytesIO(audio)) as wav:
                    report['audio']={'frames':wav.getnframes(),'sampleRate':wav.getframerate()}
                    assert wav.getnframes()>0
                deadline = time.monotonic()+20
                while True:
                    state=json.loads(request('/runtime'))
                    if state['state']=='sleeping':break
                    if time.monotonic()>deadline:raise RuntimeError('引擎未按空闲设置休眠：'+str(state))
                    time.sleep(.2)
                assert not state['model_loaded'] and state['worker_pid'] is None, state
                report['idle']=state
            report['passed']=True
        finally:
            if apple:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
            else:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
            process.wait(15)
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
