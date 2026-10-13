#!/usr/bin/env python3
"""Build a fresh source snapshot, check it, then deploy to four idle owned nodes."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.recording import RunRecorder
from scripts.lab import docker_gpu_check as lab


def check_owned_idle():
    for name in lab.NAMES:
        _, raw = lab.remote(shlex.join(['docker', 'inspect', name]))
        data = json.loads(raw)[0]
        assert data['Config']['Labels'].get('edgevisor.lab') == lab.LABEL
        mounts = {item['Destination']: item for item in data['Mounts']}
        assert mounts['/source']['Source'] == lab.BASE + '/source' and not mounts['/source']['RW']
        _, processes = lab.remote(shlex.join(['docker', 'exec', '--user', lab.USER, name,
            '/bin/bash', '-lc', "pgrep -af '^./(dllama(-api)?|reserved-join-test|delayed-sample-worker-test)( |$)' || true"]))
        if processes.strip():
            raise RuntimeError('Owned inference still running: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--image', required=True, help='Existing development image; never pulled')
    parser.add_argument('--architecture', default='75')
    args = parser.parse_args()
    if not args.architecture.isdigit():
        parser.error('Architecture must be numeric')
    config = json.loads(Path(args.config).read_text())
    run = RunRecorder(args.output, {'purpose': 'build-deploy', 'host': config['host'],
                                   'image': args.image, 'architecture': args.architecture})
    try:
        lab.configure(args.config, run.directory / 'deployment', ROOT)
        check_owned_idle()
        inputs = run.directory / 'inputs'
        subprocess.run([sys.executable, str(ROOT / 'scripts/lab/package_sources.py'), str(inputs)], check=True)
        info = json.loads((inputs / 'build_source.json').read_text())
        run.manifest['metadata']['source'] = info
        _, image_id = lab.remote(shlex.join(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}']))
        run.manifest['metadata']['image_id'] = image_id.strip()
        name = 'build_' + str(time.time_ns()) + '_' + info['source_manifest_sha256'][:12]
        remote_build = lab.BASE + '/builds/' + name
        lab.remote('mkdir -p ' + shlex.quote(lab.BASE + '/builds') + ' && mkdir ' + shlex.quote(remote_build))
        subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
                        str(inputs / 'source.tar.gz'), str(inputs / 'source_manifest.json'),
                        str(inputs / 'build_source.json'), lab.HOST + ':' + remote_build + '/'], check=True)
        lab.remote('tar -xzf ' + shlex.quote(remote_build + '/source.tar.gz') + ' -C ' + shlex.quote(remote_build))
        command = ['docker', 'run', '--rm', '--pull=never', '--network', 'none',
                   '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                   '--user', lab.USER, '--cpus', '4', '--memory', '6g', '--pids-limit', '256',
                   '-v', remote_build + ':/source', '-w', '/source', args.image, '/bin/bash', '-lc',
                   'make -C EdgeVisor -j4 DLLAMA_CUDA=1 DLLAMA_VULKAN=0 CUDA_ARCHS=' + args.architecture + ' dllama dllama-api check']
        run.event('build_start', {'remote_build': remote_build, 'command': command})
        print('Building fresh source snapshot ' + info['source_manifest_sha256'], flush=True)
        code, output = lab.remote(shlex.join(command) + ' > ' + shlex.quote(remote_build + '/build.log') + ' 2>&1',
                                  timeout=900, check=False)
        _, log = lab.remote('cat ' + shlex.quote(remote_build + '/build.log'))
        (run.directory / 'build.log').write_text(log)
        run.event('build_end', {'exit_code': code})
        if code:
            raise RuntimeError('Build/check failed; see preserved build.log')
        check_owned_idle()
        lab.remote('cp -a ' + shlex.quote(remote_build + '/EdgeVisor/.') + ' ' + shlex.quote(lab.BASE + '/source/EdgeVisor/') +
                   ' && cp ' + shlex.quote(remote_build + '/Makefile') + ' ' + shlex.quote(lab.BASE + '/source/Makefile') +
                   ' && cp ' + shlex.quote(remote_build + '/source_manifest.json') + ' ' + shlex.quote(lab.BASE + '/source_manifest.json'))
        lab.validate()
        run.event('deployment_validated', {'remote_build': remote_build})
        run.manifest['metadata']['remote_build'] = remote_build
        run.seal('passed')
        print('Fresh CUDA/API build, all checks and deployed source validation passed.', flush=True)
    except BaseException as error:
        if not run.closed:
            run.seal('failed', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
