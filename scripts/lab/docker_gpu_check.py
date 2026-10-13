#!/usr/bin/env python3
"""Run bounded model-backed checks in four explicitly owned GPU containers."""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import time

if not __debug__:
    raise SystemExit("Run without -O: ownership and result checks must stay enabled.")
HOST = LABEL = BASE = MODEL = TOKENIZER = USER = None
OUT = None
NAMES = []
IPS = []
PORT = 19393
SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]

def configure(path, output, source_root):
    global HOST, LABEL, BASE, MODEL, TOKENIZER, USER, OUT, NAMES, PORT, SOURCE_ROOT
    config = json.loads(pathlib.Path(path).read_text())
    HOST = config['host']
    LABEL = config['label']
    BASE = config['remote_base']
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
    USER = config['user']
    NAMES = config['containers']
    PORT = int(config.get('port', 19393))
    assert re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.@-]*', HOST)
    assert re.fullmatch('[A-Za-z0-9_.-]+', LABEL)
    assert re.fullmatch('[0-9]+:[0-9]+', USER)
    assert len(NAMES) == 4 and len(set(NAMES)) == 4
    assert all((re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', name) for name in NAMES))
    assert all((x.startswith('/') for x in [BASE, MODEL, TOKENIZER]))
    assert 1024 <= PORT <= 65535 and PORT != 19400
    OUT = pathlib.Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT = pathlib.Path(source_root).resolve()
    (OUT / 'lab_config.json').write_text(json.dumps(config, indent=2))

def remote(command, timeout=30, check=True):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', HOST, command], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(command + '\n' + p.stdout + p.stderr)
    return (p.returncode, p.stdout + p.stderr)

def node(i, command, timeout=30, check=True, as_root=False):
    options = [] if as_root else ['--user', USER, '--env', 'CUDA_CACHE_PATH=/tmp/edgevisor-cache']
    return remote(shlex.join(['docker', 'exec', *options, NAMES[i], '/bin/bash', '-lc', command]), timeout, check)

def validate():
    _, s = remote('cat ' + shlex.quote(BASE + '/source_manifest.json'))
    deployed = json.loads(s)
    allowed = {'.cpp', '.hpp', '.h', '.cu', '.comp', '.inc', '.inl', '.c'}
    actual = {str(p.relative_to(SOURCE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE_ROOT / 'EdgeVisor/src').rglob('*') if p.is_file() and p.suffix in allowed}
    for name in ['Makefile', 'EdgeVisor/Makefile']:
        actual[name] = hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
    assert actual == deployed, 'Container source differs from the local checkout: deploy and rebuild before testing.'
    _, raw = remote('cd ' + shlex.quote(BASE + '/source') + ' && sha256sum ' + shlex.join(sorted(actual)))
    remote_hash = {line.split(None, 1)[1].strip(): line.split(None, 1)[0] for line in raw.splitlines()}
    assert remote_hash == actual, 'Remote source does not match its manifest.'
    _, binary = remote('sha256sum ' + shlex.quote(BASE + '/source/EdgeVisor/dllama'))
    (OUT / 'source_manifest.json').write_text(json.dumps(actual, indent=2))
    (OUT / 'binary.sha256').write_text(binary)
    gpu_uuids = []
    for i, name in enumerate(NAMES):
        _, s = remote(shlex.join(['docker', 'inspect', name]))
        j = json.loads(s)[0]
        assert j['Config']['Labels'].get('edgevisor.lab') == LABEL
        assert not j['HostConfig']['Privileged'] and j['HostConfig']['NetworkMode'] != 'host'
        assert {'NET_ADMIN'} == {x.removeprefix('CAP_') for x in j['HostConfig']['CapAdd']}
        assert 'ALL' in j['HostConfig']['CapDrop']
        assert j['HostConfig']['ReadonlyRootfs'] and (not j['HostConfig']['PortBindings'])
        assert 'no-new-privileges' in j['HostConfig']['SecurityOpt']
        mounts = {m['Destination']: m for m in j['Mounts']}
        assert set(mounts) == {'/source', '/models', '/labtools', '/out'}
        assert all((not mounts[x]['RW'] for x in ['/source', '/models', '/labtools']))
        assert mounts['/source']['Source'] == BASE + '/source'
        assert mounts['/labtools']['Source'] == BASE + '/tools'
        assert mounts['/out']['Source'] == BASE + '/results/node' + str(i) and mounts['/out']['RW']
        networks = j['NetworkSettings']['Networks']
        assert len(networks) == 1
        _, network = remote(shlex.join(['docker', 'network', 'inspect', next(iter(networks))]))
        network = json.loads(network)[0]
        assert network['Internal'] and network['Labels'].get('edgevisor.lab') == LABEL
        (OUT / ('node%d_container.json' % i)).write_text(json.dumps(j, indent=2))
        IPS.append(next(iter(networks.values()))['IPAddress'])
        _, g = node(i, 'nvidia-smi --query-gpu=uuid --format=csv,noheader')
        assert len(g.strip().splitlines()) == 1
        gpu_uuids.append(g.strip())
        (OUT / ('node%d_gpu.txt' % i)).write_text(g)
    assert len(set(gpu_uuids)) == 4, 'Each container must have a distinct physical GPU.'
    (OUT / 'layout.json').write_text(json.dumps({'host': HOST, 'label': LABEL, 'names': NAMES, 'ips': IPS}, indent=2))

def reset(i):
    node(i, 'tc qdisc del dev eth0 root 2>/dev/null || true', as_root=True)

def shape(i, kind):
    reset(i)
    if kind == 'tbf5':
        node(i, 'tc qdisc replace dev eth0 root tbf rate 5mbit burst 32kb latency 400ms', as_root=True)
    elif kind == 'netem20':
        node(i, 'tc qdisc replace dev eth0 root netem delay 50ms rate 20mbit limit 2000', as_root=True)
    elif kind is not None:
        raise ValueError(kind)

def stop_worker(i, name):
    pid = '/out/' + name + '.pid'
    cmd = 'if test -s ' + shlex.quote(pid) + '; then p=$(cat ' + shlex.quote(pid) + '); exe=$(readlink /proc/$p/exe 2>/dev/null || true); if test "$exe" = /source/EdgeVisor/dllama; then kill -TERM "$p"; fi; fi'
    node(i, cmd)

def start_worker(i, name):
    _, s = node(i, "ss -lnt | awk '$4 ~ /:" + str(PORT) + "$/ {print}'")
    assert not s.strip(), s
    cmd = 'echo $$ > /out/' + name + '.pid; exec stdbuf -oL -eL ./dllama worker --port ' + str(PORT) + ' --model ' + shlex.quote(MODEL) + ' --nthreads 1 --backend cuda > /out/' + name + '.log 2>&1'
    remote(shlex.join(['docker', 'exec', '-d', '--user', USER, '--env', 'CUDA_CACHE_PATH=/tmp/edgevisor-cache', NAMES[i], '/bin/bash', '-lc', cmd]))
    try:
        for _ in range(20):
            _, s = node(i, "ss -lnt | awk '$4 ~ /:" + str(PORT) + "$/ {print}'")
            if s.strip():
                return
            time.sleep(0.2)
        raise TimeoutError('worker listen')
    except Exception:
        stop_worker(i, name)
        raise

def tokens(txt):
    return re.findall('\\[token-e2e\\] pos=(\\d+) token=(\\d+)', txt)

def save(name, result, txt):
    (OUT / (name + '.log')).write_text(txt)
    (OUT / (name + '.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

def infer(name, distributed=True, kind=None, auto=False, fault=None, steps=64, prompt='List 20 detailed facts about Paris. Number each fact.'):
    for i in range(4):
        reset(i)
    if kind:
        shape(3, kind)
    workers = [1, 2, 3] if distributed else []
    started = []
    events = []
    proc = None
    start = time.monotonic()
    node(0, ': > /out/' + name + '.log')
    try:
        for i in workers:
            start_worker(i, name)
            started.append(i)
        args = ['./dllama', 'inference', '--model', MODEL, '--tokenizer', TOKENIZER, '--prompt', prompt, '--steps', str(steps), '--nthreads', '1', '--backend', 'cuda', '--max-seq-len', '512', '--temperature', '0', '--seed', '42']
        if distributed:
            args += ['--ratios', '1@7*1@7*1@7*1@7', '--workers', *[ip + ':' + str(PORT) for ip in IPS[1:]]]
        if auto:
            args += ['--auto']
        command = 'echo $$ > /out/' + name + '.root.pid; exec env DLLAMA_TOKEN_TIMING_PRINT=1 DLLAMA_PLAN_CTRL_SOCKET=/tmp/' + name + '.sock timeout -k 5 150 stdbuf -oL -eL ' + shlex.join(args) + ' > /out/' + name + '.log 2>&1'
        remote_cmd = shlex.join(['docker', 'exec', '--user', USER, '--env', 'CUDA_CACHE_PATH=/tmp/edgevisor-cache', NAMES[0], '/bin/bash', '-lc', command])
        proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', HOST, remote_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while proc.poll() is None:
            if fault is not None and (not events):
                _, txt = node(0, 'cat /out/' + name + '.log 2>/dev/null || true')
                ts = tokens(txt)
                if ts and int(ts[-1][0]) >= 24:
                    stop_worker(fault, name)
                    events.append({'stop_worker': fault, 'observed_position': int(ts[-1][0])})
            if time.monotonic() - start > 165:
                raise TimeoutError(name)
            time.sleep(0.3 if fault is not None else 1)
        stdout, stderr = proc.communicate(timeout=5)
        _, txt = node(0, 'cat /out/' + name + '.log 2>/dev/null || true')
        ts = tokens(txt)
        _, q = node(3, 'tc -s qdisc show dev eth0')
        (OUT / (name + '.qdisc.txt')).write_text(q)
        wall = re.findall('\\[token-e2e\\].*wall=([0-9.]+)ms', txt)
        tps = re.findall('tokens/s:\\s*([0-9.]+)', txt)
        result = {'case': name, 'exit_code': proc.returncode, 'total_wall_s': round(time.monotonic() - start, 3), 'generated_tokens': len(ts), 'token_wall_mean_ms': sum(map(float, wall)) / len(wall) if wall else None, 'reported_tokens_per_s': float(tps[-1]) if tps else None, 'tail_shape': kind, 'auto': auto, 'events': events, 'session_restarts': txt.count('[failover] session-restart workers='), 'stderr': stderr}
        (OUT / (name + '.command.txt')).write_text(command)
        save(name, result, txt)
        assert result['exit_code'] == 0 and len(ts) >= 1, txt[-2000:]
        if fault is not None:
            assert events and result['session_restarts'] == 1, txt[-2000:]
        else:
            assert result['session_restarts'] == 0, txt[-2000:]
        return (ts, result)
    finally:
        if proc is not None and proc.poll() is None:
            root_pid = '/out/' + name + '.root.pid'
            node(0, 'if test -s ' + shlex.quote(root_pid) + '; then p=$(cat ' + shlex.quote(root_pid) + '); if test "$(readlink /proc/$p/exe 2>/dev/null)" = /usr/bin/timeout; then kill -TERM "$p"; fi; fi')
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
        for i in started:
            stop_worker(i, name)
            _, txt = node(i, 'cat /out/' + name + '.log')
            (OUT / (name + '_node%d.log' % i)).write_text(txt)
        for i in range(4):
            reset(i)

def network_probes():
    results = []
    try:
        for kind in [None, 'tbf5', 'netem20', None]:
            name = 'net_' + (kind or ('reset' if results else 'baseline'))
            shape(0, kind)
            node(1, ': > /out/' + name + '.log')
            cmd = 'echo $$ > /out/net_probe.pid; exec timeout -k 2 35 /labtools/net_probe serve 0.0.0.0 19400 4194304 > /out/' + name + '.log 2>&1'
            remote(shlex.join(['docker', 'exec', '-d', '--user', USER, '--env', 'CUDA_CACHE_PATH=/tmp/edgevisor-cache', NAMES[1], '/bin/bash', '-lc', cmd]))
            for _ in range(20):
                _, s = node(1, 'cat /out/' + name + '.log 2>/dev/null || true')
                if 'READY' in s:
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError('net probe listen')
            _, txt = node(0, 'timeout -k 2 35 /labtools/net_probe send ' + IPS[1] + ' 19400 4194304', timeout=40)
            data = json.loads(txt)
            assert data['bytes'] == 4194304
            _, q = node(0, 'tc -s qdisc show dev eth0')
            (OUT / (name + '.qdisc.txt')).write_text(q)
            data.update({'case': name, 'shape': kind})
            save(name, data, txt)
            results.append(data)
        assert results[0]['mbps'] > 100 and 3.5 < results[1]['mbps'] < 6.5 and (12 < results[2]['mbps'] < 24) and (results[3]['mbps'] > 100), results
        (OUT / 'network_comparison.json').write_text(json.dumps(results, indent=2))
    finally:
        node(1, 'if test -s /out/net_probe.pid; then p=$(cat /out/net_probe.pid); if test "$(readlink /proc/$p/exe 2>/dev/null)" = /usr/bin/timeout; then kill -TERM "$p" 2>/dev/null || true; fi; fi')
        reset(0)

def inference_suite():
    comparison = {}
    infer('single_paris', distributed=False, prompt='The capital of France is')
    single, _ = infer('single_cuda', distributed=False)
    ref, r = infer('four_unshaped')
    comparison['single_vs_four_identical'] = single == ref
    for name, kind, auto in [('four_tail_5mbit', 'tbf5', False), ('four_tail_20mbit_delay50', 'netem20', False), ('four_reset', None, False), ('four_auto', None, True)]:
        ts, m = infer(name, kind=kind, auto=auto)
        comparison[name] = {'identical_to_four_baseline': ts == ref, 'metrics': m}
        if not auto:
            assert ts == ref, (name, 'tokens changed')
    infer('four_middle_drop', fault=1, steps=96)
    (OUT / 'inference_comparison.json').write_text(json.dumps(comparison, indent=2))

def cleanup_check():
    for i in range(4):
        _, s = node(i, "ss -lnt | awk '$4 ~ /:(" + str(PORT) + "|19400)$/ {print}'")
        assert not s.strip(), s
        _, s = node(i, 'pgrep -af "^./dllama |^/labtools/net_probe " || true')
        assert not s.strip(), s
        _, s = node(i, 'tc qdisc show dev eth0')
        assert 'tbf' not in s and 'netem' not in s, s
    print('All owned workers/probes stopped; traffic control cleared.', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model-backed smoke checks in four owned, single-GPU Docker containers.')
    parser.add_argument('action', choices=['network', 'inference', 'fault', 'cleanup', 'all'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-root', default=str(SOURCE_ROOT))
    args = parser.parse_args()
    configure(args.config, args.output, args.source_root)
    validate()
    if args.action in ['network', 'all']:
        network_probes()
    if args.action in ['inference', 'all']:
        inference_suite()
    if args.action == 'fault':
        infer('four_middle_drop', fault=1, steps=96)
    cleanup_check()
