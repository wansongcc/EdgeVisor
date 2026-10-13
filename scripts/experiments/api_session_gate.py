#!/usr/bin/env python3
"""Verify serial API token lineage, sampling and capacity on actual Q40 models."""
import argparse
import http.client
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.recording import RunRecorder
from scripts.experiments.build_identity import validate_build_identity
from scripts.experiments.timeline import validate_trace
from scripts.lab import docker_gpu_check as lab


class Server:
    def __init__(self, run, label, distributed, tail):
        self.run, self.label = run, label
        self.name = run.run_id + '_' + label
        self.distributed, self.tail = distributed, tail
        self.started, self.tunnel = [], None
        self.requests = []
        self.local_port = None

    def __enter__(self):
        try:
            for index in range(4):
                lab.reset(index)
            if self.distributed:
                for index in (1, 2, 3):
                    lab.start_worker(index, self.name); self.started.append(index)
            args = ['./dllama-api', '--model', lab.MODEL, '--tokenizer', lab.TOKENIZER,
                    '--port', '19401', '--backend', 'cuda', '--nthreads', '1',
                    '--max-seq-len', '512', '--temperature', '0', '--seed', '42']
            if self.distributed:
                args += ['--ratios', '1@7*1@7*1@7*1@7', '--workers',
                         *[ip + ':' + str(lab.PORT) for ip in lab.IPS[1:]]]
            if self.tail:
                args += ['--last-stage-sampling', '1']
            env = ['DLLAMA_API_BIND=0.0.0.0', 'DLLAMA_NBATCHES=1',
                   'EDGEVISOR_RUN_ID=' + self.run.run_id,
                   'EDGEVISOR_TRACE_PATH=/out/' + self.name + '.trace.jsonl']
            command = 'echo $$ > /out/' + self.name + '.api.pid; exec env ' + shlex.join(env) + \
                ' timeout -k 5 360 stdbuf -oL -eL ' + shlex.join(args) + ' > /out/' + self.name + '.log 2>&1'
            _, inspect = lab.remote(shlex.join(['docker', 'inspect', lab.NAMES[0]]))
            if json.loads(inspect)[0]['Config']['Labels'].get('edgevisor.lab') != lab.LABEL:
                raise ValueError('API container ownership changed')
            _, port = lab.node(0, "ss -lnt | awk '$4 ~ /:19401$/ {print}'")
            if port.strip():
                raise ValueError('API port already in use')
            lab.remote(shlex.join(['docker', 'exec', '-d', '--user', lab.USER,
                '--env', 'CUDA_CACHE_PATH=/tmp/edgevisor-cache', lab.NAMES[0], '/bin/bash', '-lc', command]))
            (lab.OUT / (self.name + '.command.txt')).write_text(command)
            with socket.socket() as local:
                local.bind(('127.0.0.1', 0)); self.local_port = local.getsockname()[1]
            self.tunnel = subprocess.Popen(['ssh', '-N', '-o', 'BatchMode=yes',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'ExitOnForwardFailure=yes',
                '-o', 'ConnectTimeout=10', '-L', '127.0.0.1:' + str(self.local_port) + ':' + lab.IPS[0] + ':19401', lab.HOST],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if self.tunnel.poll() is not None:
                    raise RuntimeError('SSH tunnel failed: ' + self.tunnel.communicate()[1])
                try:
                    status, _ = self.http('GET', '/v1/models', None, timeout=2, record=False)
                    if status == 200:
                        return self
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(0.2)
            raise TimeoutError('API model startup')
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def http(self, method, path, payload, timeout=120, record=True):
        conn = http.client.HTTPConnection('127.0.0.1', self.local_port, timeout=timeout)
        body = json.dumps(payload) if isinstance(payload, dict) else payload
        start = time.monotonic_ns()
        try:
            conn.request(method, path, body=body,
                         headers={'Content-Type': 'application/json'} if body is not None else {})
            response = conn.getresponse(); raw = response.read().decode()
            result = json.loads(raw) if response.getheader('Content-Type', '').startswith('application/json') else raw
            if record:
                self.requests.append({'request': payload, 'status': response.status, 'response': result,
                    'client_start_ns': start, 'client_end_ns': time.monotonic_ns()})
                (lab.OUT / (self.name + '.http.json')).write_text(json.dumps(self.requests, indent=2))
            return response.status, result
        finally:
            conn.close()

    def completion(self, messages, session='case', **options):
        payload = {'messages': messages, 'max_tokens': 48, 'temperature': 0,
                   'top_p': 0.9, 'seed': 42, 'stream': False, **options}
        if session is not None:
            payload['edgevisor_session_id'] = session
        status, result = self.http('POST', '/v1/chat/completions', payload)
        if status != 200 or not isinstance(result, dict) or result.get('edgevisor', {}).get('status') != 'completed':
            raise ValueError(('Completion failed', status, result))
        usage, metadata = result['usage'], result['edgevisor']
        if metadata['kv_frontier_count'] != usage['prompt_tokens'] - 1 + usage['completion_tokens']:
            raise ValueError('API frontier includes an unconsumed sampled token')
        return result

    def __exit__(self, kind, error, traceback):
        pid = '/out/' + self.name + '.api.pid'
        lab.node(0, 'if test -s ' + shlex.quote(pid) + '; then p=$(cat ' + shlex.quote(pid) + '); '
                 'if test "$(readlink /proc/$p/exe 2>/dev/null)" = /usr/bin/timeout; then kill -TERM "$p"; fi; fi')
        if self.tunnel is not None:
            self.tunnel.terminate()
            try: self.tunnel.communicate(timeout=5)
            except subprocess.TimeoutExpired: self.tunnel.kill(); self.tunnel.communicate()
        for index in self.started:
            lab.stop_worker(index, self.name)
            _, log = lab.node(index, 'cat /out/' + self.name + '.log 2>/dev/null || true')
            (lab.OUT / (self.name + '_node' + str(index) + '.log')).write_text(log)
        for suffix in ('.log', '.trace.jsonl'):
            _, data = lab.node(0, 'cat /out/' + self.name + suffix + ' 2>/dev/null || true')
            (lab.OUT / (self.name + suffix)).write_text(data)
        for index in range(4):
            lab.reset(index)
        if error is None:
            events = [json.loads(line) for line in (lab.OUT / (self.name + '.trace.jsonl')).read_text().splitlines() if line]
            summary = validate_trace(events)
            if any(event['type'] == 'end' and event['attributes'].get('status') != 'ok' for event in events):
                raise ValueError('API request trace contains an engine error')
            if summary['run_id'] != self.run.run_id:
                raise ValueError('API trace trial mismatch')
            (lab.OUT / (self.name + '.timeline.json')).write_text(json.dumps(summary, indent=2))
        return False


def answer(response):
    return response['choices'][0]['message']['content']


def suite(server):
    messages = [{'role': 'system', 'content': 'Answer briefly and accurately.'},
                {'role': 'user', 'content': 'What is 17 plus 25? Reply with a short sentence.'}]
    first = server.completion(messages, 'warm')
    history = messages + [{'role': 'assistant', 'content': answer(first)},
                          {'role': 'user', 'content': 'Multiply that result by 3. Reply briefly.'}]
    warm = server.completion(history, 'warm')
    a, b = first['edgevisor'], warm['edgevisor']
    if b['lineage_id'] != a['lineage_id'] or b['cached_input_tokens'] != a['kv_frontier_count']:
        raise ValueError('Continuous request did not reuse exactly the consumed token history')
    cold = server.completion(history, 'cold')
    if answer(cold) != answer(warm) or cold['edgevisor']['cached_input_tokens'] != 0:
        raise ValueError('Warm API continuation differs from canonical cold prefill')
    changed = server.completion(messages, 'cold')
    if changed['edgevisor']['reset_reason'] != 'token_prefix_changed':
        raise ValueError('Changed token prefix retained stale KV')
    for payload in ('{invalid-json', {'messages': []}, {'messages': messages, 'temperature': -1},
                    {'messages': messages, 'top_p': 0}, {'messages': messages, 'seed': -1},
                    {'messages': messages, 'max_tokens': 0}, {'messages': messages, 'max_tokens': 1000},
                    {'messages': messages, 'max_tokens': 1.5}, {'messages': messages, 'max_tokens': 4294967297},
                    {'messages': [{'role': 'user', 'content': 'x ' * 1200}]}):
        status, _ = server.http('POST', '/v1/chat/completions', payload)
        if status != 400: raise ValueError(('Invalid request was not rejected', status))
    # Rejected requests must not mutate the valid ledger or advance generation.
    after = server.completion(history, 'cold')
    if after['edgevisor']['generation_id'] != changed['edgevisor']['generation_id'] + 1 or \
       after['edgevisor']['cached_input_tokens'] != changed['edgevisor']['kv_frontier_count']:
        raise ValueError('Rejected request changed live token history')
    for _ in range(2):
        stateless = server.completion(messages, None)
        if stateless['edgevisor']['cached_input_tokens'] or stateless['edgevisor']['reset_reason'] != 'session_not_provided':
            raise ValueError('Anonymous clients shared a hidden token lineage')
    stochastic = [{'role': 'user', 'content': 'Invent a short name for a friendly robot and explain it.'}]
    sample1 = server.completion(stochastic, 'sampling1', temperature=1.1, top_p=0.8, seed=123)
    sample2 = server.completion(stochastic, 'sampling2', temperature=1.1, top_p=0.8, seed=123)
    if answer(sample1) != answer(sample2):
        raise ValueError('Per-request seed was not reset consistently')
    if sample1['edgevisor']['temperature'] != sample2['edgevisor']['temperature'] or \
            abs(sample1['edgevisor']['temperature'] - 1.1) > 1e-6:
        raise ValueError('Request sampling parameters were not applied')
    streaming = {'messages': messages, 'max_tokens': 48, 'temperature': 0, 'seed': 42,
                 'stream': True, 'edgevisor_session_id': 'stream'}
    status, raw = server.http('POST', '/v1/chat/completions', streaming)
    pieces, metadata = [], None
    for line in raw.splitlines():
        if not line.startswith('data: ') or line == 'data: [DONE]': continue
        chunk = json.loads(line[6:])
        for choice in chunk['choices']: pieces.append(choice.get('delta', {}).get('content', ''))
        metadata = chunk.get('edgevisor', metadata)
    if status != 200 or ''.join(pieces) != answer(first) or not metadata:
        raise ValueError('Streaming output or lineage metadata differs')
    return {'first_answer': answer(first), 'warm_answer': answer(warm),
            'sample_answer': answer(sample1), 'first': a, 'warm': b,
            'requests': len(server.requests)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--build-evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with RunRecorder(args.output, {'purpose': 'api_session_correctness', 'prefill_batch': 1,
                                  'model_layout': '28-layer-4x7', 'quality_claim': False}) as run:
        lab.configure(args.config, run.directory / 'evidence', ROOT); lab.validate()
        source = json.loads((lab.OUT/'source_manifest.json').read_text())
        _, api_hash = lab.node(0, 'sha256sum /source/EdgeVisor/dllama-api')
        cli_hash = (lab.OUT/'binary.sha256').read_text().split()[0]
        build = validate_build_identity(args.build_evidence, source, cli_hash, api_hash.split()[0])
        run.manifest['metadata'].update(build=build['metadata'], api_binary_sha256=api_hash.split()[0])
        results = {}
        try:
            for label, distributed, tail in [('single', False, False), ('four_root', True, False), ('four_tail', True, True)]:
                run.event('api_variant_start', {'variant': label})
                with Server(run, label, distributed, tail) as server:
                    results[label] = suite(server)
                (run.directory / 'comparison.json').write_text(json.dumps(results, indent=2))
                run.event('api_variant_passed', {'variant': label})
                print(label + ': lineage, capacity, greedy/stochastic sampling and streaming passed.', flush=True)
            for label in ('four_root', 'four_tail'):
                if any(results[label][key] != results['single'][key]
                       for key in ('first_answer', 'warm_answer', 'sample_answer')):
                    raise ValueError(label + ': API outputs differ from single GPU reference')
        finally:
            lab.cleanup_check()
        run.seal('passed')
        print('Serial API sessions passed on single GPU, four-node root and tail sampling.')


if __name__ == '__main__':
    main()
