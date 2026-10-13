#!/usr/bin/env python3
"""Correctness gate for the four-node, 28-layer CUDA lab; not a performance study."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.recording import RunRecorder, measurement, verify_artifacts
from scripts.experiments.timeline import validate_trace
from scripts.lab import docker_gpu_check as lab


def check_trace(path, run_id, observed_tokens, expected_migrations=None):
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    summary = validate_trace(events)
    if summary['run_id'] != run_id or any(item.get('role') != 'root' for item in events):
        raise ValueError('Trace is from another trial or process role')
    for phase in ('loading', 'generation', 'prefill', 'decode', 'token', 'forward'):
        if phase not in summary['phase_union_ms']:
            raise ValueError('Missing required engine phase: ' + phase)
    if any(item['type'] == 'end' and item['attributes'].get('status') != 'ok' for item in events):
        raise ValueError('Engine spans did not complete successfully')
    emitted = [(str(item['attributes']['position']), str(item['attributes']['token']))
               for item in summary['points'] if item['phase'] == 'token_emitted']
    if not emitted or emitted != observed_tokens:
        raise ValueError('Trace token ledger differs from observed output')
    commits = [item['attributes'] for item in summary['points'] if item['phase'] == 'ownership_committed']
    results = [item['attributes'] for item in summary['points'] if item['phase'] == 'migration_result']
    if expected_migrations is not None and len(commits) != expected_migrations:
        raise ValueError('Requested migrations did not actually commit')
    if len(commits) != len(results) or any(not item['applied'] for item in results):
        raise ValueError('Migration failed or lacks a matching commit')
    for commit, result in zip(commits, results):
        if any(commit[key] != result[key] for key in ('from_node', 'to_node', 'layers')):
            raise ValueError('Migration result is for a different committed route')
        if result['kv_frontier_count'] != result['position'] + 1:
            raise ValueError('KV frontier must count all completed rows including position zero')
    summary['token_count'] = len(emitted)
    summary['commits'] = commits
    summary['migration_results'] = results
    return summary


def plan(seq, source, target, position, layer):
    return {'op': 'set_pp_migration', 'cmd': {'seq': seq, 'mode': 'exact',
        'fromNodeIndex': source, 'toNodeIndex': target, 'triggerPos': position,
        'triggerLayer': layer, 'layerCount': 3}}


def gpu_snapshot():
    _, data = lab.remote('nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--build-evidence', required=True, help='Passed deploy_build.py output directory')
    parser.add_argument('--output', required=True, help='New exclusive run directory')
    parser.add_argument('--network', action='store_true', help='Verify real TC probe and shaped static inference')
    args = parser.parse_args()
    with RunRecorder(args.output, {'purpose': 'correctness_gate', 'model_layout': '28-layer-4x7',
            'seed': 42, 'temperature': 0, 'max_seq_len': 512, 'steps': 128,
            'independent_episodes_per_case': 1,
            'power_energy': measurement(unit='J'),
            'gpu_clock_control': measurement(status='unsupported', unit='MHz')}) as run:
        lab.configure(args.config, run.directory / 'evidence', ROOT)
        lab.validate()
        build = verify_artifacts(args.build_evidence)
        if build['status'] != 'passed':
            raise ValueError('Cannot test a failed or incomplete build')
        source = json.loads((lab.OUT / 'source_manifest.json').read_text())
        source_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        if source_hash != build['metadata']['source']['source_manifest_sha256']:
            raise ValueError('Deployed code differs from accepted build')
        binary = (lab.OUT / 'binary.sha256').read_text().split()[0]
        built_binary = (Path(args.build_evidence) / 'deployment/binary.sha256').read_text().split()[0]
        if binary != built_binary:
            raise ValueError('Deployed binary differs from accepted build')
        run.manifest['metadata'].update(build=build['metadata'], binary_sha256=binary)
        _, model_hashes = lab.node(0, shlex.join(['sha256sum', lab.MODEL, lab.TOKENIZER]), timeout=90)
        (run.directory / 'model_tokenizer.sha256').write_text(model_hashes)
        (run.directory / 'gpu_before.csv').write_text(gpu_snapshot())
        common = ['--enable-plan-barrier', '1', '--enable-pp-migration', '1',
                  '--runtime-redundant-boundary-layers', '6', '--jit-mode', 'greedy']
        summaries = {}
        try:
            # The uninstrumented baseline catches trace-induced changes in behavior.
            base, _ = lab.infer(run.run_id + '_untraced', steps=128)
            def case(label, reference=base, **options):
                name = run.run_id + '_' + label
                run.event('case_start', {'case': label, 'options': options})
                tokens, result = lab.infer(name, steps=128, trace_run_id=run.run_id, **options)
                if tokens != reference:
                    raise ValueError(label + ': output differs from static reference')
                summary = check_trace(lab.OUT / (name + '.trace.jsonl'), run.run_id,
                                      tokens, options.get('expect_migrations'))
                summaries[label] = {'result': result, 'trace': summary, 'tokens_identical': True}
                (run.directory / 'comparison.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
                run.event('case_passed', {'case': label, 'tokens': len(tokens),
                                         'commits': len(summary['commits'])})
                return tokens
            case('traced_static', expect_migrations=0)
            for label, routes, layer in [('root_roundtrip', [(0, 1), (1, 0)], 4),
                                         ('tail_roundtrip', [(3, 2), (2, 3)], 21)]:
                schedule = [{'after_position': after,
                             'request': plan(i + 1, source, target, trigger, layer)}
                            for i, ((source, target), after, trigger) in
                            enumerate(zip(routes, (24, 80), (64, 112)))]
                case(label, extra_args=common + ['--last-stage-sampling', '1'],
                     plan_schedule=schedule, expect_migrations=2)
            case('automatic', auto=True)
            if not summaries['automatic']['trace']['commits']:
                raise ValueError('Automatic policy was not exercised')
            if args.network:
                run.event('network_probe_start')
                lab.network_probes()
                run.event('network_probe_passed')
                case('shaped_static', kind='netem20', expect_migrations=0)
        finally:
            # Sealing remains failed if cleanup fails. Preserve unsuccessful logs.
            lab.cleanup_check()
            (run.directory / 'gpu_after.csv').write_text(gpu_snapshot())
        run.seal('passed')
        print('Passed: exact output, actual commits, raw intervals, source/binary identity and cleanup.')


if __name__ == '__main__':
    main()
