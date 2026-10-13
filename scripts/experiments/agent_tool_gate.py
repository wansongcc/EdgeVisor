#!/usr/bin/env python3
"""Small live autonomous-tool correctness gate, not an aggregate quality benchmark."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent_bench.autonomous import run_autonomous_episode
from agent_bench.backends import GenerationResult
from scripts.experiments.api_session_gate import Server, answer
from scripts.experiments.recording import RunRecorder
from scripts.experiments.build_identity import validate_build_identity
from scripts.lab import docker_gpu_check as lab


class LabApiBackend:
    def __init__(self, server, session):
        self.server, self.session, self.name = server, session, server.label
    def generate(self, messages, max_tokens, **kwargs):
        response = self.server.completion(messages, self.session, max_tokens=max_tokens)
        return GenerationResult(backend=self.name, content=answer(response), rc=0,
            metrics={'api_token_lineage': response['edgevisor']})
    def close(self):
        pass # Server context owns the lab process and SSH tunnel.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--build-evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with RunRecorder(args.output, {'purpose': 'real_tool_correctness', 'execution_mode': 'autonomous',
                                  'independent_episodes_per_case': 1, 'aggregate_quality_claim': False}) as run:
        lab.configure(args.config, run.directory/'evidence', ROOT); lab.validate()
        source = json.loads((lab.OUT/'source_manifest.json').read_text())
        _, api_hash = lab.node(0, 'sha256sum /source/EdgeVisor/dllama-api')
        cli_hash = (lab.OUT/'binary.sha256').read_text().split()[0]
        build = validate_build_identity(args.build_evidence, source, cli_hash, api_hash.split()[0])
        run.manifest['metadata'].update(build=build['metadata'], api_binary_sha256=api_hash.split()[0])
        results = {}
        try:
            for label, distributed, tail in [('single', False, False), ('four_root', True, False), ('four_tail', True, True)]:
                with Server(run, label, distributed, tail) as server:
                    episode = {'id': 'live_calculator', 'allowed_tools': ['calculator'],
                        'task': 'Use calculator to compute (17+25)*3. Then give final_answer with only the integer string.',
                        'max_tokens': 96, 'max_llm_turns': 4, 'required_tools': ['calculator'],
                        'tool_observation_role': 'user', 'expected_answer': '126'}
                    trace = run_autonomous_episode(episode, LabApiBackend(server, run.run_id+'_agent'),
                                                   run.directory/'episodes'/label)
                    metrics = trace['agent_metrics']
                    if metrics['tool_call_count'] < 1:
                        raise ValueError((label, 'Real tool path was not exercised', metrics))
                    generations = [e for e in trace['events'] if e['type'] == 'llm_generation']
                    lineage = [e['metrics']['api_token_lineage'] for e in generations]
                    if len(lineage) < 2 or lineage[1]['cached_input_tokens'] != lineage[0]['kv_frontier_count'] or \
                            lineage[1]['lineage_id'] != lineage[0]['lineage_id'] or \
                            lineage[1]['runtime_context_key'] != lineage[0]['runtime_context_key']:
                        raise ValueError('The real tool call did not preserve its API KV lineage')
                    results[label] = {'metrics': metrics, 'lineage': lineage, 'final_answer': trace['final_answer']}
                    run.event('live_tool_passed', {'variant': label, 'actual_tool_calls': metrics['tool_call_count']})
                (run.directory/'comparison.json').write_text(json.dumps(results, indent=2))
                print(label + ': tool and continuous KV passed; task_success=' + str(metrics['task_success']), flush=True)
        finally:
            lab.cleanup_check()
        if any(item['metrics']['task_success'] is not True for item in results.values()):
            raise ValueError('Autonomous quality failed; all model results are retained')
        run.seal('passed')
        print('Autonomous calculator gate passed; no model outputs or arguments were overridden.')


if __name__ == '__main__': main()
