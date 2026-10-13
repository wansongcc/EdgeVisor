import copy
import json
from pathlib import Path
import tempfile
import unittest

from agent_bench.autonomous import parse_action, run_autonomous_episode
from agent_bench.backends import GenerationResult
from agent_bench.tools import ToolError, calculator
from scripts.experiments.recording import verify_artifacts


class FakeBackend:
    name = 'fixture_model'
    def __init__(self, outputs):
        self.outputs = iter(outputs); self.inputs = []; self.closed = False
    def generate(self, messages, **kwargs):
        self.inputs.append(copy.deepcopy(messages))
        return GenerationResult(backend=self.name, content=next(self.outputs), metrics={})
    def close(self): self.closed = True


class AutonomousTests(unittest.TestCase):
    def test_strict_actions_do_not_repair_partial_or_ambiguous_output(self):
        bad = ['```json\n{"tool":"final_answer","arguments":{"answer":"42"}}\n```',
               '{"tool":"final_answer","arguments":{"answer":"42',
               '{"tool":"calculator","tool":"final_answer","arguments":{"answer":"42"}}',
               '{"tool":"calculator","arguments":{"expression":"1+1"}} {}',
               '{"tool":"calculator","arguments":{"expression":NaN}}']
        for text in bad:
            with self.assertRaises(ValueError): parse_action(text, {'calculator'})
        with self.assertRaises(TypeError):
            parse_action('{"tool":"calculator","arguments":{"wrong_key":"1+1"}}', {'calculator'})
        action = parse_action('{"tool":"text_stats","arguments":{"text":"Edge Visor  agent backend"}}', {'text_stats'})
        self.assertEqual(action['arguments']['text'], 'Edge Visor  agent backend')

    def test_wrong_model_arguments_are_executed_and_scored_without_forcing(self):
        backend = FakeBackend(['{"tool":"calculator","arguments":{"expression":"2+2"}}',
                               '{"tool":"final_answer","arguments":{"answer":"5"}}'])
        episode = {'id': 'wrong_argument', 'task': 'Calculate 3+2 using calculator.',
                   'allowed_tools': ['calculator'], 'required_tools': ['calculator'],
                   'required_tool_arguments': {'calculator': {'expression': '3+2'}},
                   'expected_answer': '5'}
        with tempfile.TemporaryDirectory() as directory:
            result = run_autonomous_episode(episode, backend, Path(directory))
            self.assertEqual(result['events'][1]['arguments'], {'expression': '2+2'})
            self.assertEqual(result['events'][1]['result'], '4')
            self.assertFalse(result['agent_metrics']['task_success'])
            self.assertEqual(result['agent_metrics']['forced_action_count'], 0)
            self.assertEqual(verify_artifacts(Path(directory)/'autonomous')['status'], 'failed')
            self.assertTrue(backend.closed)

    def test_unobserved_quality_is_not_implicit_success_or_zero_timing(self):
        backend = FakeBackend(['{"tool":"final_answer","arguments":{"answer":"done"}}'])
        with tempfile.TemporaryDirectory() as directory:
            result = run_autonomous_episode({'id': 'no_checker', 'task': 'Say done.'}, backend, Path(directory))
            self.assertIsNone(result['agent_metrics']['task_success'])
            self.assertIsNone(result['agent_metrics']['total_tool_time_ms'])
            self.assertEqual(verify_artifacts(Path(directory)/'autonomous')['status'], 'rejected')

    def test_hidden_answer_never_enters_prompt_or_repairs_final(self):
        backend = FakeBackend(['{"tool":"final_answer","arguments":{"answer":"wrong"}}'])
        with tempfile.TemporaryDirectory() as directory:
            result = run_autonomous_episode({'id': 'hidden', 'task': 'Report an answer.',
                'expected_answer': 'hidden_reference_777'}, backend, Path(directory))
            self.assertNotIn('hidden_reference_777', json.dumps(backend.inputs))
            self.assertEqual(result['final_answer'], 'wrong')
            self.assertFalse(result['agent_metrics']['task_success'])

    def test_calculator_is_real_and_bounded(self):
        self.assertEqual(calculator('(17+25)*3'), '126')
        for expression in ('10**1000000', 'True+1', '1/0', '__import__("os")', '9'*257):
            with self.assertRaises(ToolError): calculator(expression)


if __name__ == '__main__': unittest.main()
