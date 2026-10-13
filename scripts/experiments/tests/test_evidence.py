import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.experiments.recording import RunRecorder, measurement, verify_artifacts
from scripts.experiments.timeline import TraceError, validate_trace


def event(sequence, kind, phase, time_ms, span=0, parent=0, **attributes):
    return {'schema_version': 1, 'run_id': 'trial', 'clock_id': 'root-clock',
            'clock': 'steady', 'role': 'root', 'seq': sequence,
            'ts_ns': int(time_ms * 1e6), 'type': kind, 'phase': phase,
            'span_id': span, 'parent_id': parent, 'attributes': attributes}


class TimelineTests(unittest.TestCase):
    def test_nested_intervals_are_not_added(self):
        trace = [event(0, 'begin', 'episode', 0, 1),
                 event(1, 'begin', 'tool', 10, 2, 1),
                 event(2, 'begin', 'preparation', 20, 3, 2),
                 event(3, 'end', 'preparation', 40, 3, 2, duration_ns=20_000_000),
                 event(4, 'end', 'tool', 60, 2, 1, duration_ns=50_000_000),
                 event(5, 'end', 'episode', 100, 1, duration_ns=100_000_000)]
        result = validate_trace(trace)
        self.assertEqual(result['phase_union_ms'], {'episode': 100, 'tool': 50, 'preparation': 20})

    def test_component_cannot_exceed_wall(self):
        trace = [event(0, 'begin', 'episode', 0, 1),
                 event(1, 'begin', 'history', 1, 2, 1),
                 event(2, 'end', 'episode', 10, 1, duration_ns=10_000_000),
                 event(3, 'end', 'history', 28, 2, 1, duration_ns=27_000_000)]
        with self.assertRaises(TraceError):
            validate_trace(trace)

    def test_plus_18_second_duration_is_rejected(self):
        trace = [event(0, 'begin', 'episode', 0, 1),
                 event(1, 'end', 'episode', 35000, 1, duration_ns=53_000_000_000)]
        with self.assertRaises(TraceError):
            validate_trace(trace)

    def test_worker_clock_cannot_be_subtracted(self):
        trace = [event(0, 'begin', 'history', 10, 1),
                 event(1, 'end', 'history', 30, 1, duration_ns=20_000_000)]
        trace[1]['clock_id'] = 'worker-clock'
        with self.assertRaises(TraceError):
            validate_trace(trace)

    def test_incomplete_and_duplicate_events_are_rejected(self):
        with self.assertRaises(TraceError):
            validate_trace([event(0, 'begin', 'episode', 0, 1)])
        with self.assertRaises(TraceError):
            validate_trace([event(0, 'point', 'token', 0), event(0, 'point', 'token', 1)])

    def test_both_fault_latency_definitions_and_missing_confirmation(self):
        trace = [event(0, 'point', 'fault_injected', 10, transition_id='tx1'),
                 event(1, 'point', 'fault_confirmed', 15, transition_id='tx1'),
                 event(2, 'point', 'first_token_after_transition', 40, transition_id='tx1')]
        result = validate_trace(trace)['fault_to_first_token']['tx1']
        self.assertEqual(result['injection_to_token_ms']['value'], 30)
        self.assertEqual(result['confirmation_to_token_ms']['value'], 25)
        trace.pop(1)
        trace[1]['seq'] = 1
        missing = validate_trace(trace)['fault_to_first_token']['tx1']['confirmation_to_token_ms']
        self.assertEqual(missing, {'value': None, 'status': 'not_observed'})


class RecordingTests(unittest.TestCase):
    def test_failed_trial_is_retained_and_tampering_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run'
            run = RunRecorder(path, {'policy': 'fixed', 'seed': 42})
            (path / 'root.log').write_text('timeout\n')
            run.seal('failed', {'reason': 'deadline'})
            self.assertEqual(verify_artifacts(path)['status'], 'failed')
            with self.assertRaises(RuntimeError):
                run.event('fake-success')
            with self.assertRaises(FileExistsError):
                RunRecorder(path, {})
            (path / 'root.log').write_text('passed\n')
            with self.assertRaises(ValueError):
                verify_artifacts(path)

    def test_nested_manifest_is_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run'
            run = RunRecorder(path, {})
            (path / 'nested').mkdir()
            (path / 'nested/manifest.json').write_text('{}')
            run.seal('passed')
            self.assertIn('nested/manifest.json', verify_artifacts(path)['artifacts'])

    def test_interrupted_work_retains_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run'
            with self.assertRaises(RuntimeError):
                with RunRecorder(path, {}) as run:
                    (path / 'root.log').write_text('partial output')
                    raise RuntimeError('model timeout')
            self.assertEqual(verify_artifacts(path)['status'], 'failed')

    def test_symlink_cannot_seal_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run'
            run = RunRecorder(path, {})
            (path / 'alias').symlink_to(Path(directory))
            with self.assertRaises(ValueError):
                run.seal('passed')
            self.assertTrue(run.closed)
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['status'], 'rejected')
            with self.assertRaises(ValueError):
                verify_artifacts(path)

    def test_missing_sensor_is_not_zero(self):
        self.assertEqual(measurement(unit='W'), {'value': None, 'status': 'not_observed', 'unit': 'W'})
        self.assertEqual(measurement(0, 'measured', 'W')['value'], 0)
        with self.assertRaises(ValueError):
            measurement(0, 'unsupported', 'W')
        with self.assertRaises(ValueError):
            measurement(None, 'measured', 'W')


if __name__ == '__main__':
    unittest.main()
