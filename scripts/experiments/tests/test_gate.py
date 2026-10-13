import json
from pathlib import Path
import tempfile
import unittest

from scripts.experiments.research_gate import check_trace
from scripts.experiments.tests.test_evidence import event


class LiveGateTests(unittest.TestCase):
    def trace(self, points):
        events = []
        for ident, phase in enumerate(('loading', 'generation', 'prefill', 'decode', 'token', 'forward'), 1):
            events.extend([event(len(events), 'begin', phase, len(events), ident),
                           event(len(events) + 1, 'end', phase, len(events) + 1, ident,
                                 duration_ns=1_000_000, status='ok')])
        for phase, attributes in points:
            events.append(event(len(events), 'point', phase, len(events), **attributes))
        for item in events:
            item['role'] = 'root'
        return events

    def check(self, events, migrations=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            path.write_text('\n'.join(json.dumps(item) for item in events))
            return check_trace(path, 'trial', [('7', '42')], migrations)

    def test_zero_migrations_is_not_roundtrip(self):
        trace = self.trace([('token_emitted', {'position': 7, 'token': 42})])
        self.assertEqual(self.check(trace, 0)['token_count'], 1)
        with self.assertRaises(ValueError):
            self.check(trace, 2)

    def test_changed_output_is_not_success(self):
        with self.assertRaises(ValueError):
            self.check(self.trace([('token_emitted', {'position': 7, 'token': 43})]))

    def test_success_log_cannot_hide_wrong_frontier_or_failed_install(self):
        attrs = {'generation': 1, 'position': 7, 'from_node': 0, 'to_node': 1, 'layers': [4]}
        points = [('token_emitted', {'position': 7, 'token': 42}), ('ownership_committed', attrs),
                  ('migration_result', {**attrs, 'applied': True, 'kv_frontier_count': 8})]
        self.assertEqual(len(self.check(self.trace(points), 1)['commits']), 1)
        points[-1][1]['kv_frontier_count'] = 7
        with self.assertRaises(ValueError):
            self.check(self.trace(points), 1)
        points[-1][1].update(kv_frontier_count=8, applied=False)
        with self.assertRaises(ValueError):
            self.check(self.trace(points), 1)

    def test_engine_error_cannot_be_a_successful_trace(self):
        trace = self.trace([('token_emitted', {'position': 7, 'token': 42})])
        trace[1]['attributes']['status'] = 'error'
        with self.assertRaises(ValueError):
            self.check(trace, 0)


if __name__ == '__main__':
    unittest.main()
