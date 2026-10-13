"""Validate raw intervals before deriving timing; never add overlapping spans."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


class TraceError(ValueError):
    pass


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise TraceError(name + ' must be a nonnegative integer')
    return value


def union_ns(intervals):
    ordered = sorted(intervals)
    total = 0
    start = end = None
    for left, right in ordered:
        if left > right:
            raise TraceError('Negative interval')
        if end is None:
            start, end = left, right
        elif left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + (0 if end is None else end - start)


def validate_trace(events):
    if not events:
        raise TraceError('Missing raw trace')
    domain = (events[0]['run_id'], events[0]['clock_id'])
    active, finished, phases, points, actor_times = {}, {}, {}, [], {}
    previous_seq = -1
    for event in events:
        if event.get('schema_version') != 1:
            raise TraceError('Unsupported or legacy event schema')
        if (event['run_id'], event['clock_id']) != domain:
            raise TraceError('Cannot subtract clocks from different runs/processes')
        timestamp = _integer(event['ts_ns'], 'ts_ns')
        sequence = _integer(event['seq'], 'seq')
        if sequence != previous_seq + 1:
            raise TraceError('Missing, duplicate or reordered event sequence')
        previous_seq = sequence
        actor = event.get('actor_id', 'main')
        if timestamp < actor_times.get(actor, 0):
            raise TraceError('Clock reversed within an actor')
        actor_times[actor] = timestamp
        ident = _integer(event['span_id'], 'span_id')
        parent = _integer(event['parent_id'], 'parent_id')
        kind = event['type']
        if kind == 'begin':
            if ident == 0 or ident in active or ident in finished:
                raise TraceError('Duplicate span or missing identity')
            if parent and parent not in active:
                raise TraceError('Parent span is not active')
            active[ident] = event
        elif kind == 'end':
            begin = active.pop(ident, None)
            if begin is None:
                raise TraceError('End without begin')
            if event['phase'] != begin['phase'] or parent != begin['parent_id'] or actor != begin.get('actor_id', 'main'):
                raise TraceError('Span identity changed')
            duration = timestamp - begin['ts_ns']
            if duration < 0:
                raise TraceError('Negative span duration')
            if event['attributes'].get('duration_ns') != duration:
                raise TraceError('Claimed duration differs from raw endpoints')
            if any(child['parent_id'] == ident for child in active.values()):
                raise TraceError('Parent ended before its child')
            finished[ident] = (begin, event)
            phases.setdefault(event['phase'], []).append((begin['ts_ns'], timestamp))
        elif kind == 'point':
            if ident != 0:
                raise TraceError('Point must not reuse a span identity')
            if parent and parent not in active:
                raise TraceError('Point parent is not active')
            points.append(event)
        else:
            raise TraceError('Unknown event type')
    if active:
        raise TraceError('Incomplete spans: ' + str(sorted(active)))
    for begin, end in finished.values():
        parent = begin['parent_id']
        if parent:
            if parent not in finished:
                raise TraceError('Missing parent endpoints')
            pbegin, pend = finished[parent]
            if not (pbegin['ts_ns'] <= begin['ts_ns'] <= end['ts_ns'] <= pend['ts_ns']):
                raise TraceError('Nested component exceeds enclosing wall time')
    return {'schema_version': 1, 'run_id': domain[0], 'clock_id': domain[1],
            'phase_union_ms': {name: union_ns(intervals) / 1e6 for name, intervals in phases.items()},
            'span_count': len(finished), 'points': points,
            'fault_to_first_token': transition_intervals(points)}


def transition_intervals(points):
    grouped = {}
    for event in points:
        if event['phase'] not in {'fault_injected', 'fault_confirmed', 'first_token_after_transition'}:
            continue
        ident = event['attributes'].get('transition_id')
        if not ident:
            raise TraceError('Fault marker lacks transaction identity')
        group = grouped.setdefault(ident, {})
        if event['phase'] in group:
            raise TraceError('Duplicate fault milestone')
        group[event['phase']] = event['ts_ns']
    results = {}
    for ident, group in grouped.items():
        result = {}
        for label, start in [('injection_to_token_ms', 'fault_injected'),
                             ('confirmation_to_token_ms', 'fault_confirmed')]:
            if start not in group or 'first_token_after_transition' not in group:
                result[label] = {'value': None, 'status': 'not_observed'}
            else:
                elapsed = group['first_token_after_transition'] - group[start]
                if elapsed < 0:
                    raise TraceError('Token precedes fault milestone')
                result[label] = {'value': elapsed / 1e6, 'status': 'measured'}
        results[ident] = result
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trace')
    args = parser.parse_args()
    data = [json.loads(line) for line in Path(args.trace).read_text().splitlines() if line]
    print(json.dumps(validate_trace(data), indent=2, allow_nan=False))
