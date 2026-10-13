"""Portable autonomous tool loop. Expected results are used only after generation."""
from __future__ import annotations
import inspect
import json
from pathlib import Path
from typing import Any

from agent_bench.tools import TOOL_SCHEMAS, _TOOL_FUNCS, run_tool
from scripts.experiments.recording import RunRecorder
from scripts.experiments.timeline import validate_trace


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Non-JSON numeric constant: ' + value)


def parse_action(text: str, allowed_tools: set[str]) -> dict:
    # Strict JSON: no fence stripping, partial-object recovery or answer repair.
    action = json.loads(text, object_pairs_hook=_json_object, parse_constant=_invalid_constant)
    if not isinstance(action, dict) or set(action) - {'tool', 'arguments', 'reason'}:
        raise ValueError('Expected exactly one action object')
    tool, args = action.get('tool'), action.get('arguments')
    if not isinstance(tool, str) or not isinstance(args, dict):
        raise ValueError('tool and arguments are required')
    if 'reason' in action and not isinstance(action['reason'], str):
        raise ValueError('reason must be a string')
    if tool == 'final_answer':
        if set(args) != {'answer'} or not isinstance(args['answer'], str) or not args['answer'].strip():
            raise ValueError('final_answer requires a nonempty string answer')
    else:
        if tool not in allowed_tools:
            raise ValueError('Tool is not available in this episode')
        # Check supplied keys/defaults without normalizing the model's values.
        inspect.signature(_TOOL_FUNCS[tool]).bind(**args)
    return action


def check_outcome(episode: dict, events: list[dict], final_answer: str) -> dict:
    tools = [item for item in events if item['type'] == 'tool_call']
    observed = {item['name'] for item in tools}
    required = set(episode.get('required_tools', []))
    required_arguments = episode.get('required_tool_arguments', {})
    arguments_ok = all(any(item['name'] == name and item['arguments'] == args for item in tools)
                       for name, args in required_arguments.items())
    expected = episode.get('expected_answer')
    contains = episode.get('answer_contains', [])
    # Without an objective answer checker, completion is observed, quality unknown.
    answer_checked = expected is not None or bool(contains)
    answer_ok = (expected is None or final_answer.strip() == str(expected).strip()) and \
                all(str(fragment) in final_answer for fragment in contains)
    errors = [item for item in events if item['type'] in {'tool_error', 'backend_error'}]
    success = bool(final_answer) and required <= observed and arguments_ok and not errors and answer_ok
    return {'completed': bool(final_answer), 'task_success': success if answer_checked else None,
            'quality_status': 'checked' if answer_checked else 'not_observed',
            'required_tools_observed': required <= observed, 'required_arguments_match': arguments_ok,
            'tool_call_count': len(tools), 'parse_error_count': sum(e['type'] == 'parse_error' for e in events),
            'forced_action_count': 0, 'forced_final_answer_count': 0, 'execution_mode': 'autonomous',
            'autonomous_quality_eligible': True, 'independent_episodes': 1}


def run_autonomous_episode(episode: dict, backend: Any, out_dir: Path) -> dict:
    allowed = set(episode.get('allowed_tools', [item['name'] for item in TOOL_SCHEMAS]))
    if not allowed <= set(_TOOL_FUNCS):
        raise ValueError('Unknown episode tool')
    observation_role = episode.get('tool_observation_role', 'user')
    if observation_role not in {'user', 'tool'}:
        raise ValueError('Unsupported tool observation role')
    tools = [item for item in TOOL_SCHEMAS if item['name'] in allowed]
    messages = [{'role': 'system', 'content':
        'Use tools to solve the task. Return exactly one JSON object per turn, no prose. '
        'Tool results arrive as observations. Never invent or output tool results yourself. '
        'When the observations are sufficient, return final_answer instead of repeating a successful tool call. '
        'Call format: {"tool":"name","arguments":{...}}. '
        'Final format: {"tool":"final_answer","arguments":{"answer":"your answer"}}. '
        'Available tools: ' + json.dumps(tools, ensure_ascii=False)},
        {'role': 'user', 'content': episode['task']}]
    events, final_answer = [], ''
    with RunRecorder(Path(out_dir) / 'autonomous', {'purpose': 'autonomous_agent_episode',
            'episode_id': episode['id'], 'backend': backend.name, 'execution_mode': 'autonomous'}) as run:
        (run.directory/'episode.json').write_text(json.dumps(episode, indent=2, ensure_ascii=False, allow_nan=False))
        try:
            with run.span('episode'):
                for turn in range(1, int(episode.get('max_llm_turns', 8)) + 1):
                    name = f'agent_step_{turn:02d}'
                    dynamic = episode.get('edgevisor_dynamic_plan')
                    if isinstance(dynamic, dict):
                        commands = dynamic.get('commands')
                        if isinstance(commands, list):
                            selected = [cmd for cmd in commands if isinstance(cmd, dict) and
                                cmd.get('trigger_generation', dynamic.get('trigger_generation')) == name]
                            dynamic = {**dynamic, 'commands': selected, 'trigger_generation': name} if selected else None
                        elif dynamic.get('trigger_generation') != name:
                            dynamic = None
                    with run.span('generation', {'generation_id': turn}) as interval:
                        result = backend.generate(messages, max_tokens=int(episode.get('max_tokens', 96)),
                            generation_name=name, out_dir=run.directory, dynamic_plan=dynamic)
                        interval['rc'] = result.rc
                        lineage = result.metrics.get('api_token_lineage', {})
                        interval['api_token_lineage'] = lineage
                    events.append({'type': 'llm_generation', 'generation_id': turn,
                        'content': result.content, 'rc': result.rc, 'metrics': result.metrics,
                        'log_path': result.log_path, 'dynamic_events': result.dynamic_events})
                    if result.rc:
                        events.append({'type': 'backend_error', 'generation_id': turn, 'rc': result.rc})
                        break
                    messages.append({'role': 'assistant', 'content': result.content})
                    try:
                        action = parse_action(result.content, allowed)
                    except (ValueError, TypeError) as error:
                        events.append({'type': 'parse_error', 'generation_id': turn,
                                       'content': result.content, 'error': str(error)})
                        messages.append({'role': 'user', 'content': 'Invalid action JSON. Return one complete action object.'})
                        continue
                    if action['tool'] == 'final_answer':
                        final_answer = action['arguments']['answer']
                        events.append({'type': 'final_answer', 'generation_id': turn,
                                       'answer': final_answer, 'forced': False})
                        break
                    with run.span('tool', {'generation_id': turn, 'name': action['tool'],
                            'api_token_lineage': lineage}) as interval:
                        try:
                            result_tool = run_tool(action['tool'], action['arguments'])
                            observation = {'name': result_tool.name, 'result': result_tool.result}
                            events.append({'type': 'tool_call', 'generation_id': turn,
                                'name': result_tool.name, 'arguments': result_tool.arguments,
                                'result': result_tool.result, 'latency_ms': result_tool.latency_ms})
                            interval['tool_success'] = True
                        except (KeyError, TypeError, ValueError, RuntimeError) as error:
                            observation = {'name': action['tool'], 'error': str(error)}
                            events.append({'type': 'tool_error', 'generation_id': turn, 'name': action['tool'],
                                           'arguments': action['arguments'], 'error': str(error)})
                            interval['tool_success'] = False
                    # A factual tool observation contains no target action or answer.
                    messages.append({'role': observation_role, 'content': 'Tool observation: ' + json.dumps(observation, ensure_ascii=False)})
        finally:
            backend.close()
        metrics = check_outcome(episode, events, final_answer)
        trace = {'episode_id': episode['id'], 'execution_mode': 'autonomous', 'backend': backend.name,
                 'events': events, 'messages': messages, 'final_answer': final_answer, 'agent_metrics': metrics}
        raw = [json.loads(line) for line in (run.directory / 'client_events.jsonl').read_text().splitlines()]
        timeline = validate_trace(raw)
        metrics.update(episode_completion_time=timeline['phase_union_ms'].get('episode'),
                       total_generation_time_ms=timeline['phase_union_ms'].get('generation'),
                       total_tool_time_ms=timeline['phase_union_ms'].get('tool'),
                       p99_tpot=None, p99_status='not_observed')
        (run.directory / 'trace.json').write_text(json.dumps(trace, indent=2, ensure_ascii=False))
        (run.directory / 'timeline.json').write_text(json.dumps(timeline, indent=2))
        run.seal('passed' if metrics['task_success'] is True else 'failed' if metrics['task_success'] is False else 'rejected')
        return trace
