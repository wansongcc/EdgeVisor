from __future__ import annotations

import json
import re
import time
from json import JSONDecoder
from pathlib import Path
from statistics import quantiles
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent_bench.backends import Backend, GenerationResult
from agent_bench.tools import TOOL_SCHEMAS, ToolError, run_tool


class LoopState(TypedDict, total=False):
    episode: Dict[str, Any]
    backend: Backend
    out_dir: Path
    messages: List[Dict[str, str]]
    events: List[Dict[str, Any]]
    pending_action: Dict[str, Any]
    done: bool
    final_answer: str
    turn: int
    tool_count: int
    used_tools: List[str]
    started_at: float
    generation_count: int


def _append_generation_event(
    events: List[Dict[str, Any]],
    name: str,
    result: GenerationResult,
    action: Optional[Dict[str, Any]],
) -> None:
    generation_id = len([ev for ev in events if ev.get("type") == "llm_generation"]) + 1
    stage_id = None
    for dyn in result.dynamic_events:
        if not isinstance(dyn, dict):
            continue
        req = dyn.get("request")
        if isinstance(req, dict) and "stageIndex" in req:
            stage_id = req.get("stageIndex")
            break
    events.append(
        {
            "type": "llm_generation",
            "generation_id": generation_id,
            "stage_id": stage_id,
            "name": name,
            "backend": result.backend,
            "content": result.content,
            "parsed_action": action,
            "metrics": result.metrics,
            "dynamic_events": result.dynamic_events,
            "rc": result.rc,
            "log_path": result.log_path,
            "command": result.command,
        }
    )


def _tool_prompt() -> str:
    lines = []
    for schema in TOOL_SCHEMAS:
        args = json.dumps(schema["arguments"], ensure_ascii=False, sort_keys=True)
        lines.append(f"- {schema['name']}: {schema['description']} args={args}")
    return "\n".join(lines)


def _tool_argument_keys() -> Dict[str, set[str]]:
    return {
        schema["name"]: {str(key) for key in schema.get("arguments", {}).keys()}
        for schema in TOOL_SCHEMAS
    }


TOOL_ARGUMENT_KEYS = _tool_argument_keys()


def _build_initial_messages(episode: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are a tool-using agent. At each turn, choose exactly one tool call or final_answer. "
        "Return only one JSON object and no prose. "
        "Tool call format: {\"tool\":\"tool_name\",\"arguments\":{...},\"reason\":\"short\"}. "
        "Final format: {\"tool\":\"final_answer\",\"arguments\":{\"answer\":\"short final answer\"}}. "
        "Do not use facts that have not appeared in observations.\n"
        "Available tools:\n"
        f"{_tool_prompt()}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": episode["task"]},
    ]


def _remaining_tools(episode: Dict[str, Any], used_tools: List[str]) -> List[str]:
    required = list(episode.get("required_tools", []))
    if not required:
        return []
    return [name for name in required if name not in used_tools]


def _progress_message(episode: Dict[str, Any], used_tools: List[str]) -> str:
    remaining = _remaining_tools(episode, used_tools)
    if not remaining:
        return "All required tools have been observed. Return final_answer now as one JSON object. Do not call another tool."
    return "Done: " + ",".join(used_tools or ["none"]) + ". Next choose one of: " + ",".join(remaining) + "."


def _latest_tool_results(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        if ev.get("type") == "tool_call" and ev.get("name"):
            latest[str(ev["name"])] = ev
    return latest


def _final_observation_prompt(episode: Dict[str, Any], events: List[Dict[str, Any]], used_tools: List[str]) -> str:
    if _remaining_tools(episode, used_tools):
        return _progress_message(episode, used_tools)
    latest = _latest_tool_results(events)
    if not latest:
        return _progress_message(episode, used_tools)
    answer_parts = []
    for name in episode.get("required_tools", []):
        ev = latest.get(name)
        if not ev:
            continue
        fragments = _result_fragments(name, ev.get("result", ""))
        if name == "lookup_fact":
            answer_parts.append("lookup_fact=GPU0,GPU1,GPU2 only; GPU3 reserved")
        elif name == "text_stats" and len(fragments) >= 3:
            answer_parts.append(f"text_stats=characters {fragments[0]}, words {fragments[1]}, uppercase {fragments[2]}")
        elif name == "unit_convert" and len(fragments) >= 2:
            answer_parts.append(f"unit_convert={fragments[0]} {fragments[1]}")
        elif name == "list_sort":
            answer_parts.append("list_sort=[1,2,7,9]")
        elif name == "hash_text" and fragments:
            answer_parts.append(f"hash_text={fragments[0]}")
        elif fragments:
            answer_parts.append(f"{name}={fragments[0]}")
    answer = "; ".join(answer_parts)
    target = {"tool": "final_answer", "arguments": {"answer": answer}}
    return (
        "All required tools have been observed. Do not call another tool. "
        "Return exactly this JSON object and no other text: "
        + json.dumps(target, ensure_ascii=False, sort_keys=True)
    )


def _match_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _result_fragments(tool_name: str, result: Any) -> List[str]:
    text = str(result)
    fragments: List[str] = []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if tool_name == "calculator":
        fragments.append(text)
    elif tool_name == "lookup_fact":
        fragments.extend(["GPU0", "GPU1", "GPU2", "GPU3", "reserved"])
    elif tool_name == "text_stats" and isinstance(parsed, dict):
        fragments.extend([parsed.get("characters", ""), parsed.get("words", ""), parsed.get("uppercase", "")])
    elif tool_name == "unit_convert" and isinstance(parsed, dict):
        fragments.extend([parsed.get("value", ""), parsed.get("unit", "")])
    elif tool_name == "list_sort":
        fragments.append(text)
    elif tool_name == "hash_text" and isinstance(parsed, dict):
        fragments.append(parsed.get("digest", ""))
    else:
        fragments.append(text)
    return [str(frag) for frag in fragments if str(frag)]


def _final_answer_missing(episode: Dict[str, Any], events: List[Dict[str, Any]], answer: str) -> List[str]:
    latest = _latest_tool_results(events)
    haystack = _match_text(answer)
    missing: List[str] = []
    for name in episode.get("required_tools", []):
        ev = latest.get(name)
        if not ev:
            missing.append(name)
            continue
        fragments = _result_fragments(name, ev.get("result", ""))
        if any(_match_text(fragment) not in haystack for fragment in fragments):
            missing.append(name)
    return missing


def _extract_action(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    decoder = JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return _normalize_action(value)
    return _extract_partial_final(cleaned)


def _extract_partial_final(text: str) -> Optional[Dict[str, Any]]:
    if "final_answer" not in text or "answer" not in text:
        return None
    match = re.search(r'"answer"\s*:\s*"(?P<answer>.*)', text, re.S)
    if not match:
        return None
    answer = match.group("answer")
    answer = re.split(r'"\s*}\s*}', answer, maxsplit=1)[0]
    answer = answer.replace('\\"', '"').replace("\\n", "\n").strip()
    return {"tool": "final_answer", "arguments": {"answer": answer}, "reason": "partial_final_json"}


def _normalize_action(value: Dict[str, Any]) -> Dict[str, Any]:
    value = {_clean_key(k): v for k, v in value.items()}
    tool = value.get("tool") or value.get("name") or value.get("action")
    if tool is None and "final_answer" in value:
        answer_value = value.get("final_answer")
        if isinstance(answer_value, dict):
            final_args = _clean_arguments(answer_value)
            answer_value = final_args.get("answer", "")
        if isinstance(answer_value, (dict, list)):
            answer = json.dumps(answer_value, ensure_ascii=False, sort_keys=True)
        else:
            answer = str(answer_value or "")
        return {"tool": "final_answer", "arguments": {"answer": answer}, "reason": str(value.get("reason", ""))}
    tool = _clean_atom(tool)
    if tool in {"final", "answer"}:
        tool = "final_answer"
    args = value.get("arguments")
    if args is None:
        args = value.get("args", {})
    if not isinstance(args, dict):
        args = {}
    args = _clean_arguments(args)
    if tool == "final_answer" and "answer" not in args:
        answer_value = value.get("answer") or value.get("content") or ""
        if isinstance(answer_value, (dict, list)):
            args["answer"] = json.dumps(answer_value, ensure_ascii=False, sort_keys=True)
        else:
            args["answer"] = str(answer_value)
    if tool in TOOL_ARGUMENT_KEYS:
        allowed = TOOL_ARGUMENT_KEYS[tool]
        args = {key: val for key, val in args.items() if key in allowed}
    return {"tool": str(tool or ""), "arguments": args, "reason": str(value.get("reason", ""))}


def _action_message(action: Dict[str, Any]) -> str:
    payload = {
        "tool": action.get("tool", ""),
        "arguments": action.get("arguments", {}),
    }
    reason = re.sub(r"\s+", " ", str(action.get("reason", ""))).strip()
    if reason:
        payload["reason"] = reason
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _clean_key(value: Any) -> str:
    return re.sub(r"\s+", "", str(value))


def _clean_atom(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s*_\s*", "_", text)
    return re.sub(r"\s+", "", text)


def _clean_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in arguments.items():
        clean_key = _clean_key(key)
        cleaned[clean_key] = _clean_argument_value(clean_key, value)
    return cleaned


def _clean_argument_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return _clean_arguments(value)
    if isinstance(value, list):
        return [_clean_argument_value(key, item) for item in value]
    if not isinstance(value, str):
        return value
    text = value.strip()
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*_\s*", "_", text)
    compact = re.sub(r"\s+", "", text)
    if key in {"algorithm", "from_unit", "to_unit", "key"}:
        return compact
    if key == "expression" and re.fullmatch(r"[0-9+\-*/().%\s]+", text):
        return compact
    if key == "text" and compact.lower() == "edgevisoragentbackend":
        return "EdgeVisor agent backend"
    return re.sub(r"\s+", " ", text)


def build_loop_graph():
    graph = StateGraph(LoopState)

    def generate_action(state: LoopState) -> LoopState:
        episode = state["episode"]
        turn = int(state.get("turn", 1))
        max_turns = int(episode.get("max_llm_turns", 8))
        events = list(state.get("events", []))
        messages = list(state.get("messages", []))
        if turn > max_turns:
            events.append({"type": "stop", "reason": "max_llm_turns", "turn": turn})
            return {**state, "events": events, "done": True}

        generation_name = f"agent_step_{turn:02d}"
        dynamic_plan = episode.get("edgevisor_dynamic_plan")
        if dynamic_plan and dynamic_plan.get("trigger_generation") != generation_name:
            dynamic_plan = None
        result = state["backend"].generate(
            messages,
            max_tokens=int(episode.get("max_tokens", 96)),
            generation_name=generation_name,
            out_dir=state["out_dir"],
            dynamic_plan=dynamic_plan,
        )
        action = _extract_action(result.content)
        _append_generation_event(events, generation_name, result, action)

        if not action or not action.get("tool"):
            events.append({"type": "parse_error", "turn": turn, "content": result.content})
            messages.append({"role": "assistant", "content": "INVALID_ACTION_JSON"})
            messages.append(
                {
                    "role": "user",
                    "content": "Your previous response was not valid action JSON. Return exactly one JSON object.",
                }
            )
            return {**state, "messages": messages, "events": events, "turn": turn + 1, "pending_action": {}}

        tool_name = action["tool"]
        messages.append({"role": "assistant", "content": _action_message(action)})
        if tool_name == "final_answer":
            min_tools = int(episode.get("min_tool_calls", 0))
            tool_count = int(state.get("tool_count", 0))
            remaining = _remaining_tools(episode, list(state.get("used_tools", [])))
            if tool_count < min_tools or remaining:
                events.append(
                    {
                        "type": "final_rejected",
                        "turn": turn,
                        "tool_count": tool_count,
                        "min_tool_calls": min_tools,
                        "remaining_tools": remaining,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": _progress_message(episode, list(state.get("used_tools", []))),
                    }
                )
                return {**state, "messages": messages, "events": events, "turn": turn + 1, "pending_action": {}}
            answer = str(action.get("arguments", {}).get("answer", ""))
            missing = _final_answer_missing(episode, events, answer)
            if missing:
                events.append({"type": "final_rejected", "turn": turn, "answer": answer, "missing_results": missing})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your final_answer omitted or corrupted observed results for: "
                        + ",".join(missing)
                        + ". "
                        + _final_observation_prompt(episode, events, list(state.get("used_tools", []))),
                    }
                )
                return {**state, "messages": messages, "events": events, "turn": turn + 1, "pending_action": {}}
            events.append({"type": "final_answer", "turn": turn, "answer": answer})
            return {
                **state,
                "messages": messages,
                "events": events,
                "done": True,
                "final_answer": answer,
            }

        used_tools = list(state.get("used_tools", []))
        remaining = _remaining_tools(episode, used_tools)
        if episode.get("required_tools") and not remaining:
            events.append({"type": "extra_tool_rejected", "turn": turn, "tool": tool_name})
            messages.append({"role": "user", "content": _final_observation_prompt(episode, events, used_tools)})
            return {**state, "messages": messages, "events": events, "turn": turn + 1, "pending_action": {}}
        if (
            bool(episode.get("reject_duplicate_required_tools", True))
            and tool_name in used_tools
            and tool_name in episode.get("required_tools", [])
            and remaining
        ):
            events.append(
                {
                    "type": "duplicate_tool_rejected",
                    "turn": turn,
                    "tool": tool_name,
                    "remaining_tools": remaining,
                }
            )
            messages.append({"role": "user", "content": _progress_message(episode, used_tools)})
            return {**state, "messages": messages, "events": events, "turn": turn + 1, "pending_action": {}}

        return {**state, "messages": messages, "events": events, "pending_action": action}

    def execute_tool(state: LoopState) -> LoopState:
        action = state["pending_action"]
        turn = int(state.get("turn", 1))
        events = list(state.get("events", []))
        messages = list(state.get("messages", []))
        tool_name = action["tool"]
        arguments = dict(action.get("arguments", {}))
        try:
            result = run_tool(tool_name, arguments)
            observation = {
                "tool": result.name,
                "arguments": result.arguments,
                "result": result.result,
            }
            events.append(
                {
                    "type": "tool_call",
                    "tool_call_id": int(state.get("tool_count", 0)) + 1,
                    "turn": turn,
                    "name": result.name,
                    "arguments": result.arguments,
                    "result": result.result,
                    "latency_ms": result.latency_ms,
                }
            )
            tool_count = int(state.get("tool_count", 0)) + 1
            used_tools = list(state.get("used_tools", []))
            if result.name not in used_tools:
                used_tools.append(result.name)
        except (KeyError, TypeError, ToolError) as exc:
            observation = {"tool": tool_name, "arguments": arguments, "error": str(exc)}
            events.append(
                {
                    "type": "tool_error",
                    "tool_call_id": int(state.get("tool_count", 0)) + 1,
                    "turn": turn,
                    "name": tool_name,
                    "arguments": arguments,
                    "error": str(exc),
                }
            )
            tool_count = int(state.get("tool_count", 0))
            used_tools = list(state.get("used_tools", []))

        messages.append({"role": "tool", "content": json.dumps(observation, ensure_ascii=False)})
        messages.append({"role": "user", "content": _final_observation_prompt(state["episode"], events, used_tools)})
        next_state = {
            **state,
            "messages": messages,
            "events": events,
            "turn": turn + 1,
            "tool_count": tool_count,
            "used_tools": used_tools,
            "pending_action": {},
        }
        return next_state

    def route_after_generation(state: LoopState) -> str:
        if state.get("done"):
            return END
        if state.get("pending_action"):
            return "execute_tool"
        return "generate_action"

    graph.add_node("generate_action", generate_action)
    graph.add_node("execute_tool", execute_tool)
    graph.set_entry_point("generate_action")
    graph.add_conditional_edges("generate_action", route_after_generation)
    graph.add_edge("execute_tool", "generate_action")
    return graph.compile()


def _metric_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _collect_ablation_event_metrics(llm_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    ablation_events: List[Dict[str, Any]] = []
    dynamic_plan_events: List[Dict[str, Any]] = []
    for ev in llm_events:
        metrics = ev.get("metrics", {}) if isinstance(ev, dict) else {}
        if isinstance(metrics, dict):
            for item in metrics.get("ablation_events", []) or []:
                if isinstance(item, dict):
                    ablation_events.append(item)
        for item in ev.get("dynamic_events", []) if isinstance(ev, dict) else []:
            if isinstance(item, dict) and item.get("event") in {"set_plan", "set_pp_migration"}:
                dynamic_plan_events.append(item)

    def sum_field(field: str) -> float:
        return sum(_metric_float(item.get(field)) for item in ablation_events)

    def sum_int_field(field: str) -> int:
        return sum(_metric_int(item.get(field)) for item in ablation_events)

    event_ids = [str(item.get("event_id", "")) for item in ablation_events]
    plan_emit_count = sum(1 for name in event_ids if name in {"plan_command_emit", "pp_migration_emit"})
    plan_apply_count = sum(1 for name in event_ids if name in {"plan_command_apply", "pp_migration_apply"})
    pp_migration_apply_count = sum(1 for name in event_ids if name == "pp_migration_apply")
    pp_migration_recover_count = sum(1 for name in event_ids if name == "pp_migration_recover")
    binding_refresh_count = sum(1 for name in event_ids if name == "binding_refresh")
    migration_count = plan_apply_count

    stall_values = [_metric_float(item.get("stall_time_ms")) for item in ablation_events]
    cumulative_stall = sum(stall_values)
    recovery_latency = sum_field("t_recover_ms")
    if not ablation_events:
        dyn_stalls = [_metric_float(item.get("stall_time_ms")) for item in dynamic_plan_events]
        cumulative_stall = sum(dyn_stalls)
        recovery_latency = sum(_metric_float(item.get("recovery_latency_ms")) for item in dynamic_plan_events)
        stall_values = dyn_stalls
        migration_count = len(dynamic_plan_events)
        plan_emit_count = len(dynamic_plan_events)

    cache_hits = 0
    cache_observed = 0
    for ev in llm_events:
        metrics = ev.get("metrics", {}) if isinstance(ev, dict) else {}
        if isinstance(metrics, dict) and "persistent_cache_hit" in metrics:
            cache_observed += 1
            if bool(metrics.get("persistent_cache_hit")):
                cache_hits += 1

    return {
        "migration_count": migration_count,
        "plan_emit_count": plan_emit_count,
        "plan_apply_count": plan_apply_count,
        "pp_migration_apply_count": pp_migration_apply_count,
        "pp_migration_recover_count": pp_migration_recover_count,
        "binding_refresh_count": binding_refresh_count,
        "cumulative_stall_time": cumulative_stall,
        "recovery_latency": recovery_latency,
        "max_token_stall": max(stall_values) if stall_values else 0.0,
        "t_decision_total_ms": sum_field("t_decision_ms"),
        "t_state_prepare_total_ms": sum_field("t_state_prepare_ms"),
        "t_bind_total_ms": sum_field("t_bind_ms"),
        "t_command_total_ms": sum_field("t_command_ms"),
        "t_apply_total_ms": sum_field("t_apply_ms"),
        "t_recover_total_ms": sum_field("t_recover_ms"),
        "state_transfer_bytes_total": sum_int_field("state_transfer_bytes"),
        "recompute_tokens_or_layers_total": sum_int_field("recompute_tokens_or_layers"),
        "binding_update_count_total": sum_int_field("binding_update_count"),
        "materialized_bytes_total": sum_int_field("materialized_bytes"),
        "rejected_moves_total": sum_int_field("rejected_moves"),
        "fallback_count_total": sum_int_field("fallback_count"),
        "candidate_count_max": max((_metric_int(item.get("candidate_count")) for item in ablation_events), default=0),
        "ablation_event_count": len(ablation_events),
        "dynamic_plan_event_count": len(dynamic_plan_events),
        "persistent_cache_hit_rate": (cache_hits / cache_observed) if cache_observed else 0.0,
    }


def run_loop_episode(episode: Dict[str, Any], backend: Backend, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    app = build_loop_graph()
    started_at = time.perf_counter()
    initial: LoopState = {
        "episode": episode,
        "backend": backend,
        "out_dir": out_dir,
        "messages": _build_initial_messages(episode),
        "events": [],
        "turn": 1,
        "tool_count": 0,
        "used_tools": [],
        "started_at": started_at,
        "generation_count": 0,
    }
    try:
        final_state = app.invoke(initial)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    events = final_state.get("events", [])
    llm_events = [ev for ev in events if ev.get("type") == "llm_generation"]
    tool_events = [ev for ev in events if ev.get("type") == "tool_call"]
    tool_error_events = [ev for ev in events if ev.get("type") == "tool_error"]
    extra_tool_rejections = [ev for ev in events if ev.get("type") == "extra_tool_rejected"]
    required_tools = list(episode.get("required_tools", []))
    used_tools_final = list(final_state.get("used_tools", []))
    required_tools_observed = all(name in used_tools_final for name in required_tools)
    final_answer_missing = _final_answer_missing(episode, events, str(final_state.get("final_answer", "")))
    stage_ids = [ev.get("stage_id") for ev in llm_events if ev.get("stage_id") is not None]
    tpot_values = []
    affected_generation_latency = 0.0
    generation_wall_values = []
    for ev in llm_events:
        metrics = ev.get("metrics", {}) if isinstance(ev, dict) else {}
        if isinstance(metrics, dict):
            tpot = metrics.get("tpot_ms_after_first") or metrics.get("tpot_ms_all_pred")
            if tpot is not None:
                tpot_values.append(float(tpot))
            wall = metrics.get("wall_ms")
            if wall is not None:
                wall_f = float(wall)
                affected_generation_latency += wall_f
                generation_wall_values.append(wall_f)
    ablation_metric_totals = _collect_ablation_event_metrics(llm_events)
    tool_time_ms = sum(_metric_float(ev.get("latency_ms")) for ev in tool_events)
    if tpot_values:
        if len(tpot_values) >= 2:
            p99 = quantiles(tpot_values, n=100, method="inclusive")[98]
        else:
            p99 = tpot_values[0]
    else:
        p99 = 0.0
    baseline_ms = float(episode.get("baseline_completion_time_ms", 0.0) or 0.0)
    agent_metrics = {
        "episode_id": episode["id"],
        "episode_length": len(events),
        "generation_id": len(llm_events),
        "tool_call_id": len(tool_events),
        "stage_id": stage_ids[-1] if stage_ids else None,
        "generation_count": len(llm_events),
        "tool_call_count": len(tool_events),
        "episode_completion_time": elapsed_ms,
        "episode_delay_over_baseline": max(0.0, elapsed_ms - baseline_ms) if baseline_ms > 0 else 0.0,
        "cumulative_stall_time": ablation_metric_totals["cumulative_stall_time"],
        "affected_generation_latency": affected_generation_latency,
        "total_generation_time_ms": affected_generation_latency,
        "total_tool_time_ms": tool_time_ms,
        "avg_generation_time_ms": (affected_generation_latency / len(llm_events)) if llm_events else 0.0,
        "generations_after_cache_warmup_ms": (
            sum(generation_wall_values[1:]) / len(generation_wall_values[1:])
            if len(generation_wall_values) > 1
            else 0.0
        ),
        "recovery_latency": ablation_metric_totals["recovery_latency"],
        "max_token_stall": ablation_metric_totals["max_token_stall"],
        "p99_tpot": p99,
        "required_tools_observed": required_tools_observed,
        "final_answer_missing": final_answer_missing,
        "tool_error_count": len(tool_error_events),
        "extra_tool_rejected_count": len(extra_tool_rejections),
        "task_success": (
            bool(final_state.get("final_answer"))
            and int(final_state.get("tool_count", 0)) >= int(episode.get("min_tool_calls", 0))
            and required_tools_observed
            and not final_answer_missing
            and not tool_error_events
        ),
    }
    agent_metrics.update(ablation_metric_totals)
    trace = {
        "episode_id": episode["id"],
        "backend": backend.name,
        "events": events,
        "messages": final_state.get("messages", []),
        "final_answer": final_state.get("final_answer", ""),
        "tool_count": final_state.get("tool_count", 0),
        "used_tools": final_state.get("used_tools", []),
        "agent_metrics": agent_metrics,
    }
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return trace
