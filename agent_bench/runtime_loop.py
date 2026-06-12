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
        return "All required tools have been observed. You may now return final_answer as one JSON object."
    return "Done: " + ",".join(used_tools or ["none"]) + ". Next choose one of: " + ",".join(remaining) + "."


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
        args["answer"] = str(value.get("answer") or value.get("content") or "")
    return {"tool": str(tool or ""), "arguments": args, "reason": str(value.get("reason", ""))}


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
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
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
        messages.append({"role": "user", "content": _progress_message(state["episode"], used_tools)})
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
    final_state = app.invoke(initial)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    events = final_state.get("events", [])
    llm_events = [ev for ev in events if ev.get("type") == "llm_generation"]
    tool_events = [ev for ev in events if ev.get("type") == "tool_call"]
    stage_ids = [ev.get("stage_id") for ev in llm_events if ev.get("stage_id") is not None]
    tpot_values = []
    affected_generation_latency = 0.0
    cumulative_stall = 0.0
    recovery_latency = 0.0
    max_token_stall = 0.0
    for ev in llm_events:
        metrics = ev.get("metrics", {}) if isinstance(ev, dict) else {}
        if isinstance(metrics, dict):
            tpot = metrics.get("tpot_ms_after_first") or metrics.get("tpot_ms_all_pred")
            if tpot is not None:
                tpot_values.append(float(tpot))
            wall = metrics.get("wall_ms")
            if wall is not None:
                affected_generation_latency += float(wall)
        for dyn in ev.get("dynamic_events", []) if isinstance(ev, dict) else []:
            if not isinstance(dyn, dict):
                continue
            stall = dyn.get("stall_time_ms") or dyn.get("simulated_stall_ms") or 0.0
            cumulative_stall += float(stall)
            recovery_latency += float(dyn.get("recovery_latency_ms") or stall or 0.0)
            max_token_stall = max(max_token_stall, float(stall))
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
        "cumulative_stall_time": cumulative_stall,
        "affected_generation_latency": affected_generation_latency,
        "recovery_latency": recovery_latency,
        "max_token_stall": max_token_stall,
        "p99_tpot": p99,
        "task_success": bool(final_state.get("final_answer")) and int(final_state.get("tool_count", 0)) >= int(episode.get("min_tool_calls", 0)),
    }
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
