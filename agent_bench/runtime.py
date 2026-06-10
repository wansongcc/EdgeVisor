from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from agent_bench.backends import Backend, GenerationResult
from agent_bench.tools import run_tool


class EpisodeState(TypedDict, total=False):
    episode: Dict[str, Any]
    backend: Backend
    out_dir: Path
    messages: List[Dict[str, str]]
    events: List[Dict[str, Any]]
    tool_result: str


def _append_generation_event(events: List[Dict[str, Any]], name: str, result: GenerationResult) -> None:
    events.append(
        {
            "type": "llm_generation",
            "name": name,
            "backend": result.backend,
            "content": result.content,
            "metrics": result.metrics,
            "dynamic_events": result.dynamic_events,
            "rc": result.rc,
            "log_path": result.log_path,
            "command": result.command,
        }
    )


def _render_template(text: str, state: EpisodeState) -> str:
    return text.format(tool_result=state.get("tool_result", ""))


def build_graph():
    graph = StateGraph(EpisodeState)

    def generation_1(state: EpisodeState) -> EpisodeState:
        episode = state["episode"]
        gen = episode["generations"][0]
        messages = list(state.get("messages", []))
        messages.append({"role": "user", "content": _render_template(gen["prompt"], state)})
        dynamic_plan = episode.get("edgevisor_dynamic_plan")
        if dynamic_plan and dynamic_plan.get("trigger_generation") != gen["name"]:
            dynamic_plan = None
        result = state["backend"].generate(
            messages,
            max_tokens=int(gen.get("max_tokens", 32)),
            generation_name=gen["name"],
            out_dir=state["out_dir"],
            dynamic_plan=dynamic_plan,
        )
        messages.append({"role": "assistant", "content": result.content})
        events = list(state.get("events", []))
        _append_generation_event(events, gen["name"], result)
        return {**state, "messages": messages, "events": events}

    def tool_call(state: EpisodeState) -> EpisodeState:
        tool_spec = state["episode"]["tool_call"]
        result = run_tool(tool_spec["name"], dict(tool_spec.get("arguments", {})))
        messages = list(state.get("messages", []))
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "tool": result.name,
                        "arguments": result.arguments,
                        "result": result.result,
                    },
                    ensure_ascii=False,
                ),
            }
        )
        events = list(state.get("events", []))
        events.append(
            {
                "type": "tool_call",
                "name": result.name,
                "arguments": result.arguments,
                "result": result.result,
                "latency_ms": result.latency_ms,
            }
        )
        return {**state, "messages": messages, "events": events, "tool_result": result.result}

    def generation_2(state: EpisodeState) -> EpisodeState:
        episode = state["episode"]
        gen = episode["generations"][1]
        messages = list(state.get("messages", []))
        messages.append({"role": "user", "content": _render_template(gen["prompt"], state)})
        dynamic_plan = episode.get("edgevisor_dynamic_plan")
        if dynamic_plan and dynamic_plan.get("trigger_generation") != gen["name"]:
            dynamic_plan = None
        result = state["backend"].generate(
            messages,
            max_tokens=int(gen.get("max_tokens", 64)),
            generation_name=gen["name"],
            out_dir=state["out_dir"],
            dynamic_plan=dynamic_plan,
        )
        messages.append({"role": "assistant", "content": result.content})
        events = list(state.get("events", []))
        _append_generation_event(events, gen["name"], result)
        return {**state, "messages": messages, "events": events}

    graph.add_node("generation_1", generation_1)
    graph.add_node("tool_call", tool_call)
    graph.add_node("generation_2", generation_2)
    graph.set_entry_point("generation_1")
    graph.add_edge("generation_1", "tool_call")
    graph.add_edge("tool_call", "generation_2")
    graph.add_edge("generation_2", END)
    return graph.compile()


def run_episode(episode: Dict[str, Any], backend: Backend, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    app = build_graph()
    initial: EpisodeState = {
        "episode": episode,
        "backend": backend,
        "out_dir": out_dir,
        "messages": list(episode.get("initial_messages", [])),
        "events": [],
    }
    final_state = app.invoke(initial)
    trace = {
        "episode_id": episode["id"],
        "backend": backend.name,
        "events": final_state.get("events", []),
        "messages": final_state.get("messages", []),
    }
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return trace

