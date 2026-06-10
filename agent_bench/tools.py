from __future__ import annotations

import ast
import hashlib
import json
import operator
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


class ToolError(RuntimeError):
    pass


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_expr(node.operand))
    raise ToolError(f"unsupported calculator expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression without exposing Python eval."""
    tree = ast.parse(expression, mode="eval")
    value = _eval_expr(tree)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_FACTS = {
    "gpu_policy": "The benchmark may use GPU0, GPU1, and GPU2 only. GPU3 is reserved and must not be touched.",
    "backend_swap": "The LangGraph episode keeps the same tool workflow while swapping the LLM generation backend between Prima.cpp and EdgeVisor.",
    "edgevisor_dynamic": "The EdgeVisor backend can send a UDS set_plan command during a generation and continue with the adjusted allocation.",
}


def lookup_fact(key: str) -> str:
    if key not in _FACTS:
        raise ToolError(f"unknown fact key: {key}")
    return _FACTS[key]


def text_stats(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9_]+", text)
    result = {
        "characters": len(text),
        "words": len(words),
        "uppercase": text.upper(),
    }
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def unit_convert(value: Any, from_unit: str, to_unit: str) -> str:
    value_f = float(value)
    source = from_unit.lower()
    target = to_unit.lower()
    if source in {"c", "celsius"} and target in {"f", "fahrenheit"}:
        converted = value_f * 9.0 / 5.0 + 32.0
    elif source in {"f", "fahrenheit"} and target in {"c", "celsius"}:
        converted = (value_f - 32.0) * 5.0 / 9.0
    elif source in {"m", "meter", "meters"} and target in {"km", "kilometer", "kilometers"}:
        converted = value_f / 1000.0
    elif source in {"km", "kilometer", "kilometers"} and target in {"m", "meter", "meters"}:
        converted = value_f * 1000.0
    elif source in {"s", "sec", "second", "seconds"} and target in {"min", "minute", "minutes"}:
        converted = value_f / 60.0
    elif source in {"min", "minute", "minutes"} and target in {"s", "sec", "second", "seconds"}:
        converted = value_f * 60.0
    else:
        raise ToolError(f"unsupported conversion: {from_unit} -> {to_unit}")
    if converted.is_integer():
        converted = int(converted)
    return json.dumps({"value": converted, "unit": to_unit}, ensure_ascii=False, sort_keys=True)


def list_sort(items: Any, reverse: bool = False) -> str:
    parsed = json.loads(items) if isinstance(items, str) else items
    if not isinstance(parsed, list):
        raise ToolError("items must be a list")
    reverse_bool = reverse
    if isinstance(reverse, str):
        reverse_bool = reverse.lower() in {"1", "true", "yes"}
    return json.dumps(sorted(parsed, reverse=bool(reverse_bool)), ensure_ascii=False)


def regex_extract(pattern: str, text: str) -> str:
    return json.dumps(re.findall(pattern, text), ensure_ascii=False)


def string_replace(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


def json_get(data: Any, path: str) -> str:
    current = json.loads(data) if isinstance(data, str) else data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise ToolError(f"cannot descend into {type(current).__name__}")
    return json.dumps(current, ensure_ascii=False, sort_keys=True)


def compare_numbers(a: Any, b: Any) -> str:
    left = float(a)
    right = float(b)
    relation = "equal"
    if left < right:
        relation = "less"
    elif left > right:
        relation = "greater"
    return json.dumps({"a": left, "b": right, "relation": relation}, ensure_ascii=False, sort_keys=True)


def hash_text(text: str, algorithm: str = "sha256") -> str:
    algo = algorithm.lower()
    if algo not in {"md5", "sha1", "sha256"}:
        raise ToolError(f"unsupported hash algorithm: {algorithm}")
    digest = hashlib.new(algo, text.encode("utf-8")).hexdigest()
    return json.dumps({"algorithm": algo, "digest": digest}, ensure_ascii=False, sort_keys=True)


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "calculator",
        "description": "Evaluate a simple arithmetic expression.",
        "arguments": {"expression": "string"},
    },
    {
        "name": "lookup_fact",
        "description": "Lookup a local fact by key. Keys: gpu_policy, backend_swap, edgevisor_dynamic.",
        "arguments": {"key": "string"},
    },
    {
        "name": "text_stats",
        "description": "Return character count, word count, and uppercase form for text.",
        "arguments": {"text": "string"},
    },
    {
        "name": "unit_convert",
        "description": "Convert celsius/fahrenheit, meters/kilometers, or seconds/minutes.",
        "arguments": {"value": "number", "from_unit": "string", "to_unit": "string"},
    },
    {
        "name": "list_sort",
        "description": "Sort a JSON list.",
        "arguments": {"items": "list", "reverse": "boolean optional"},
    },
    {
        "name": "regex_extract",
        "description": "Extract regex matches from text.",
        "arguments": {"pattern": "string", "text": "string"},
    },
    {
        "name": "string_replace",
        "description": "Replace all occurrences of old with new in text.",
        "arguments": {"text": "string", "old": "string", "new": "string"},
    },
    {
        "name": "json_get",
        "description": "Read a dot path from a JSON object or array.",
        "arguments": {"data": "object or JSON string", "path": "string"},
    },
    {
        "name": "compare_numbers",
        "description": "Compare two numbers.",
        "arguments": {"a": "number", "b": "number"},
    },
    {
        "name": "hash_text",
        "description": "Hash text with md5, sha1, or sha256.",
        "arguments": {"text": "string", "algorithm": "string optional"},
    },
]


_TOOL_FUNCS: Dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "lookup_fact": lookup_fact,
    "text_stats": text_stats,
    "unit_convert": unit_convert,
    "list_sort": list_sort,
    "regex_extract": regex_extract,
    "string_replace": string_replace,
    "json_get": json_get,
    "compare_numbers": compare_numbers,
    "hash_text": hash_text,
}


@dataclass
class ToolResult:
    name: str
    arguments: Dict[str, Any]
    result: str
    latency_ms: float


def run_tool(name: str, arguments: Dict[str, Any]) -> ToolResult:
    start = time.perf_counter()
    if name not in _TOOL_FUNCS:
        raise ToolError(f"unknown tool: {name}")
    result = _TOOL_FUNCS[name](**arguments)
    return ToolResult(
        name=name,
        arguments=arguments,
        result=result,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
