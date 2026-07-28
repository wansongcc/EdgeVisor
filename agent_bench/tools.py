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


# ---------------------------------------------------------------------------
# web_search: real web search when the network is reachable, offline corpus
# fallback otherwise.
#
# Network probing on the experiment host (2026-07): duckduckgo.com and
# wikipedia.org time out, but cn.bing.com is reachable (HTTP 200, ~0.5s).
# We therefore use the Bing CN HTML endpoint with plain `requests` (no extra
# dependency). Latency is the natural network latency (~0.5-3s).
# ---------------------------------------------------------------------------

_BING_SEARCH_URL = "https://cn.bing.com/search"
_BING_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_BING_ALGO_RE = re.compile(r'<li class="b_algo".*?</li>', re.DOTALL)
_BING_TITLE_RE = re.compile(r"<h2>.*?<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.DOTALL)
_BING_SNIPPET_RE = re.compile(r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', re.DOTALL)
_BING_SNIPPET_FALLBACK_RE = re.compile(r"<p>(.*?)</p>", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(fragment: str) -> str:
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub("", fragment)).strip()


def _bing_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    import requests  # local import: tools.py must stay importable without requests

    resp = requests.get(
        _BING_SEARCH_URL,
        params={"q": query, "setlang": "en"},
        headers={"User-Agent": _BING_UA},
        timeout=15.0,
    )
    resp.raise_for_status()
    results: List[Dict[str, Any]] = []
    for block in _BING_ALGO_RE.findall(resp.text):
        m = _BING_TITLE_RE.search(block)
        if not m:
            continue
        url, title = m.group(1), _strip_html(m.group(2))
        sm = _BING_SNIPPET_RE.search(block) or _BING_SNIPPET_FALLBACK_RE.search(block)
        snippet = _strip_html(sm.group(1)) if sm else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


# Offline fallback corpus: a tiny bundled document set used ONLY when the
# network is unreachable. Retrieval is keyword-overlap scoring over the docs;
# latency is drawn from a realistic web-search distribution (uniform 0.5-3s)
# to keep tool-window experiments representative. This is NOT a real search.
_CORPUS: List[Dict[str, str]] = [
    {"title": "Paris", "url": "corpus://paris",
     "snippet": "Paris is the capital and most populous city of France. It is situated on the River Seine, in northern France."},
    {"title": "France", "url": "corpus://france",
     "snippet": "France, officially the French Republic, is a country in Western Europe. Its capital is Paris."},
    {"title": "Emmanuel Macron", "url": "corpus://macron",
     "snippet": "Emmanuel Macron is a French politician who has served as President of France since 2017."},
    {"title": "Eiffel Tower", "url": "corpus://eiffel",
     "snippet": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France, completed in 1889."},
    {"title": "Seine", "url": "corpus://seine",
     "snippet": "The Seine is a 777-kilometre-long river in northern France, flowing through Paris."},
    {"title": "Tokyo", "url": "corpus://tokyo",
     "snippet": "Tokyo is the capital and most populous city of Japan, located on the eastern coast of Honshu."},
    {"title": "Japan", "url": "corpus://japan",
     "snippet": "Japan is an island country in East Asia, located in the Pacific Ocean. Its capital is Tokyo."},
    {"title": "Mount Fuji", "url": "corpus://fuji",
     "snippet": "Mount Fuji is the highest mountain in Japan, standing 3,776 meters tall, an active stratovolcano."},
    {"title": "Large language model", "url": "corpus://llm",
     "snippet": "A large language model (LLM) is a neural network trained on massive text corpora for language tasks."},
    {"title": "Pipeline parallelism", "url": "corpus://pp",
     "snippet": "Pipeline parallelism partitions a deep neural network across stages, each holding a subset of layers."},
    {"title": "KV cache", "url": "corpus://kvcache",
     "snippet": "The KV cache stores key and value tensors of previous tokens to avoid recomputation in autoregressive decoding."},
    {"title": "NVIDIA T4", "url": "corpus://t4",
     "snippet": "The NVIDIA T4 is a Turing-architecture datacenter GPU with 16 GB of GDDR6 memory, built for inference."},
]


def _corpus_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    import random

    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) > 2]
    scored = []
    for doc in _CORPUS:
        hay = (doc["title"] + " " + doc["snippet"]).lower()
        score = sum(hay.count(t) for t in terms)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    docs = [dict(doc) for _, doc in scored[:max_results]]
    # Realistic web-search latency distribution (offline fallback only).
    time.sleep(random.uniform(0.5, 3.0))
    return docs


def web_search(query: str, max_results: Any = 3) -> str:
    limit = int(max_results)
    mode = "bing"
    try:
        results = _bing_search(query, limit)
        if not results:
            mode = "corpus_fallback"
            results = _corpus_search(query, limit)
    except Exception:
        # Offline fallback: bundled corpus + simulated realistic latency.
        mode = "corpus_fallback"
        results = _corpus_search(query, limit)
    return json.dumps({"query": query, "mode": mode, "results": results}, ensure_ascii=False, sort_keys=True)


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
    {
        "name": "web_search",
        "description": "Search the web for a query and return top results (title, url, snippet).",
        "arguments": {"query": "string", "max_results": "number optional"},
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
    "web_search": web_search,
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
