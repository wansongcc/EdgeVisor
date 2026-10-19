#!/usr/bin/env python3
"""Experiment A: measure real preparation opportunity in agentic workloads.

The study runs three tool-backed workloads through an OpenAI-compatible
EdgeVisor API and records, per episode:

  * tool-window duration (tool call -> observation available),
  * generation duration and token counts,
  * the observed work (generated tokens) after each tool window,
  * the fraction of episode wall time spent in tools vs generation.

It deliberately uses real local tools (subprocess tests, BM25 retrieval, a
read-only SQLite database) instead of fixed sleeps. Coding solutions run in a
macOS sandbox on a private temp directory; on other platforms the runner is
still time- and CPU-limited but not sandboxed, which the manifest records.

This script is a measurement harness, not a paper-quality benchmark. It does
not decide task quality beyond the objective checks in the fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES = ROOT.parent / "opportunity_20261019" / "fixtures"
DEFAULT_OUT = ROOT.parent / "opportunity_20261019" / "runs"


# ---------------------------------------------------------------------------
# Task specifications
# ---------------------------------------------------------------------------

CODING_SPECS = {
    "coding_word_count": {
        "signature": "def word_count(text: str) -> dict[str, int]",
        "description": "Count lowercase [a-z0-9_]+ words. Return a dict of word -> count.",
    },
    "coding_top_k": {
        "signature": "def top_k(nums: list[int], k: int) -> list[int]",
        "description": "Return the k largest integers in descending order.",
    },
    "coding_matrix": {
        "signature": "def matmul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]",
        "description": "Multiply two square integer matrices.",
    },
    "coding_logparse": {
        "signature": "def parse_log(lines: list[str]) -> dict[str, int]",
        "description": "Lines look like 'LEVEL message'. Return level -> count.",
    },
    "coding_dedupe": {
        "signature": "def dedupe(items: list[int]) -> list[int]",
        "description": "Remove duplicates while preserving first-occurrence order.",
    },
    "coding_binsearch": {
        "signature": "def binsearch(arr: list[int], target: int) -> int",
        "description": "Return the index of target in a sorted list, or -1.",
    },
    "coding_fix_slug": {
        "signature": "def slugify(text: str) -> str",
        "description": "Lowercase, replace runs of non-alphanumerics with '-', strip leading/trailing '-'.",
    },
    "coding_fix_rle": {
        "signature": "def rle(text: str) -> str",
        "description": "Run-length encode: 'aaabb' -> 'a3b2'.",
    },
    "coding_fix_median": {
        "signature": "def median(nums: list[float]) -> float",
        "description": "Return the median; even-length input uses the mean of the two middle values.",
    },
}

SEARCH_SPECS = [
    {"id": "search_001", "region": "R000123"},
    {"id": "search_002", "region": "R004567"},
    {"id": "search_003", "region": "R010011"},
    {"id": "search_004", "region": "R022222"},
    {"id": "search_005", "region": "R033333"},
    {"id": "search_006", "region": "R044444"},
]

SQL_SPECS = [
    {"id": "sql_001", "question": "Which region has the highest total sales amount? Answer with the region only."},
    {"id": "sql_002", "question": "How many sales rows have day between 2025-03-01 and 2025-03-31 (inclusive)? Answer with the integer."},
    {"id": "sql_003", "question": "Which day has the highest total sales amount? Answer with YYYY-MM-DD."},
    {"id": "sql_004", "question": "Which region has the highest average sale amount (at least 100 rows)? Answer with the region only."},
    {"id": "sql_005", "question": "How many distinct regions are in the table? Answer with the integer."},
    {"id": "sql_006", "question": "What is the total sales amount for region R03? Answer with the number rounded to two decimals."},
]


MULTI_TASKS = [
    {
        "id": "coding_fix_slug", "workload": "coding", "allowed": {"run_tests"}, "min_tool_calls": 1,
        "prompt": ("The starter code below has a bug. Fix it, then call run_tests with "
                   "task='coding_fix_slug' and your corrected code. Iterate until both hidden tests pass, "
                   "then final_answer with 'done'.\n```python\n"
                   "import re\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower())\n```"),
    },
    {
        "id": "coding_fix_rle", "workload": "coding", "allowed": {"run_tests"}, "min_tool_calls": 1,
        "prompt": ("The starter code below has a bug. Fix it, then call run_tests with "
                   "task='coding_fix_rle' and your corrected code. Iterate until both hidden tests pass, "
                   "then final_answer with 'done'.\n```python\n"
                   "def rle(text):\n    if not text:\n        return ''\n    out=[]; count=1\n"
                   "    for i in range(1, len(text)):\n        if text[i] == text[i-1]:\n"
                   "            count += 1\n        else:\n            out.append(text[i-1]); out.append(str(count)); count=0\n"
                   "    out.append(text[-1]); out.append(str(count))\n    return ''.join(out)\n```"),
    },
    {
        "id": "coding_fix_median", "workload": "coding", "allowed": {"run_tests"}, "min_tool_calls": 1,
        "prompt": ("The starter code below has a bug. Fix it, then call run_tests with "
                   "task='coding_fix_median' and your corrected code. Iterate until both hidden tests pass, "
                   "then final_answer with 'done'.\n```python\n"
                   "def median(nums):\n    return nums[len(nums)//2]\n```"),
    },
    {
        "id": "sql_chain_region_day_count", "workload": "sql", "allowed": {"sql_query"}, "min_tool_calls": 3,
        "prompt": ("Use exactly three separate sql_query calls, in order. (1) Find the region with the highest "
                   "total sales amount and remember its region string. (2) For that region, find the day with the "
                   "highest total sales amount. (3) For that region and day, count the number of rows. Then "
                   "final_answer with 'region|day|count'."),
    },
    {
        "id": "sql_chain_avg_region_total", "workload": "sql", "allowed": {"sql_query"}, "min_tool_calls": 3,
        "prompt": ("Use exactly three sql_query calls. (1) Find the region with the highest average sale amount "
                   "among regions with at least 100 rows. (2) For that region, find the day with the highest total "
                   "sales amount. (3) Find the total sales amount for that region and day. Then final_answer with "
                   "'region|day|total'."),
    },
    {
        "id": "search_chain_top_region", "workload": "search", "allowed": {"sql_query", "search_docs"},
        "min_tool_calls": 2,
        "prompt": ("First use sql_query to find the region with the highest total sales amount. Then use "
                   "search_docs to find that region's capital. final_answer with the city name only."),
    },
    {
        "id": "search_chain_avg_region", "workload": "search", "allowed": {"sql_query", "search_docs"},
        "min_tool_calls": 2,
        "prompt": ("First use sql_query to find the region with the highest average sale amount among regions "
                   "with at least 100 rows. Then use search_docs to find that region's capital. final_answer "
                   "with the city name only."),
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUNNER = r'''
import importlib.util, json, sys, traceback
from pathlib import Path

base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("solution", base / "solution.py")
solution = importlib.util.module_from_spec(spec)
sys.modules["solution"] = solution
result = {"passed": 0, "failed": 0, "errors": []}
try:
    spec.loader.exec_module(solution)
except Exception as error:
    result["failed"] = 1
    result["errors"].append("import: " + type(error).__name__ + ": " + str(error))
    print(json.dumps(result))
    sys.exit(0)

spec_tests = importlib.util.spec_from_file_location("tests", base / "tests.py")
tests = importlib.util.module_from_spec(spec_tests)
sys.modules["tests"] = tests
try:
    spec_tests.loader.exec_module(tests)
except Exception as error:
    result["failed"] = 1
    result["errors"].append("tests: " + type(error).__name__ + ": " + str(error))
    print(json.dumps(result))
    sys.exit(0)

for name in sorted(dir(tests)):
    if not name.startswith("test_"):
        continue
    function = getattr(tests, name)
    if not callable(function):
        continue
    try:
        function(solution)
        result["passed"] += 1
    except Exception as error:
        result["failed"] += 1
        if len(result["errors"]) < 3:
            result["errors"].append(name + ": " + type(error).__name__ + ": " + str(error)[:300])
print(json.dumps(result))
'''


def _coding_tests(task: str) -> str:
    if task == "coding_word_count":
        return r'''
import re

def test_small(solution):
    assert solution.word_count("Alpha beta_2 GAMMA alpha") == {"alpha": 2, "beta_2": 1, "gamma": 1}

def test_large(solution):
    text = ("Alpha beta gamma delta. " * 100000) + "omega"
    counts = solution.word_count(text)
    assert counts["alpha"] == 100000, counts.get("alpha")
    assert counts["omega"] == 1
'''
    if task == "coding_top_k":
        return r'''
import random

def test_small(solution):
    assert solution.top_k([3, 1, 4, 1, 5, 9, 2, 6], 3) == [9, 6, 5]

def test_large(solution):
    rng = random.Random(7)
    nums = [rng.randrange(-1000000, 1000000) for _ in range(1000000)]
    expected = sorted(nums, reverse=True)[:100]
    assert solution.top_k(nums, 100) == expected
'''
    if task == "coding_matrix":
        return r'''
import random

def test_small(solution):
    assert solution.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]

def test_large(solution):
    rng = random.Random(11)
    n = 180
    a = [[rng.randrange(10) for _ in range(n)] for _ in range(n)]
    b = [[rng.randrange(10) for _ in range(n)] for _ in range(n)]
    expected = [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    assert solution.matmul(a, b) == expected
'''
    if task == "coding_logparse":
        return r'''
def test_small(solution):
    assert solution.parse_log(["INFO a", "WARN b", "INFO c", "ERROR d"]) == {"INFO": 2, "WARN": 1, "ERROR": 1}

def test_large(solution):
    lines = ["INFO message"] * 400000 + ["WARN message"] * 200000 + ["ERROR message"] * 100000
    counts = solution.parse_log(lines)
    assert counts == {"INFO": 400000, "WARN": 200000, "ERROR": 100000}, counts
'''
    if task == "coding_dedupe":
        return r'''
import random

def test_small(solution):
    assert solution.dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]

def test_large(solution):
    rng = random.Random(13)
    items = [rng.randrange(50000) for _ in range(1000000)]
    expected = list(dict.fromkeys(items))
    assert solution.dedupe(items) == expected
'''
    if task == "coding_binsearch":
        return r'''
def test_small(solution):
    assert solution.binsearch([1, 3, 5, 7], 5) == 2
    assert solution.binsearch([1, 3, 5, 7], 4) == -1

def test_large(solution):
    arr = list(range(0, 8000000, 2))
    for target in range(0, 8000000, 4000):
        index = solution.binsearch(arr, target)
        assert index == target // 2, (target, index)
    assert solution.binsearch(arr, 1) == -1
'''
    if task == "coding_fix_slug":
        return r'''
def test_basic(solution):
    assert solution.slugify(" Hello, World! ") == "hello-world"

def test_runs(solution):
    assert solution.slugify("A--B__C") == "a-b-c"
    assert solution.slugify("---") == ""
'''
    if task == "coding_fix_rle":
        return r'''
def test_basic(solution):
    assert solution.rle("aaabb") == "a3b2"
    assert solution.rle("abc") == "a1b1c1"
    assert solution.rle("") == ""

def test_runs(solution):
    assert solution.rle("zzzzzzzzzzzz") == "z12"
'''
    if task == "coding_fix_median":
        return r'''
def test_odd(solution):
    assert solution.median([3, 1, 2]) == 2

def test_even(solution):
    assert solution.median([4, 1, 3, 2]) == 2.5
    assert solution.median([5]) == 5
'''
    raise KeyError(task)


def prepare_coding(root: Path) -> dict:
    coding = root / "coding"
    coding.mkdir(parents=True, exist_ok=True)
    (coding / "runner.py").write_text(RUNNER, encoding="utf-8")
    manifest = {}
    for task, spec in CODING_SPECS.items():
        folder = coding / task
        folder.mkdir(exist_ok=True)
        (folder / "tests.py").write_text(_coding_tests(task), encoding="utf-8")
        (folder / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        manifest[task] = {"tests_sha256": _sha256(folder / "tests.py")}
    return manifest


def prepare_search(root: Path, documents: int = 120000) -> dict:
    search = root / "search"
    search.mkdir(parents=True, exist_ok=True)
    docs_path = search / "docs.pkl"
    index_path = search / "index.pkl"
    if docs_path.exists() and index_path.exists():
        return {"documents": documents, "docs_sha256": _sha256(docs_path)}
    rng = random.Random(23)
    docs = []
    for i in range(documents):
        region = f"R{i % 50000:06d}"
        city = f"City_{i % 50000:06d}"
        docs.append({
            "id": i,
            "region": region,
            "text": (f"Document {i}. The capital of region {region} is {city}. "
                     f"Region {region} is known for its port, its annual festival, and a population of "
                     f"{1000 + (i * 37) % 900000}. The local council of {region} meets in {city}."),
        })
    index: dict[str, list[int]] = defaultdict(list)
    lengths = []
    for doc in docs:
        tokens = re.findall(r"[a-z0-9_]+", doc["text"].lower())
        lengths.append(len(tokens))
        for token in set(tokens):
            index[token].append(doc["id"])
    payload = {"docs": docs, "index": dict(index), "lengths": lengths,
               "avg_len": sum(lengths) / max(1, len(lengths))}
    with docs_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    index_path.write_bytes(b"present")
    return {"documents": len(docs), "docs_sha256": _sha256(docs_path)}


def prepare_sql(root: Path, rows: int = 1000000) -> dict:
    sql = root / "sql"
    sql.mkdir(parents=True, exist_ok=True)
    db_path = sql / "sales.db"
    if db_path.exists():
        return {"rows": rows, "db_sha256": _sha256(db_path)}
    rng = random.Random(31)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount REAL, day TEXT)")
    regions = [f"R{i:06d}" for i in range(20)]
    batch = []
    for i in range(rows):
        day = f"2025-{(i * 7) % 12 + 1:02d}-{(i * 13) % 28 + 1:02d}"
        batch.append((i, regions[(i * 17) % len(regions)], round(rng.random() * 1000, 2), day))
        if len(batch) >= 50000:
            connection.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", batch)
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", batch)
    connection.execute("CREATE INDEX sales_day ON sales(day)")
    connection.execute("CREATE INDEX sales_region ON sales(region)")
    connection.commit()
    connection.close()
    return {"rows": rows, "db_sha256": _sha256(db_path)}


def prepare_all(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"platform": platform.platform(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "coding": prepare_coding(root), "search": prepare_search(root), "sql": prepare_sql(root)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_SEARCH_CACHE: dict[str, Any] | None = None


def _fixtures_root() -> Path:
    return Path(os.environ.get("EDGEVISOR_OPPORTUNITY_ROOT", str(DEFAULT_FIXTURES))).expanduser()


def _sandbox_profile(temp: Path) -> str:
    return ("(version 1)(allow default)(deny network*)(deny file-write*)"
            f'(allow file-write* (subpath "{temp}"))')


def run_tests(task: str, code: str, timeout_s: float = 12.0) -> str:
    if task not in CODING_SPECS:
        raise ValueError("unknown coding task: " + str(task))
    if not isinstance(code, str) or len(code) > 40000:
        raise ValueError("code must be a string under 40000 chars")
    root = _fixtures_root() / "coding" / task
    tests = root / "tests.py"
    if not tests.exists():
        raise RuntimeError("coding fixture missing: " + str(tests))
    temp = Path(tempfile.mkdtemp(prefix="edgevisor_opp_", dir="/private/tmp" if platform.system() == "Darwin" else None))
    try:
        (temp / "solution.py").write_text(code, encoding="utf-8")
        shutil.copyfile(tests, temp / "tests.py")
        shutil.copyfile(_fixtures_root() / "coding" / "runner.py", temp / "runner.py")
        command = [sys.executable, "-I", "-B", "runner.py"]
        if platform.system() == "Darwin":
            command = ["sandbox-exec", "-p", _sandbox_profile(temp)] + command
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=temp, capture_output=True, text=True,
                                       timeout=timeout_s, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return json.dumps({"task": task, "passed": 0, "failed": 1, "errors": ["tool timeout"],
                               "duration_ms": round((time.perf_counter() - started) * 1000, 2)})
        elapsed = (time.perf_counter() - started) * 1000
        output = completed.stdout.strip().splitlines()
        payload: dict[str, Any]
        try:
            payload = json.loads(output[-1]) if output else {"passed": 0, "failed": 1, "errors": ["no output"]}
        except json.JSONDecodeError:
            payload = {"passed": 0, "failed": 1, "errors": [completed.stdout[-500:] + completed.stderr[-500:]]}
        payload.update({"task": task, "duration_ms": round(elapsed, 2), "returncode": completed.returncode})
        return json.dumps(payload, ensure_ascii=False)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def search_docs(query: str, top_k: int = 3) -> str:
    global _SEARCH_CACHE
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("query must be a nonempty string under 500 chars")
    if _SEARCH_CACHE is None:
        path = _fixtures_root() / "search" / "docs.pkl"
        if not path.exists():
            raise RuntimeError("search fixture missing: " + str(path))
        with path.open("rb") as fh:
            _SEARCH_CACHE = pickle.load(fh)
    docs = _SEARCH_CACHE["docs"]
    index = _SEARCH_CACHE["index"]
    lengths = _SEARCH_CACHE["lengths"]
    avg_len = _SEARCH_CACHE["avg_len"]
    tokens = re.findall(r"[a-z0-9_]+", query.lower())
    if not tokens:
        return json.dumps({"query": query, "results": []})
    candidates: set[int] = set()
    for token in tokens:
        candidates.update(index.get(token, ()))
    if not candidates:
        return json.dumps({"query": query, "results": []})
    k1, b = 1.5, 0.75
    total = len(docs)
    scored = []
    for doc_id in candidates:
        doc_tokens = Counter(re.findall(r"[a-z0-9_]+", docs[doc_id]["text"].lower()))
        score = 0.0
        for token in tokens:
            tf = doc_tokens.get(token, 0)
            if not tf:
                continue
            df = len(index.get(token, ()))
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * lengths[doc_id] / avg_len))
        scored.append((score, doc_id))
    scored.sort(reverse=True)
    results = [{"id": docs[doc_id]["id"], "region": docs[doc_id]["region"],
                "text": docs[doc_id]["text"][:400], "score": round(score, 4)}
               for score, doc_id in scored[:top_k]]
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


_SQL_TIMEOUT_MS = 5000


def sql_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise ValueError("query must be a nonempty string under 4000 chars")
    path = _fixtures_root() / "sql" / "sales.db"
    if not path.exists():
        raise RuntimeError("sql fixture missing: " + str(path))
    connection = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
    deadline = time.perf_counter() + _SQL_TIMEOUT_MS / 1000.0
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
        cursor = connection.execute(query)
        columns = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(20)
        return json.dumps({"columns": columns, "rows": [list(row) for row in rows], "truncated": len(rows) == 20},
                          ensure_ascii=False, default=str)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# API client and agent loop
# ---------------------------------------------------------------------------

class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, url: str, timeout_s: float = 180.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def chat(self, messages: list[dict], max_tokens: int, temperature: float = 0.0, seed: int = 42) -> dict:
        body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "seed": seed, "stream": False}).encode("utf-8")
        request = urllib.request.Request(self.url + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            elapsed = (time.perf_counter() - started) * 1000
            raise ApiError(json.dumps({"status": error.code, "body": raw[:1000], "duration_ms": round(elapsed, 2)}))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            elapsed = (time.perf_counter() - started) * 1000
            raise ApiError(json.dumps({"status": 0, "body": str(error), "duration_ms": round(elapsed, 2)}))
        elapsed = (time.perf_counter() - started) * 1000
        payload = json.loads(raw)
        choice = (payload.get("choices") or [{}])[0]
        return {"content": (choice.get("message") or {}).get("content", ""),
                "finish_reason": choice.get("finish_reason"),
                "edgevisor": payload.get("edgevisor", {}), "usage": payload.get("usage", {}),
                "duration_ms": round(elapsed, 2), "raw": payload}


def parse_action(text: str, allowed: set[str]) -> dict:
    # Qwen3's chat template may prepend an empty or truncated think block.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("<think>", "").replace("</think>", "").strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("action must be an object")
    extra = set(payload) - {"tool", "arguments", "reason"}
    if extra:
        raise ValueError("unexpected keys: " + ",".join(sorted(extra)))
    tool, arguments = payload.get("tool"), payload.get("arguments")
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        raise ValueError("tool and arguments are required")
    if tool != "final_answer" and tool not in allowed:
        raise ValueError("tool is not available: " + tool)
    return payload


def _call_tool(tool: str, arguments: dict) -> str:
    if tool == "run_tests":
        return run_tests(str(arguments.get("task", "")), str(arguments.get("code", "")))
    if tool == "search_docs":
        return search_docs(str(arguments.get("query", "")))
    if tool == "sql_query":
        return sql_query(str(arguments.get("query", "")))
    raise ValueError("unknown tool: " + tool)


TOOL_DESCRIPTIONS = [
    {"name": "run_tests", "arguments": {"task": "coding task id", "code": "complete Python source for the solution"},
     "description": "Run the hidden tests for a coding task against your solution."},
    {"name": "search_docs", "arguments": {"query": "search query"},
     "description": "Search the local document corpus. Returns the top three documents."},
    {"name": "sql_query", "arguments": {"query": "read-only SQLite SELECT query"},
     "description": "Run a read-only SQL query against table sales(id, region, amount, day)."},
]


def system_prompt(allowed: set[str]) -> str:
    tools = [item for item in TOOL_DESCRIPTIONS if item["name"] in allowed]
    return ("You are an agent solving a task with tools. Return exactly one JSON object per turn, no prose. "
            "Format: {\"tool\":\"name\",\"arguments\":{...}} or "
            "{\"tool\":\"final_answer\",\"arguments\":{\"answer\":\"...\"}}. "
            "Use observations, do not invent tool results. When the task is done, return final_answer. "
            "Available tools: " + json.dumps(tools, ensure_ascii=False) + " /no_think")


def task_specs(fixtures: Path, workloads: list[str], limit: int, suite: str = "base") -> list[dict]:
    if suite == "multi":
        tasks = [task for task in MULTI_TASKS if task["workload"] in workloads]
        return tasks[:limit]
    tasks = []
    if "coding" in workloads:
        for task_id, spec in CODING_SPECS.items():
            tasks.append({"id": task_id, "workload": "coding", "allowed": {"run_tests"},
                          "prompt": (f"Task: implement `{spec['signature']}`. {spec['description']} "
                                     f"Call run_tests with task='{task_id}' and code=<your complete Python source>. "
                                     "You must call run_tests at least once before final_answer. "
                                     "Iterate on failures. When all tests pass, return final_answer with 'done'.")})
    if "search" in workloads:
        for spec in SEARCH_SPECS:
            tasks.append({"id": spec["id"], "workload": "search", "allowed": {"search_docs"},
                          "prompt": (f"Use search_docs to find the capital of region {spec['region']}. "
                                     "Return final_answer with only the city name.")})
    if "sql" in workloads:
        for spec in SQL_SPECS:
            tasks.append({"id": spec["id"], "workload": "sql", "allowed": {"sql_query"},
                          "prompt": (f"Table sales(id INTEGER, region TEXT, amount REAL, day TEXT) has rows for 2025. "
                                     f"{spec['question']} Use sql_query and return final_answer.")})
    by_workload = defaultdict(list)
    for task in tasks:
        by_workload[task["workload"]].append(task)
    selected = []
    for workload in workloads:
        selected.extend(by_workload[workload][:limit])
    return selected


def run_episode(task: dict, client: ApiClient, out_dir: Path, max_turns: int, max_tokens: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": system_prompt(task["allowed"])},
                {"role": "user", "content": task["prompt"]}]
    events: list[dict] = []
    final_answer = ""
    last_signature = None
    tool_calls = 0
    started = time.perf_counter()
    for turn in range(1, max_turns + 1):
        generation_started = time.perf_counter()
        try:
            response = client.chat(messages, max_tokens=max_tokens)
        except ApiError as error:
            events.append({"type": "api_error", "turn": turn, "error": str(error),
                           "duration_ms": round((time.perf_counter() - generation_started) * 1000, 2)})
            break
        generation_finished = time.perf_counter()
        content = response["content"]
        events.append({"type": "generation", "turn": turn, "content": content,
                       "duration_ms": response["duration_ms"],
                       "finish_reason": response["finish_reason"],
                       "edgevisor": response["edgevisor"], "usage": response["usage"],
                       "start_offset_ms": round((generation_started - started) * 1000, 2),
                       "end_offset_ms": round((generation_finished - started) * 1000, 2)})
        messages.append({"role": "assistant", "content": content})
        try:
            action = parse_action(content, task["allowed"])
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            events.append({"type": "parse_error", "turn": turn, "error": str(error)})
            messages.append({"role": "user", "content": "Invalid action JSON. Return one complete action object."})
            continue
        if action["tool"] == "final_answer":
            required_calls = int(task.get("min_tool_calls", 1))
            if task["allowed"] and tool_calls < required_calls:
                events.append({"type": "premature_final", "turn": turn,
                               "tool_calls": tool_calls, "required": required_calls})
                messages.append({"role": "user", "content":
                    f"You have used {tool_calls} of {required_calls} required tool calls. "
                    "Continue with the next required step before final_answer."})
                continue
            final_answer = str(action["arguments"].get("answer", ""))
            events.append({"type": "final_answer", "turn": turn, "answer": final_answer})
            break
        signature = json.dumps([action["tool"], action["arguments"]], sort_keys=True)
        if signature == last_signature:
            events.append({"type": "repeated_action", "turn": turn, "tool": action["tool"]})
            messages.append({"role": "user", "content":
                "You already received this exact observation. Do not repeat it; use the observation and return final_answer."})
            continue
        last_signature = signature
        tool_calls += 1
        tool_started = time.perf_counter()
        try:
            result = _call_tool(action["tool"], action["arguments"])
            tool_error = None
        except Exception as error:  # tool failures are observations, not crashes
            result = json.dumps({"error": str(error)})
            tool_error = str(error)
        tool_finished = time.perf_counter()
        events.append({"type": "tool", "turn": turn, "name": action["tool"],
                       "arguments": action["arguments"], "result": result, "error": tool_error,
                       "duration_ms": round((tool_finished - tool_started) * 1000, 2),
                       "start_offset_ms": round((tool_started - started) * 1000, 2),
                       "end_offset_ms": round((tool_finished - started) * 1000, 2)})
        messages.append({"role": "user",
                         "content": "Tool observation: " + json.dumps({"name": action["tool"], "result": result},
                                                                     ensure_ascii=False)})
    episode = {"id": task["id"], "workload": task["workload"], "prompt": task["prompt"],
               "max_turns": max_turns, "max_tokens": max_tokens,
               "events": events, "messages": messages, "final_answer": final_answer,
               "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
    (out_dir / "episode.json").write_text(json.dumps(episode, indent=2, ensure_ascii=False), encoding="utf-8")
    return episode


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def analyze(episodes: list[dict]) -> dict:
    per_workload: dict[str, list[dict]] = defaultdict(list)
    tool_windows: dict[str, list[float]] = defaultdict(list)
    generation_windows: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        workload = episode["workload"]
        tools = [event for event in episode["events"] if event["type"] == "tool"]
        generations = [event for event in episode["events"] if event["type"] == "generation"]
        tool_time = sum(event["duration_ms"] for event in tools)
        generation_time = sum(event["duration_ms"] for event in generations)
        episode_time = max(1.0, episode["duration_ms"])
        # Generated tokens observed after each tool call, plus tokens per turn.
        for index, event in enumerate(tools):
            later = [gen for gen in generations if gen["turn"] > event["turn"]]
            event["observed_tokens_after"] = sum(int((gen.get("usage") or {}).get("completion_tokens", 0)) for gen in later)
            tool_windows[f"{workload}:{event['name']}"].append(event["duration_ms"])
            tool_windows[workload].append(event["duration_ms"])
        for event in generations:
            generation_windows[workload].append(event["duration_ms"])
        per_workload[workload].append({
            "id": episode["id"], "turns": len(generations), "tool_calls": len(tools),
            "tool_time_ms": round(tool_time, 2), "generation_time_ms": round(generation_time, 2),
            "episode_ms": round(episode_time, 2),
            "tool_fraction": round(tool_time / episode_time, 4),
            "generation_fraction": round(generation_time / episode_time, 4),
            "final_answer": episode["final_answer"],
            "completed": bool(episode["final_answer"]),
        })
    summary = {"episodes": len(episodes), "by_workload": {}, "tool_windows_ms": {}, "generation_ms": {}}
    for workload, rows in per_workload.items():
        tool_values = tool_windows.get(workload, [])
        summary["by_workload"][workload] = {
            "episodes": len(rows),
            "completed": sum(1 for row in rows if row["completed"]),
            "tool_calls": sum(row["tool_calls"] for row in rows),
            "tool_fraction_median": percentile([row["tool_fraction"] for row in rows], 0.5),
            "tool_time_ms_median": percentile([row["tool_time_ms"] for row in rows], 0.5),
            "episode_ms_median": percentile([row["episode_ms"] for row in rows], 0.5),
        }
        summary["tool_windows_ms"][workload] = {
            "n": len(tool_values), "p50": percentile(tool_values, 0.5),
            "p95": percentile(tool_values, 0.95), "max": max(tool_values) if tool_values else None,
            "mean": sum(tool_values) / len(tool_values) if tool_values else None,
        }
        summary["generation_ms"][workload] = {
            "n": len(generation_windows[workload]),
            "p50": percentile(generation_windows[workload], 0.5),
            "p95": percentile(generation_windows[workload], 0.95),
        }
    for key, values in sorted(tool_windows.items()):
        if ":" in key and values:
            summary["tool_windows_ms"][key] = {"n": len(values), "p50": percentile(values, 0.5),
                                                "p95": percentile(values, 0.95), "max": max(values)}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:19690/v1")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "pilot")
    parser.add_argument("--workloads", default="coding,search,sql")
    parser.add_argument("--limit", type=int, default=6, help="tasks per workload")
    parser.add_argument("--suite", choices=["base", "multi"], default="base")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--prepare-fixtures", action="store_true")
    args = parser.parse_args()

    os.environ["EDGEVISOR_OPPORTUNITY_ROOT"] = str(args.fixtures)
    if args.prepare_fixtures:
        manifest = prepare_all(args.fixtures)
        print(json.dumps(manifest, indent=2))
        return 0
    if not (args.fixtures / "manifest.json").exists():
        raise SystemExit("fixtures missing; run with --prepare-fixtures")
    workloads = [item.strip() for item in args.workloads.split(",") if item.strip()]
    tasks = task_specs(args.fixtures, workloads, args.limit, args.suite)
    args.out.mkdir(parents=True, exist_ok=True)
    client = ApiClient(args.api_url, args.timeout)
    episodes = []
    for task in tasks:
        print("run", task["id"], flush=True)
        episode = run_episode(task, client, args.out / task["id"], args.max_turns, args.max_tokens)
        episodes.append(episode)
        print("  turns", len([e for e in episode["events"] if e["type"] == "generation"]),
              "tools", len([e for e in episode["events"] if e["type"] == "tool"]),
              "final", bool(episode["final_answer"]), flush=True)
    summary = analyze(episodes)
    summary["tasks"] = [{"id": task["id"], "workload": task["workload"]} for task in tasks]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
