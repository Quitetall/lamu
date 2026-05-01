"""
Agentic benchmark runner — compare Opus solo vs swarm on coding tasks.

Three modes:
  1. swebench   — SWE-bench Lite (300 real GitHub issues with test suites)
  2. custom     — tasks from your own repos (provide a tasks.json)
  3. builtin    — hand-crafted tasks covering common patterns

Usage:
  python -m agents.bench run --suite builtin --config opus-solo
  python -m agents.bench run --suite builtin --config swarm
  python -m agents.bench compare results/opus-solo/ results/swarm/
  python -m agents.bench run --suite custom --tasks path/to/tasks.json
  python -m agents.bench run --suite swebench --limit 10

Task format (tasks.json):
  [
    {
      "id": "fix-auth-001",
      "description": "Fix the JWT token expiration check in auth.py",
      "repo": "https://github.com/user/repo.git",
      "commit": "abc123",
      "test_cmd": "python -m pytest tests/test_auth.py -v",
      "expected": "pass"
    }
  ]
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

RESULTS_DIR = Path(__file__).parent / "bench_results"

# ── Built-in task suite ──────────────────────────────────────────────────

BUILTIN_TASKS = [
    {
        "id": "calc-001",
        "description": "Create a Python module `calculator.py` with functions add, subtract, multiply, divide. Division by zero should raise ValueError.",
        "setup": "mkdir -p src tests",
        "setup_files": {
            "tests/test_calculator.py": '''import pytest
from src.calculator import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0.1, 0.2) == pytest.approx(0.3)

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 0) == 0

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(-2, 3) == -6
    assert multiply(0, 100) == 0

def test_divide():
    assert divide(10, 2) == 5
    assert divide(7, 2) == 3.5
    with pytest.raises(ValueError):
        divide(1, 0)
''',
            "src/__init__.py": "",
        },
        "test_cmd": "python -m pytest tests/test_calculator.py -v",
    },
    {
        "id": "stack-002",
        "description": "Implement a Stack class in `src/stack.py` with push, pop, peek, is_empty, and size methods. Pop and peek on empty stack should raise IndexError.",
        "setup": "mkdir -p src tests",
        "setup_files": {
            "tests/test_stack.py": '''import pytest
from src.stack import Stack

def test_push_pop():
    s = Stack()
    s.push(1)
    s.push(2)
    assert s.pop() == 2
    assert s.pop() == 1

def test_peek():
    s = Stack()
    s.push(42)
    assert s.peek() == 42
    assert s.size() == 1

def test_is_empty():
    s = Stack()
    assert s.is_empty()
    s.push(1)
    assert not s.is_empty()

def test_size():
    s = Stack()
    assert s.size() == 0
    s.push(1)
    s.push(2)
    assert s.size() == 2

def test_empty_pop():
    s = Stack()
    with pytest.raises(IndexError):
        s.pop()

def test_empty_peek():
    s = Stack()
    with pytest.raises(IndexError):
        s.peek()
''',
            "src/__init__.py": "",
        },
        "test_cmd": "python -m pytest tests/test_stack.py -v",
    },
    {
        "id": "csv-003",
        "description": "Create `src/csv_stats.py` with a function `summarize(path)` that reads a CSV file and returns a dict with keys: row_count, column_names, numeric_means (dict of column->mean for numeric columns).",
        "setup": "mkdir -p src tests data",
        "setup_files": {
            "data/sample.csv": "name,age,score\nAlice,30,85.5\nBob,25,92.0\nCharlie,35,78.5\n",
            "tests/test_csv_stats.py": '''import pytest
from src.csv_stats import summarize

def test_summarize():
    result = summarize("data/sample.csv")
    assert result["row_count"] == 3
    assert result["column_names"] == ["name", "age", "score"]
    assert result["numeric_means"]["age"] == pytest.approx(30.0)
    assert result["numeric_means"]["score"] == pytest.approx(85.333, rel=1e-2)
    assert "name" not in result["numeric_means"]
''',
            "src/__init__.py": "",
        },
        "test_cmd": "python -m pytest tests/test_csv_stats.py -v",
    },
    {
        "id": "api-004",
        "description": "Create `src/api.py` — a FastAPI app with: GET /health returning {\"status\":\"ok\"}, POST /echo that returns the JSON body it receives, GET /fibonacci/{n} returning {\"result\": fib(n)}. Fibonacci of 0=0, 1=1.",
        "setup": "mkdir -p src tests",
        "setup_files": {
            "tests/test_api.py": '''import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_echo():
    payload = {"message": "hello", "count": 3}
    r = client.post("/echo", json=payload)
    assert r.status_code == 200
    assert r.json() == payload

def test_fibonacci():
    assert client.get("/fibonacci/0").json() == {"result": 0}
    assert client.get("/fibonacci/1").json() == {"result": 1}
    assert client.get("/fibonacci/10").json() == {"result": 55}
    assert client.get("/fibonacci/20").json() == {"result": 6765}
''',
            "src/__init__.py": "",
        },
        "test_cmd": "python -m pytest tests/test_api.py -v",
    },
    {
        "id": "linked-list-005",
        "description": "Implement a singly linked list in `src/linked_list.py` with: append, prepend, delete(value), find(value)->bool, to_list()->list, reverse(), and __len__.",
        "setup": "mkdir -p src tests",
        "setup_files": {
            "tests/test_linked_list.py": '''from src.linked_list import LinkedList

def test_append_and_to_list():
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    assert ll.to_list() == [1, 2, 3]

def test_prepend():
    ll = LinkedList()
    ll.prepend(3)
    ll.prepend(2)
    ll.prepend(1)
    assert ll.to_list() == [1, 2, 3]

def test_delete():
    ll = LinkedList()
    for v in [1, 2, 3, 2, 4]:
        ll.append(v)
    ll.delete(2)
    assert ll.to_list() == [1, 3, 2, 4]

def test_find():
    ll = LinkedList()
    ll.append(10)
    ll.append(20)
    assert ll.find(10) is True
    assert ll.find(30) is False

def test_len():
    ll = LinkedList()
    assert len(ll) == 0
    ll.append(1)
    ll.append(2)
    assert len(ll) == 2

def test_reverse():
    ll = LinkedList()
    for v in [1, 2, 3, 4]:
        ll.append(v)
    ll.reverse()
    assert ll.to_list() == [4, 3, 2, 1]

def test_delete_head():
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.delete(1)
    assert ll.to_list() == [2]
''',
            "src/__init__.py": "",
        },
        "test_cmd": "python -m pytest tests/test_linked_list.py -v",
    },
    {
        "id": "lru-cache-006",
        "description": "Implement an LRU Cache class in `src/lru_cache.py` with get(key), put(key, value), and a capacity limit. get/put must be O(1).",
        "setup": "mkdir -p src tests",
        "setup_files": {
            "tests/test_lru_cache.py": '''from src.lru_cache import LRUCache

def test_basic():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)  # evicts key 2
    assert cache.get(2) == -1
    cache.put(4, 4)  # evicts key 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4

def test_update():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(1, 10)  # update, key 1 is now most recent
    cache.put(3, 3)   # evicts key 2 (not 1)
    assert cache.get(2) == -1
    assert cache.get(1) == 10

def test_single_capacity():
    cache = LRUCache(1)
    cache.put(1, 1)
    assert cache.get(1) == 1
    cache.put(2, 2)
    assert cache.get(1) == -1
    assert cache.get(2) == 2
''',
            "src/__init__.py": "",
        },
        "test_cmd": "python -m pytest tests/test_lru_cache.py -v",
    },
    {
        "id": "json-parser-007",
        "description": "Create `src/json_parser.py` with a function `parse(s)` that parses a JSON string into Python objects. Support strings, numbers, booleans, null, arrays, and objects. No imports allowed (implement from scratch).",
        "setup": "mkdir -p src tests",
        "setup_files": {
            "tests/test_json_parser.py": '''from src.json_parser import parse

def test_string():
    assert parse('"hello"') == "hello"
    assert parse('"with \\\\"escapes\\\\""') == 'with "escapes"'

def test_number():
    assert parse("42") == 42
    assert parse("3.14") == 3.14
    assert parse("-7") == -7

def test_bool_null():
    assert parse("true") is True
    assert parse("false") is False
    assert parse("null") is None

def test_array():
    assert parse("[1, 2, 3]") == [1, 2, 3]
    assert parse("[]") == []
    assert parse('[1, "two", true, null]') == [1, "two", True, None]

def test_object():
    assert parse('{"a": 1}') == {"a": 1}
    assert parse('{}') == {}
    result = parse('{"name": "test", "values": [1, 2], "nested": {"x": true}}')
    assert result["name"] == "test"
    assert result["values"] == [1, 2]
    assert result["nested"]["x"] is True
''',
            "src/__init__.py": "",
        },
        "test_cmd": "python -m pytest tests/test_json_parser.py -v",
    },
    {
        "id": "binary-search-tree-008",
        "description": "Implement a Binary Search Tree in `src/bst.py` with insert, search, delete, inorder traversal (returns sorted list), min, and max.",
        "setup": "mkdir -p src tests",
        "setup_files": {
            "tests/test_bst.py": '''from src.bst import BST

def test_insert_search():
    tree = BST()
    for v in [5, 3, 7, 1, 4]:
        tree.insert(v)
    assert tree.search(3) is True
    assert tree.search(6) is False

def test_inorder():
    tree = BST()
    for v in [5, 3, 7, 1, 4, 6, 8]:
        tree.insert(v)
    assert tree.inorder() == [1, 3, 4, 5, 6, 7, 8]

def test_min_max():
    tree = BST()
    for v in [5, 3, 7, 1, 9]:
        tree.insert(v)
    assert tree.min() == 1
    assert tree.max() == 9

def test_delete_leaf():
    tree = BST()
    for v in [5, 3, 7]:
        tree.insert(v)
    tree.delete(3)
    assert tree.inorder() == [5, 7]

def test_delete_with_children():
    tree = BST()
    for v in [5, 3, 7, 1, 4]:
        tree.insert(v)
    tree.delete(3)
    assert tree.inorder() == [1, 4, 5, 7]

def test_delete_root():
    tree = BST()
    for v in [5, 3, 7]:
        tree.insert(v)
    tree.delete(5)
    assert tree.search(5) is False
    assert sorted(tree.inorder()) == [3, 7]
''',
            "src/__init__.py": "",
        },
        "test_cmd": "python -m pytest tests/test_bst.py -v",
    },
]


# ── SWE-bench integration ───────────────────────────────────────────────

def load_swebench_tasks(limit: int = 10) -> list[dict]:
    """Load SWE-bench Lite tasks. Requires swebench package."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return []

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    tasks = []
    for item in ds.select(range(min(limit, len(ds)))):
        tasks.append({
            "id": item["instance_id"],
            "description": f"Fix this issue in {item['repo']}:\n\n{item['problem_statement']}",
            "repo": item["repo"],
            "commit": item["base_commit"],
            "test_cmd": f"python -m pytest {item['test_patch']}",
            "swebench": True,
        })
    return tasks


# ── Runner ───────────────────────────────────────────────────────────────

async def run_task_opus_solo(task: dict, workdir: Path) -> dict:
    """Run a task with just Opus (cloud-only, no local workers)."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    bifrost_url = os.getenv("BIFROST_URL", "http://localhost:8080/v1")
    bifrost_key = os.getenv("BIFROST_KEY", "sk-local")
    model = os.getenv("PLANNER_MODEL", "anthropic/claude-opus-4-7")

    llm = ChatOpenAI(
        model=model,
        openai_api_base=bifrost_url,
        openai_api_key=bifrost_key,
        temperature=0.3,
        max_tokens=8192,
    )

    # Load context
    tree = subprocess.run(
        ["find", ".", "-name", "*.py", "-not", "-path", "*/__pycache__/*"],
        capture_output=True, text=True, cwd=str(workdir)
    ).stdout

    response = await llm.ainvoke([
        SystemMessage(content="You are an expert programmer. Implement exactly what is described. For each file, output:\n```path/to/file.py\n<complete content>\n```"),
        HumanMessage(content=f"Task: {task['description']}\n\nFile tree:\n{tree}"),
    ])

    # Parse and write files
    import re
    for m in re.finditer(r"```(\S+\.py)\n(.*?)```", response.content, re.DOTALL):
        path = workdir / m.group(1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(m.group(2))

    # Run tests
    result = subprocess.run(
        task["test_cmd"].split(),
        capture_output=True, text=True,
        cwd=str(workdir), timeout=60,
    )
    return {
        "passed": result.returncode == 0,
        "output": (result.stdout + result.stderr)[-3000:],
    }


async def run_task_swarm(task: dict, workdir: Path) -> dict:
    """Run a task with the full swarm (Opus planner + local workers)."""
    from agents.swarm import invoke_swarm

    result = await invoke_swarm(
        task=task["description"],
        repo_path=str(workdir),
        test_cmd=task["test_cmd"],
    )

    return {
        "passed": result.get("test_passed", False) and result.get("status") == "success",
        "output": result.get("test_output", "")[-3000:],
        "retries": result.get("retry_count", 0),
        "planner_loops": result.get("planner_loops", 0),
    }


async def run_suite(
    tasks: list[dict],
    config: str,
    run_name: Optional[str] = None,
) -> dict:
    """Run a benchmark suite with a given config."""
    run_name = run_name or f"{config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    passed = 0
    total = len(tasks)

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{total}] {task['id']}: {task['description'][:60]}...")

        # Create isolated workdir
        workdir = Path(tempfile.mkdtemp(prefix=f"bench_{task['id']}_"))

        # Set up test files if builtin task
        if "setup_files" in task:
            for path, content in task["setup_files"].items():
                fp = workdir / path
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content)

        if task.get("setup"):
            subprocess.run(task["setup"], shell=True, cwd=str(workdir))

        start = time.time()
        try:
            if config == "opus-solo":
                result = await run_task_opus_solo(task, workdir)
            elif config == "swarm":
                result = await run_task_swarm(task, workdir)
            else:
                raise ValueError(f"Unknown config: {config}")
        except Exception as e:
            result = {"passed": False, "output": f"Error: {e}"}

        elapsed = time.time() - start

        result["task_id"] = task["id"]
        result["elapsed"] = round(elapsed, 1)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passed += 1
        print(f"  {status} ({elapsed:.1f}s)")

    # Save results
    summary = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "results": results,
    }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*50}")
    print(f"  {config}: {passed}/{total} passed ({summary['pass_rate']}%)")
    print(f"  Results: {run_dir}/results.json")
    return summary


# ── Compare ──────────────────────────────────────────────────────────────

def compare(dir_a: str, dir_b: str):
    """Compare two benchmark runs side by side."""
    a = json.loads((Path(dir_a) / "results.json").read_text())
    b = json.loads((Path(dir_b) / "results.json").read_text())

    print(f"\n{'Task':<25} {'Config A':>12} {'Config B':>12}")
    print("-" * 50)

    a_map = {r["task_id"]: r for r in a["results"]}
    b_map = {r["task_id"]: r for r in b["results"]}

    all_ids = sorted(set(list(a_map.keys()) + list(b_map.keys())))
    for tid in all_ids:
        ra = a_map.get(tid, {})
        rb = b_map.get(tid, {})
        sa = "PASS" if ra.get("passed") else "FAIL" if ra else "-"
        sb = "PASS" if rb.get("passed") else "FAIL" if rb else "-"
        print(f"  {tid:<23} {sa:>12} {sb:>12}")

    print("-" * 50)
    print(f"  {'TOTAL':<23} {a['passed']}/{a['total']:>8} {b['passed']}/{b['total']:>8}")
    print(f"  {'Pass rate':<23} {a['pass_rate']:>11}% {b['pass_rate']:>11}%")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agentic benchmark: Opus solo vs swarm")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a benchmark suite")
    run_p.add_argument("--suite", choices=["builtin", "swebench", "custom"], default="builtin")
    run_p.add_argument("--config", choices=["opus-solo", "swarm"], required=True)
    run_p.add_argument("--tasks", help="Path to tasks.json (for custom suite)")
    run_p.add_argument("--limit", type=int, default=5, help="Max tasks for swebench")
    run_p.add_argument("--name", help="Run name (for results dir)")

    cmp_p = sub.add_parser("compare", help="Compare two benchmark runs")
    cmp_p.add_argument("dir_a", help="First results directory")
    cmp_p.add_argument("dir_b", help="Second results directory")

    sub.add_parser("list", help="List available builtin tasks")

    args = parser.parse_args()

    if args.command == "run":
        if args.suite == "builtin":
            tasks = BUILTIN_TASKS
        elif args.suite == "swebench":
            tasks = load_swebench_tasks(args.limit)
        elif args.suite == "custom":
            tasks = json.loads(Path(args.tasks).read_text())
        else:
            parser.error(f"Unknown suite: {args.suite}")

        asyncio.run(run_suite(tasks, args.config, args.name))

    elif args.command == "compare":
        compare(args.dir_a, args.dir_b)

    elif args.command == "list":
        for t in BUILTIN_TASKS:
            print(f"  {t['id']}: {t['description'][:70]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
