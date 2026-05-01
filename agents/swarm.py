"""
Agent swarm — cloud orchestrator + local workers + test-driven validation.

Architecture:
  PLANNER (Opus 4.7)        plans subtasks, assigns to workers
  WORKERS (local Qwen)      implement in parallel, free and fast
  INTEGRATOR (local Qwen)   merges parallel outputs
  TEST RUNNER (pytest)       deterministic arbiter of correctness
  CRITIC (Sonnet/GPT)       reviews passing code for quality
  TRAINER (local)           collects data, fine-tunes LoRA adapters

Graph topology:
  planner → workers → integrator → test_runner
    ↑ FAIL (retries left)  ←─────────┘  │
    ↑ FAIL (exhausted)     ←─────────┘  │
                                         ↓ PASS
                                       critic
    ↑ REJECT               ←────────────┘  │
                                            ↓ APPROVE
                                          commit → collect training data → END

Usage:
  python -m agents.swarm "Fix the auth bug" --repo /path/to/project
  python -m agents.swarm "Add pagination" --repo . --test "pytest tests/test_api.py -v"
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

load_dotenv(Path(__file__).parent / ".env")

# ── Config ───────────────────────────────────���──────────────────────────

BIFROST_URL = os.getenv("BIFROST_URL", "http://localhost:8080/v1")
BIFROST_KEY = os.getenv("BIFROST_KEY", "sk-local")

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "anthropic/claude-opus-4-7")
WORKER_MODEL = os.getenv("WORKER_MODEL", "qwen/qwen3.6-27b-uncensored")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "anthropic/claude-sonnet-4-6")

MAX_WORKER_RETRIES = 3
MAX_PLANNER_LOOPS = 2
CONTEXT_BUDGET = 400_000  # chars (~100k tokens — utilizes the 108-262K context window)


# ── State ────��──────────────────────────────────���───────────────────────

class SwarmState(TypedDict):
    # Input
    task: str
    repo_path: str
    test_cmd: str
    context: str

    # Plan
    plan: list[dict]
    integration_notes: str

    # Implementation
    implementations: list[dict]
    integrated_diff: str
    applied_files: dict[str, str]  # path → full content

    # Test
    test_passed: bool
    test_output: str
    retry_count: int

    # Review
    critic_approved: bool
    critic_feedback: str
    planner_loops: int

    # Outcome
    status: str  # "pending" | "success" | "failed"


# ── Context loader ─────────────���────────────────────────────────────────

def load_context(repo_path: str, task: str) -> str:
    """Build surgical context: file tree + keyword-relevant source files."""
    repo = Path(repo_path).resolve()

    # File tree (source only, skip noise)
    tree = subprocess.run(
        ["find", ".", "-type", "f",
         "(", "-name", "*.py", "-o", "-name", "*.js", "-o", "-name", "*.ts",
         "-o", "-name", "*.yaml", "-o", "-name", "*.toml", "-o", "-name", "*.json", ")",
         "-not", "-path", "*/.venv/*", "-not", "-path", "*/__pycache__/*",
         "-not", "-path", "*/node_modules/*", "-not", "-path", "*/.git/*"],
        capture_output=True, text=True, cwd=str(repo)
    ).stdout.strip()

    # Keyword search — pull files that mention task-relevant terms
    keywords = [w for w in task.lower().split() if len(w) > 3 and w.isalpha()]
    relevant = set()
    for kw in keywords:
        hits = subprocess.run(
            ["rg", "-l", "-i", kw, "--type", "py", "--type", "js", "--type", "ts"],
            capture_output=True, text=True, cwd=str(repo)
        )
        if hits.stdout.strip():
            relevant.update(hits.stdout.strip().split("\n"))

    # Always include test files
    tests = subprocess.run(
        ["find", ".", "-type", "f", "(", "-name", "test_*.py", "-o", "-name", "*_test.py", ")"],
        capture_output=True, text=True, cwd=str(repo)
    )
    if tests.stdout.strip():
        relevant.update(tests.stdout.strip().split("\n"))

    # Assemble within budget
    parts = [f"=== File Tree ===\n{tree}\n"]
    total = len(parts[0])

    for f in sorted(relevant):
        if not f:
            continue
        path = repo / f
        if not path.exists() or path.stat().st_size > 100_000:
            continue
        content = path.read_text(errors="replace")
        entry = f"\n=== {f} ===\n{content}\n"
        if total + len(entry) > CONTEXT_BUDGET:
            break
        parts.append(entry)
        total += len(entry)

    return "".join(parts)


# ── LLM factory ────────────��────────────────────────────────────────────

def strip_think(text: str) -> str:
    """Strip <think>...</think> blocks from model output."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def llm(model: str, temperature: float = 0.3, max_tokens: int = 4096) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_base=BIFROST_URL,
        openai_api_key=BIFROST_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ── Nodes ─────────────────────────────────────���─────────────────────────

PLANNER_SYSTEM = """You are a senior software architect directing implementation workers.

Given a task and codebase context, produce a plan. Workers have NO context beyond what you provide — be precise about which files to read and what to change.

Output ONLY valid JSON:
{
    "subtasks": [
        {
            "id": 1,
            "description": "Exact implementation instruction",
            "files_to_modify": ["path/to/file.py"],
            "files_to_read": ["path/to/related.py"],
            "acceptance_criteria": "What should work when done"
        }
    ],
    "integration_notes": "How to merge if subtasks overlap"
}"""

WORKER_SYSTEM = """You are an expert programmer. Implement exactly what is described.

For EACH file you modify, output a fenced block with the file path as the language tag:

```path/to/file.py
<complete file content — every line, not just changes>
```

Output the COMPLETE file, not a diff. Include ALL lines."""

INTEGRATOR_SYSTEM = """Merge these parallel implementations into one unified set of files.
Combine changes carefully when multiple subtasks touch the same file.

For EACH file, output:
```path/to/file.py
<complete merged file content>
```"""

CRITIC_SYSTEM = """You are a senior code reviewer. The implementation PASSED all tests.

Review for:
1. Correctness beyond what tests cover
2. Architecture and design quality
3. Edge cases, error handling, security
4. Whether the implementation matches the original task intent

Respond with ONLY valid JSON:
{"approved": true, "feedback": ""}
or
{"approved": false, "feedback": "Specific actionable feedback"}"""


async def planner_node(state: SwarmState) -> dict:
    """Opus reads the task + context and produces a structured plan."""
    model = llm(PLANNER_MODEL, temperature=0.3, max_tokens=8192)

    extra = ""
    if state.get("critic_feedback"):
        extra += f"\n\n--- Critic feedback (address these) ---\n{state['critic_feedback']}"
    if state.get("test_output") and not state.get("test_passed"):
        extra += f"\n\n--- Test failures ---\n{state['test_output'][-3000:]}"

    response = await model.ainvoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Task: {state['task']}\n\nCodebase:\n{state['context']}{extra}"),
    ])

    # Strip think blocks + markdown fences, extract JSON
    text = strip_think(response.content)
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    plan = json.loads(text)
    return {
        "plan": plan["subtasks"],
        "integration_notes": plan.get("integration_notes", ""),
        "implementations": [],
        "planner_loops": state.get("planner_loops", 0) + 1,
    }


async def worker_node(state: SwarmState) -> dict:
    """Local Qwen implements all subtasks in parallel."""
    model = llm(WORKER_MODEL, temperature=0.2, max_tokens=4096)
    repo = Path(state["repo_path"])

    async def run_subtask(subtask: dict) -> dict:
        # Load relevant file contents
        files = []
        for f in subtask.get("files_to_read", []) + subtask.get("files_to_modify", []):
            path = repo / f
            if path.exists():
                files.append(f"--- {f} ---\n{path.read_text(errors='replace')}")

        context = "\n\n".join(files) if files else "(no files loaded)"

        test_hint = ""
        if state.get("test_output") and not state.get("test_passed"):
            test_hint = f"\n\nFailing tests from last attempt:\n{state['test_output'][-2000:]}"

        response = await model.ainvoke([
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(content=f"Subtask: {subtask['description']}\n\nCurrent files:\n{context}{test_hint}"),
        ])

        return {"subtask_id": subtask["id"], "output": strip_think(response.content)}

    results = await asyncio.gather(*[run_subtask(s) for s in state["plan"]])
    return {"implementations": list(results)}


async def integrator_node(state: SwarmState) -> dict:
    """Merge parallel implementations. Skip if single subtask."""
    impls = state["implementations"]

    if len(impls) == 1:
        parsed = _parse_file_blocks(impls[0]["output"])
        return {"integrated_diff": impls[0]["output"], "applied_files": parsed}

    model = llm(WORKER_MODEL, temperature=0.1, max_tokens=8192)

    impl_text = "\n\n".join(
        f"=== Subtask {i['subtask_id']} ===\n{i['output']}" for i in impls
    )

    response = await model.ainvoke([
        SystemMessage(content=INTEGRATOR_SYSTEM),
        HumanMessage(content=f"Notes: {state['integration_notes']}\n\n{impl_text}"),
    ])

    clean = strip_think(response.content)
    parsed = _parse_file_blocks(clean)
    return {"integrated_diff": clean, "applied_files": parsed}


def _parse_file_blocks(text: str) -> dict[str, str]:
    """Extract ```path/to/file.ext\\ncontent\\n``` blocks."""
    files = {}
    pattern = r"```(\S+\.(?:py|js|ts|jsx|tsx|yaml|toml|json|sh|sql|css|html|md))\n(.*?)```"
    for m in re.finditer(pattern, text, re.DOTALL):
        files[m.group(1)] = m.group(2)
    return files


async def test_runner_node(state: SwarmState) -> dict:
    """Apply changes, run pytest. No LLM — deterministic."""
    repo = Path(state["repo_path"])

    # Write files
    for rel_path, content in state.get("applied_files", {}).items():
        full = repo / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)

    # Run tests
    cmd = state.get("test_cmd", "python -m pytest tests/ -v --tb=short")
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd.split(),
            capture_output=True, text=True,
            cwd=str(repo), timeout=300,
        )
        passed = result.returncode == 0
        output = (result.stdout + result.stderr)[-5000:]
    except subprocess.TimeoutExpired:
        passed = False
        output = "TEST TIMEOUT (300s)"

    return {
        "test_passed": passed,
        "test_output": output,
        "retry_count": state.get("retry_count", 0) + 1,
    }


async def critic_node(state: SwarmState) -> dict:
    """Cloud model reviews passing implementation."""
    model = llm(CRITIC_MODEL, temperature=0.3, max_tokens=4096)

    files_text = "\n".join(
        f"--- {p} ---\n{c[:4000]}" for p, c in state.get("applied_files", {}).items()
    )

    response = await model.ainvoke([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Plan:\n{json.dumps(state['plan'], indent=2)}\n\n"
            f"Files:\n{files_text}\n\n"
            f"Test output:\n{state['test_output'][-2000:]}"
        )),
    ])

    text = strip_think(response.content)
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    review = json.loads(text)
    return {
        "critic_approved": review["approved"],
        "critic_feedback": review.get("feedback", ""),
    }


async def commit_node(state: SwarmState) -> dict:
    """Final step — save training data for future fine-tuning."""
    _save_training_data(state)
    return {"status": "success"}


async def fail_node(state: SwarmState) -> dict:
    """Terminal — swarm exhausted retries."""
    return {"status": "failed"}


# ── Training data collection ─────────��──────────────────────────────────

def _save_training_data(state: SwarmState):
    """Persist successful (task → implementation) pair for fine-tuning."""
    data_dir = Path(__file__).parent / "training_data"
    data_dir.mkdir(exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "task": state["task"],
        "plan": state["plan"],
        "applied_files": state.get("applied_files", {}),
        "test_output": state.get("test_output", ""),
        "retry_count": state.get("retry_count", 0),
        "planner_loops": state.get("planner_loops", 0),
    }

    path = data_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(entry, indent=2))


# ── Routing ───────────��─────────────────────────────────────────────────

def route_after_tests(state: SwarmState) -> str:
    if state["test_passed"]:
        return "critic"
    if state["retry_count"] >= MAX_WORKER_RETRIES:
        # Workers exhausted retries — escalate to planner for replan
        if state.get("planner_loops", 1) >= MAX_PLANNER_LOOPS:
            return "fail"  # full exhaustion
        return "planner"
    return "worker"  # retry with failure context


def route_after_critic(state: SwarmState) -> str:
    if state["critic_approved"]:
        return "commit"
    if state.get("planner_loops", 1) >= MAX_PLANNER_LOOPS:
        return "commit"  # accept after max loops
    return "planner"  # replan with critic feedback


# ── Graph assembly ──────────────────────────────────────────────────────

def build_swarm():
    graph = StateGraph(SwarmState)

    graph.add_node("planner", planner_node)
    graph.add_node("worker", worker_node)
    graph.add_node("integrator", integrator_node)
    graph.add_node("test_runner", test_runner_node)
    graph.add_node("critic", critic_node)
    graph.add_node("commit", commit_node)
    graph.add_node("fail", fail_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "worker")
    graph.add_edge("worker", "integrator")
    graph.add_edge("integrator", "test_runner")

    graph.add_conditional_edges("test_runner", route_after_tests, {
        "critic": "critic",
        "worker": "worker",
        "planner": "planner",
        "fail": "fail",
    })

    graph.add_conditional_edges("critic", route_after_critic, {
        "commit": "commit",
        "planner": "planner",
    })

    graph.add_edge("commit", END)
    graph.add_edge("fail", END)

    return graph.compile()


# ── Public API ───────���──────────────────────────────────────────────────

async def invoke_swarm(
    task: str,
    repo_path: str,
    test_cmd: str = "python -m pytest tests/ -v --tb=short",
) -> SwarmState:
    """Run the full swarm on a task. Returns final state."""
    print(f"Loading context from {repo_path}...")
    context = load_context(repo_path, task)
    print(f"  Context: {len(context)} chars from {context.count('=== ./')} files")

    graph = build_swarm()

    print(f"Invoking swarm: {task[:80]}...")
    result = await graph.ainvoke({
        "task": task,
        "repo_path": str(Path(repo_path).resolve()),
        "test_cmd": test_cmd,
        "context": context,
        "plan": [],
        "integration_notes": "",
        "implementations": [],
        "integrated_diff": "",
        "applied_files": {},
        "test_passed": False,
        "test_output": "",
        "retry_count": 0,
        "critic_approved": False,
        "critic_feedback": "",
        "planner_loops": 0,
        "status": "pending",
    })

    return result


# ── CLI ───────────────��────────────────────────���────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Agent swarm: cloud planner + local workers + test-driven loop"
    )
    parser.add_argument("task", help="Task description")
    parser.add_argument("--repo", required=True, help="Path to target repository")
    parser.add_argument("--test", default="python -m pytest tests/ -v --tb=short",
                        help="Test command (default: pytest)")
    parser.add_argument("--planner", default=None, help="Override planner model")
    parser.add_argument("--worker", default=None, help="Override worker model")
    parser.add_argument("--critic", default=None, help="Override critic model")
    args = parser.parse_args()

    if args.planner:
        global PLANNER_MODEL
        PLANNER_MODEL = args.planner
    if args.worker:
        global WORKER_MODEL
        WORKER_MODEL = args.worker
    if args.critic:
        global CRITIC_MODEL
        CRITIC_MODEL = args.critic

    result = asyncio.run(invoke_swarm(args.task, args.repo, args.test))

    print("\n" + "=" * 60)
    if result["status"] == "success":
        files = list(result.get("applied_files", {}).keys())
        print(f"DONE — {len(files)} files modified: {files}")
        print(f"  Retries: {result['retry_count']}  Planner loops: {result['planner_loops']}")
        if result.get("critic_feedback"):
            print(f"  Critic: {result['critic_feedback'][:200]}")
    else:
        print(f"FAILED after {result['retry_count']} retries, {result['planner_loops']} planner loops")
        print(f"  Last test output:\n{result['test_output'][-500:]}")


if __name__ == "__main__":
    main()
