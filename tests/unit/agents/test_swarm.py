"""Tests for agents.swarm — pure helpers + smoke."""
from __future__ import annotations

import pytest


def test_module_imports():
    import agents.swarm as s
    assert hasattr(s, "build_swarm")
    assert hasattr(s, "main")
    assert hasattr(s, "invoke_swarm")


def test_strip_think_no_block():
    from agents.swarm import strip_think
    assert strip_think("hello") == "hello"


def test_strip_think_removes_block():
    from agents.swarm import strip_think
    assert strip_think("<think>thinking</think>actual answer") == "actual answer"


def test_parse_file_blocks_single():
    from agents.swarm import _parse_file_blocks
    text = "intro\n```src/x.py\nprint('hi')\n```\noutro"
    blocks = _parse_file_blocks(text)
    assert "src/x.py" in blocks
    assert "print('hi')" in blocks["src/x.py"]


def test_parse_file_blocks_multi():
    from agents.swarm import _parse_file_blocks
    text = "```a.py\nA=1\n```\n```b.py\nB=2\n```"
    blocks = _parse_file_blocks(text)
    assert set(blocks.keys()) == {"a.py", "b.py"}


def test_parse_file_blocks_empty():
    from agents.swarm import _parse_file_blocks
    assert _parse_file_blocks("no fences here") == {}


def test_route_after_tests_pass():
    from agents.swarm import route_after_tests
    state = {"test_passed": True, "retry_count": 0}
    assert route_after_tests(state) == "critic"


def test_route_after_tests_retry_with_failure():
    from agents.swarm import route_after_tests
    state = {"test_passed": False, "retry_count": 0}
    assert route_after_tests(state) == "worker"


def test_route_after_tests_replans_after_worker_exhaustion():
    from agents.swarm import route_after_tests, MAX_WORKER_RETRIES
    state = {"test_passed": False, "retry_count": MAX_WORKER_RETRIES, "planner_loops": 1}
    assert route_after_tests(state) == "planner"


def test_route_after_tests_full_exhaustion_fails():
    from agents.swarm import route_after_tests, MAX_WORKER_RETRIES, MAX_PLANNER_LOOPS
    state = {"test_passed": False, "retry_count": MAX_WORKER_RETRIES,
             "planner_loops": MAX_PLANNER_LOOPS}
    assert route_after_tests(state) == "fail"


def test_route_after_critic_approves():
    from agents.swarm import route_after_critic
    assert route_after_critic({"critic_approved": True, "planner_loops": 0}) == "commit"


def test_route_after_critic_replans_when_under_limit():
    from agents.swarm import route_after_critic
    assert route_after_critic({"critic_approved": False, "planner_loops": 0}) == "planner"


def test_route_after_critic_accepts_after_max_loops():
    from agents.swarm import route_after_critic, MAX_PLANNER_LOOPS
    state = {"critic_approved": False, "planner_loops": MAX_PLANNER_LOOPS}
    assert route_after_critic(state) == "commit"


def test_swarm_step_error_class_exists():
    """Phase C: SwarmStepError exists for non-recoverable swarm failures."""
    from lamu.core.errors import SwarmStepError
    assert issubclass(SwarmStepError, Exception)


def test_test_runner_typed_error_on_pytest_crash(tmp_path):
    """Phase C: subprocess.SubprocessError → SwarmStepError."""
    import asyncio
    from unittest.mock import patch
    from agents.swarm import test_runner_node
    from lamu.core.errors import SwarmStepError

    state = {
        "repo_path": str(tmp_path),
        "test_cmd": "python -m pytest tests/",
        "applied_files": {},
        "retry_count": 0,
    }
    with patch("asyncio.to_thread", side_effect=OSError("exec format error")):
        with pytest.raises(SwarmStepError):
            asyncio.run(test_runner_node(state))
