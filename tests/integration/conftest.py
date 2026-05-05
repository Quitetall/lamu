"""Integration-test fixtures (slow, end-to-end-ish)."""
from __future__ import annotations

import pytest

# Re-export the unit-test fixtures we need. They live in tests/unit/conftest.py
# but Pytest fixtures are scoped by directory; copy the bits we need here.
from tests.unit.conftest import make_entry, make_model_entry, sample_registry  # noqa: F401


@pytest.fixture(autouse=True)
def _mark_slow(request):
    """All tests under tests/integration/ get the `slow` marker implicitly."""
    request.node.add_marker(pytest.mark.slow)
