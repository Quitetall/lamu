"""Tests for web.data_layer — SQLite persistence."""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys

import pytest


_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]


@pytest.fixture
def data_layer(tmp_path, monkeypatch):
    """Load web/data_layer.py via importlib (web/ has no __init__.py)."""
    db = tmp_path / "chats.db"
    web_dir = str(_REPO_ROOT / "web")
    monkeypatch.syspath_prepend(web_dir)
    for name in ("web.data_layer", "web", "data_layer"):
        sys.modules.pop(name, None)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "data_layer", str(_REPO_ROOT / "web" / "data_layer.py"),
    )
    dl = importlib.util.module_from_spec(spec)
    sys.modules["data_layer"] = dl
    spec.loader.exec_module(dl)
    monkeypatch.setattr(dl, "DB_PATH", str(db))
    dl._init_db()
    return dl


def test_module_imports(data_layer):
    assert hasattr(data_layer, "SQLiteDataLayer")
    assert hasattr(data_layer, "_init_db")
    assert hasattr(data_layer, "_now")


def test_init_db_creates_tables(data_layer):
    con = sqlite3.connect(data_layer.DB_PATH)
    tables = {
        row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    con.close()
    assert {"users", "threads", "steps", "elements", "feedback"} <= tables


def test_now_returns_iso(data_layer):
    s = data_layer._now()
    assert "T" in s
    # ISO timestamps end with offset like '+00:00' or 'Z'
    assert "+" in s or "Z" in s


@pytest.mark.asyncio
async def test_create_and_get_user(data_layer):
    layer = data_layer.SQLiteDataLayer()
    User = type("User", (), {"identifier": "alice", "display_name": "Alice", "metadata": {}})
    user = User()
    persisted = await layer.create_user(user)
    assert persisted is not None
    fetched = await layer.get_user("alice")
    assert fetched is not None
    assert fetched.identifier == "alice"


@pytest.mark.asyncio
async def test_get_user_missing(data_layer):
    layer = data_layer.SQLiteDataLayer()
    assert await layer.get_user("ghost") is None


@pytest.mark.asyncio
async def test_update_thread_inserts_or_updates(data_layer):
    """update_thread should persist a row that subsequent calls can read."""
    layer = data_layer.SQLiteDataLayer()
    await layer.update_thread("t1", name="hello", user_id="u1")
    con = sqlite3.connect(data_layer.DB_PATH)
    row = con.execute(
        "SELECT id, name FROM threads WHERE id=?", ("t1",),
    ).fetchone()
    con.close()
    assert row is not None
    assert row[1] == "hello"


@pytest.mark.asyncio
async def test_create_user_db_error_raises_typed(data_layer, monkeypatch):
    """Phase C: sqlite3.OperationalError → DataLayerError (single catch point)."""
    from lamu.core.errors import DataLayerError
    layer = data_layer.SQLiteDataLayer()
    User = type("User", (), {"identifier": "x", "display_name": "X", "metadata": {}})

    def boom(*a, **k):
        raise sqlite3.OperationalError("disk i/o")
    monkeypatch.setattr(sqlite3, "connect", boom)
    with pytest.raises(DataLayerError):
        await layer.create_user(User())
