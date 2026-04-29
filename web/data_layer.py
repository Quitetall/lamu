"""
SQLite-backed Chainlit data layer.
Persists threads (chat sessions) and steps (messages/tool calls) locally.
"""
import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import chainlit as cl
from chainlit.data.base import BaseDataLayer
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

DB_PATH = "/home/brianklam/local-llm/web/chats.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(fn):
    """Run a synchronous SQLite function in a thread."""
    return asyncio.to_thread(fn)


def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            identifier TEXT UNIQUE NOT NULL,
            display_name TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT NOT NULL,
            user_id TEXT,
            metadata TEXT DEFAULT '{}',
            tags TEXT DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS steps (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            parent_id TEXT,
            name TEXT,
            type TEXT,
            input TEXT,
            output TEXT,
            metadata TEXT DEFAULT '{}',
            is_error INTEGER DEFAULT 0,
            created_at TEXT,
            start_time TEXT,
            end_time TEXT,
            FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS elements (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            step_id TEXT,
            type TEXT,
            name TEXT,
            url TEXT,
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            step_id TEXT,
            value INTEGER,
            comment TEXT
        );
    """)
    con.commit()
    con.close()


_init_db()


class SQLiteDataLayer(BaseDataLayer):
    # ── Users ──────────────────────────────────────────────────────────

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        def _get():
            con = sqlite3.connect(DB_PATH)
            row = con.execute(
                "SELECT id, identifier, display_name, metadata, created_at FROM users WHERE identifier = ?",
                (identifier,),
            ).fetchone()
            con.close()
            return row

        row = await _run(_get)
        if not row:
            return None
        return PersistedUser(
            id=row[0],
            identifier=row[1],
            display_name=row[2],
            metadata=json.loads(row[3] or "{}"),
            createdAt=row[4],
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        uid = str(uuid.uuid4())
        now = _now()

        def _create():
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "INSERT OR IGNORE INTO users (id, identifier, display_name, metadata, created_at) VALUES (?,?,?,?,?)",
                (uid, user.identifier, user.display_name, json.dumps(user.metadata), now),
            )
            con.commit()
            row = con.execute(
                "SELECT id, created_at FROM users WHERE identifier = ?",
                (user.identifier,),
            ).fetchone()
            con.close()
            return row

        row = await _run(_create)
        return PersistedUser(
            id=row[0],
            identifier=user.identifier,
            display_name=user.display_name,
            metadata=user.metadata,
            createdAt=row[1],
        )

    # ── Threads ────────────────────────────────────────────────────────

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        def _update():
            con = sqlite3.connect(DB_PATH)
            # Ensure thread exists
            existing = con.execute(
                "SELECT id FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
            if not existing:
                con.execute(
                    "INSERT INTO threads (id, created_at) VALUES (?, ?)",
                    (thread_id, _now()),
                )
            updates = []
            params = []
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if user_id is not None:
                updates.append("user_id = ?")
                params.append(user_id)
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            if tags is not None:
                updates.append("tags = ?")
                params.append(json.dumps(tags))
            if updates:
                params.append(thread_id)
                con.execute(f"UPDATE threads SET {', '.join(updates)} WHERE id = ?", params)
            con.commit()
            con.close()

        await _run(_update)

    async def delete_thread(self, thread_id: str):
        def _delete():
            con = sqlite3.connect(DB_PATH)
            con.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            con.commit()
            con.close()

        await _run(_delete)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        def _list():
            con = sqlite3.connect(DB_PATH)
            where = []
            params: list = []
            if filters.userId:
                where.append("t.user_id = ?")
                params.append(filters.userId)
            if filters.search:
                where.append("t.name LIKE ?")
                params.append(f"%{filters.search}%")
            cursor_clause = ""
            if pagination.cursor:
                cursor_clause = "AND t.created_at < (SELECT created_at FROM threads WHERE id = ?)"
                params.append(pagination.cursor)
            where_str = ("WHERE " + " AND ".join(where)) if where else ""
            rows = con.execute(
                f"""
                SELECT t.id, t.name, t.created_at, t.user_id, t.metadata, t.tags
                FROM threads t
                {where_str}
                {cursor_clause}
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                params + [pagination.first + 1],
            ).fetchall()
            con.close()
            return rows

        rows = await _run(_list)
        has_next = len(rows) > pagination.first
        rows = rows[: pagination.first]
        threads: List[ThreadDict] = [
            {
                "id": r[0],
                "name": r[1],
                "createdAt": r[2],
                "userId": r[3],
                "userIdentifier": None,
                "metadata": json.loads(r[4] or "{}"),
                "tags": json.loads(r[5] or "[]"),
                "steps": [],
                "elements": [],
            }
            for r in rows
        ]
        return PaginatedResponse(
            data=threads,
            pageInfo=PageInfo(
                hasNextPage=has_next,
                startCursor=threads[0]["id"] if threads else None,
                endCursor=threads[-1]["id"] if threads else None,
            ),
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        def _get():
            con = sqlite3.connect(DB_PATH)
            t = con.execute(
                "SELECT id, name, created_at, user_id, metadata, tags FROM threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
            if not t:
                con.close()
                return None, []
            steps = con.execute(
                "SELECT id, thread_id, parent_id, name, type, input, output, metadata, is_error, created_at, start_time, end_time FROM steps WHERE thread_id = ? ORDER BY created_at ASC",
                (thread_id,),
            ).fetchall()
            con.close()
            return t, steps

        t, steps = await _run(_get)
        if not t:
            return None

        step_dicts = [
            {
                "id": s[0],
                "threadId": s[1],
                "parentId": s[2],
                "name": s[3],
                "type": s[4],
                "input": s[5] or "",
                "output": s[6] or "",
                "metadata": json.loads(s[7] or "{}"),
                "isError": bool(s[8]),
                "createdAt": s[9],
                "start": s[10],
                "end": s[11],
            }
            for s in steps
        ]
        return {
            "id": t[0],
            "name": t[1],
            "createdAt": t[2],
            "userId": t[3],
            "userIdentifier": None,
            "metadata": json.loads(t[4] or "{}"),
            "tags": json.loads(t[5] or "[]"),
            "steps": step_dicts,
            "elements": [],
        }

    async def get_thread_author(self, thread_id: str) -> str:
        def _get():
            con = sqlite3.connect(DB_PATH)
            row = con.execute(
                "SELECT user_id FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
            con.close()
            return row[0] if row else ""

        return await _run(_get)

    # ── Steps ──────────────────────────────────────────────────────────

    async def create_step(self, step_dict: dict):
        def _create():
            con = sqlite3.connect(DB_PATH)
            con.execute(
                """INSERT OR REPLACE INTO steps
                   (id, thread_id, parent_id, name, type, input, output, metadata, is_error, created_at, start_time, end_time)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    step_dict.get("id"),
                    step_dict.get("threadId"),
                    step_dict.get("parentId"),
                    step_dict.get("name"),
                    step_dict.get("type"),
                    step_dict.get("input", ""),
                    step_dict.get("output", ""),
                    json.dumps(step_dict.get("metadata") or {}),
                    int(step_dict.get("isError", False)),
                    step_dict.get("createdAt", _now()),
                    step_dict.get("start"),
                    step_dict.get("end"),
                ),
            )
            con.commit()
            con.close()

        await _run(_create)

    async def update_step(self, step_dict: dict):
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        def _delete():
            con = sqlite3.connect(DB_PATH)
            con.execute("DELETE FROM steps WHERE id = ?", (step_id,))
            con.commit()
            con.close()

        await _run(_delete)

    # ── Elements ───────────────────────────────────────────────────────

    async def create_element(self, element):
        def _create():
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "INSERT OR REPLACE INTO elements (id, thread_id, step_id, type, name, url, metadata) VALUES (?,?,?,?,?,?,?)",
                (
                    element.id,
                    getattr(element, "thread_id", None),
                    getattr(element, "for_id", None),
                    element.type,
                    element.name,
                    getattr(element, "url", None),
                    "{}",
                ),
            )
            con.commit()
            con.close()

        await _run(_create)

    async def get_element(self, thread_id: str, element_id: str) -> Optional[dict]:
        return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        def _delete():
            con = sqlite3.connect(DB_PATH)
            con.execute("DELETE FROM elements WHERE id = ?", (element_id,))
            con.commit()
            con.close()

        await _run(_delete)

    # ── Feedback ───────────────────────────────────────────────────────

    async def upsert_feedback(self, feedback: Feedback) -> str:
        fid = feedback.id or str(uuid.uuid4())

        def _upsert():
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "INSERT OR REPLACE INTO feedback (id, step_id, value, comment) VALUES (?,?,?,?)",
                (fid, feedback.forId, feedback.value, feedback.comment),
            )
            con.commit()
            con.close()

        await _run(_upsert)
        return fid

    async def delete_feedback(self, feedback_id: str) -> bool:
        def _delete():
            con = sqlite3.connect(DB_PATH)
            con.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
            con.commit()
            con.close()

        await _run(_delete)
        return True

    # ── Misc ───────────────────────────────────────────────────────────

    async def get_favorite_steps(self, user_id: str) -> List[dict]:
        return []

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass
