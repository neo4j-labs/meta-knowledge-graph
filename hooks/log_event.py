#!/usr/bin/env python3
"""Hook: log every Claude Code lifecycle event to Neo4j as a linked list.

Wired in .claude/settings.json for SessionStart, UserPromptSubmit, PreToolUse,
PostToolUse, Notification, Stop, SubagentStop, PreCompact, and SessionEnd.
Reads the hook payload from stdin and appends an Event to the per-session chain.

Graph shape::

    (Session)-[:FIRST_EVENT]->(Event)-[:NEXT]->(Event)->...
    (Session)-[:LATEST_EVENT]->(latest Event)

Adapted from https://github.com/tomasonjo/agent-memory-hooks-neo4j to reuse this
project's existing NEO4J_* env vars (loaded from .env) instead of HOOKS_NEO4J_*.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import (  # noqa: E402
    ensure_project_schema,
    link_event_to_project,
    load_dotenv,
    resolve_project,
)

MAX_RESPONSE_CHARS = 4000


def _neo4j_config() -> tuple[str, str, str, str]:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    return uri, user, password, database


def _serialize_tool_response(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) > MAX_RESPONSE_CHARS:
        text = text[:MAX_RESPONSE_CHARS] + f"...[truncated {len(text) - MAX_RESPONSE_CHARS} chars]"
    return text


def _read_transcript(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return None


def _ensure_constraints(tx) -> None:
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE")
    tx.run(
        "CREATE FULLTEXT INDEX memory_fulltext IF NOT EXISTS "
        "FOR (m:Memory) ON EACH [m.content, m.path]"
    )
    ensure_project_schema(tx)


def _append_event(tx, session_id: str, client: str, event_props: dict) -> None:
    tx.run(
        """
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET s.created_at = $timestamp, s.client = $client
        SET s.client = coalesce(s.client, $client)
        WITH s
        CREATE (e:Event $event_props)
        CREATE (s)-[:HAS_EVENT]->(e)
        WITH s, e
        OPTIONAL MATCH (s)-[old_latest:LATEST_EVENT]->(prev:Event)
        DELETE old_latest
        WITH s, e, prev
        FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
            CREATE (prev)-[:NEXT]->(e)
        )
        FOREACH (_ IN CASE WHEN prev IS NULL THEN [1] ELSE [] END |
            CREATE (s)-[:FIRST_EVENT]->(e)
        )
        CREATE (s)-[:LATEST_EVENT]->(e)
        """,
        session_id=session_id,
        client=client,
        timestamp=event_props.get("timestamp"),
        event_props=event_props,
    )


def log_event(data: dict, client: str) -> None:
    from neo4j import GraphDatabase

    session_id = data.get("session_id", "unknown")
    event_name = data.get("hook_event_name", "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()
    event_id = f"{client}_{session_id}_{timestamp}_{event_name}"
    project_root = Path(__file__).resolve().parents[1]
    project = resolve_project(data, project_root)

    event_props = {
        "event_id": event_id,
        "event_name": event_name,
        "client": client,
        "timestamp": timestamp,
        "project_id": project.id if project else None,
        "cwd": data.get("cwd"),
        "tool_name": data.get("tool_name"),
        "tool_use_id": data.get("tool_use_id"),
        "tool_input": json.dumps(data.get("tool_input")) if data.get("tool_input") else None,
        "tool_response": (
            _serialize_tool_response(data.get("tool_response"))
            if data.get("tool_response") is not None
            else None
        ),
        "prompt": data.get("prompt"),
        "model": data.get("model"),
        "source": data.get("source"),
        "turn_id": data.get("turn_id"),
        "last_assistant_message": data.get("last_assistant_message"),
        "stop_hook_active": data.get("stop_hook_active"),
        "transcript_path": data.get("transcript_path"),
        "transcript": _read_transcript(data.get("transcript_path")),
    }
    event_props = {k: v for k, v in event_props.items() if v is not None}

    uri, user, password, database = _neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(_ensure_constraints)
            session.execute_write(_append_event, session_id, client, event_props)
            if project:
                session.execute_write(
                    link_event_to_project,
                    project,
                    session_id,
                    event_id,
                    timestamp,
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="claude_code")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        log_event(data, client=args.client)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[log_event] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
