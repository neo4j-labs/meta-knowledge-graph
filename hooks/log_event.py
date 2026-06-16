#!/usr/bin/env python3
"""Hook: log every Claude Code lifecycle event to Neo4j as a linked list.

Wired in .claude/settings.json for SessionStart, UserPromptSubmit, PreToolUse,
PostToolUse, Notification, Stop, SubagentStop, PreCompact, and SessionEnd.
Reads the hook payload from stdin and appends a SessionEvent to the per-session chain.

Graph shape::

    (Session)-[:FIRST_EVENT]->(SessionEvent)-[:NEXT]->(SessionEvent)->...
    (Session)-[:LATEST_EVENT]->(latest SessionEvent)
    (Learning|Decision)-[:INJECTED_AT]->(SessionEvent)  # back-filled for
        SessionStart / UserPromptSubmit events whose parallel inject hook
        already marked memory as injected in this session.

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
    injection_window_start,
    link_event_to_project,
    load_dotenv,
    resolve_project,
)

MAX_RESPONSE_CHARS = 4000
# Hook events that inject_project_context.py uses to inject memory; only these
# events can be the target of an INJECTED_AT back-fill.
INJECTION_CONTEXT_EVENTS = {"SessionStart", "UserPromptSubmit"}


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
    ensure_project_schema(tx)


def _append_event(tx, session_id: str, client: str, event_props: dict) -> None:
    tx.run(
        """
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET s.created_at = $timestamp, s.client = $client
        SET s.client = coalesce(s.client, $client)
        WITH s
        CREATE (e:SessionEvent $event_props)
        CREATE (s)-[:HAS_EVENT]->(e)
        WITH s, e
        OPTIONAL MATCH (s)-[old_latest:LATEST_EVENT]->(prev:SessionEvent)
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


def _link_injected_memory(tx, session_id: str, event_id: str, event_name: str) -> None:
    """Back-fill (memory)-[:INJECTED_AT]->(event) for injections the parallel
    inject_project_context hook recorded before this event node existed."""
    since = injection_window_start()
    tx.run(
        """
        MATCH (s:Session {session_id: $session_id})
              -[:HAS_EVENT]->(e:SessionEvent {event_id: $event_id})
        MATCH (m)-[inj:INJECTED_IN]->(s)
        WHERE (m:Learning OR m:Decision)
          AND inj.hook_event = $event_name
          AND inj.last_injected_at >= datetime($since)
          AND NOT EXISTS {
              MATCH (m)-[:INJECTED_AT]->(recent:SessionEvent)
              WHERE recent.timestamp >= $since
          }
        MERGE (m)-[r:INJECTED_AT]->(e)
        ON CREATE SET r.injected_at = datetime()
        """,
        session_id=session_id,
        event_id=event_id,
        event_name=event_name,
        since=since,
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
            if event_name in INJECTION_CONTEXT_EVENTS and session_id != "unknown":
                session.execute_write(
                    _link_injected_memory, session_id, event_id, event_name
                )
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
