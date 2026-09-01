#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0"]
# ///
"""Hook: link learnings written by the MCP tool to their originating session.

Wired as a PostToolUse hook for the ``project_add_learning`` MCP tool. The MCP
server has no notion of the harness session, so learnings captured through the
tool carry no ``FROM_SESSION`` edge — and that edge is exactly what the recall
filter in inject_project_context uses to keep a conversation's own notes from
being echoed back into it. Without the link, a note captured in turn 3
resurfaces as "relevant memory" in turn 7 of the same session, pure redundancy
while the original is still in the context window. This hook reads the tool
result, resolves the learning id from it, and merges
``(:Learning)-[:FROM_SESSION]->(:Session)`` so same-session recall skips the
note; every later session still retrieves it normally.

Graph shape::

    (:Learning)-[:FROM_SESSION]->(:Session)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import (  # noqa: E402
    ensure_project_schema,
    load_mkg_env,
    neo4j_config,
    normalize_tool_failure_payload,
)

LEARNING_TOOL_SUFFIX = "project_add_learning"
LEARNING_ID_PREFIX = "learning:"


def _parse_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _collect_learning_ids(value: Any, ids: list[str]) -> None:
    if isinstance(value, str):
        parsed = _parse_json(value)
        if parsed is not None:
            _collect_learning_ids(parsed, ids)
        return
    if isinstance(value, list):
        for item in value:
            _collect_learning_ids(item, ids)
        return
    if isinstance(value, dict):
        node_id = value.get("id")
        if isinstance(node_id, str) and node_id.startswith(LEARNING_ID_PREFIX):
            ids.append(node_id)
        for item in value.values():
            _collect_learning_ids(item, ids)


def extract_learning_ids(tool_response: Any) -> list[str]:
    """Learning ids embedded in a ``project_add_learning`` tool result.

    The success payload is the server's JSON row (``{"id": "learning:...", ...}``),
    but each client wraps it differently — a raw string, an MCP content-block
    list, or a ``{result: "..."}`` envelope — so the walk is shape-agnostic:
    recurse everywhere, parse every string as JSON, and keep any ``id`` with the
    learning prefix.
    """
    ids: list[str] = []
    _collect_learning_ids(tool_response, ids)
    return list(dict.fromkeys(ids))


def write_session_links(tx, session_id: str, learning_ids: list[str], timestamp: str) -> None:
    tx.run(
        """
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET s.created_at = datetime($timestamp)
        WITH s
        UNWIND $learning_ids AS learning_id
        MATCH (l:Learning {id: learning_id})
        MERGE (l)-[r:FROM_SESSION]->(s)
        ON CREATE SET r.created_at = datetime($timestamp)
        SET l.last_source_session_id = $session_id
        """,
        session_id=session_id,
        learning_ids=learning_ids,
        timestamp=timestamp,
    )


def link(payload: dict[str, Any]) -> int:
    payload = normalize_tool_failure_payload(payload)
    tool_name = str(payload.get("tool_name") or "")
    if LEARNING_TOOL_SUFFIX not in tool_name.lower():
        return 0
    # A failed or interrupted call never reached the MERGE that writes the
    # learning, so there is nothing to link.
    if payload.get("tool_error") is True or payload.get("is_interrupt") is True:
        return 0
    session_id = str(payload.get("session_id") or "")
    if not session_id or session_id == "unknown":
        return 0
    learning_ids = extract_learning_ids(payload.get("tool_response"))
    if not learning_ids:
        return 0

    from neo4j import GraphDatabase

    timestamp = datetime.now(timezone.utc).isoformat()
    uri, user, password, database = neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_project_schema)
            session.execute_write(write_session_links, session_id, learning_ids, timestamp)
    return len(learning_ids)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        linked = link(payload)
        if linked:
            print(f"[link_learning_session] linked {linked} learning(s) to session")
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[link_learning_session] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
