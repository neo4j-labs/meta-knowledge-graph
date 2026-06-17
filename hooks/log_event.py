#!/usr/bin/env python3
"""Hook: log every Claude Code lifecycle event to Neo4j as a linked list.

Wired in .claude/settings.json for SessionStart, UserPromptSubmit, PreToolUse,
PostToolUse, Notification, Stop, SubagentStop, PreCompact, and SessionEnd.
Reads the hook payload from stdin and appends a SessionEvent to the per-session chain.

Graph shape::

    (Session)-[:FIRST_EVENT]->(SessionEvent)-[:NEXT]->(SessionEvent)->...
    (Session)-[:LATEST_EVENT]->(latest SessionEvent)
    (Session)-[:HAS_SUBAGENT]->(Session)
    (SubagentSession)-[:TRIGGERED_BY|STARTED_AT|ENDED_AT]->(SessionEvent)
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
from hashlib import sha1
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import (  # noqa: E402
    agent_context_props,
    ensure_project_schema,
    in_extraction_subprocess,
    injection_window_start,
    link_event_to_project,
    load_mkg_env,
    resolve_project,
)

MAX_RESPONSE_CHARS = 4000
# Hook events that inject_project_context.py uses to inject memory; only these
# events can be the target of an INJECTED_AT back-fill.
INJECTION_CONTEXT_EVENTS = {"SessionStart", "UserPromptSubmit"}
SPAWN_AGENT_TOOL_NAMES = {"spawn_agent", "multi_agent_v1.spawn_agent"}


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


def _event_session_id(parent_session_id: str, event_props: dict) -> str:
    if event_props.get("is_subagent") and event_props.get("agent_id"):
        return str(event_props["agent_id"])
    return parent_session_id


def _session_props(
    event_session_id: str,
    parent_session_id: str,
    client: str,
    event_props: dict,
) -> dict:
    props = {
        "client": client,
        "agent_kind": event_props.get("agent_kind"),
        "agent_id": event_props.get("agent_id") or event_session_id,
        "agent_transcript_id": event_props.get("agent_transcript_id"),
        "agent_type": event_props.get("agent_type"),
    }
    if event_session_id != parent_session_id:
        props.update(
            {
                "agent_kind": "subagent",
                "agent_id": event_session_id,
                "parent_session_id": parent_session_id,
                "parent_agent_id": event_props.get("parent_agent_id")
                or parent_session_id,
            }
        )
        if event_props.get("agent_transcript_id") != event_session_id:
            props.pop("agent_transcript_id", None)
    return {k: v for k, v in props.items() if v is not None}


def _json_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _spawned_subagent_id(data: dict, event_name: str) -> str | None:
    if event_name != "PostToolUse":
        return None
    tool_name = str(data.get("tool_name") or "")
    if tool_name not in SPAWN_AGENT_TOOL_NAMES and not tool_name.endswith("spawn_agent"):
        return None
    agent_id = _json_dict(data.get("tool_response")).get("agent_id")
    return agent_id.strip() if isinstance(agent_id, str) and agent_id.strip() else None


def _ensure_constraints(tx) -> None:
    ensure_project_schema(tx)


def _append_event(tx, session_id: str, client: str, event_props: dict) -> None:
    event_session_id = _event_session_id(session_id, event_props)
    session_props = _session_props(event_session_id, session_id, client, event_props)
    tx.run(
        """
        MERGE (s:Session {session_id: $event_session_id})
        ON CREATE SET s.created_at = datetime($timestamp)
        SET s += $session_props
        WITH s
        OPTIONAL MATCH (dup:SessionEvent {event_id: $event_id})
        WITH s, dup
        WHERE dup IS NULL
        CREATE (e:SessionEvent $event_props)
        SET e.timestamp = datetime($timestamp)
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
        event_session_id=event_session_id,
        session_props=session_props,
        timestamp=event_props.get("timestamp"),
        event_props=event_props,
        event_id=event_props.get("event_id"),
    )
    if event_session_id != session_id:
        _link_subagent_session(
            tx,
            parent_session_id=session_id,
            subagent_session_id=event_session_id,
            client=client,
            event_props=event_props,
        )
    spawned_subagent_id = event_props.get("spawned_subagent_id")
    if spawned_subagent_id:
        _link_spawned_subagent_session(
            tx,
            parent_session_id=session_id,
            subagent_session_id=str(spawned_subagent_id),
            client=client,
            event_props=event_props,
        )


def _link_subagent_session(
    tx,
    parent_session_id: str,
    subagent_session_id: str,
    client: str,
    event_props: dict,
) -> None:
    parent_agent_id = event_props.get("parent_agent_id") or parent_session_id
    tx.run(
        """
        MATCH (child:Session {session_id: $subagent_session_id})
        MERGE (parent:Session {session_id: $parent_session_id})
        ON CREATE SET parent.created_at = datetime($timestamp)
        SET parent.client = coalesce(parent.client, $client),
            parent.agent_kind = coalesce(parent.agent_kind, 'main'),
            parent.agent_id = coalesce(parent.agent_id, $parent_agent_id)
        MERGE (parent)-[has:HAS_SUBAGENT]->(child)
        ON CREATE SET has.created_at = datetime($timestamp)
        SET has.updated_at = datetime($timestamp),
            has.agent_id = $subagent_session_id,
            has.parent_agent_id = $parent_agent_id
        MERGE (child)-[of:SUBAGENT_OF]->(parent)
        ON CREATE SET of.created_at = datetime($timestamp)
        SET of.updated_at = datetime($timestamp),
            of.parent_agent_id = $parent_agent_id
        WITH child
        MATCH (e:SessionEvent {event_id: $event_id})
        FOREACH (_ IN CASE WHEN $event_name = 'SubagentStart' THEN [1] ELSE [] END |
            MERGE (child)-[:STARTED_AT]->(e)
        )
        FOREACH (_ IN CASE WHEN $event_name = 'SubagentStop' THEN [1] ELSE [] END |
            MERGE (child)-[:ENDED_AT]->(e)
        )
        """,
        parent_session_id=parent_session_id,
        subagent_session_id=subagent_session_id,
        parent_agent_id=parent_agent_id,
        client=client,
        timestamp=event_props.get("timestamp"),
        event_id=event_props.get("event_id"),
        event_name=event_props.get("event_name"),
    )


def _link_spawned_subagent_session(
    tx,
    parent_session_id: str,
    subagent_session_id: str,
    client: str,
    event_props: dict,
) -> None:
    parent_agent_id = event_props.get("agent_id") or parent_session_id
    tx.run(
        """
        MERGE (parent:Session {session_id: $parent_session_id})
        ON CREATE SET parent.created_at = datetime($timestamp)
        SET parent.client = coalesce(parent.client, $client),
            parent.agent_kind = coalesce(parent.agent_kind, 'main'),
            parent.agent_id = coalesce(parent.agent_id, $parent_agent_id)
        MERGE (child:Session {session_id: $subagent_session_id})
        ON CREATE SET child.created_at = datetime($timestamp)
        SET child.client = coalesce(child.client, $client),
            child.agent_kind = 'subagent',
            child.agent_id = $subagent_session_id,
            child.parent_session_id = $parent_session_id,
            child.parent_agent_id = $parent_agent_id
        MERGE (parent)-[has:HAS_SUBAGENT]->(child)
        ON CREATE SET has.created_at = datetime($timestamp)
        SET has.updated_at = datetime($timestamp),
            has.agent_id = $subagent_session_id,
            has.parent_agent_id = $parent_agent_id
        MERGE (child)-[of:SUBAGENT_OF]->(parent)
        ON CREATE SET of.created_at = datetime($timestamp)
        SET of.updated_at = datetime($timestamp),
            of.parent_agent_id = $parent_agent_id
        WITH child
        MATCH (e:SessionEvent {event_id: $event_id})
        MERGE (child)-[:TRIGGERED_BY]->(e)
        """,
        parent_session_id=parent_session_id,
        subagent_session_id=subagent_session_id,
        parent_agent_id=parent_agent_id,
        client=client,
        timestamp=event_props.get("timestamp"),
        event_id=event_props.get("event_id"),
    )


def _link_injected_memory(tx, session_id: str, event_id: str, event_name: str) -> None:
    """Back-fill (memory)-[:INJECTED_AT]->(event) for injections the parallel
    inject_project_context hook recorded before this event node existed."""
    since = injection_window_start()
    tx.run(
        """
        MATCH (e:SessionEvent {event_id: $event_id})
        MATCH (s:Session {session_id: $session_id})
        MATCH (m)-[inj:INJECTED_IN]->(s)
        WHERE (m:Learning OR m:Decision)
          AND inj.hook_event = $event_name
          AND inj.last_injected_at >= datetime($since)
          AND NOT EXISTS {
              MATCH (m)-[:INJECTED_AT]->(recent:SessionEvent)
              WHERE recent.timestamp >= datetime($since)
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

    session_id = str(data.get("session_id") or "unknown")
    event_name = str(data.get("hook_event_name") or "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()
    # Deterministic identity: the *same* lifecycle event delivered to two hook
    # configs at once (e.g. the repo's .claude/settings.json and an installed
    # plugin both active) collapses to one node instead of double-logging.
    # Claude Code hands each matching hook the identical stdin payload, so
    # hashing it yields the same id from both firings; the SessionEvent.event_id
    # uniqueness constraint makes the dedupe atomic under the race, and
    # _append_event skips the insert when the id already exists in the session.
    payload_sig = sha1(
        json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    event_id = f"{client}:{session_id}:{event_name}:{payload_sig}"
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
    event_props.update(agent_context_props(data, session_id, event_name))
    spawned_subagent_id = _spawned_subagent_id(data, event_name)
    if spawned_subagent_id:
        event_props["spawned_subagent_id"] = spawned_subagent_id
    event_props = {k: v for k, v in event_props.items() if v is not None}
    event_session_id = _event_session_id(session_id, event_props)

    uri, user, password, database = _neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(_ensure_constraints)
            session.execute_write(_append_event, session_id, client, event_props)
            if event_name in INJECTION_CONTEXT_EVENTS and session_id != "unknown":
                session.execute_write(
                    _link_injected_memory, event_session_id, event_id, event_name
                )
                if event_session_id != session_id:
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
                if event_session_id != session_id:
                    session.execute_write(
                        link_event_to_project,
                        project,
                        event_session_id,
                        event_id,
                        timestamp,
                    )
                if spawned_subagent_id:
                    session.execute_write(
                        link_event_to_project,
                        project,
                        spawned_subagent_id,
                        event_id,
                        timestamp,
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="claude_code")
    args = parser.parse_args()

    # Don't record the nested claude_cli extraction session's own events.
    if in_extraction_subprocess():
        return 0

    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)

    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        log_event(data, client=args.client)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[log_event] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
