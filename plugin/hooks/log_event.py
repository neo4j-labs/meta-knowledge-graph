#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0"]
# ///
"""Hook: log every Claude Code lifecycle event to Neo4j as a linked list.

Wired in hooks/hooks.json for SessionStart, InstructionsLoaded,
UserPromptSubmit, UserPromptExpansion, PreToolUse, PostToolUse,
PostToolUseFailure, PermissionDenied, Notification, Stop, StopFailure,
SubagentStart, SubagentStop, TaskCreated, TaskCompleted, PreCompact,
PostCompact, PostModelSwitch, and SessionEnd. WorktreeCreate and
WorktreeRemove stay unwired: a hook there replaces the harness's own worktree
handling. Reads the hook payload from stdin and appends a SessionEvent to
the per-session chain. PostToolUseFailure (Claude Code-only) is normalized into
a PostToolUse event with an ``isError`` tool_response plus ``tool_error`` /
``is_interrupt`` / ``source_event`` props, so failed calls look the same for
Claude Code and Codex.

Background work. Subagents run in the background by default, so a
subagent's report reaches the session as a prompt the harness submits, as do
teammate, cross-session, and channel messages. Each UserPromptSubmit records
its ``prompt_origin`` (``user``, or the wrapper tag; see
``project_common.prompt_origin``), so extraction never reads a subagent's
report as the user speaking. The main agent's Stop records the work still in
flight, ``background_tasks`` and ``session_crons``, so a turn that paused for
its agents is not mistaken for a finished one. A StopFailure keeps the API
error that ended the turn as ``error`` / ``error_details``.

Restarts. Every event records the ``harness_pid`` (``CLAUDE_PID``) that fired
it, and the id hash includes it: one Claude Code process gives all its hooks
the same value, so parallel hook configs still collapse, while a restarted
session (supervisor wake-up, ``claude respawn``, ``--resume``) repeating its
resume SessionStart word for word is still its own event. A background
session records its ``job_id`` (from ``CLAUDE_JOB_DIR``) on the Session. The
hook ignores SIGTERM, which a headless run sends to the hooks of its last
event (often a StopFailure) as it exits.

Graph shape::

    (Session)-[:FIRST_EVENT]->(SessionEvent)-[:NEXT]->(SessionEvent)->...
    (Session)-[:LATEST_EVENT]->(latest SessionEvent)
    (Session)-[:HAS_SUBAGENT]->(Session)
    (SubagentSession)-[:TRIGGERED_BY|STARTED_AT|ENDED_AT]->(SessionEvent)
    (Learning)-[:INJECTED_AT]->(SessionEvent)  # back-filled for
        SessionStart / UserPromptSubmit events whose parallel inject hook
        already marked memory as injected in this session.

Adapted from https://github.com/tomasonjo/agent-memory-hooks-neo4j to reuse this
project's existing NEO4J_* env vars (loaded from .env) instead of HOOKS_NEO4J_*.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import (  # noqa: E402
    CAPTURE_ONLY_EVENTS,
    PROJECT_NEUTRAL_LIFECYCLE_EVENTS,
    SUBAGENT_HOOK_EVENTS,
    agent_context_props,
    ensure_project_schema,
    filter_dialog_transcript,
    in_extraction_subprocess,
    injection_window_start,
    link_event_to_project,
    load_mkg_env,
    merge_user_sessions,
    normalize_tool_failure_payload,
    project_git_root,
    prompt_origin,
    resolve_project,
    resolve_user,
)

MAX_RESPONSE_CHARS = 4000
# Long free-text fields of the newer events (a compaction summary, a task
# description, a command's arguments), capped like a tool response.
MAX_FIELD_CHARS = 8000
TRUNCATED_FIELDS = (
    "compact_summary",
    "custom_instructions",
    "command_args",
    "task_description",
    "message",
)
# Why a turn ended, why a session ended, why auto mode denied a call: enough
# to say why, never a wall of output.
MAX_ERROR_CHARS = 1000
ERROR_FIELDS = ("error", "error_details", "reason")
# Stop's view of the work still in flight: one line per entry, enough to say
# what is running, never a whole workflow's task list.
MAX_LISTED = 20
MAX_LINE_CHARS = 200
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


def _truncate(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) > limit:
        text = text[:limit] + f"...[truncated {len(text) - limit} chars]"
    return text


def _effort_level(data: dict):
    effort = data.get("effort")
    return effort.get("level") if isinstance(effort, dict) else effort


def _listed(entries, line) -> list[str] | None:
    """Up to MAX_LISTED entries as one-line strings, or None when empty."""
    if not isinstance(entries, list):
        return None
    lines = [
        _truncate(" ".join(line(entry).split()), MAX_LINE_CHARS)
        for entry in entries
        if isinstance(entry, dict)
    ]
    if len(lines) > MAX_LISTED:
        lines = lines[:MAX_LISTED] + [f"... and {len(lines) - MAX_LISTED} more"]
    return lines or None


def _task_line(task: dict) -> str:
    head = f"{task.get('type') or 'task'} ({task.get('status') or 'unknown'})"
    who = (
        task.get("name")
        or task.get("agent_type")
        or "/".join(filter(None, (task.get("server"), task.get("tool"))))
    )
    what = task.get("description") or task.get("command")
    body = " · ".join(str(part) for part in (who, what) if part)
    return f"{head}: {body}" if body else head


def _cron_line(cron: dict) -> str:
    when = "recurring" if cron.get("recurring") else "once"
    return f"{cron.get('schedule') or '?'} ({when}): {cron.get('prompt') or ''}"


def _harness_pid(client: str) -> str | None:
    """Process id of the Claude Code process that fired the hook.

    Every hook of one process gets the same ``CLAUDE_PID``, and a restarted
    process gets a new one. Read for Claude Code only: a Codex hook run from
    inside a Claude Code shell would inherit the variable."""
    if client != "claude_code":
        return None
    return (os.getenv("CLAUDE_PID") or "").strip() or None


def _job_id(client: str) -> str | None:
    """The background-session job this hook runs in, or None.

    Claude Code sets ``CLAUDE_JOB_DIR`` to ``~/.claude/jobs/<id>`` in every
    background session (``claude --bg``, ``/background``, a dispatch from
    agent view), and the hook inherits it."""
    if client != "claude_code":
        return None
    return Path((os.getenv("CLAUDE_JOB_DIR") or "").strip()).name or None


def _read_transcript(path: str | None) -> str | None:
    """Dialog-only snapshot of the transcript at event time.

    Only conversation text records are stored — tool outputs, tool inputs,
    attachments, thinking blocks, and meta records are dropped before the
    event is written, so they never enter the graph.
    """
    if not path:
        return None
    try:
        with open(path, "r") as f:
            content = f.read()
    except Exception:
        return None
    return filter_dialog_transcript(content) or None


def _event_session_id(parent_session_id: str, event_props: dict) -> str:
    # SubagentStart/SubagentStop are markers on the *parent* timeline: the start
    # is where the parent triggered the subagent and the stop is where control
    # returned. They stay owned by the parent session even though they describe a
    # subagent (the subagent id is preserved on the event + via HAS_SUBAGENT).
    if event_props.get("event_name") in SUBAGENT_HOOK_EVENTS:
        return parent_session_id
    # A subagent's own internal events (tool calls, prompts, etc.) are owned by
    # the subagent session so "what did subagent X do" is answerable directly.
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
        "user_id": event_props.get("user_id"),
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


def _append_event(
    tx,
    session_id: str,
    client: str,
    event_props: dict,
    job_id: str | None = None,
) -> None:
    event_name = event_props.get("event_name")
    is_lifecycle = event_name in SUBAGENT_HOOK_EVENTS
    raw_subagent_id = event_props.get("agent_id")
    subagent_id = str(raw_subagent_id) if raw_subagent_id else None
    event_session_id = _event_session_id(session_id, event_props)
    if is_lifecycle:
        # The marker is owned by the parent timeline, so it must not stamp the
        # subagent's actor identity (agent_kind=subagent, agent_id, ...) onto the
        # parent Session node. The subagent session is described by
        # _link_subagent_session instead.
        session_props = {
            k: v
            for k, v in {"client": client, "user_id": event_props.get("user_id")}.items()
            if v is not None
        }
    else:
        session_props = _session_props(event_session_id, session_id, client, event_props)
    # The job is the background session's, so only its own Session carries it;
    # the subagents it runs inherit CLAUDE_JOB_DIR but are not the job.
    if job_id and event_session_id == session_id:
        session_props["job_id"] = job_id
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
    if is_lifecycle and subagent_id and subagent_id != session_id:
        # Lifecycle marker lives on the parent timeline but still links to the
        # subagent session it describes (HAS_SUBAGENT + STARTED_AT/ENDED_AT).
        _link_subagent_session(
            tx,
            parent_session_id=session_id,
            subagent_session_id=subagent_id,
            client=client,
            event_props=event_props,
        )
    elif event_session_id != session_id:
        # Subagent-internal event owned by the subagent session.
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
        MERGE (child:Session {session_id: $subagent_session_id})
        ON CREATE SET child.created_at = datetime($timestamp)
        SET child.client = coalesce(child.client, $client),
            child.agent_kind = 'subagent',
            child.agent_id = coalesce(child.agent_id, $subagent_session_id),
            child.parent_session_id = coalesce(child.parent_session_id, $parent_session_id),
            child.parent_agent_id = coalesce(child.parent_agent_id, $parent_agent_id),
            child.agent_type = coalesce(child.agent_type, $agent_type)
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
        agent_type=event_props.get("agent_type"),
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
        MATCH (m:Learning)-[inj:INJECTED_IN]->(s)
        WHERE inj.hook_event = $event_name
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


def _event_project_id(project, event_name: str) -> str | None:
    if event_name in PROJECT_NEUTRAL_LIFECYCLE_EVENTS:
        return None
    return project.id if project else None


def log_event(data: dict, client: str) -> None:
    from neo4j import GraphDatabase

    # Claude Code delivers failed tool calls as PostToolUseFailure (CC-only)
    # instead of PostToolUse. Fold them into the unified PostToolUse shape
    # before hashing so the event id stays deterministic across hook configs
    # and the stored event matches what a Codex failure looks like.
    data = normalize_tool_failure_payload(data)
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
    # The harness process is part of the hashed content: shared by every hook
    # of one process, so the parallel firings above still collapse, and new on
    # each restart, whose resume SessionStart repeats the last one word for word.
    harness_pid = _harness_pid(client)
    signed = {**data, "harness_pid": harness_pid} if harness_pid else data
    payload_sig = sha1(
        json.dumps(signed, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    event_id = f"{client}:{session_id}:{event_name}:{payload_sig}"
    project_root = Path(__file__).resolve().parents[1]
    project = resolve_project(data, project_root)
    # Who is driving this session. Stamped on the event and the session so
    # every downstream artifact (learnings, observations, profile) can be
    # traced to a person; resolved once per firing, never from the payload.
    user = resolve_user(project_git_root(project), client)
    prompt = data.get("prompt")
    tool_event = "tool_name" in data
    # Background work is the parent session's, so only the main agent's Stop
    # records it: every SubagentStop of a workflow carries the same list again.
    stop = event_name == "Stop"

    event_props = {
        "event_id": event_id,
        "event_name": event_name,
        "client": client,
        "user_id": user.id,
        "timestamp": timestamp,
        "project_id": _event_project_id(project, event_name),
        "harness_pid": harness_pid,
        "cwd": data.get("cwd"),
        "permission_mode": data.get("permission_mode"),
        "effort_level": _effort_level(data),
        "tool_name": data.get("tool_name"),
        "tool_use_id": data.get("tool_use_id"),
        "tool_input": json.dumps(data.get("tool_input")) if data.get("tool_input") else None,
        "tool_response": (
            _serialize_tool_response(data.get("tool_response"))
            if data.get("tool_response") is not None
            else None
        ),
        "prompt": prompt,
        # Background agents, teammates, and other sessions reach the session
        # as prompts the harness submits; see project_common.prompt_origin.
        "prompt_origin": (
            prompt_origin(prompt) if event_name == "UserPromptSubmit" and prompt else None
        ),
        # UserPromptExpansion: the slash command or MCP prompt a typed prompt
        # expanded into.
        "expansion_type": data.get("expansion_type"),
        "command_name": data.get("command_name"),
        "command_args": data.get("command_args"),
        "command_source": data.get("command_source"),
        "model": data.get("model"),
        "source": data.get("source"),
        "source_event": data.get("source_event"),
        "tool_error": data.get("tool_error"),
        # A failed tool call's reason is already its tool_response; outside a
        # tool call, ``error`` is the API error type that ended a turn
        # (StopFailure), with ``error_details`` beside it.
        "error": None if tool_event else data.get("error"),
        "error_details": data.get("error_details"),
        # Why a session ended, or why auto mode denied a call.
        "reason": data.get("reason"),
        "is_interrupt": data.get("is_interrupt"),
        "turn_id": data.get("turn_id"),
        "last_assistant_message": data.get("last_assistant_message"),
        "stop_hook_active": data.get("stop_hook_active"),
        "background_tasks": (
            _listed(data.get("background_tasks"), _task_line) if stop else None
        ),
        "session_crons": _listed(data.get("session_crons"), _cron_line) if stop else None,
        "agent_transcript_path": data.get("agent_transcript_path"),
        # TaskCreated and TaskCompleted.
        "task_id": data.get("task_id"),
        "task_subject": data.get("task_subject"),
        "task_description": data.get("task_description"),
        "teammate_name": data.get("teammate_name"),
        # PreCompact and PostCompact; the harness's own summary of the
        # conversation it compacted.
        "trigger": data.get("trigger"),
        "custom_instructions": data.get("custom_instructions"),
        "compact_summary": data.get("compact_summary"),
        # PostModelSwitch.
        "from_model": data.get("from_model"),
        "to_model": data.get("to_model"),
        "requested_model": data.get("requested_model"),
        # InstructionsLoaded: which CLAUDE.md or rules file entered context.
        "file_path": data.get("file_path"),
        "memory_type": data.get("memory_type"),
        "load_reason": data.get("load_reason"),
        "trigger_file_path": data.get("trigger_file_path"),
        "parent_file_path": data.get("parent_file_path"),
        "notification_type": data.get("notification_type"),
        "message": data.get("message"),
        "transcript_path": data.get("transcript_path"),
        # Extraction never reads the capture-only events, so a dialog snapshot
        # on one (InstructionsLoaded fires once per file loaded) is dead weight.
        "transcript": (
            None
            if event_name in CAPTURE_ONLY_EVENTS
            else _read_transcript(data.get("transcript_path"))
        ),
    }
    for field in TRUNCATED_FIELDS:
        if event_props.get(field) is not None:
            event_props[field] = _truncate(event_props[field], MAX_FIELD_CHARS)
    for field in ERROR_FIELDS:
        if event_props.get(field) is not None:
            event_props[field] = _truncate(event_props[field], MAX_ERROR_CHARS)
    event_props.update(agent_context_props(data, session_id, event_name))
    spawned_subagent_id = _spawned_subagent_id(data, event_name)
    if spawned_subagent_id:
        event_props["spawned_subagent_id"] = spawned_subagent_id
    event_props = {k: v for k, v in event_props.items() if v is not None}
    event_session_id = _event_session_id(session_id, event_props)

    # SubagentStart/Stop are owned by the parent, so the subagent session they
    # describe still needs its own project and user links.
    lifecycle_subagent_id = (
        str(event_props.get("agent_id"))
        if event_name in SUBAGENT_HOOK_EVENTS and event_props.get("agent_id")
        else None
    )

    uri, db_user, password, database = _neo4j_config()
    with GraphDatabase.driver(uri, auth=(db_user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(_ensure_constraints)
            session.execute_write(
                _append_event, session_id, client, event_props, _job_id(client)
            )
            session.execute_write(
                merge_user_sessions,
                user,
                [
                    sid
                    for sid in (
                        session_id,
                        event_session_id,
                        lifecycle_subagent_id,
                        spawned_subagent_id,
                    )
                    if sid
                ],
                timestamp,
            )
            if event_name in INJECTION_CONTEXT_EVENTS and session_id != "unknown":
                session.execute_write(
                    _link_injected_memory, event_session_id, event_id, event_name
                )
                if event_session_id != session_id:
                    session.execute_write(
                        _link_injected_memory, session_id, event_id, event_name
                    )
            if project and event_name not in PROJECT_NEUTRAL_LIFECYCLE_EVENTS:
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
                if lifecycle_subagent_id and lifecycle_subagent_id != session_id:
                    session.execute_write(
                        link_event_to_project,
                        project,
                        lifecycle_subagent_id,
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
    # A headless run (-p, the Agent SDK) exits as soon as an API error ends its
    # turn and sends SIGTERM to the StopFailure hooks still running, which
    # would kill the one record of why the turn ended before its write.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
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
