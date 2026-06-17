#!/usr/bin/env python3
"""Project hook: derive a typed tool-call graph from raw session events.

The raw ``:SessionEvent`` stream remains append-only provenance. This processor reads
hook-captured ``PreToolUse`` / ``PostToolUse`` pairs and writes a deterministic
projection that is easier to query:

    (:Session)-[:HAS_TURN]->(:Turn)-[:ISSUED]->(:ToolCall)
    (:ToolCall)-[:USES_TOOL]->(:Tool)
    (:ToolCall)-[:RETURNED]->(:ToolResult)
    (:ToolCall)-[:HAS_RATIONALE]->(:ToolRationale)
    (:ToolCall)-[:TARGETS]->(:Resource)

``ToolRationale`` stores an observable rationale inferred from the user prompt,
assistant-visible tool input, and tool result. It must not be treated as hidden
chain-of-thought.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import (  # noqa: E402
    ProjectRef,
    agent_context_props,
    ensure_project_schema,
    load_mkg_env,
    merge_project_and_session,
    neo4j_config,
    resolve_project,
    slugify,
    truncate,
)


MAX_TEXT = 1200
MAX_SUMMARY = 500


def _read_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def _event_text(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    return value if isinstance(value, str) else ""


def _event_time(event: dict[str, Any]) -> str:
    return _event_text(event, "timestamp") or ""


def _json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stable_id(prefix: str, *values: object, limit: int = 16) -> str:
    digest = sha1("\n".join(str(value or "") for value in values).encode("utf-8"))
    return f"{prefix}:{digest.hexdigest()[:limit]}"


def _tool_namespace(tool_name: str) -> str | None:
    if "__" not in tool_name:
        return None
    parts = [part for part in tool_name.split("__") if part]
    if len(parts) < 2:
        return None
    return parts[0]


def _tool_kind(tool_name: str) -> str:
    if tool_name == "Bash":
        return "shell"
    if tool_name in {"apply_patch", "Edit", "MultiEdit", "Write"}:
        return "filesystem_write"
    if tool_name.startswith("mcp__"):
        return "mcp"
    if "neo4j" in tool_name.lower():
        return "graph"
    return "tool"


def _parse_command(tool_input: dict[str, Any]) -> str:
    for key in ("cmd", "command"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return ""


def _command_verb(command: str) -> str:
    if not command:
        return ""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return parts[0] if parts else ""


def _looks_like_path(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    suffixes = (
        ".py",
        ".md",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".txt",
        ".csv",
        ".tsx",
        ".ts",
        ".js",
        ".jsx",
    )
    return "/" in token or token.endswith(suffixes)


def _command_paths(command: str) -> list[str]:
    if not command:
        return []
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    paths: list[str] = []
    for token in parts[1:]:
        cleaned = token.strip("'\"")
        if _looks_like_path(cleaned):
            paths.append(cleaned)
    return paths[:5]


def _resource_rows_for_call(call_id: str, tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    command = _parse_command(tool_input)
    for path in _command_paths(command):
        rows.append(
            {
                "id": f"resource:file:{path}",
                "call_id": call_id,
                "type": "file",
                "name": path,
                "value": path,
                "confidence": 0.85,
            }
        )

    query = tool_input.get("query")
    if isinstance(query, str) and query.strip():
        rows.append(
            {
                "id": _stable_id("resource:cypher", query),
                "call_id": call_id,
                "type": "cypher_query",
                "name": truncate(query, 120),
                "value": query,
                "confidence": 0.9,
            }
        )

    for key in ("path", "file", "uri", "url", "ref_id"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            rows.append(
                {
                    "id": _stable_id(f"resource:{key}", value),
                    "call_id": call_id,
                    "type": key,
                    "name": truncate(value, 120),
                    "value": value,
                    "confidence": 0.8,
                }
            )

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        deduped.append(row)
    return deduped[:5]


def _known_tool_intent(tool_name: str, tool_input: dict[str, Any]) -> tuple[str, str, float]:
    command = _parse_command(tool_input)
    verb = _command_verb(command)
    lowered_name = tool_name.lower()

    if tool_name == "Bash":
        if verb in {"sed", "nl", "cat", "head", "tail"}:
            return "read repository file context", "file contents", 0.82
        if verb == "rg":
            return "locate relevant files or references", "matching files or lines", 0.8
        if verb == "git":
            return "inspect repository state", "git status or history output", 0.78
        if verb in {"uv", "pytest", "python", "python3"}:
            return "run project verification", "test or script output", 0.78
        if verb in {"wc", "ls", "find"}:
            return "inspect local filesystem shape", "file listing or counts", 0.74
        return "run a shell command for project work", "command output", 0.62

    if "project_get_context" in lowered_name:
        return "recall scoped project memory", "relevant learnings and decisions", 0.9
    if "project_add_learning" in lowered_name:
        return "persist durable project context", "created or updated learning", 0.92
    if "neo4j_get_schema" in lowered_name:
        return "inspect live graph schema", "node labels, relationships, and properties", 0.9
    if "neo4j_read_cypher" in lowered_name:
        return "query current graph state", "read-only graph query results", 0.88
    if "apply_patch" in lowered_name or tool_name in {"Edit", "MultiEdit", "Write"}:
        return "edit repository files", "applied file changes", 0.82

    return "use a tool to advance the current task", "tool response", 0.55


def _input_summary(tool_input: dict[str, Any], raw_tool_input: str) -> str:
    command = _parse_command(tool_input)
    if command:
        return truncate(command, MAX_SUMMARY)
    query = tool_input.get("query")
    if isinstance(query, str) and query.strip():
        return truncate(query, MAX_SUMMARY)
    return truncate(raw_tool_input, MAX_SUMMARY)


def _result_summary(tool_response: str) -> str:
    if not tool_response:
        return ""
    parsed = _json_object(tool_response)
    if parsed:
        text = json.dumps(parsed, default=str)
    else:
        text = tool_response
    return truncate(text, MAX_SUMMARY)


def _result_success(tool_response: str) -> bool | None:
    if not tool_response:
        return None
    lowered = tool_response.lower()
    if "process exited with code 0" in lowered or '"status": "success"' in lowered:
        return True
    if "process exited with code " in lowered or '"status": "error"' in lowered:
        return False
    return None


def _turn_node_id(session_id: str, turn_id: str | None) -> str:
    value = turn_id or "unknown"
    return f"turn:{session_id}:{value}"


def _call_node_id(session_id: str, event: dict[str, Any]) -> str:
    tool_use_id = _event_text(event, "tool_use_id")
    if tool_use_id:
        return f"tool-call:{session_id}:{tool_use_id}"
    return _stable_id(
        f"tool-call:{session_id}",
        event.get("event_id"),
        event.get("tool_name"),
        event.get("tool_input"),
    )


def _agent_fields(event: dict[str, Any], session_id: str) -> dict[str, Any]:
    return agent_context_props(event, session_id, _event_text(event, "event_name"))


def build_event_enrichment_projection(
    project: ProjectRef,
    session_id: str,
    mode: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    sorted_events = sorted(events, key=lambda event: (_event_time(event), event.get("event_id") or ""))
    turn_prompts: dict[str | None, dict[str, str]] = {}
    last_prompt: dict[str, str] | None = None

    for event in sorted_events:
        if event.get("event_name") != "UserPromptSubmit":
            continue
        prompt = _event_text(event, "prompt")
        if not prompt:
            continue
        prompt_ref = {
            "prompt": prompt,
            "prompt_event_id": _event_text(event, "event_id"),
            "timestamp": _event_time(event),
        }
        prompt_ref.update(_agent_fields(event, session_id))
        turn_prompts[event.get("turn_id")] = prompt_ref
        last_prompt = prompt_ref

    pre_events = [event for event in sorted_events if event.get("event_name") == "PreToolUse"]
    post_events = [event for event in sorted_events if event.get("event_name") == "PostToolUse"]
    posts_by_use_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in post_events:
        tool_use_id = _event_text(event, "tool_use_id")
        if tool_use_id:
            posts_by_use_id[tool_use_id].append(event)

    used_post_ids: set[str] = set()
    turn_rows: dict[str, dict[str, Any]] = {}
    tool_call_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    rationale_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    follows_rows: list[dict[str, Any]] = []
    dependency_rows: list[dict[str, Any]] = []
    calls_by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, pre_event in enumerate(pre_events, start=1):
        tool_name = _event_text(pre_event, "tool_name") or "unknown"
        raw_tool_input = _event_text(pre_event, "tool_input")
        tool_input = _json_object(raw_tool_input)
        tool_use_id = _event_text(pre_event, "tool_use_id")
        post_event = None
        if tool_use_id and posts_by_use_id.get(tool_use_id):
            for candidate in posts_by_use_id[tool_use_id]:
                candidate_id = _event_text(candidate, "event_id")
                if candidate_id not in used_post_ids:
                    post_event = candidate
                    break
        if post_event is None:
            for candidate in post_events:
                candidate_id = _event_text(candidate, "event_id")
                if candidate_id in used_post_ids:
                    continue
                if _event_time(candidate) < _event_time(pre_event):
                    continue
                if _event_text(candidate, "tool_name") != tool_name:
                    continue
                if _event_text(candidate, "tool_input") != raw_tool_input:
                    continue
                post_event = candidate
                break
        if post_event:
            used_post_ids.add(_event_text(post_event, "event_id"))

        turn_id = pre_event.get("turn_id") or (post_event or {}).get("turn_id")
        prompt_ref = turn_prompts.get(turn_id) or last_prompt or {}
        prompt = prompt_ref.get("prompt", "")
        agent_fields = _agent_fields(pre_event, session_id)
        if not agent_fields.get("agent_id") and prompt_ref.get("agent_id"):
            agent_fields["agent_id"] = prompt_ref.get("agent_id")
        if not agent_fields.get("agent_transcript_id") and prompt_ref.get("agent_transcript_id"):
            agent_fields["agent_transcript_id"] = prompt_ref.get("agent_transcript_id")
        if not agent_fields.get("parent_session_id") and prompt_ref.get("parent_session_id"):
            agent_fields["parent_session_id"] = prompt_ref.get("parent_session_id")
        turn_node_id = _turn_node_id(session_id, str(turn_id) if turn_id else None)
        if turn_node_id not in turn_rows:
            turn_rows[turn_node_id] = {
                "id": turn_node_id,
                "session_id": session_id,
                "turn_id": str(turn_id) if turn_id else None,
                "agent_kind": agent_fields.get("agent_kind"),
                "agent_id": agent_fields.get("agent_id"),
                "agent_transcript_id": agent_fields.get("agent_transcript_id"),
                "parent_session_id": agent_fields.get("parent_session_id"),
                "is_subagent": agent_fields.get("is_subagent", False),
                "prompt": truncate(prompt, MAX_TEXT) if prompt else None,
                "prompt_event_id": prompt_ref.get("prompt_event_id"),
                "first_seen_at": prompt_ref.get("timestamp") or _event_time(pre_event),
            }

        call_id = _call_node_id(session_id, pre_event)
        tool_id = f"tool:{slugify(tool_name)}"
        intent, expected_output, confidence = _known_tool_intent(tool_name, tool_input)
        input_summary = _input_summary(tool_input, raw_tool_input)
        prompt_fragment = truncate(prompt, 180) if prompt else "current session task"
        observable_rationale = (
            f"{intent} in response to the active request: {prompt_fragment}"
        )
        result_text = _event_text(post_event or {}, "tool_response")

        call_row = {
            "id": call_id,
            "session_id": session_id,
            "turn_node_id": turn_node_id,
            "turn_id": str(turn_id) if turn_id else None,
            "agent_kind": agent_fields.get("agent_kind"),
            "agent_id": agent_fields.get("agent_id"),
            "agent_transcript_id": agent_fields.get("agent_transcript_id"),
            "parent_session_id": agent_fields.get("parent_session_id"),
            "is_subagent": agent_fields.get("is_subagent", False),
            "tool_id": tool_id,
            "tool_name": tool_name,
            "tool_namespace": _tool_namespace(tool_name),
            "tool_kind": _tool_kind(tool_name),
            "tool_use_id": tool_use_id or None,
            "pre_event_id": _event_text(pre_event, "event_id"),
            "post_event_id": _event_text(post_event or {}, "event_id") or None,
            "started_at": _event_time(pre_event) or None,
            "ended_at": _event_time(post_event or {}) or None,
            "operation": _command_verb(_parse_command(tool_input))
            or str(tool_input.get("query") and "cypher_query" or ""),
            "input_summary": input_summary,
            "tool_input": truncate(raw_tool_input, MAX_TEXT) if raw_tool_input else None,
            "intent": intent,
            "expected_output": expected_output,
            "observable_rationale": truncate(observable_rationale, MAX_TEXT),
            "rationale_confidence": confidence,
            "order": index,
        }
        tool_call_rows.append(call_row)
        calls_by_turn[turn_node_id].append(call_row)

        rationale_rows.append(
            {
                "id": f"tool-rationale:{call_id}",
                "call_id": call_id,
                "intent": intent,
                "expected_output": expected_output,
                "observable_rationale": truncate(observable_rationale, MAX_TEXT),
                "confidence": confidence,
                "source": "heuristic_event_enrichment",
            }
        )

        if post_event:
            result_id = f"tool-result:{call_id}"
            result_rows.append(
                {
                    "id": result_id,
                    "call_id": call_id,
                    "post_event_id": _event_text(post_event, "event_id"),
                    "summary": _result_summary(result_text),
                    "response_chars": len(result_text),
                    "succeeded": _result_success(result_text),
                    "created_at": _event_time(post_event) or None,
                }
            )
            if result_text:
                resource_rows.append(
                    {
                        "id": f"evidence:{call_id}",
                        "call_id": call_id,
                        "type": "tool_response",
                        "name": f"{tool_name} response",
                        "value": _result_summary(result_text),
                        "confidence": 0.7,
                    }
                )

        resource_rows.extend(_resource_rows_for_call(call_id, tool_input))

    for turn_node_id, calls in calls_by_turn.items():
        ordered = sorted(calls, key=lambda row: row["order"])
        for previous, current in zip(ordered, ordered[1:]):
            follows_rows.append(
                {
                    "from_id": previous["id"],
                    "to_id": current["id"],
                    "turn_node_id": turn_node_id,
                    "confidence": 0.7,
                    "source": "event_order",
                }
            )

    resources_by_call: dict[str, set[str]] = defaultdict(set)
    for row in resource_rows:
        if row["type"] == "tool_response":
            continue
        resources_by_call[row["call_id"]].add(row["id"])
    for turn_node_id, calls in calls_by_turn.items():
        ordered = sorted(calls, key=lambda row: row["order"])
        for idx, current in enumerate(ordered):
            current_resources = resources_by_call.get(current["id"], set())
            if not current_resources:
                continue
            for previous in ordered[:idx]:
                shared = current_resources & resources_by_call.get(previous["id"], set())
                if shared:
                    dependency_rows.append(
                        {
                            "from_id": current["id"],
                            "to_id": previous["id"],
                            "turn_node_id": turn_node_id,
                            "resource_ids": sorted(shared),
                            "confidence": 0.65,
                            "source": "shared_target_resource",
                        }
                    )

    event_ids = [_event_text(event, "event_id") for event in sorted_events if event.get("event_id")]
    digest = sha1("\n".join(event_ids).encode("utf-8")).hexdigest()[:16]
    enrichment_id = f"event-enrichment:{project.id}:{session_id}:{mode}:{digest}"
    summary = (
        f"Enriched {len(event_ids)} {mode} events into "
        f"{len(tool_call_rows)} tool calls, {len(turn_rows)} turns, "
        f"{len(rationale_rows)} rationales, and {len(resource_rows)} resources."
    )

    return {
        "id": enrichment_id,
        "summary": summary,
        "event_ids": event_ids,
        "turns": list(turn_rows.values()),
        "tool_calls": tool_call_rows,
        "results": result_rows,
        "rationales": rationale_rows,
        "resources": resource_rows,
        "follows": follows_rows,
        "dependencies": dependency_rows,
    }


def ensure_event_enrichment_schema(tx) -> None:
    ensure_project_schema(tx)
    for stmt in (
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Turn) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (tc:ToolCall) REQUIRE tc.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (tr:ToolResult) REQUIRE tr.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (tool:Tool) REQUIRE tool.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:ToolRationale) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (res:Resource) REQUIRE res.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ee:EventEnrichment) REQUIRE ee.id IS UNIQUE",
        "CREATE FULLTEXT INDEX tool_call_fulltext IF NOT EXISTS "
        "FOR (tc:ToolCall) ON EACH [tc.intent, tc.input_summary, tc.observable_rationale]",
        "CREATE FULLTEXT INDEX tool_rationale_fulltext IF NOT EXISTS "
        "FOR (r:ToolRationale) ON EACH [r.intent, r.observable_rationale, r.expected_output]",
    ):
        tx.run(stmt)


def _fetch_unprocessed_events(
    driver,
    database: str,
    project_id: str,
    session_id: str,
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    result = driver.execute_query(
        """
        MATCH (s:Session {session_id: $session_id})
        OPTIONAL MATCH (s)-[:HAS_SUBAGENT*1..]->(sub:Session)
        WITH [s] + collect(DISTINCT sub) AS sessions
        UNWIND sessions AS scoped_session
        WITH DISTINCT scoped_session
        WHERE scoped_session IS NOT NULL
        MATCH (scoped_session)-[:HAS_EVENT]->(e:SessionEvent)
        WHERE NOT EXISTS {
            MATCH (:EventEnrichment {project_id: $project_id, mode: $mode})
                  -[:PROCESSED_EVENT]->(e)
        }
        RETURN properties(e) AS event
        ORDER BY e.timestamp
        LIMIT $limit
        """,
        project_id=project_id,
        session_id=session_id,
        mode=mode,
        limit=limit,
        database_=database,
    )
    return [dict(record["event"]) for record in result.records]


def _write_event_enrichment(
    tx,
    project: ProjectRef,
    session_id: str,
    mode: str,
    projection: dict[str, Any],
    timestamp: str,
) -> None:
    tx.run(
        """
        MERGE (p:Project {id: $project_id})
        ON CREATE SET p.created_at = $timestamp
        SET p.updated_at = $timestamp,
            p.last_activity_at = $timestamp
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET s.created_at = $timestamp
        MERGE (p)-[:HAS_SESSION]->(s)
        MERGE (ee:EventEnrichment {id: $enrichment_id})
        ON CREATE SET ee.created_at = $timestamp
        SET ee.project_id = $project_id,
            ee.session_id = $session_id,
            ee.mode = $mode,
            ee.summary = $summary,
            ee.event_count = size($event_ids),
            ee.turn_count = size($turns),
            ee.tool_call_count = size($tool_calls),
            ee.rationale_count = size($rationales),
            ee.resource_count = size($resources),
            ee.updated_at = $timestamp
        MERGE (p)-[:HAS_EVENT_ENRICHMENT]->(ee)
        MERGE (s)-[:HAS_EVENT_ENRICHMENT]->(ee)
        WITH ee
        UNWIND $event_ids AS event_id
        MATCH (e:SessionEvent {event_id: event_id})
        MERGE (ee)-[:PROCESSED_EVENT]->(e)
        """,
        project_id=project.id,
        session_id=session_id,
        mode=mode,
        enrichment_id=projection["id"],
        summary=projection["summary"],
        event_ids=projection["event_ids"],
        turns=projection["turns"],
        tool_calls=projection["tool_calls"],
        rationales=projection["rationales"],
        resources=projection["resources"],
        timestamp=timestamp,
    )

    if projection["turns"]:
        tx.run(
            """
            MATCH (s:Session {session_id: $session_id})
            UNWIND $turns AS row
            MERGE (t:Turn {id: row.id})
            ON CREATE SET t.created_at = $timestamp
            SET t.session_id = row.session_id,
                t.turn_id = row.turn_id,
                t.agent_kind = row.agent_kind,
                t.agent_id = row.agent_id,
                t.agent_transcript_id = row.agent_transcript_id,
                t.parent_session_id = row.parent_session_id,
                t.is_subagent = row.is_subagent,
                t.prompt = row.prompt,
                t.first_seen_at = row.first_seen_at,
                t.updated_at = $timestamp
            MERGE (s)-[:HAS_TURN]->(t)
            WITH t, row
            OPTIONAL MATCH (promptEvent:SessionEvent {event_id: row.prompt_event_id})
            FOREACH (_ IN CASE WHEN promptEvent IS NULL THEN [] ELSE [1] END |
                MERGE (t)-[:PROMPT_EVENT]->(promptEvent)
            )
            """,
            session_id=session_id,
            turns=projection["turns"],
            timestamp=timestamp,
        )

    if projection["tool_calls"]:
        tx.run(
            """
            MATCH (ee:EventEnrichment {id: $enrichment_id})
            UNWIND $tool_calls AS row
            MATCH (t:Turn {id: row.turn_node_id})
            MATCH (preEvent:SessionEvent {event_id: row.pre_event_id})
            OPTIONAL MATCH (postEvent:SessionEvent {event_id: row.post_event_id})
            MERGE (tool:Tool {id: row.tool_id})
            ON CREATE SET tool.created_at = $timestamp
            SET tool.name = row.tool_name,
                tool.namespace = row.tool_namespace,
                tool.kind = row.tool_kind,
                tool.updated_at = $timestamp
            MERGE (call:ToolCall {id: row.id})
            ON CREATE SET call.created_at = $timestamp
            SET call.session_id = row.session_id,
                call.turn_id = row.turn_id,
                call.agent_kind = row.agent_kind,
                call.agent_id = row.agent_id,
                call.agent_transcript_id = row.agent_transcript_id,
                call.parent_session_id = row.parent_session_id,
                call.is_subagent = row.is_subagent,
                call.tool_use_id = row.tool_use_id,
                call.tool_name = row.tool_name,
                call.tool_kind = row.tool_kind,
                call.operation = row.operation,
                call.input_summary = row.input_summary,
                call.tool_input = row.tool_input,
                call.started_at = row.started_at,
                call.ended_at = row.ended_at,
                call.intent = row.intent,
                call.expected_output = row.expected_output,
                call.observable_rationale = row.observable_rationale,
                call.rationale_confidence = row.rationale_confidence,
                call.call_order = row.order,
                call.updated_at = $timestamp
            MERGE (t)-[:ISSUED]->(call)
            MERGE (call)-[:USES_TOOL]->(tool)
            MERGE (call)-[:LOGGED_BY]->(preEvent)
            MERGE (ee)-[:PRODUCED_TOOL_CALL]->(call)
            FOREACH (_ IN CASE WHEN postEvent IS NULL THEN [] ELSE [1] END |
                MERGE (call)-[:COMPLETED_BY]->(postEvent)
            )
            """,
            enrichment_id=projection["id"],
            tool_calls=projection["tool_calls"],
            timestamp=timestamp,
        )

    if projection["results"]:
        tx.run(
            """
            UNWIND $results AS row
            MATCH (call:ToolCall {id: row.call_id})
            MATCH (postEvent:SessionEvent {event_id: row.post_event_id})
            MERGE (result:ToolResult {id: row.id})
            ON CREATE SET result.created_at = $timestamp
            SET result.summary = row.summary,
                result.response_chars = row.response_chars,
                result.succeeded = row.succeeded,
                result.result_at = row.created_at,
                result.updated_at = $timestamp
            MERGE (call)-[:RETURNED]->(result)
            MERGE (result)-[:LOGGED_BY]->(postEvent)
            """,
            results=projection["results"],
            timestamp=timestamp,
        )

    if projection["rationales"]:
        tx.run(
            """
            UNWIND $rationales AS row
            MATCH (call:ToolCall {id: row.call_id})
            MERGE (rationale:ToolRationale {id: row.id})
            ON CREATE SET rationale.created_at = $timestamp
            SET rationale.intent = row.intent,
                rationale.expected_output = row.expected_output,
                rationale.observable_rationale = row.observable_rationale,
                rationale.confidence = row.confidence,
                rationale.source = row.source,
                rationale.updated_at = $timestamp
            MERGE (call)-[:HAS_RATIONALE]->(rationale)
            """,
            rationales=projection["rationales"],
            timestamp=timestamp,
        )

    if projection["resources"]:
        tx.run(
            """
            UNWIND $resources AS row
            MATCH (call:ToolCall {id: row.call_id})
            MERGE (resource:Resource {id: row.id})
            ON CREATE SET resource.created_at = $timestamp
            SET resource.type = row.type,
                resource.name = row.name,
                resource.value = row.value,
                resource.updated_at = $timestamp
            MERGE (call)-[target:TARGETS]->(resource)
            SET target.confidence = row.confidence,
                target.source = 'event_enrichment',
                target.updated_at = $timestamp
            """,
            resources=projection["resources"],
            timestamp=timestamp,
        )

    if projection["follows"]:
        tx.run(
            """
            UNWIND $follows AS row
            MATCH (previous:ToolCall {id: row.from_id})
            MATCH (current:ToolCall {id: row.to_id})
            MERGE (previous)-[r:NEXT_TOOL_CALL]->(current)
            SET r.turn_id = row.turn_node_id,
                r.confidence = row.confidence,
                r.source = row.source,
                r.updated_at = $timestamp
            """,
            follows=projection["follows"],
            timestamp=timestamp,
        )

    if projection["dependencies"]:
        tx.run(
            """
            UNWIND $dependencies AS row
            MATCH (current:ToolCall {id: row.from_id})
            MATCH (previous:ToolCall {id: row.to_id})
            MERGE (current)-[r:DEPENDS_ON]->(previous)
            SET r.turn_id = row.turn_node_id,
                r.resource_ids = row.resource_ids,
                r.confidence = row.confidence,
                r.source = row.source,
                r.updated_at = $timestamp
            """,
            dependencies=projection["dependencies"],
            timestamp=timestamp,
        )


def enrich_events(payload: dict[str, Any], mode: str, limit: int) -> None:
    project_root = Path(__file__).resolve().parents[1]
    project = resolve_project(payload, project_root)
    if not project:
        return

    session_id = payload.get("session_id", "unknown")
    if not session_id or session_id == "unknown":
        return

    from neo4j import GraphDatabase

    timestamp = datetime.now(timezone.utc).isoformat()
    uri, user, password, database = neo4j_config()

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_event_enrichment_schema)
            session.execute_write(merge_project_and_session, project, session_id, timestamp)
            events = _fetch_unprocessed_events(
                driver,
                database,
                project_id=project.id,
                session_id=session_id,
                mode=mode,
                limit=limit,
            )
            if not events:
                return
            projection = build_event_enrichment_projection(project, session_id, mode, events)
            session.execute_write(
                _write_event_enrichment,
                project,
                session_id,
                mode,
                projection,
                timestamp,
            )


def _spawn_background(mode: str, limit: int, session_id: str) -> None:
    if not session_id or session_id == "unknown":
        return

    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--limit",
        str(limit),
        "--session-id",
        session_id,
    ]
    with open(os.devnull, "rb") as stdin, open(os.devnull, "wb") as output:
        subprocess.Popen(
            command,
            cwd=str(project_root),
            env=os.environ.copy(),
            stdin=stdin,
            stdout=output,
            stderr=output,
            start_new_session=True,
            close_fds=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["turn", "session"], required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--session-id")
    parser.add_argument(
        "--background",
        action="store_true",
        help="Spawn the enrichment processor in the background and return immediately.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)
    payload = _read_payload()
    if args.session_id:
        payload["session_id"] = args.session_id

    if args.background:
        _spawn_background(
            mode=args.mode,
            limit=args.limit,
            session_id=str(payload.get("session_id") or "unknown"),
        )
        return 0

    try:
        enrich_events(payload, mode=args.mode, limit=args.limit)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[enrich_events] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
