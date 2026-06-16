#!/usr/bin/env python3
"""Process completed project work into compact candidate learnings.

The raw hook stream remains append-only. This processor runs at Stop and
SessionEnd, reads the completed batch of events, and writes a small curated
projection for future retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import (  # noqa: E402
    ProjectRef,
    decision_id,
    decision_namespace,
    ensure_project_schema,
    fetch_project_decisions,
    fetch_project_learnings,
    learning_id,
    learning_namespace,
    llm_model,
    llm_ready,
    load_dotenv,
    merge_project_and_session,
    neo4j_config,
    normalize_scope,
    resolve_project,
    truncate,
)

import capture_query_failures  # noqa: E402


DEFAULT_MEMORY_EXTRACTION_PROMPT_NAME = "default"
MEMORY_EXTRACTION_PROMPT_TOKENS = (
    "[[PROJECT_NAME]]",
    "[[PROJECT_ID]]",
    "[[MODE]]",
    "[[CORPUS]]",
    "[[EXISTING_MEMORY]]",
)

DEFAULT_MEMORY_EXTRACTION_PROMPT = """Project: [[PROJECT_NAME]] ([[PROJECT_ID]])
Processing scope: [[MODE]]

Current completed work:
[[CORPUS]]

[[EXISTING_MEMORY]]

Decide whether this completed work contains durable memory worth storing.
Prefer updating existing memory when it is materially the same idea. Create a new
item only for a distinct reusable learning or major decision. Return ignore when
the work is routine, transient, or already covered without needing reinforcement.

Classify each candidate into the most specific applicable bucket. Do not record
the same signal in multiple buckets. Use this routing precedence:
1. decisions: explicit decisions ("Decision:", "we decided", "going forward
   must/should") or stable policy choices with implementation impact.
   Do not also create a learning for the same signal.
2. learnings: reusable facts, environment quirks, domain observations, durable
   user preferences, or task patterns that are not better represented as a
   decision.

Every learning has a scope:
- "user": a durable fact about the *person* you are working with that holds
  across projects — their role, communication/workflow preferences, broad
  interests, recurring constraints, or domain priorities. Only record these when
  the signal is explicit or repeatedly reinforced.
- "project": a fact, quirk, observation, or task pattern specific to this
  project or its environment.
Default to "project" unless the signal is clearly about the person themselves.
Never store secrets, sensitive personal data, transient details, or one-off task
context in either scope.

Every decision also has a scope:
- "user": a stable working agreement, preference, or operating policy about how
  to collaborate with this person across projects.
- "project": a project-specific policy or implementation choice.
Default to "project" unless the decision is clearly about the person or their
cross-project working preferences.

Return JSON only with this shape:
{
  "learnings": [
    {
      "action": "create|update|ignore",
      "existing_id": "learning id when action is update, otherwise null",
      "scope": "project|user",
      "text": "concise durable learning, or null",
      "task_pattern": "short reusable task pattern, or null",
      "confidence": 0.0,
      "reason": "why this action"
    }
  ],
  "decisions": [
    {
      "action": "create|update|ignore",
      "existing_id": "decision id when action is update, otherwise null",
      "scope": "project|user",
      "text": "concise major decision, or null",
      "rationale": "why the decision matters, or null",
      "task_pattern": "short reusable task pattern, or null",
      "related_learning_id": "optional related learning id, or null",
      "confidence": 0.0,
      "reason": "why this action"
    }
  ]
}
"""


def _search_query(*values: object) -> str:
    words: list[str] = []
    seen: set[str] = set()
    for value in values:
        for word in re_words(str(value or "")):
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
            if len(words) >= 12:
                return " ".join(words)
    return " ".join(words)


def re_words(value: str) -> list[str]:
    import re

    return re.findall(r"[a-zA-Z0-9]{3,}", value.lower())


def _read_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def _event_text(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    return value if isinstance(value, str) else ""


def _event_corpus(events: list[dict[str, Any]], limit: int = 12000) -> str:
    parts: list[str] = []
    for event in events:
        event_name = _event_text(event, "event_name")
        if event_name:
            parts.append(f"Event: {event_name}")
        for key in ("prompt", "last_assistant_message", "tool_name", "tool_input", "tool_response"):
            text = _event_text(event, key)
            if text:
                parts.append(f"{key}: {truncate(text, 1200)}")
    joined = "\n".join(parts)
    if len(joined) <= limit:
        return joined
    return "[earlier events elided]\n" + joined[-(limit - len("[earlier events elided]\n")):]


def _format_existing_memory(
    learnings: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> str:
    lines: list[str] = ["Existing similar learnings:"]
    if learnings:
        for item in learnings:
            lines.append(
                "- "
                f"id={item.get('id')}; "
                f"scope={item.get('scope') or 'project'}; "
                f"status={item.get('status')}; "
                f"task_pattern={item.get('task_pattern')}; "
                f"text={truncate(str(item.get('text') or ''), 300)}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Existing similar decisions:")
    if decisions:
        for item in decisions:
            lines.append(
                "- "
                f"id={item.get('id')}; "
                f"scope={item.get('scope') or 'project'}; "
                f"task_pattern={item.get('task_pattern')}; "
                f"text={truncate(str(item.get('text') or ''), 260)}; "
                f"rationale={truncate(str(item.get('rationale') or ''), 260)}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def memory_extraction_prompt_is_valid(content: str) -> bool:
    return all(token in content for token in MEMORY_EXTRACTION_PROMPT_TOKENS)


def render_memory_extraction_prompt(
    template: str,
    project: ProjectRef,
    mode: str,
    events: list[dict[str, Any]],
    similar_learnings: list[dict[str, Any]],
    similar_decisions: list[dict[str, Any]],
) -> str:
    corpus = _event_corpus(events)
    existing = _format_existing_memory(similar_learnings, similar_decisions)
    replacements = {
        "[[PROJECT_NAME]]": project.name,
        "[[PROJECT_ID]]": project.id,
        "[[MODE]]": mode,
        "[[CORPUS]]": corpus,
        "[[EXISTING_MEMORY]]": existing,
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def load_or_seed_memory_extraction_prompt(
    tx,
    name: str,
    default_content: str,
    now: str,
) -> dict[str, Any]:
    record = tx.run(
        """
        MERGE (p:MemoryExtractionPrompt {name: $name})
        ON CREATE SET p.content = $default_content,
                      p.version = 1,
                      p.created_at = datetime($now),
                      p.updated_at = datetime($now)
        SET p.content = coalesce(p.content, $default_content),
            p.version = coalesce(p.version, 1)
        RETURN p.content AS content,
               coalesce(p.version, 1) AS version
        """,
        name=name,
        default_content=default_content,
        now=now,
    ).single()
    if not record:
        return {"content": default_content, "version": 1}
    return {"content": str(record["content"] or default_content), "version": int(record["version"])}


def build_memory_extraction_prompt(
    project: ProjectRef,
    mode: str,
    events: list[dict[str, Any]],
    similar_learnings: list[dict[str, Any]],
    similar_decisions: list[dict[str, Any]],
    template: str | None = None,
) -> str:
    active_template = template or DEFAULT_MEMORY_EXTRACTION_PROMPT
    if not memory_extraction_prompt_is_valid(active_template):
        active_template = DEFAULT_MEMORY_EXTRACTION_PROMPT
    return render_memory_extraction_prompt(
        active_template,
        project,
        mode,
        events,
        similar_learnings,
        similar_decisions,
    )


def _json_from_llm_text(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"learnings": [], "decisions": []}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {"learnings": [], "decisions": []}
    if not isinstance(parsed, dict):
        return {"learnings": [], "decisions": []}
    return parsed


def ask_llm_for_memory_actions(prompt: str) -> dict[str, Any]:
    if not llm_ready():
        return {"learnings": [], "decisions": []}

    import litellm

    response = litellm.completion(
        model=llm_model(),
        messages=[
            {
                "role": "system",
                "content": "You extract durable project memory for an agent. Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return _json_from_llm_text(content)


def _confidence(value: object, default: float = 0.6) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _scope_from_action(item: dict[str, Any], existing_id: str) -> str:
    scope_value = str(item.get("scope") or "").strip().lower()
    if scope_value:
        return normalize_scope(scope_value)
    if existing_id.startswith(("learning:user:", "decision:user:")):
        return "user"
    return "project"


def _memory_rows_from_actions(
    project: ProjectRef,
    mode: str,
    actions: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    learning_rows: list[dict[str, Any]] = []
    for item in actions.get("learnings") or []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").lower()
        if action not in {"create", "update"}:
            continue
        text = truncate(str(item.get("text") or "").strip())
        existing_id = str(item.get("existing_id") or "").strip()
        if action == "update" and not existing_id:
            continue
        if action == "create" and not text:
            continue
        scope = _scope_from_action(item, existing_id)
        row_id = (
            existing_id
            if action == "update"
            else learning_id(learning_namespace(project.id, scope), text)
        )
        learning_rows.append(
            {
                "id": row_id,
                "action": action,
                "text": text or None,
                "task_pattern": item.get("task_pattern"),
                "confidence": _confidence(item.get("confidence")),
                "status": "candidate",
                "scope": scope,
                "source": f"project_{mode}_llm",
                "summary": text or item.get("reason"),
                "reason": item.get("reason"),
            }
        )

    decision_rows: list[dict[str, Any]] = []
    for item in actions.get("decisions") or []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").lower()
        if action not in {"create", "update"}:
            continue
        text = truncate(str(item.get("text") or "").strip())
        existing_id = str(item.get("existing_id") or "").strip()
        if action == "update" and not existing_id:
            continue
        if action == "create" and not text:
            continue
        scope = _scope_from_action(item, existing_id)
        row_id = (
            existing_id
            if action == "update"
            else decision_id(decision_namespace(project.id, scope), text)
        )
        decision_rows.append(
            {
                "id": row_id,
                "action": action,
                "text": text or None,
                "rationale": truncate(str(item.get("rationale") or ""), 500) or None,
                "task_pattern": item.get("task_pattern"),
                "confidence": _confidence(item.get("confidence")),
                "scope": scope,
                "source": f"project_{mode}_llm",
                "summary": text or item.get("reason"),
                "related_learning_id": item.get("related_learning_id"),
                "reason": item.get("reason"),
            }
        )
    return learning_rows[:3], decision_rows[:5]


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
        MATCH (s:Session {session_id: $session_id})-[:HAS_EVENT]->(e:SessionEvent)
        WHERE NOT EXISTS {
            MATCH (:ProjectProcessing {project_id: $project_id, mode: $mode})
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


def _write_processing(
    tx,
    project: ProjectRef,
    session_id: str,
    mode: str,
    events: list[dict[str, Any]],
    learning_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    memory_extraction_prompt_name: str,
    memory_extraction_prompt_version: int,
    timestamp: str,
) -> None:
    event_ids = [event["event_id"] for event in events if event.get("event_id")]
    digest = sha1("\n".join(event_ids).encode("utf-8")).hexdigest()[:16]
    processing_id = f"processing:{project.id}:{session_id}:{mode}:{digest}"
    summary = (
        f"Processed {len(event_ids)} {mode} events and produced "
        f"{len(learning_rows)} learning actions and {len(decision_rows)} decision actions."
    )

    tx.run(
        """
        MERGE (p:Project {id: $project_id})
        ON CREATE SET p.created_at = $timestamp
        SET p.updated_at = $timestamp,
            p.last_activity_at = $timestamp
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET s.created_at = $timestamp
        MERGE (p)-[:HAS_SESSION]->(s)
        MERGE (pp:ProjectProcessing {id: $processing_id})
        ON CREATE SET pp.created_at = $timestamp
        SET pp.project_id = $project_id,
            pp.session_id = $session_id,
            pp.mode = $mode,
            pp.status = 'ok',
            pp.type = 'processing',
            pp.event_count = $event_count,
            pp.learning_count = $learning_count,
            pp.decision_count = $decision_count,
            pp.memory_extraction_prompt_name = $memory_extraction_prompt_name,
            pp.memory_extraction_prompt_version = $memory_extraction_prompt_version,
            pp.summary = $summary,
            pp.updated_at = $timestamp
        MERGE (p)-[:HAS_PROCESSING]->(pp)
        MERGE (s)-[:HAS_PROCESSING]->(pp)
        WITH pp
        UNWIND $event_ids AS event_id
        MATCH (e:SessionEvent {event_id: event_id})
        MERGE (pp)-[:PROCESSED_EVENT]->(e)
        """,
        project_id=project.id,
        session_id=session_id,
        processing_id=processing_id,
        mode=mode,
        event_count=len(event_ids),
        learning_count=len(learning_rows),
        decision_count=len(decision_rows),
        memory_extraction_prompt_name=memory_extraction_prompt_name,
        memory_extraction_prompt_version=memory_extraction_prompt_version,
        summary=summary,
        event_ids=event_ids,
        timestamp=timestamp,
    )

    learning_create_rows = [row for row in learning_rows if row.get("action") == "create"]
    learning_update_rows = [row for row in learning_rows if row.get("action") == "update"]
    decision_create_rows = [row for row in decision_rows if row.get("action") == "create"]
    decision_update_rows = [row for row in decision_rows if row.get("action") == "update"]

    if learning_create_rows:
        tx.run(
            """
            MATCH (p:Project {id: $project_id})
            MATCH (s:Session {session_id: $session_id})
            MATCH (pp:ProjectProcessing {id: $processing_id})
            UNWIND $learnings AS row
            MERGE (l:Learning {id: row.id})
            ON CREATE SET l.created_at = $timestamp,
                          l.text = row.text,
                          l.task_pattern = row.task_pattern,
                          l.summary = row.summary,
                          l.source = row.source,
                          l.status = row.status,
                          l.scope = row.scope,
                          l.use_count = 0,
                          l.support_count = 0
            SET l.text = row.text,
                l.task_pattern = row.task_pattern,
                l.summary = row.summary,
                l.last_source = row.source,
                l.last_source_session_id = $session_id,
                l.last_reason = row.reason,
                l.project_id = $project_id,
                l.updated_at = $timestamp,
                l.support_count = coalesce(l.support_count, 0) + 1,
                l.confidence = CASE
                    WHEN coalesce(l.confidence, 0.0) < row.confidence THEN row.confidence
                    ELSE l.confidence
                END
            MERGE (p)-[:HAS_LEARNING]->(l)
            MERGE (l)-[:FROM_SESSION]->(s)
            MERGE (pp)-[:PRODUCED_LEARNING]->(l)
            """,
            project_id=project.id,
            session_id=session_id,
            processing_id=processing_id,
            learnings=learning_create_rows,
            timestamp=timestamp,
        )

    if learning_update_rows:
        tx.run(
            """
            MATCH (p:Project {id: $project_id})
            MATCH (s:Session {session_id: $session_id})
            MATCH (pp:ProjectProcessing {id: $processing_id})
            UNWIND $learnings AS row
            MATCH (l:Learning {id: row.id})
            SET l.text = coalesce(row.text, l.text),
                l.task_pattern = coalesce(row.task_pattern, l.task_pattern),
                l.summary = coalesce(row.summary, l.summary),
                l.last_source = row.source,
                l.last_source_session_id = $session_id,
                l.last_reason = row.reason,
                l.project_id = $project_id,
                l.updated_at = $timestamp,
                l.support_count = coalesce(l.support_count, 0) + 1,
                l.confidence = CASE
                    WHEN coalesce(l.confidence, 0.0) < row.confidence THEN row.confidence
                    ELSE l.confidence
                END
            MERGE (p)-[:HAS_LEARNING]->(l)
            MERGE (l)-[:FROM_SESSION]->(s)
            MERGE (pp)-[:UPDATED_LEARNING]->(l)
            """,
            project_id=project.id,
            session_id=session_id,
            processing_id=processing_id,
            learnings=learning_update_rows,
            timestamp=timestamp,
        )

    if decision_create_rows:
        tx.run(
            """
            MATCH (p:Project {id: $project_id})
            MATCH (s:Session {session_id: $session_id})
            MATCH (pp:ProjectProcessing {id: $processing_id})
            UNWIND $decisions AS row
            MERGE (d:Decision {id: row.id})
            ON CREATE SET d.created_at = $timestamp,
                          d.text = row.text,
                          d.rationale = row.rationale,
                          d.task_pattern = row.task_pattern,
                          d.summary = row.summary,
                          d.source = row.source,
                          d.scope = row.scope,
                          d.support_count = 0
            SET d.text = row.text,
                d.rationale = row.rationale,
                d.task_pattern = row.task_pattern,
                d.summary = row.summary,
                d.scope = row.scope,
                d.last_source = row.source,
                d.last_source_session_id = $session_id,
                d.last_reason = row.reason,
                d.project_id = $project_id,
                d.updated_at = $timestamp,
                d.support_count = coalesce(d.support_count, 0) + 1,
                d.confidence = CASE
                    WHEN coalesce(d.confidence, 0.0) < row.confidence THEN row.confidence
                    ELSE d.confidence
                END
            MERGE (p)-[:HAS_DECISION]->(d)
            MERGE (d)-[:FROM_SESSION]->(s)
            MERGE (pp)-[:PRODUCED_DECISION]->(d)
            """,
            project_id=project.id,
            session_id=session_id,
            processing_id=processing_id,
            decisions=decision_create_rows,
            timestamp=timestamp,
        )

    if decision_update_rows:
        tx.run(
            """
            MATCH (p:Project {id: $project_id})
            MATCH (s:Session {session_id: $session_id})
            MATCH (pp:ProjectProcessing {id: $processing_id})
            UNWIND $decisions AS row
            MATCH (d:Decision {id: row.id})
            SET d.text = coalesce(row.text, d.text),
                d.rationale = coalesce(row.rationale, d.rationale),
                d.task_pattern = coalesce(row.task_pattern, d.task_pattern),
                d.summary = coalesce(row.summary, d.summary),
                d.scope = coalesce(row.scope, d.scope, 'project'),
                d.last_source = row.source,
                d.last_source_session_id = $session_id,
                d.last_reason = row.reason,
                d.project_id = $project_id,
                d.updated_at = $timestamp,
                d.support_count = coalesce(d.support_count, 0) + 1,
                d.confidence = CASE
                    WHEN coalesce(d.confidence, 0.0) < row.confidence THEN row.confidence
                    ELSE d.confidence
                END
            MERGE (p)-[:HAS_DECISION]->(d)
            MERGE (d)-[:FROM_SESSION]->(s)
            MERGE (pp)-[:UPDATED_DECISION]->(d)
            """,
            project_id=project.id,
            session_id=session_id,
            processing_id=processing_id,
            decisions=decision_update_rows,
            timestamp=timestamp,
        )

    if learning_rows and decision_rows:
        tx.run(
            """
            UNWIND $decisions AS decision
            UNWIND $learnings AS learning
            WITH decision, learning
            WHERE decision.task_pattern IS NOT NULL
              AND decision.task_pattern = learning.task_pattern
            MATCH (d:Decision {id: decision.id})
            MATCH (l:Learning {id: learning.id})
            MERGE (d)-[:INFORMS_LEARNING]->(l)
            """,
            decisions=decision_rows,
            learnings=learning_rows,
        )


def _write_processing_error(
    tx,
    project: ProjectRef,
    session_id: str,
    mode: str,
    events: list[dict[str, Any]],
    error_text: str,
    timestamp: str,
) -> None:
    """Record a failed processing run as a :ProjectProcessing {status:'error'} node.

    Mirrors the success node's label and provenance edges so failures are queryable
    alongside successful runs, but deliberately omits PROCESSED_EVENT edges: the
    events stay unprocessed and backfill on a later healthy run.
    """
    event_ids = [event["event_id"] for event in events if event.get("event_id")]
    seed = "\n".join(event_ids) if event_ids else f"{session_id}:{mode}:{timestamp}"
    digest = sha1(seed.encode("utf-8")).hexdigest()[:16]
    processing_id = f"processing:{project.id}:{session_id}:{mode}:error:{digest}"
    summary = (
        f"Memory extraction failed for {len(event_ids)} {mode} events "
        f"({truncate(error_text, 200)}); events left unprocessed for backfill."
    )

    tx.run(
        """
        MERGE (p:Project {id: $project_id})
        ON CREATE SET p.created_at = $timestamp
        SET p.updated_at = $timestamp,
            p.last_activity_at = $timestamp
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET s.created_at = $timestamp
        MERGE (p)-[:HAS_SESSION]->(s)
        MERGE (pp:ProjectProcessing {id: $processing_id})
        ON CREATE SET pp.created_at = $timestamp,
                      pp.first_error_at = $timestamp
        SET pp.project_id = $project_id,
            pp.session_id = $session_id,
            pp.mode = $mode,
            pp.status = 'error',
            pp.type = 'error',
            pp.event_count = $event_count,
            pp.learning_count = 0,
            pp.decision_count = 0,
            pp.error = $error_text,
            pp.summary = $summary,
            pp.attempt_count = coalesce(pp.attempt_count, 0) + 1,
            pp.last_error_at = $timestamp,
            pp.updated_at = $timestamp
        MERGE (p)-[:HAS_PROCESSING]->(pp)
        MERGE (s)-[:HAS_PROCESSING]->(pp)
        """,
        project_id=project.id,
        session_id=session_id,
        processing_id=processing_id,
        mode=mode,
        event_count=len(event_ids),
        error_text=truncate(error_text, 1000),
        summary=summary,
        timestamp=timestamp,
    )


def process_project(payload: dict[str, Any], mode: str, limit: int) -> None:
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
            session.execute_write(ensure_project_schema)
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
            try:
                prompt_name = DEFAULT_MEMORY_EXTRACTION_PROMPT_NAME
                prompt_record = session.execute_write(
                    load_or_seed_memory_extraction_prompt,
                    name=prompt_name,
                    default_content=DEFAULT_MEMORY_EXTRACTION_PROMPT,
                    now=timestamp,
                )
                prompt_template = str(prompt_record.get("content") or DEFAULT_MEMORY_EXTRACTION_PROMPT)
                prompt_version = int(prompt_record.get("version") or 1)
                if not memory_extraction_prompt_is_valid(prompt_template):
                    print(
                        "[process_project] stored memory extraction prompt is missing "
                        "required tokens; using default template for this run",
                        file=sys.stderr,
                    )
                    prompt_template = DEFAULT_MEMORY_EXTRACTION_PROMPT
                corpus = _event_corpus(events)
                search_query = _search_query(corpus)
                similar_learnings = fetch_project_learnings(
                    driver,
                    database,
                    project_id=project.id,
                    query=search_query,
                    statuses=["approved", "candidate"],
                    limit=8,
                )
                similar_decisions = fetch_project_decisions(
                    driver,
                    database,
                    project_id=project.id,
                    query=search_query,
                    limit=8,
                )
                prompt = build_memory_extraction_prompt(
                    project,
                    mode,
                    events,
                    similar_learnings,
                    similar_decisions,
                    template=prompt_template,
                )
                actions = ask_llm_for_memory_actions(prompt)
                learning_rows, decision_rows = _memory_rows_from_actions(
                    project,
                    mode,
                    actions,
                )
                session.execute_write(
                    _write_processing,
                    project,
                    session_id,
                    mode,
                    events,
                    learning_rows,
                    decision_rows,
                    prompt_name,
                    prompt_version,
                    timestamp,
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                print(
                    f"[process_project] memory extraction failed; recording error node: {error_text}",
                    file=sys.stderr,
                )
                error_timestamp = datetime.now(timezone.utc).isoformat()
                try:
                    session.execute_write(
                        _write_processing_error,
                        project,
                        session_id,
                        mode,
                        events,
                        error_text,
                        error_timestamp,
                    )
                except Exception as write_exc:
                    print(
                        f"[process_project] failed to record error node: {write_exc}",
                        file=sys.stderr,
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
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--session-id")
    parser.add_argument(
        "--background",
        action="store_true",
        help="Spawn the processor in the background and return immediately.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    payload = _read_payload()
    if args.session_id:
        payload["session_id"] = args.session_id

    # Deterministic query-failure capture runs synchronously in the foreground
    # hook: it is LLM-free, and only the foreground payload carries
    # transcript_path (the background re-invocation gets only --session-id).
    # PostToolUse never fires for isError tool results, so the transcript is the
    # sole complete record of failed queries.
    transcript_path = payload.get("transcript_path")
    if transcript_path:
        try:
            captured = capture_query_failures.capture_transcript(
                transcript_path,
                str(payload.get("session_id") or "unknown"),
                payload,
                project_root,
            )
            if captured:
                print(f"[process_project] captured {captured} query issue(s) from transcript")
        except Exception as exc:  # pragma: no cover - hook must never crash the session
            print(
                f"[process_project] transcript failure capture failed: {exc}",
                file=sys.stderr,
            )

    if args.background:
        _spawn_background(
            mode=args.mode,
            limit=args.limit,
            session_id=str(payload.get("session_id") or "unknown"),
        )
        return 0

    try:
        process_project(payload, mode=args.mode, limit=args.limit)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[process_project] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
