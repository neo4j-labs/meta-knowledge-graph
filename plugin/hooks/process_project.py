#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0", "litellm>=1.40.0"]
# ///
"""Process completed project work into compact candidate learnings.

The raw hook stream remains append-only. This processor runs at Stop, reads the
completed batch of events, and writes a small curated projection for future
retrieval.
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
    embed_text,
    embed_texts,
    extraction_model_label,
    fetch_project_decisions,
    fetch_project_learnings,
    has_project_work_events,
    in_extraction_subprocess,
    learning_id,
    learning_namespace,
    llm_complete,
    llm_readiness_status,
    llm_model,
    load_mkg_env,
    merge_project_and_session,
    neo4j_config,
    normalize_scope,
    observation_id,
    project_env,
    resolve_llm_model,
    resolve_project,
    truncate,
)

import capture_query_failures  # noqa: E402
from consistency_gate import (  # noqa: E402
    attach_candidate_embeddings,
    ensure_memory_vector_indexes,
    run_consistency_gate,
    sweep_ungated_candidates,
)


# Stable, verbose provenance tag for memory written by the Stop-hook extractor.
# Kept uniform across modes and clients (Codex and Claude) so learnings/decisions
# minted by the hook are always identifiable as `hooks-stop`, paralleling the
# MCP tool's `agent-mcp` tag.
HOOK_LEARNING_SOURCE = "hooks-stop"

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
  across projects — their role and identity, communication/workflow
  preferences, language and tooling preferences, coding and writing style,
  broad interests, recurring constraints, or domain priorities. When the signal
  is unambiguously about the person, scope it to "user" on the first explicit
  mention; do not wait for it to be repeated. Require repetition only for
  borderline traits you are inferring rather than ones the person stated.
- "project": a fact, quirk, observation, or task pattern specific to this
  project or its environment.
Default to "project" only when it is genuinely unclear whether the signal is
about the person or the project; a clearly personal fact always takes "user".
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


# Episodic observations: unconditional per-window timeline records. Unlike the
# memory extraction prompt this template is a code constant, not a stored /
# user-evolvable :MemoryExtractionPrompt — "be selective" (memory) and "always
# record" (episodes) must never share a template.
OBSERVATION_TYPES = (
    "change",
    "bugfix",
    "feature",
    "refactor",
    "discovery",
    "decision",
    "problem",
)
DEFAULT_OBSERVATION_TYPE = "change"
MAX_OBSERVATIONS_PER_WINDOW = 3
MAX_OBSERVATION_FACTS = 6

DEFAULT_OBSERVATION_EXTRACTION_PROMPT = """Project: [[PROJECT_NAME]] ([[PROJECT_ID]])

Completed work window:
[[CORPUS]]

Write an episodic record of this work window for the project timeline.
Unlike durable memory, this is unconditional: always return at least one
observation describing what happened, even for routine work. Return up to
three when the window contains clearly distinct efforts.

Each observation:
- "type": one of change | bugfix | feature | refactor | discovery | decision | problem.
  Use "decision" for the moment a choice was made, "discovery" for new
  understanding gained, "problem" for an issue that was hit, otherwise the
  closest work type.
- "title": one short line naming what happened.
- "facts": 2-6 short, concrete, self-contained statements about what was done,
  found, or produced. No speculation.
- "narrative": 2-4 sentences telling what was asked, what was done, and how it
  ended (done, in progress, or blocked).

Describe only what the window shows. Return JSON only with this shape:
{"observations": [{"type": "...", "title": "...", "facts": ["..."], "narrative": "..."}]}
"""


def _read_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def _event_text(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    return value if isinstance(value, str) else ""


def _ts_sort_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return text


def _transcript_records(snapshot: str):
    for line in snapshot.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _dialog_entry(record: Any) -> tuple[str, str, str] | None:
    """Classify a transcript record as ('user'|'assistant', timestamp, text).

    Conversation text only: tool_use/tool_result blocks, Codex function_call /
    mcp_tool_call_end records, and thinking blocks never produce an entry.
    """
    if not isinstance(record, dict):
        return None
    timestamp = str(record.get("timestamp") or "")

    # Claude-style records: {message: {role, content}}.
    message = record.get("message")
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        texts: list[str] = []
        has_tool_result = False
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_result":
                    has_tool_result = True
                elif block_type in ("text", "output_text", "input_text"):
                    if isinstance(block.get("text"), str):
                        texts.append(block["text"])
        text = "\n".join(part for part in texts if part.strip()).strip()
        if role == "assistant" and text:
            return ("assistant", timestamp, text)
        # A user record carrying tool_result blocks is a tool output envelope,
        # not a user message.
        if role == "user" and text and not has_tool_result:
            return ("user", timestamp, text)
        return None

    # Codex rollout records: {type, payload}.
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    payload_type = payload.get("type")
    if record.get("type") == "response_item" and payload_type == "message":
        role = payload.get("role")
        content = payload.get("content")
        texts = []
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") in ("text", "output_text", "input_text")
                    and isinstance(block.get("text"), str)
                ):
                    texts.append(block["text"])
        text = "\n".join(part for part in texts if part.strip()).strip()
        if role == "assistant" and text:
            return ("assistant", timestamp, text)
        if role == "user" and text:
            return ("user", timestamp, text)
        return None
    if record.get("type") == "event_msg" and payload_type in ("agent_message", "user_message"):
        raw = payload.get("message")
        if not isinstance(raw, str):
            raw = payload.get("text") if isinstance(payload.get("text"), str) else ""
        if raw.strip():
            kind = "assistant" if payload_type == "agent_message" else "user"
            return (kind, timestamp, raw.strip())
    return None


def _window_assistant_texts(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(timestamp, text) of assistant messages written during this event window.

    Sources the transcript snapshots log_event stores on events. Snapshots are
    grouped per transcript file (parent and subagent sessions each have their
    own); within a group the window's new content is the last snapshot minus
    the first-snapshot prefix, falling back to everything after the last user
    message when no prefix relation holds. Texts equal to a window event's
    last_assistant_message are dropped — that field is already in the corpus.
    """
    try:
        known = {
            str(event.get("last_assistant_message") or "").strip()
            for event in events
            if event.get("last_assistant_message")
        }
        groups: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            snapshot = event.get("transcript")
            if isinstance(snapshot, str) and snapshot:
                key = str(event.get("transcript_path") or "")
                groups.setdefault(key, []).append(event)

        results: list[tuple[str, str]] = []
        for group in groups.values():
            first = str(group[0].get("transcript") or "")
            last = str(group[-1].get("transcript") or "")
            if len(group) > 1 and last.startswith(first):
                segment = last[len(first):]
                entries = [
                    entry
                    for entry in (_dialog_entry(r) for r in _transcript_records(segment))
                    if entry
                ]
                selected = [(ts, text) for kind, ts, text in entries if kind == "assistant"]
            else:
                entries = [
                    entry
                    for entry in (_dialog_entry(r) for r in _transcript_records(last))
                    if entry
                ]
                last_user = -1
                for index, (kind, _, _) in enumerate(entries):
                    if kind == "user":
                        last_user = index
                selected = [
                    (ts, text)
                    for kind, ts, text in entries[last_user + 1 :]
                    if kind == "assistant"
                ]
            current_ts = str(group[0].get("timestamp") or "")
            for ts, text in selected:
                current_ts = ts or current_ts
                if text.strip() in known:
                    continue
                results.append((current_ts, text))
        return results
    except Exception:  # pragma: no cover - corpus building must never fail extraction
        return []


def _event_corpus(events: list[dict[str, Any]], limit: int = 12000) -> str:
    entries: list[tuple[str, int, str]] = []
    for order, event in enumerate(events):
        parts: list[str] = []
        event_name = _event_text(event, "event_name")
        if event_name:
            parts.append(f"Event: {event_name}")
        # Tool outputs are deliberately excluded: memory extraction reads only
        # user prompts, assistant messages, and tool inputs.
        for key in ("prompt", "last_assistant_message", "tool_name", "tool_input"):
            text = _event_text(event, key)
            if text:
                parts.append(f"{key}: {truncate(text, 1200)}")
        if parts:
            entries.append((_ts_sort_key(event.get("timestamp")), order, "\n".join(parts)))
    for ts, text in _window_assistant_texts(events):
        entries.append((_ts_sort_key(ts), len(events), f"assistant: {truncate(text, 1200)}"))
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    joined = "\n".join(block for _, _, block in entries)
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
    actions, _, _ = ask_llm_for_memory_actions_with_model(prompt)
    return actions


def ask_llm_for_memory_actions_with_model(
    prompt: str,
    model: str | None = None,
) -> tuple[dict[str, Any], str | None, dict[str, str | None]]:
    model_name = model or llm_model()
    model_label = extraction_model_label(model_name)
    ready, reason = llm_readiness_status(model_name)
    if not ready:
        return (
            {"learnings": [], "decisions": []},
            model_label,
            {"status": "skipped", "skip_reason": reason, "error": None},
        )

    content = llm_complete(
        [
            {
                "role": "system",
                "content": "You extract durable project memory for an agent. Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model_name,
    )
    return (
        _json_from_llm_text(content),
        model_label,
        {"status": "called", "skip_reason": None, "error": None},
    )


def _llm_model_for_processing_events(events: list[dict[str, Any]]) -> str:
    clients = [str(event.get("client") or "").strip().lower() for event in events]
    client = "claude_code" if "claude_code" in clients else next(
        (value for value in clients if value),
        None,
    )
    return llm_model(client=client)


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
    llm_model: str | None = None,
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
                "source": HOOK_LEARNING_SOURCE,
                "summary": text or item.get("reason"),
                "reason": item.get("reason"),
                "llm_model": llm_model,
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
                "status": "candidate",
                "scope": scope,
                "source": HOOK_LEARNING_SOURCE,
                "summary": text or item.get("reason"),
                "related_learning_id": item.get("related_learning_id"),
                "reason": item.get("reason"),
                "llm_model": llm_model,
            }
        )
    return learning_rows[:3], decision_rows[:5]


def _window_digest(events: list[dict[str, Any]]) -> str:
    event_ids = [event["event_id"] for event in events if event.get("event_id")]
    return sha1("\n".join(event_ids).encode("utf-8")).hexdigest()[:16]


def _window_bounds(events: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    timestamps = sorted(
        _ts_sort_key(event.get("timestamp"))
        for event in events
        if event.get("timestamp")
    )
    if not timestamps:
        return None, None
    return timestamps[0], timestamps[-1]


def build_observation_prompt(project: ProjectRef, corpus: str) -> str:
    return (
        DEFAULT_OBSERVATION_EXTRACTION_PROMPT.replace("[[PROJECT_NAME]]", project.name)
        .replace("[[PROJECT_ID]]", project.id)
        .replace("[[CORPUS]]", corpus)
    )


def _observation_items_from_text(text: str) -> list[dict[str, Any]]:
    parsed = _json_from_llm_text(text)
    items = parsed.get("observations")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def ask_llm_for_observations(
    prompt: str,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    """One unconditional episodic-record call. No fallback: when the LLM is
    unavailable or fails, the window simply gets no observation."""
    model_name = model or llm_model()
    ready, reason = llm_readiness_status(model_name)
    if not ready:
        return [], {"status": "skipped", "skip_reason": reason}

    content = llm_complete(
        [
            {
                "role": "system",
                "content": (
                    "You write concise episodic records of completed agent work. "
                    "Return strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model=model_name,
    )
    return _observation_items_from_text(content), {"status": "called", "skip_reason": None}


def _observation_rows_from_items(
    project: ProjectRef,
    session_id: str,
    events: list[dict[str, Any]],
    items: list[dict[str, Any]],
    llm_model: str | None = None,
) -> list[dict[str, Any]]:
    digest = _window_digest(events)
    started_at, ended_at = _window_bounds(events)
    rows: list[dict[str, Any]] = []
    for item in items:
        title = truncate(str(item.get("title") or "").strip(), 200)
        if not title:
            continue
        type_value = str(item.get("type") or "").strip().lower()
        if type_value not in OBSERVATION_TYPES:
            type_value = DEFAULT_OBSERVATION_TYPE
        raw_facts = item.get("facts") if isinstance(item.get("facts"), list) else []
        facts = [
            truncate(str(fact).strip(), 300)
            for fact in raw_facts
            if isinstance(fact, (str, int, float)) and str(fact).strip()
        ][:MAX_OBSERVATION_FACTS]
        narrative = truncate(str(item.get("narrative") or "").strip(), 1000) or None
        rows.append(
            {
                "id": observation_id(project.id, session_id, digest, len(rows)),
                "type": type_value,
                "title": title,
                "facts": facts,
                "narrative": narrative,
                "source": HOOK_LEARNING_SOURCE,
                "llm_model": llm_model,
                "started_at": started_at,
                "ended_at": ended_at,
            }
        )
        if len(rows) >= MAX_OBSERVATIONS_PER_WINDOW:
            break
    return rows


def attach_observation_embeddings(rows: list[dict[str, Any]]) -> None:
    """Embed title + facts + narrative so observations are hybrid-searchable.
    Failures leave ``embedding`` unset; the write degrades gracefully."""
    if not rows:
        return
    texts = [
        "\n".join(
            part
            for part in (
                str(row.get("title") or ""),
                " ".join(row.get("facts") or []),
                str(row.get("narrative") or ""),
            )
            if part.strip()
        )
        for row in rows
    ]
    vectors = embed_texts(texts)
    for row, vector in zip(rows, vectors):
        if vector:
            row["embedding"] = vector


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
        WITH s, collect(DISTINCT sub) AS subs
        UNWIND ([s] + subs) AS scoped_session
        WITH DISTINCT scoped_session
        WHERE scoped_session IS NOT NULL
        MATCH (scoped_session)-[:HAS_EVENT]->(e:SessionEvent)
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
    llm_model: str | None,
    llm_status: str | None,
    llm_skip_reason: str | None,
    llm_error: str | None,
    timestamp: str,
    observation_rows: list[dict[str, Any]] | None = None,
) -> None:
    observation_rows = observation_rows or []
    event_ids = [event["event_id"] for event in events if event.get("event_id")]
    digest = _window_digest(events)
    processing_id = f"processing:{project.id}:{session_id}:{mode}:{digest}"
    summary = (
        f"Processed {len(event_ids)} {mode} events and produced "
        f"{len(learning_rows)} learning actions, {len(decision_rows)} decision actions, "
        f"and {len(observation_rows)} observations."
    )

    tx.run(
        """
        MERGE (p:Project {id: $project_id})
        ON CREATE SET p.created_at = datetime($timestamp)
        SET p.updated_at = datetime($timestamp),
            p.last_activity_at = datetime($timestamp)
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET s.created_at = datetime($timestamp)
        MERGE (p)-[:HAS_SESSION]->(s)
        MERGE (pp:ProjectProcessing {id: $processing_id})
        ON CREATE SET pp.created_at = datetime($timestamp)
        SET pp.project_id = $project_id,
            pp.session_id = $session_id,
            pp.mode = $mode,
            pp.status = 'ok',
            pp.type = 'processing',
            pp.event_count = $event_count,
            pp.learning_count = $learning_count,
            pp.decision_count = $decision_count,
            pp.observation_count = $observation_count,
            pp.memory_extraction_prompt_name = $memory_extraction_prompt_name,
            pp.memory_extraction_prompt_version = $memory_extraction_prompt_version,
            pp.llm_model = $llm_model,
            pp.llm_status = $llm_status,
            pp.llm_skip_reason = $llm_skip_reason,
            pp.llm_error = $llm_error,
            pp.summary = $summary,
            pp.updated_at = datetime($timestamp)
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
        observation_count=len(observation_rows),
        memory_extraction_prompt_name=memory_extraction_prompt_name,
        memory_extraction_prompt_version=memory_extraction_prompt_version,
        llm_model=llm_model,
        llm_status=llm_status,
        llm_skip_reason=truncate(str(llm_skip_reason or ""), 1000) or None,
        llm_error=truncate(str(llm_error or ""), 1000) or None,
        summary=summary,
        event_ids=event_ids,
        timestamp=timestamp,
    )

    tx.run(
        """
        MATCH (pp:ProjectProcessing {id: $processing_id})
        MATCH (mep:MemoryExtractionPrompt {name: $memory_extraction_prompt_name})
        SET mep.last_used_at = datetime($timestamp),
            mep.last_used_project_id = $project_id,
            mep.last_used_session_id = $session_id,
            mep.last_used_version = $memory_extraction_prompt_version,
            mep.last_used_model = coalesce($llm_model, mep.last_used_model),
            mep.last_used_llm_status = $llm_status,
            mep.last_used_llm_skip_reason = $llm_skip_reason,
            mep.last_used_llm_error = $llm_error
        MERGE (pp)-[r:USED_MEMORY_EXTRACTION_PROMPT]->(mep)
        SET r.version = $memory_extraction_prompt_version,
            r.llm_model = $llm_model,
            r.llm_status = $llm_status,
            r.llm_skip_reason = $llm_skip_reason,
            r.llm_error = $llm_error,
            r.used_at = datetime($timestamp)
        """,
        project_id=project.id,
        session_id=session_id,
        processing_id=processing_id,
        memory_extraction_prompt_name=memory_extraction_prompt_name,
        memory_extraction_prompt_version=memory_extraction_prompt_version,
        llm_model=llm_model,
        llm_status=llm_status,
        llm_skip_reason=truncate(str(llm_skip_reason or ""), 1000) or None,
        llm_error=truncate(str(llm_error or ""), 1000) or None,
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
            ON CREATE SET l.created_at = datetime($timestamp),
                          l.text = row.text,
                          l.task_pattern = row.task_pattern,
                          l.summary = row.summary,
                          l.source = row.source,
                          l.created_by_model = row.llm_model,
                          l.status = row.status,
                          l.scope = row.scope,
                          l.use_count = 0,
                          l.support_count = 0
            SET l.text = row.text,
                l.task_pattern = row.task_pattern,
                l.summary = row.summary,
                l.embedding = coalesce(row.embedding, l.embedding),
                l.last_source = row.source,
                l.last_source_session_id = $session_id,
                l.last_reason = row.reason,
                l.last_llm_model = coalesce(row.llm_model, l.last_llm_model),
                l.project_id = $project_id,
                l.updated_at = datetime($timestamp),
                l.support_count = coalesce(l.support_count, 0) + 1,
                l.confidence = CASE
                    WHEN coalesce(l.confidence, 0.0) < row.confidence THEN row.confidence
                    ELSE l.confidence
                END
            MERGE (p)-[:HAS_LEARNING]->(l)
            MERGE (l)-[:FROM_SESSION]->(s)
            MERGE (pp)-[produced:PRODUCED_LEARNING]->(l)
            SET produced.llm_model = row.llm_model,
                produced.created_at = datetime($timestamp)
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
                l.last_llm_model = coalesce(row.llm_model, l.last_llm_model),
                l.project_id = $project_id,
                l.updated_at = datetime($timestamp),
                l.support_count = coalesce(l.support_count, 0) + 1,
                l.confidence = CASE
                    WHEN coalesce(l.confidence, 0.0) < row.confidence THEN row.confidence
                    ELSE l.confidence
                END
            MERGE (p)-[:HAS_LEARNING]->(l)
            MERGE (l)-[:FROM_SESSION]->(s)
            MERGE (pp)-[updated:UPDATED_LEARNING]->(l)
            SET updated.llm_model = row.llm_model,
                updated.updated_at = datetime($timestamp)
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
            ON CREATE SET d.created_at = datetime($timestamp),
                          d.text = row.text,
                          d.rationale = row.rationale,
                          d.task_pattern = row.task_pattern,
                          d.summary = row.summary,
                          d.source = row.source,
                          d.created_by_model = row.llm_model,
                          d.status = row.status,
                          d.scope = row.scope,
                          d.support_count = 0
            SET d.text = row.text,
                d.rationale = row.rationale,
                d.task_pattern = row.task_pattern,
                d.summary = row.summary,
                d.embedding = coalesce(row.embedding, d.embedding),
                d.scope = row.scope,
                d.last_source = row.source,
                d.last_source_session_id = $session_id,
                d.last_reason = row.reason,
                d.last_llm_model = coalesce(row.llm_model, d.last_llm_model),
                d.project_id = $project_id,
                d.updated_at = datetime($timestamp),
                d.support_count = coalesce(d.support_count, 0) + 1,
                d.confidence = CASE
                    WHEN coalesce(d.confidence, 0.0) < row.confidence THEN row.confidence
                    ELSE d.confidence
                END
            MERGE (p)-[:HAS_DECISION]->(d)
            MERGE (d)-[:FROM_SESSION]->(s)
            MERGE (pp)-[produced:PRODUCED_DECISION]->(d)
            SET produced.llm_model = row.llm_model,
                produced.created_at = datetime($timestamp)
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
                d.scope = row.scope,
                d.last_source = row.source,
                d.last_source_session_id = $session_id,
                d.last_reason = row.reason,
                d.last_llm_model = coalesce(row.llm_model, d.last_llm_model),
                d.project_id = $project_id,
                d.updated_at = datetime($timestamp),
                d.support_count = coalesce(d.support_count, 0) + 1,
                d.confidence = CASE
                    WHEN coalesce(d.confidence, 0.0) < row.confidence THEN row.confidence
                    ELSE d.confidence
                END
            MERGE (p)-[:HAS_DECISION]->(d)
            MERGE (d)-[:FROM_SESSION]->(s)
            MERGE (pp)-[updated:UPDATED_DECISION]->(d)
            SET updated.llm_model = row.llm_model,
                updated.updated_at = datetime($timestamp)
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

    if observation_rows:
        # Episodic timeline entries: append-only, never gated, never deduped.
        # Written in this transaction so an observation can never be lost after
        # the window's events are marked PROCESSED_EVENT.
        tx.run(
            """
            MATCH (p:Project {id: $project_id})
            MATCH (s:Session {session_id: $session_id})
            MATCH (pp:ProjectProcessing {id: $processing_id})
            UNWIND $observations AS row
            MERGE (o:Observation {id: row.id})
            ON CREATE SET o.created_at = datetime($timestamp)
            SET o.type = row.type,
                o.title = row.title,
                o.facts = row.facts,
                o.narrative = row.narrative,
                o.embedding = coalesce(row.embedding, o.embedding),
                o.project_id = $project_id,
                o.session_id = $session_id,
                o.scope = 'project',
                o.status = 'recorded',
                o.source = row.source,
                o.created_by_model = coalesce(row.llm_model, o.created_by_model),
                o.started_at = CASE
                    WHEN row.started_at IS NULL THEN o.started_at
                    ELSE datetime(row.started_at)
                END,
                o.ended_at = CASE
                    WHEN row.ended_at IS NULL THEN o.ended_at
                    ELSE datetime(row.ended_at)
                END,
                o.updated_at = datetime($timestamp)
            MERGE (p)-[:HAS_OBSERVATION]->(o)
            MERGE (o)-[:FROM_SESSION]->(s)
            MERGE (pp)-[:PRODUCED_OBSERVATION]->(o)
            """,
            project_id=project.id,
            session_id=session_id,
            processing_id=processing_id,
            observations=observation_rows,
            timestamp=timestamp,
        )
        pairs = [
            {"prev": before["id"], "next": after["id"]}
            for before, after in zip(observation_rows, observation_rows[1:])
        ]
        if pairs:
            tx.run(
                """
                UNWIND $pairs AS pair
                MATCH (a:Observation {id: pair.prev})
                MATCH (b:Observation {id: pair.next})
                MERGE (a)-[:NEXT]->(b)
                """,
                pairs=pairs,
            )
        tx.run(
            """
            MATCH (p:Project {id: $project_id})
            OPTIONAL MATCH (p)-[latest:LATEST_OBSERVATION]->(prev:Observation)
            DELETE latest
            WITH p, prev
            MATCH (head:Observation {id: $head_id})
            FOREACH (_ IN CASE
                WHEN prev IS NOT NULL
                     AND NOT prev.id IN $window_ids
                THEN [1] ELSE [] END |
                MERGE (prev)-[:NEXT]->(head)
            )
            WITH p
            MATCH (tail:Observation {id: $tail_id})
            MERGE (p)-[:LATEST_OBSERVATION]->(tail)
            """,
            project_id=project.id,
            head_id=observation_rows[0]["id"],
            tail_id=observation_rows[-1]["id"],
            window_ids=[row["id"] for row in observation_rows],
        )


def _write_processing_error(
    tx,
    project: ProjectRef,
    session_id: str,
    mode: str,
    events: list[dict[str, Any]],
    error_text: str,
    llm_model: str | None,
    llm_status: str | None,
    llm_skip_reason: str | None,
    llm_error: str | None,
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
        ON CREATE SET p.created_at = datetime($timestamp)
        SET p.updated_at = datetime($timestamp),
            p.last_activity_at = datetime($timestamp)
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET s.created_at = datetime($timestamp)
        MERGE (p)-[:HAS_SESSION]->(s)
        MERGE (pp:ProjectProcessing {id: $processing_id})
        ON CREATE SET pp.created_at = datetime($timestamp),
                      pp.first_error_at = datetime($timestamp)
        SET pp.project_id = $project_id,
            pp.session_id = $session_id,
            pp.mode = $mode,
            pp.status = 'error',
            pp.type = 'error',
            pp.event_count = $event_count,
            pp.learning_count = 0,
            pp.decision_count = 0,
            pp.llm_model = $llm_model,
            pp.llm_status = $llm_status,
            pp.llm_skip_reason = $llm_skip_reason,
            pp.llm_error = $llm_error,
            pp.error = $error_text,
            pp.summary = $summary,
            pp.attempt_count = coalesce(pp.attempt_count, 0) + 1,
            pp.last_error_at = datetime($timestamp),
            pp.updated_at = datetime($timestamp)
        MERGE (p)-[:HAS_PROCESSING]->(pp)
        MERGE (s)-[:HAS_PROCESSING]->(pp)
        """,
        project_id=project.id,
        session_id=session_id,
        processing_id=processing_id,
        mode=mode,
        event_count=len(event_ids),
        llm_model=llm_model,
        llm_status=llm_status,
        llm_skip_reason=truncate(str(llm_skip_reason or ""), 1000) or None,
        llm_error=truncate(str(llm_error or ""), 1000) or None,
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
            ensure_memory_vector_indexes(driver, database)
            events = _fetch_unprocessed_events(
                driver,
                database,
                project_id=project.id,
                session_id=session_id,
                mode=mode,
                limit=limit,
            )
            if not events or not has_project_work_events(events):
                return
            session.execute_write(merge_project_and_session, project, session_id, timestamp)
            llm_model_used: str | None = None
            llm_status: str | None = None
            llm_skip_reason: str | None = None
            llm_error: str | None = None
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
                query_vector = embed_text(corpus)
                similar_learnings = fetch_project_learnings(
                    driver,
                    database,
                    project_id=project.id,
                    query=corpus,
                    statuses=["approved", "candidate"],
                    limit=8,
                    query_vector=query_vector,
                )
                similar_decisions = fetch_project_decisions(
                    driver,
                    database,
                    project_id=project.id,
                    query=corpus,
                    limit=8,
                    query_vector=query_vector,
                )
                prompt = build_memory_extraction_prompt(
                    project,
                    mode,
                    events,
                    similar_learnings,
                    similar_decisions,
                    template=prompt_template,
                )
                model_name = resolve_llm_model(_llm_model_for_processing_events(events))
                llm_model_used = extraction_model_label(model_name)
                actions, llm_model_used, llm_meta = ask_llm_for_memory_actions_with_model(
                    prompt,
                    model=model_name,
                )
                llm_status = llm_meta.get("status")
                llm_skip_reason = llm_meta.get("skip_reason")
                llm_error = llm_meta.get("error")
                learning_rows, decision_rows = _memory_rows_from_actions(
                    project,
                    mode,
                    actions,
                    llm_model=llm_model_used,
                )
                # Embed candidates before the write so each is stored searchable
                # and the consistency gate can retrieve prior approved neighbours.
                attach_candidate_embeddings(learning_rows, decision_rows)
                # Episodic observation for the same window. Failures must not
                # block semantic memory: the window then has no observation.
                observation_rows: list[dict[str, Any]] = []
                try:
                    observation_items, _observation_meta = ask_llm_for_observations(
                        build_observation_prompt(project, corpus),
                        model=model_name,
                    )
                    observation_rows = _observation_rows_from_items(
                        project,
                        session_id,
                        events,
                        observation_items,
                        llm_model=llm_model_used,
                    )
                    attach_observation_embeddings(observation_rows)
                except Exception as observation_exc:
                    observation_rows = []
                    print(
                        f"[process_project] observation extraction failed: {observation_exc}",
                        file=sys.stderr,
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
                    llm_model_used,
                    llm_status,
                    llm_skip_reason,
                    llm_error,
                    timestamp,
                    observation_rows=observation_rows,
                )
                # Auto-approval gate: promote consistent candidates, invalidate
                # older items they supersede. No-op if embeddings/judge/index
                # are unavailable (candidates simply stay 'candidate'). Runs only
                # at Stop ('turn'), never at SessionEnd ('session').
                if mode == "turn":
                    run_consistency_gate(
                        driver,
                        database,
                        project=project,
                        learning_rows=learning_rows,
                        decision_rows=decision_rows,
                        model=model_name,
                        timestamp=timestamp,
                    )
                    # Then sweep candidates that entered the graph without a
                    # gate run — MCP memory tool writes and rows an
                    # earlier judge failure skipped. This run's own rows are
                    # excluded: they were just attempted, so a row the judge
                    # failed on moments ago is not immediately retried.
                    sweep_ungated_candidates(
                        driver,
                        database,
                        project=project,
                        model=model_name,
                        timestamp=timestamp,
                        exclude_ids=[
                            row["id"]
                            for row in (*learning_rows, *decision_rows)
                            if row.get("id")
                        ],
                    )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                if llm_model_used and llm_status is None:
                    llm_status = "error"
                    llm_error = error_text
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
                        llm_model_used,
                        llm_status,
                        llm_skip_reason,
                        llm_error,
                        error_timestamp,
                    )
                except Exception as write_exc:
                    print(
                        f"[process_project] failed to record error node: {write_exc}",
                        file=sys.stderr,
                    )


def _spawn_background(
    mode: str,
    limit: int,
    session_id: str,
    project: ProjectRef | None = None,
) -> None:
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
            env=project_env(project),
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

    # No-op inside a claude_cli extraction subprocess: the nested `claude -p`
    # loads MKG's hooks, and re-running the processor here would spawn another
    # extraction (infinite recursion).
    if in_extraction_subprocess():
        return 0

    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)
    payload = _read_payload()
    if args.session_id:
        payload["session_id"] = args.session_id
    project = resolve_project(payload, project_root)

    # Deterministic query-failure capture runs synchronously in the foreground
    # hook: it is LLM-free, and only the foreground payload carries
    # transcript_path (the background re-invocation gets only --session-id).
    # Claude Code captures failures live via PostToolUseFailure, but Codex has
    # no failure event (PostToolUse never fires for isError results there), so
    # this transcript scan stays as the Codex path — and as an idempotent
    # safety net when a live hook never ran; shared stable ids make re-capture
    # converge instead of duplicating.
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
            project=project,
        )
        return 0

    try:
        process_project(payload, mode=args.mode, limit=args.limit)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[process_project] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
