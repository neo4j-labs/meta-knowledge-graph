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
    UserRef,
    dialog_entry,
    ensure_project_schema,
    embed_text,
    embed_texts,
    extraction_model_label,
    fetch_project_learnings,
    fetch_user_learnings,
    has_project_work_events,
    in_extraction_subprocess,
    is_error_tool_response,
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
    project_git_root,
    resolve_llm_model,
    resolve_project,
    resolve_user,
    tool_failure_text,
    tool_key,
    transcript_records,
    truncate,
)

from consistency_gate import (  # noqa: E402
    attach_candidate_embeddings,
    ensure_memory_vector_indexes,
    run_consistency_gate,
    sweep_ungated_candidates,
)


# Stable, verbose provenance tag for memory written by the Stop-hook extractor.
# Kept uniform across modes and clients (Codex and Claude) so learnings minted by
# the hook are always identifiable as `hooks-stop`, paralleling the MCP tool's
# `agent-mcp` tag.
HOOK_LEARNING_SOURCE = "hooks-stop"

# The memory-extraction prompt is a fixed code constant — there is no
# Neo4j-backed, self-improving prompt loop. Edit the template here to change it.
DEFAULT_MEMORY_EXTRACTION_PROMPT = """Project: [[PROJECT_NAME]] ([[PROJECT_ID]])
Processing scope: [[MODE]]

Current completed work:
[[CORPUS]]

[[EXISTING_MEMORY]]

Decide whether this completed work contains durable memory worth storing.
Prefer updating existing memory when it is materially the same idea. The existing
list spans both scopes; a restatement of a listed user fact is an update to that
id, not a new item. Create a new item only for a distinct reusable learning.
Return ignore when the work is routine, transient, or already covered without
needing reinforcement.

A learning is any durable, reusable fact worth recalling in a future session:
reusable facts, environment quirks, domain observations, durable user
preferences, task patterns, and decisions or stable policy choices with
implementation impact ("Decision:", "we decided", "going forward must/should").
When the signal is a decision, fold the reason it matters into the learning text
rather than recording it separately.

Every learning has a scope:
- "user": a durable fact about the *person* you are working with that holds
  across projects — their role and identity, communication/workflow
  preferences, language and tooling preferences, coding and writing style,
  broad interests, recurring constraints, domain priorities, or a stable working
  agreement about how to collaborate with them. When the signal is unambiguously
  about the person, scope it to "user" on the first explicit mention; do not wait
  for it to be repeated. Require repetition only for borderline traits you are
  inferring rather than ones the person stated.
- "project": a fact, quirk, observation, task pattern, or policy choice specific
  to this project or its environment.
Default to "project" only when it is genuinely unclear whether the signal is
about the person or the project; a clearly personal fact always takes "user".
Never store secrets, sensitive personal data, transient details, or one-off task
context in either scope.

A learning may also carry a task_pattern: the short, general name of the *kind
of task* it applies to. This is a grouping label, not a summary of the work.
Learnings from unrelated sessions should be able to carry the exact same
pattern verbatim — that shared label is what later lets them compile into one
reusable skill, so a pattern used once is a pattern wasted. Write it as a
lowercase noun phrase of at most 6 words naming the recurring theme:
"hook pipeline debugging", "cypher schema migration", "flaky test triage",
"release checklist". Never put steps, conditions, ids, or instructions in it —
the procedure belongs in text. Prefer the more general theme whenever two
phrasings would both fit. Use null when the learning is a standalone fact with
no recurring task behind it; a pattern nothing else will ever share is worse
than none.

Failures are learnings too. Every learning has a kind: "fact" (the default) or
"error". The corpus keeps the output of failed tool calls only, marked
tool_error, with the tool's tool_key and — when a later call to the same tool
succeeded — that call's input. When a failure was corrected in this work (a
changed retry succeeded, or the assistant worked around it), record the
error-and-fix as one learning with "kind": "error": the text names the tool,
what fails, and the correction that worked ("neo4j_read_cypher: datetime
properties serialize as {} — project them with toString(...)"); "tool_key"
copies the tool_key shown with the failure; "error_signature" is a one-line
normalized form of the error message; "resolved" is true. A failure that was
never resolved is worth recording with "resolved": false only when it is a
durable, deterministic failure future sessions will hit again — then the text
states what fails and that no fix is known. Never record transient failures
(timeouts, rate limits, connectivity, resource exhaustion, permissions, user
interrupts) as learnings. When a listed existing error learning has the same
signature, update it instead of creating another; an unresolved one becomes
resolved when the fix appears. Facts leave the error fields null.

Return JSON only with this shape:
{
  "learnings": [
    {
      "action": "create|update|ignore",
      "existing_id": "learning id when action is update, otherwise null",
      "scope": "project|user",
      "kind": "fact|error",
      "text": "concise durable learning, or null",
      "task_pattern": "general recurring theme, <=6 words, or null",
      "tool_key": "for kind error: the tool_key shown with the failure; else null",
      "error_signature": "for kind error: one-line normalized error message; else null",
      "resolved": "for kind error: true when the work found the fix, false otherwise; null for facts",
      "confidence": 0.0,
      "reason": "why this action"
    }
  ]
}
"""


# Episodic observations: unconditional per-window timeline records. This template
# is a separate code constant from the memory-extraction prompt — "be selective"
# (memory) and "always record" (episodes) must never share a template.
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


def _window_assistant_texts(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(timestamp, text) of assistant messages written during this event window.

    Sources the dialog-only transcript snapshots log_event stores on events
    (conversation text records only — see ``filter_dialog_transcript``).
    Snapshots are grouped per transcript file (parent and subagent sessions
    each have their own); within a group the window's new content is the last
    snapshot minus the first-snapshot prefix, falling back to everything after
    the last user message when no prefix relation holds. Texts equal to a
    window event's last_assistant_message are dropped — that field is already
    in the corpus.
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
                    for entry in (dialog_entry(r) for r in transcript_records(segment))
                    if entry
                ]
                selected = [(ts, text) for kind, ts, text in entries if kind == "assistant"]
            else:
                entries = [
                    entry
                    for entry in (dialog_entry(r) for r in transcript_records(last))
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


# Failed tool output is the one tool output the corpus keeps: the error text is
# what an error learning is built from. Capped so a wall of stack trace never
# crowds out the dialog.
MAX_ERROR_TEXT = 600
MAX_RETRY_INPUT_TEXT = 600


def _is_failed_tool_event(event: dict[str, Any]) -> bool:
    """A tool call that failed and was not a user interrupt: the live
    ``PostToolUseFailure`` normalization (``tool_error``) or an MCP error
    envelope stored on a plain ``PostToolUse`` event."""
    if event.get("is_interrupt") is True:
        return False
    if event.get("tool_error") is True:
        return True
    return is_error_tool_response(event.get("tool_response"))


def _later_success_same_tool(events: list[dict[str, Any]], index: int) -> str | None:
    """Input of the first later successful call to the same tool, if any.

    The corrected retry is the evidence that pairs an error with its fix, so
    the corpus states the pairing outright instead of leaving the extractor to
    infer it from event order."""
    failed = events[index]
    tool_name = str(failed.get("tool_name") or "")
    if not tool_name:
        return None
    for later in events[index + 1 :]:
        if str(later.get("event_name") or "") != "PostToolUse":
            continue
        if str(later.get("tool_name") or "") != tool_name:
            continue
        if _is_failed_tool_event(later):
            continue
        return _event_text(later, "tool_input") or None
    return None


def _event_corpus(events: list[dict[str, Any]], limit: int = 12000) -> str:
    entries: list[tuple[str, int, str]] = []
    for order, event in enumerate(events):
        parts: list[str] = []
        event_name = _event_text(event, "event_name")
        if event_name:
            parts.append(f"Event: {event_name}")
        # Successful tool outputs are deliberately excluded: memory extraction
        # reads user prompts, assistant messages, tool inputs — and, below,
        # the error text of failed calls only.
        for key in ("prompt", "last_assistant_message", "tool_name", "tool_input"):
            text = _event_text(event, key)
            if text:
                parts.append(f"{key}: {truncate(text, 1200)}")
        if _is_failed_tool_event(event):
            error_text = tool_failure_text(event.get("tool_response"), MAX_ERROR_TEXT)
            if error_text:
                parts.append(f"tool_key: {tool_key(event.get('tool_name')) or 'unknown'}")
                parts.append(f"tool_error: {error_text}")
                retry_input = _later_success_same_tool(events, order)
                if retry_input:
                    parts.append(
                        "later_call_to_same_tool_succeeded_with_input: "
                        f"{truncate(retry_input, MAX_RETRY_INPUT_TEXT)}"
                    )
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
) -> str:
    lines: list[str] = ["Existing similar learnings:"]
    if learnings:
        for item in learnings:
            error_fields = ""
            if item.get("kind") == "error":
                error_fields = (
                    "kind=error; "
                    f"tool_key={item.get('tool_key')}; "
                    f"resolved={bool(item.get('resolved'))}; "
                )
            lines.append(
                "- "
                f"id={item.get('id')}; "
                f"scope={item.get('scope') or 'project'}; "
                f"status={item.get('status')}; "
                f"task_pattern={item.get('task_pattern')}; "
                f"{error_fields}"
                f"text={truncate(str(item.get('text') or ''), 300)}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def render_memory_extraction_prompt(
    template: str,
    project: ProjectRef,
    mode: str,
    events: list[dict[str, Any]],
    similar_learnings: list[dict[str, Any]],
) -> str:
    corpus = _event_corpus(events)
    existing = _format_existing_memory(similar_learnings)
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


def build_memory_extraction_prompt(
    project: ProjectRef,
    mode: str,
    events: list[dict[str, Any]],
    similar_learnings: list[dict[str, Any]],
) -> str:
    return render_memory_extraction_prompt(
        DEFAULT_MEMORY_EXTRACTION_PROMPT,
        project,
        mode,
        events,
        similar_learnings,
    )


def _json_from_llm_text(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"learnings": []}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {"learnings": []}
    if not isinstance(parsed, dict):
        return {"learnings": []}
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
            {"learnings": []},
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
    if existing_id.startswith("learning:user:"):
        return "user"
    return "project"


# ``task_pattern`` is the grouping key skill distillation clusters on: it
# resolves to a ``(:TaskPattern)`` node by exact normalized match first, then by
# embedding similarity over this very string. A paragraph-length value matches
# nothing and never will, so its learning becomes a permanent singleton that no
# skill can ever form from — the field silently stops working while still
# looking populated. Anything past a short phrase is dropped rather than
# clipped: a null pattern leaves the learning skill-ineligible, which is the
# honest outcome for one with no reusable theme, whereas a truncated paragraph
# is a bad key forever.
MAX_TASK_PATTERN_CHARS = 60
MAX_TASK_PATTERN_WORDS = 6


def _task_pattern(value: object) -> str | None:
    pattern = " ".join(str(value or "").split())
    if not pattern:
        return None
    if len(pattern) > MAX_TASK_PATTERN_CHARS:
        return None
    if len(pattern.split(" ")) > MAX_TASK_PATTERN_WORDS:
        return None
    return pattern


LEARNING_KINDS = frozenset({"fact", "error"})
MAX_ERROR_SIGNATURE_CHARS = 300


def _learning_kind(value: object) -> str | None:
    """``"fact"`` / ``"error"`` when the extractor stated a kind, else None so
    an update never silently rewrites the kind of an existing learning."""
    kind = str(value or "").strip().lower()
    return kind if kind in LEARNING_KINDS else None


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return None


def _error_fields(item: dict[str, Any], action: str) -> dict[str, Any]:
    """The error-learning columns of one extractor item.

    A fact carries nulls. An error learning carries the canonical tool key,
    a one-line error signature, and ``resolved`` — defaulting to False on
    create (an error with no stated fix is unresolved) and left None on
    update so an omitted field keeps the stored value."""
    kind = _learning_kind(item.get("kind"))
    if action == "create" and kind is None:
        kind = "fact"
    if kind != "error":
        return {"kind": kind, "tool_key": None, "error_signature": None, "resolved": None}
    signature = " ".join(str(item.get("error_signature") or "").split()) or None
    if signature and len(signature) > MAX_ERROR_SIGNATURE_CHARS:
        signature = signature[: MAX_ERROR_SIGNATURE_CHARS - 3].rstrip() + "..."
    resolved = _bool_or_none(item.get("resolved"))
    if action == "create" and resolved is None:
        resolved = False
    return {
        "kind": "error",
        "tool_key": tool_key(item.get("tool_key")),
        "error_signature": signature,
        "resolved": resolved,
    }


def _memory_rows_from_actions(
    project: ProjectRef,
    mode: str,
    actions: dict[str, Any],
    llm_model: str | None = None,
    *,
    user_id: str,
) -> list[dict[str, Any]]:
    """Rows for the graph write. ``user_id`` is the person this window belongs
    to: it keys user-scoped ids (``learning:user:<user_id>:...``) so the same
    sentence about two people never collides, and it is stamped on every row
    as provenance."""
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
            else learning_id(learning_namespace(project.id, scope, user_id), text)
        )
        learning_rows.append(
            {
                "id": row_id,
                "action": action,
                "text": text or None,
                "task_pattern": _task_pattern(item.get("task_pattern")),
                **_error_fields(item, action),
                "confidence": _confidence(item.get("confidence")),
                "status": "candidate",
                "scope": scope,
                "source": HOOK_LEARNING_SOURCE,
                "summary": text or item.get("reason"),
                "reason": item.get("reason"),
                "llm_model": llm_model,
                "user_id": user_id,
            }
        )
    return learning_rows[:3]


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
    *,
    user_id: str,
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
                "user_id": user_id,
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
    llm_model: str | None,
    llm_status: str | None,
    llm_skip_reason: str | None,
    llm_error: str | None,
    timestamp: str,
    observation_rows: list[dict[str, Any]] | None = None,
    *,
    user_id: str,
) -> None:
    observation_rows = observation_rows or []
    event_ids = [event["event_id"] for event in events if event.get("event_id")]
    digest = _window_digest(events)
    processing_id = f"processing:{project.id}:{session_id}:{mode}:{digest}"
    summary = (
        f"Processed {len(event_ids)} {mode} events and produced "
        f"{len(learning_rows)} learning actions "
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
        SET s.user_id = coalesce(s.user_id, $user_id)
        MERGE (p)-[:HAS_SESSION]->(s)
        MERGE (pp:ProjectProcessing {id: $processing_id})
        ON CREATE SET pp.created_at = datetime($timestamp)
        SET pp.project_id = $project_id,
            pp.session_id = $session_id,
            pp.user_id = $user_id,
            pp.mode = $mode,
            pp.status = 'ok',
            pp.type = 'processing',
            pp.event_count = $event_count,
            pp.learning_count = $learning_count,
            pp.observation_count = $observation_count,
            pp.llm_model = $llm_model,
            pp.llm_status = $llm_status,
            pp.llm_skip_reason = $llm_skip_reason,
            pp.llm_error = $llm_error,
            pp.summary = $summary,
            pp.updated_at = datetime($timestamp)
        MERGE (p)-[:HAS_PROCESSING]->(pp)
        MERGE (s)-[:HAS_PROCESSING]->(pp)
        MERGE (u:User {id: $user_id})
        ON CREATE SET u.created_at = datetime($timestamp)
        SET u.last_seen_at = datetime($timestamp)
        MERGE (u)-[:HAS_SESSION]->(s)
        WITH pp
        UNWIND $event_ids AS event_id
        MATCH (e:SessionEvent {event_id: event_id})
        MERGE (pp)-[:PROCESSED_EVENT]->(e)
        """,
        project_id=project.id,
        session_id=session_id,
        user_id=user_id,
        processing_id=processing_id,
        mode=mode,
        event_count=len(event_ids),
        learning_count=len(learning_rows),
        observation_count=len(observation_rows),
        llm_model=llm_model,
        llm_status=llm_status,
        llm_skip_reason=truncate(str(llm_skip_reason or ""), 1000) or None,
        llm_error=truncate(str(llm_error or ""), 1000) or None,
        summary=summary,
        event_ids=event_ids,
        timestamp=timestamp,
    )

    learning_create_rows = [row for row in learning_rows if row.get("action") == "create"]
    learning_update_rows = [row for row in learning_rows if row.get("action") == "update"]

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
                          l.kind = row.kind,
                          l.tool_key = row.tool_key,
                          l.error_signature = row.error_signature,
                          l.resolved = row.resolved,
                          l.summary = row.summary,
                          l.source = row.source,
                          l.created_by_model = row.llm_model,
                          l.status = row.status,
                          l.scope = row.scope,
                          l.user_id = row.user_id,
                          l.use_count = 0,
                          l.support_count = 0
            SET l.text = row.text,
                l.task_pattern = row.task_pattern,
                l.kind = coalesce(row.kind, l.kind, 'fact'),
                l.tool_key = coalesce(row.tool_key, l.tool_key),
                l.error_signature = coalesce(row.error_signature, l.error_signature),
                l.resolved = coalesce(row.resolved, l.resolved),
                l.summary = row.summary,
                l.embedding = coalesce(row.embedding, l.embedding),
                l.last_source = row.source,
                l.last_source_session_id = $session_id,
                l.last_user_id = coalesce(row.user_id, l.last_user_id),
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
            // A user fact belongs to the person it describes: hang it off
            // their (:User) the way a project fact hangs off its (:Project).
            FOREACH (_ IN CASE WHEN row.scope = 'user' THEN [1] ELSE [] END |
                MERGE (u:User {id: row.user_id})
                ON CREATE SET u.created_at = datetime($timestamp)
                MERGE (u)-[:HAS_LEARNING]->(l)
            )
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
            // Ownership guard: automation may only update memory this project
            // owns, or project-independent user-scoped facts. Blocks one
            // project's session silently rewriting another project's memory
            // through an LLM-supplied id.
            WHERE l.scope = 'user' OR (p)-[:HAS_LEARNING]->(l)
            WITH p, s, pp, l, row,
                 // A content change to an approved note is a NEW, unreviewed
                 // claim. Demote it back to candidate so it re-enters the gate
                 // (and, if ambiguous, the human queue) instead of inheriting
                 // the approved badge and human provenance.
                 (l.status = 'approved'
                  AND row.text IS NOT NULL
                  AND row.text <> l.text) AS demoted
            SET l.text = coalesce(row.text, l.text),
                l.task_pattern = coalesce(row.task_pattern, l.task_pattern),
                l.kind = coalesce(row.kind, l.kind),
                l.tool_key = coalesce(row.tool_key, l.tool_key),
                l.error_signature = coalesce(row.error_signature, l.error_signature),
                l.resolved = coalesce(row.resolved, l.resolved),
                l.summary = coalesce(row.summary, l.summary),
                l.last_source = row.source,
                l.last_source_session_id = $session_id,
                l.last_user_id = coalesce(row.user_id, l.last_user_id),
                l.last_reason = row.reason,
                l.last_llm_model = coalesce(row.llm_model, l.last_llm_model),
                l.project_id = coalesce(l.project_id, $project_id),
                l.updated_at = datetime($timestamp),
                l.support_count = coalesce(l.support_count, 0) + 1,
                l.status = CASE WHEN demoted THEN 'candidate' ELSE l.status END,
                l.reviewed_by = CASE WHEN demoted THEN null ELSE l.reviewed_by END,
                l.reviewed_at = CASE WHEN demoted THEN null ELSE l.reviewed_at END,
                // Re-open the demoted item for the gate/sweep (which key on a
                // null consistency_checked_at); the embedding is kept so it
                // stays searchable and can be re-judged.
                l.consistency_status = CASE WHEN demoted THEN null ELSE l.consistency_status END,
                l.consistency_checked_at = CASE WHEN demoted THEN null ELSE l.consistency_checked_at END,
                l.confidence = CASE
                    WHEN coalesce(l.confidence, 0.0) < row.confidence THEN row.confidence
                    ELSE l.confidence
                END
            MERGE (p)-[:HAS_LEARNING]->(l)
            MERGE (l)-[:FROM_SESSION]->(s)
            MERGE (pp)-[updated:UPDATED_LEARNING]->(l)
            SET updated.llm_model = row.llm_model,
                updated.updated_at = datetime($timestamp),
                updated.demoted_from_approved = demoted
            """,
            project_id=project.id,
            session_id=session_id,
            processing_id=processing_id,
            learnings=learning_update_rows,
            timestamp=timestamp,
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
                o.user_id = coalesce(row.user_id, o.user_id),
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

    # The person this window belongs to. Background runs get it pinned in
    # MKG_USER_ID by the foreground hook; a direct run resolves it here.
    user = resolve_user(project_git_root(project))
    timestamp = datetime.now(timezone.utc).isoformat()
    uri, db_user, password, database = neo4j_config()

    with GraphDatabase.driver(uri, auth=(db_user, password)) as driver:
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
            session.execute_write(
                merge_project_and_session, project, session_id, timestamp, user.id
            )
            llm_model_used: str | None = None
            llm_status: str | None = None
            llm_skip_reason: str | None = None
            llm_error: str | None = None
            try:
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
                # User facts belong in the same shortlist as project ones: this
                # list is what the extractor deduplicates against, and a scope
                # it cannot see is a scope it can only ever "create" into. Left
                # out, every restatement of a durable user fact mints a new node
                # (the id is a hash of the text, so paraphrases never collide)
                # and lands in the human review queue as a fresh discovery.
                similar_user_learnings = fetch_user_learnings(
                    driver,
                    database,
                    query=corpus,
                    statuses=["approved", "candidate"],
                    limit=5,
                    query_vector=query_vector,
                    include_consolidated=True,
                    user_id=user.id,
                )
                prompt = build_memory_extraction_prompt(
                    project,
                    mode,
                    events,
                    similar_learnings + similar_user_learnings,
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
                learning_rows = _memory_rows_from_actions(
                    project,
                    mode,
                    actions,
                    llm_model=llm_model_used,
                    user_id=user.id,
                )
                # Embed candidates before the write so each is stored searchable
                # and the consistency gate can retrieve prior approved neighbours.
                attach_candidate_embeddings(learning_rows)
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
                        user_id=user.id,
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
                    llm_model_used,
                    llm_status,
                    llm_skip_reason,
                    llm_error,
                    timestamp,
                    observation_rows=observation_rows,
                    user_id=user.id,
                )
                # Autonomous gate: safety-screen each candidate (blocking
                # laundered instructions, privilege grabs, and secrets as
                # recorded tombstones), then promote consistent survivors and
                # invalidate older items they supersede — both scopes, no human
                # queue. No-op if embeddings/judge/index are unavailable
                # (candidates simply stay 'candidate'). Runs only at Stop
                # ('turn'), never at SessionEnd ('session').
                if mode == "turn":
                    run_consistency_gate(
                        driver,
                        database,
                        project=project,
                        learning_rows=learning_rows,
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
                            for row in learning_rows
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
    user: UserRef | None = None,
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
            env=project_env(project, user),
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

    if args.background:
        _spawn_background(
            mode=args.mode,
            limit=args.limit,
            session_id=str(payload.get("session_id") or "unknown"),
            project=project,
            user=resolve_user(project_git_root(project)),
        )
        return 0

    try:
        process_project(payload, mode=args.mode, limit=args.limit)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[process_project] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
