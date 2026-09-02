#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0", "litellm>=1.40.0"]
# ///
"""Stop / SessionEnd hook: rate-limited query-error consolidation service.

``capture_query_failures.py`` records every failed query as raw
``(:QueryExecution)-[:HAS_ISSUE]->(:QueryIssue)`` artifacts. This service turns
that raw failure log into durable, reusable guidance: it folds unconsolidated
*invalid-query* failures into per-tool ``(:QueryErrorPattern)`` nodes — a known
error signature, its root cause, and the actionable fix — that the
``inject_query_error_context.py`` recall hook replays the next time the same
tool produces a similar error.

Only failures deterministically caused by the query text itself are
consolidated (syntax errors, schema mismatches, missing procedures/functions,
serialization projections). Transient or environmental failures — timeouts,
rate limits, resource exhaustion, permissions — are excluded twice: hard-coded
issue classes never enter the queue, and the LLM is instructed to discard
transient one-offs hiding in the generic error bucket.

Patterns are grouped by tool: each ``(:ToolErrorProfile)`` anchors the patterns
of one (project, tool) pair, so recall only ever surfaces guidance for the tool
that actually failed. Like the system-prompt service it runs in the background
on every Stop / SessionEnd but does real work rarely:

1. Rate limit. A per-tool cooldown window
   (``MKG_QUERY_ERROR_CONSOLIDATION_INTERVAL_HOURS``, default 6h) lives on the
   ``:ToolErrorProfile`` node.
2. Threshold gate. A tool consolidates only when *more than*
   ``MKG_QUERY_ERROR_CONSOLIDATION_THRESHOLD`` (default 1) of its failures are
   pending — captured, consolidatable, and not yet folded into a pattern.
3. Consolidate. The tool's existing patterns plus its pending failures go to
   the LLM, which merges same-root-cause failures into one pattern, updates
   existing patterns instead of duplicating them, and skips transients.
4. Keep provenance. Patterns link ``[:DERIVED_FROM]`` to their source
   executions; consumed executions are stamped ``error_consolidated_at`` so
   they drop out of the queue either way.

Graph shape::

    (:Project)-[:HAS_TOOL_ERROR_PROFILE]->(:ToolErrorProfile {tool_key})
    (:ToolErrorProfile)-[:HAS_ERROR_PATTERN]->(:QueryErrorPattern)
    (:QueryErrorPattern)-[:DERIVED_FROM]->(:QueryExecution)

Like ``process_project.py`` it swallows its own errors so a Neo4j or LLM outage
never blocks the session.
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

from capture_query_failures import (  # noqa: E402
    BIGQUERY_SUFFIX,
    CONSOLIDATABLE_ISSUE_TYPES,
    NEO4J_SUFFIX,
    ensure_query_failure_schema,
    tool_key,
)
from project_common import (  # noqa: E402
    ProjectRef,
    _env_float,
    embed_texts,
    embedding_dimensions,
    ensure_project_schema,
    extraction_model_label,
    in_extraction_subprocess,
    llm_complete,
    llm_readiness_status,
    load_mkg_env,
    neo4j_config,
    project_env,
    project_git_root,
    resolve_project,
    resolve_user,
    truncate,
)

QUERY_ERROR_CONSOLIDATION_THRESHOLD = 1
QUERY_ERROR_CONSOLIDATION_INTERVAL_HOURS = 6.0
MAX_PENDING_PER_TOOL = 20
MAX_EXISTING_PATTERNS = 12
MAX_ERROR_EXCERPT = 600
MAX_QUERY_TEXT = 800

QUERY_ERROR_PATTERN_VECTOR_INDEX = "query_error_pattern_vector"
QUERY_ERROR_PATTERN_FULLTEXT_INDEX = "query_error_pattern_fulltext"

_ENGINE_BY_TOOL_KEY = {BIGQUERY_SUFFIX: "bigquery", NEO4J_SUFFIX: "neo4j"}

CONSOLIDATION_INSTRUCTION = """\
You maintain a library of known query-failure patterns for one query tool used
by an AI agent. Your patterns are injected as guidance the next time the agent
hits a similar error on this tool, so they must teach the fix, not just
describe the failure.

Tool: [[TOOL]] (engine: [[ENGINE]])

Below are (A) the existing known error patterns for this tool and (B) new
failed query executions captured since the last consolidation.

Fold the new failures into the pattern library:
- Keep only failures deterministically caused by the query text itself:
  invalid syntax, wrong table/dataset/label/column/property names, unknown or
  missing functions and procedures, invalid value projections (e.g. temporal
  values serialized as empty objects).
- Discard transient or environmental failures — timeouts, rate limits,
  connectivity problems, resource exhaustion, permission or credential issues,
  cancelled queries — by listing their execution ids in
  "skipped_execution_ids". Never build a pattern from them.
- Merge failures with the same root cause into ONE pattern. When new failures
  match an existing pattern, return that pattern's id with refreshed content;
  create a new pattern (id null) only for a genuinely new root cause.
- "resolution" must be actionable: state the correct query form for this tool
  and engine. Fill "example_fix" with a corrected version of the failing query
  whenever you can infer one.

Return ONLY a JSON object, no commentary and no code fences:
{
  "patterns": [
    {
      "id": "existing pattern id to update, or null for a new pattern",
      "title": "short name of the failure mode",
      "error_signature": "normalized one-line form of the error message",
      "root_cause": "why queries of this shape fail",
      "resolution": "how to write the query correctly",
      "example_query": "a failing query taken from the executions",
      "example_fix": "the corrected query, or null",
      "issue_types": ["syntax_error"],
      "confidence": 0.9,
      "source_execution_ids": ["query-execution:..."]
    }
  ],
  "skipped_execution_ids": ["query-execution:..."]
}

(A) EXISTING PATTERNS:
[[EXISTING_PATTERNS]]

(B) NEW FAILED EXECUTIONS:
[[PENDING_FAILURES]]
"""


def consolidation_threshold() -> int:
    return int(
        _env_float(
            "MKG_QUERY_ERROR_CONSOLIDATION_THRESHOLD",
            QUERY_ERROR_CONSOLIDATION_THRESHOLD,
        )
    )


def consolidation_interval_hours() -> float:
    return _env_float(
        "MKG_QUERY_ERROR_CONSOLIDATION_INTERVAL_HOURS",
        QUERY_ERROR_CONSOLIDATION_INTERVAL_HOURS,
    )


def profile_id(project_id: str, tool: str) -> str:
    return f"tool-error-profile:{project_id}:{tool}"


def pattern_id(project_id: str, tool: str, signature: str) -> str:
    digest = sha1(
        f"{project_id}\n{tool}\n{' '.join(signature.lower().split())}".encode("utf-8")
    ).hexdigest()[:16]
    return f"query-error-pattern:{project_id}:{tool}:{digest}"


def consolidation_gate(
    pending_count: int,
    threshold: int,
    last_consolidated_at: Any,
    interval_hours: float,
    now: datetime,
) -> tuple[bool, str]:
    """Per-tool decision: threshold first, then the cooldown. Pure for tests."""
    if pending_count <= threshold:
        return False, (
            f"{pending_count} pending query failure(s) "
            f"(need more than {threshold}); skipping"
        )
    if last_consolidated_at:
        last = _parse_iso(last_consolidated_at)
        if last is not None:
            elapsed_hours = (now - last).total_seconds() / 3600.0
            if elapsed_hours < interval_hours:
                return False, (
                    f"rate-limited: last consolidation {elapsed_hours:.1f}h ago "
                    f"(< {interval_hours:.0f}h cooldown)"
                )
    return True, (
        f"{pending_count} pending query failure(s) (> {threshold}); consolidating"
    )


def _parse_iso(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_native"):  # neo4j.time.DateTime
        parsed = value.to_native()
    else:
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def resolve_tool_key(row: dict[str, Any]) -> str | None:
    """Tool identity of a captured execution; tolerant of pre-``tool_key`` nodes."""
    explicit = row.get("tool_key")
    if isinstance(explicit, str) and explicit:
        return explicit
    derived = tool_key(str(row.get("tool_name") or ""))
    if derived:
        return derived
    engine = str(row.get("engine") or "")
    for key, engine_name in _ENGINE_BY_TOOL_KEY.items():
        if engine == engine_name:
            return key
    return None


def group_pending_by_tool(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = resolve_tool_key(row)
        if key is None:
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def ensure_query_error_pattern_schema(driver, database: str) -> None:
    """Constraints plus the retrieval indexes for ``:QueryErrorPattern``.

    The vector index declares ``project_id`` / ``tool_key`` as in-index metadata
    so recall can pre-filter to the failing tool inside the ``SEARCH`` clause
    (same native filtered vector search the memory indexes use, Neo4j 2026.02+).
    Each statement runs in its own session so one schema statement never blocks
    the next.
    """
    dims = embedding_dimensions()
    statements = (
        "CREATE CONSTRAINT IF NOT EXISTS "
        "FOR (t:ToolErrorProfile) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS "
        "FOR (e:QueryErrorPattern) REQUIRE e.id IS UNIQUE",
        f"CREATE FULLTEXT INDEX {QUERY_ERROR_PATTERN_FULLTEXT_INDEX} IF NOT EXISTS "
        "FOR (e:QueryErrorPattern) "
        "ON EACH [e.title, e.error_signature, e.root_cause, e.resolution, e.example_query]",
        f"CREATE VECTOR INDEX {QUERY_ERROR_PATTERN_VECTOR_INDEX} IF NOT EXISTS "
        "FOR (e:QueryErrorPattern) ON e.embedding "
        "WITH [e.project_id, e.tool_key, e.status] "
        f"OPTIONS {{indexConfig: {{"
        f"`vector.dimensions`: {dims}, "
        f"`vector.similarity_function`: 'cosine'}}}}",
    )
    for stmt in statements:
        with driver.session(database=database) as session:
            session.run(stmt).consume()


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def fetch_pending_failures(
    driver,
    database: str,
    project_id: str,
    limit: int = 120,
) -> list[dict[str, Any]]:
    """Captured executions that are consolidatable and not yet consolidated.

    Transient issue classes (timeouts, resource limits, permissions, ...) are
    excluded here: an execution enters the queue only when at least one of its
    issues is a consolidatable invalid-query class.
    """
    records = driver.execute_query(
        """
        MATCH (:Project {id: $project_id})-[:HAS_QUERY_EXECUTION]->(q:QueryExecution)
        WHERE q.error_consolidated_at IS NULL
          AND any(t IN q.issue_types WHERE t IN $consolidatable)
        RETURN q.id AS id,
               q.tool_name AS tool_name,
               q.tool_key AS tool_key,
               q.engine AS engine,
               q.query_text AS query_text,
               q.response_excerpt AS response_excerpt,
               q.issue_types AS issue_types
        ORDER BY toString(coalesce(q.last_seen_at, q.created_at)) DESC
        LIMIT $limit
        """,
        database_=database,
        project_id=project_id,
        consolidatable=sorted(CONSOLIDATABLE_ISSUE_TYPES),
        limit=limit,
    )
    return [dict(record) for record in getattr(records, "records", records) or []]


def fetch_profile_last_consolidated_at(
    driver, database: str, profile: str
) -> str | None:
    records = driver.execute_query(
        """
        MATCH (t:ToolErrorProfile {id: $profile_id})
        RETURN t.last_consolidated_at AS last_consolidated_at
        """,
        database_=database,
        profile_id=profile,
    )
    rows = list(getattr(records, "records", records) or [])
    return rows[0]["last_consolidated_at"] if rows else None


def fetch_existing_patterns(
    driver,
    database: str,
    project_id: str,
    tool: str,
    limit: int = MAX_EXISTING_PATTERNS,
) -> list[dict[str, Any]]:
    records = driver.execute_query(
        """
        MATCH (e:QueryErrorPattern {project_id: $project_id, tool_key: $tool_key})
        WHERE e.status = 'active'
        RETURN e.id AS id,
               e.title AS title,
               e.error_signature AS error_signature,
               e.root_cause AS root_cause,
               e.resolution AS resolution,
               e.example_query AS example_query,
               e.example_fix AS example_fix,
               e.issue_types AS issue_types,
               e.confidence AS confidence
        ORDER BY coalesce(e.occurrence_count, 0) DESC,
                 toString(coalesce(e.updated_at, e.created_at)) DESC
        LIMIT $limit
        """,
        database_=database,
        project_id=project_id,
        tool_key=tool,
        limit=limit,
    )
    return [dict(record) for record in getattr(records, "records", records) or []]


# --------------------------------------------------------------------------- #
# LLM round trip
# --------------------------------------------------------------------------- #
def build_consolidation_prompt(
    tool: str,
    engine: str,
    existing_patterns: list[dict[str, Any]],
    pending: list[dict[str, Any]],
) -> str:
    if existing_patterns:
        existing_lines = [
            json.dumps(
                {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "error_signature": row.get("error_signature"),
                    "root_cause": row.get("root_cause"),
                    "resolution": row.get("resolution"),
                    "issue_types": row.get("issue_types"),
                },
                default=str,
            )
            for row in existing_patterns
        ]
        existing_text = "\n".join(existing_lines)
    else:
        existing_text = "(none yet)"

    failure_blocks = []
    for row in pending:
        failure_blocks.append(
            "\n".join(
                [
                    f"execution_id: {row.get('id')}",
                    f"issue_types: {', '.join(row.get('issue_types') or [])}",
                    f"query: {truncate(str(row.get('query_text') or ''), MAX_QUERY_TEXT)}",
                    f"error: {truncate(str(row.get('response_excerpt') or ''), MAX_ERROR_EXCERPT)}",
                ]
            )
        )
    pending_text = "\n---\n".join(failure_blocks) if failure_blocks else "(none)"

    return (
        CONSOLIDATION_INSTRUCTION.replace("[[TOOL]]", tool)
        .replace("[[ENGINE]]", engine)
        .replace("[[EXISTING_PATTERNS]]", existing_text)
        .replace("[[PENDING_FAILURES]]", pending_text)
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    for candidate in (stripped,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _clean_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def parse_consolidation_response(
    text: str,
    project_id: str,
    tool: str,
    pending_ids: set[str],
    existing_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Validate the LLM's pattern list into write-ready rows.

    Hallucinated pattern ids fall back to a deterministic content-addressed id;
    source/skipped execution ids are filtered to the batch that was actually in
    the prompt. Returns ``None`` when the response is not usable at all.
    """
    parsed = _parse_json_object(text)
    if parsed is None:
        return None

    raw_patterns = parsed.get("patterns")
    if not isinstance(raw_patterns, list):
        return None

    patterns: list[dict[str, Any]] = []
    for raw in raw_patterns:
        if not isinstance(raw, dict):
            continue
        title = _clean_text(raw.get("title"), 200)
        signature = _clean_text(raw.get("error_signature"), 500)
        resolution = _clean_text(raw.get("resolution"), 1500)
        if not title or not signature or not resolution:
            continue
        raw_id = raw.get("id")
        row_id = (
            raw_id
            if isinstance(raw_id, str) and raw_id in existing_ids
            else pattern_id(project_id, tool, signature)
        )
        issue_types = [
            item
            for item in (raw.get("issue_types") or [])
            if isinstance(item, str) and item
        ]
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence"))))
        except (TypeError, ValueError):
            confidence = 0.7
        source_ids = [
            item
            for item in (raw.get("source_execution_ids") or [])
            if isinstance(item, str) and item in pending_ids
        ]
        patterns.append(
            {
                "id": row_id,
                "title": title,
                "error_signature": signature,
                "root_cause": _clean_text(raw.get("root_cause"), 1000),
                "resolution": resolution,
                "example_query": _clean_text(raw.get("example_query"), MAX_QUERY_TEXT),
                "example_fix": _clean_text(raw.get("example_fix"), MAX_QUERY_TEXT),
                "issue_types": issue_types,
                "confidence": confidence,
                "source_execution_ids": source_ids,
            }
        )

    skipped = [
        item
        for item in (parsed.get("skipped_execution_ids") or [])
        if isinstance(item, str) and item in pending_ids
    ]
    return patterns, skipped


def attach_pattern_embeddings(patterns: list[dict[str, Any]]) -> None:
    """Embed each pattern so recall can match it semantically; degrades to
    fulltext-only retrieval when embeddings are unavailable."""
    texts = [
        "\n".join(
            part
            for part in (
                row.get("title"),
                row.get("error_signature"),
                row.get("resolution"),
                row.get("example_query"),
            )
            if part
        )
        for row in patterns
    ]
    for row, vector in zip(patterns, embed_texts(texts)):
        if vector:
            row["embedding"] = vector


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def write_consolidation(
    tx,
    project: ProjectRef,
    tool: str,
    engine: str,
    patterns: list[dict[str, Any]],
    consumed_execution_ids: list[str],
    model: str,
    timestamp: str,
) -> None:
    tx.run(
        """
        MERGE (p:Project {id: $project_id})
        ON CREATE SET p.created_at = datetime($timestamp)
        MERGE (t:ToolErrorProfile {id: $profile_id})
        ON CREATE SET t.created_at = datetime($timestamp)
        SET t.project_id = $project_id,
            t.tool_key = $tool_key,
            t.engine = $engine,
            t.updated_at = datetime($timestamp),
            t.last_consolidated_at = datetime($timestamp),
            t.last_consolidation_model = $model
        MERGE (p)-[:HAS_TOOL_ERROR_PROFILE]->(t)
        """,
        project_id=project.id,
        profile_id=profile_id(project.id, tool),
        tool_key=tool,
        engine=engine,
        model=model,
        timestamp=timestamp,
    )
    if patterns:
        tx.run(
            """
            MATCH (t:ToolErrorProfile {id: $profile_id})
            UNWIND $patterns AS row
            MERGE (e:QueryErrorPattern {id: row.id})
            ON CREATE SET e.created_at = datetime($timestamp),
                          e.status = 'active'
            SET e.title = row.title,
                e.error_signature = row.error_signature,
                e.root_cause = row.root_cause,
                e.resolution = row.resolution,
                e.example_query = row.example_query,
                e.example_fix = row.example_fix,
                e.issue_types = row.issue_types,
                e.confidence = row.confidence,
                e.project_id = $project_id,
                e.tool_key = $tool_key,
                e.engine = $engine,
                e.embedding = coalesce(row.embedding, e.embedding),
                e.last_consolidation_model = $model,
                e.updated_at = datetime($timestamp),
                e.last_consolidated_at = datetime($timestamp)
            MERGE (t)-[:HAS_ERROR_PATTERN]->(e)
            WITH e, row
            UNWIND coalesce(row.source_execution_ids, []) AS execution_id
            MATCH (q:QueryExecution {id: execution_id})
            MERGE (e)-[:DERIVED_FROM]->(q)
            // count(*) is an eager aggregation: every provenance MERGE above is
            // applied before any count below, so the COUNT subquery sees the
            // full edge set instead of the first streamed row per pattern.
            WITH e, count(*) AS merged_edges
            SET e.occurrence_count = COUNT { (e)-[:DERIVED_FROM]->() }
            """,
            profile_id=profile_id(project.id, tool),
            patterns=patterns,
            project_id=project.id,
            tool_key=tool,
            engine=engine,
            model=model,
            timestamp=timestamp,
        )
    if consumed_execution_ids:
        tx.run(
            """
            UNWIND $execution_ids AS execution_id
            MATCH (q:QueryExecution {id: execution_id})
            SET q.error_consolidated_at = datetime($timestamp)
            """,
            execution_ids=consumed_execution_ids,
            timestamp=timestamp,
        )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def consolidate(payload: dict[str, Any]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    project = resolve_project(payload, project_root)
    if project is None:
        return

    from neo4j import GraphDatabase

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    threshold = consolidation_threshold()
    interval_hours = consolidation_interval_hours()

    uri, user, password, database = neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_project_schema)
            session.execute_write(ensure_query_failure_schema)
        ensure_query_error_pattern_schema(driver, database)

        pending = fetch_pending_failures(driver, database, project.id)
        grouped = group_pending_by_tool(pending)
        if not grouped:
            print("[consolidate_query_errors] no pending query failures; skipping")
            return

        llm_checked = False
        for tool, rows in sorted(grouped.items()):
            last_run = fetch_profile_last_consolidated_at(
                driver, database, profile_id(project.id, tool)
            )
            proceed, reason = consolidation_gate(
                pending_count=len(rows),
                threshold=threshold,
                last_consolidated_at=last_run,
                interval_hours=interval_hours,
                now=now,
            )
            print(f"[consolidate_query_errors] {tool}: {reason}")
            if not proceed:
                continue

            if not llm_checked:
                ready, not_ready_reason = llm_readiness_status()
                llm_checked = True
                if not ready:
                    print(
                        "[consolidate_query_errors] LLM unavailable "
                        f"({not_ready_reason}); leaving failures pending.",
                        file=sys.stderr,
                    )
                    return

            batch = rows[:MAX_PENDING_PER_TOOL]
            engine = str(batch[0].get("engine") or _ENGINE_BY_TOOL_KEY.get(tool, ""))
            existing = fetch_existing_patterns(driver, database, project.id, tool)
            prompt = build_consolidation_prompt(tool, engine, existing, batch)
            try:
                response = llm_complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You consolidate query failures into reusable "
                                "error patterns. Return only the JSON object."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
            except Exception as exc:
                print(
                    f"[consolidate_query_errors] {tool}: LLM call failed: {exc}",
                    file=sys.stderr,
                )
                continue

            result = parse_consolidation_response(
                response,
                project_id=project.id,
                tool=tool,
                pending_ids={str(row["id"]) for row in batch},
                existing_ids={str(row["id"]) for row in existing},
            )
            if result is None:
                print(
                    f"[consolidate_query_errors] {tool}: unusable LLM response; "
                    "leaving failures pending.",
                    file=sys.stderr,
                )
                continue

            patterns, skipped = result
            attach_pattern_embeddings(patterns)
            consumed = [str(row["id"]) for row in batch]
            with driver.session(database=database) as session:
                session.execute_write(
                    write_consolidation,
                    project,
                    tool,
                    engine,
                    patterns,
                    consumed,
                    extraction_model_label(),
                    timestamp,
                )
            print(
                f"[consolidate_query_errors] {tool}: wrote {len(patterns)} "
                f"pattern(s) from {len(consumed)} failure(s) "
                f"({len(skipped)} judged transient)"
            )


def _read_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def _spawn_background(session_id: str, project: ProjectRef | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--session-id",
        session_id,
    ]
    with open(os.devnull, "rb") as stdin, open(os.devnull, "wb") as output:
        subprocess.Popen(
            command,
            cwd=str(project_root),
            env=project_env(project, resolve_user(project_git_root(project))),
            stdin=stdin,
            stdout=output,
            stderr=output,
            start_new_session=True,
            close_fds=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id")
    parser.add_argument(
        "--background",
        action="store_true",
        help="Spawn the consolidation in the background and return immediately.",
    )
    args = parser.parse_args()

    # No-op inside a claude_cli extraction subprocess (see process_project).
    if in_extraction_subprocess():
        return 0

    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)
    payload = _read_payload()
    if args.session_id:
        payload["session_id"] = args.session_id

    if args.background:
        project = resolve_project(payload, project_root)
        _spawn_background(str(payload.get("session_id") or "unknown"), project)
        return 0

    try:
        consolidate(payload)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[consolidate_query_errors] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
