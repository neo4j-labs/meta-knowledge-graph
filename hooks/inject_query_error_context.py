#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0", "litellm>=1.40.0"]
# ///
"""PostToolUse / PostToolUseFailure hook: recall consolidated query-error fixes.

Counterpart of ``consolidate_query_errors.py``. When a query tool fails with an
*invalid-query* class of error (the same consolidatable classes the capture
hook detects — syntax, schema, capability, serialization; never timeouts or
other transient errors), this hook retrieves the ``(:QueryErrorPattern)``
guidance previously consolidated **for that same tool** and feeds it back to
the model so it can apply the known fix instead of retrying blind variations.

Matching is hybrid: the failing query text plus the error message are searched
against the per-tool pattern library via the vector index and the fulltext
index, fused with RRF — so a pattern is recalled both when the *error* looks
similar and when the *query* looks similar. Retrieval is always scoped to the
failing tool's ``tool_key`` (and the active project), so guidance for one tool
never leaks into another's failures.

Delivery depends on the event:

- ``PostToolUse`` (error-shaped ``isError`` results, e.g. on Codex) supports
  ``hookSpecificOutput.additionalContext``.
- ``PostToolUseFailure`` (Claude Code's live failure event) does not support
  ``additionalContext``; the documented channel is a top-level
  ``decision: "block"`` whose ``reason`` is shown to the model — the tool call
  already failed, so nothing is actually blocked.
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

from capture_query_failures import (  # noqa: E402
    CONSOLIDATABLE_ISSUE_TYPES,
    _engine,
    _query_text,
    detect_issues,
    normalize_response,
    tool_key,
)
from consolidate_query_errors import (  # noqa: E402
    QUERY_ERROR_PATTERN_FULLTEXT_INDEX,
    QUERY_ERROR_PATTERN_VECTOR_INDEX,
)
from project_common import (  # noqa: E402
    _hybrid_keyword_query,
    embed_text,
    load_mkg_env,
    neo4j_config,
    normalize_tool_failure_payload,
    resolve_project,
    truncate,
)

MAX_PATTERNS = 2
RRF_K = 60.0
RANK_LIMIT = 8
MAX_SEARCH_TEXT = 2000

_PATTERN_RETURN = """
        RETURN node.id AS id,
               node.title AS title,
               node.error_signature AS error_signature,
               node.root_cause AS root_cause,
               node.resolution AS resolution,
               node.example_query AS example_query,
               node.example_fix AS example_fix,
               node.occurrence_count AS occurrence_count,
               score
"""

_VECTOR_BRANCH = f"""
        MATCH (node:QueryErrorPattern)
        SEARCH node IN (
            VECTOR INDEX {QUERY_ERROR_PATTERN_VECTOR_INDEX}
            FOR $query_vector
            WHERE node.project_id = $project_id
              AND node.tool_key = $tool_key
            LIMIT $rank_limit
        ) SCORE AS raw_score
        WHERE node.status = 'active'
        WITH node, raw_score
        ORDER BY raw_score DESC
        WITH collect({{node: node, raw_score: raw_score}}) AS rows
        UNWIND range(0, size(rows) - 1) AS idx
        WITH rows[idx] AS row, idx + 1 AS rank
        RETURN row.node AS node, rank, 'vector' AS source
"""

_FULLTEXT_BRANCH = f"""
        CALL db.index.fulltext.queryNodes('{QUERY_ERROR_PATTERN_FULLTEXT_INDEX}', $search_query)
        YIELD node, score AS raw_score
        WHERE node:QueryErrorPattern
          AND node.project_id = $project_id
          AND node.tool_key = $tool_key
          AND node.status = 'active'
        WITH node, raw_score
        ORDER BY raw_score DESC
        LIMIT $rank_limit
        WITH collect({{node: node, raw_score: raw_score}}) AS rows
        UNWIND range(0, size(rows) - 1) AS idx
        WITH rows[idx] AS row, idx + 1 AS rank
        RETURN row.node AS node, rank, 'keyword' AS source
"""


def build_recall_request(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Decide whether this tool result warrants pattern recall.

    Returns the retrieval request (tool key, query, search text) only when the
    call is a query tool whose response classifies as a consolidatable
    invalid-query failure. Successes, interrupts, and transient-only failures
    (timeouts, resource limits, permissions, ...) return ``None`` — those never
    have consolidated fixes, so injecting invalid-query guidance would be
    noise.
    """
    payload = normalize_tool_failure_payload(payload)
    if payload.get("is_interrupt") is True:
        return None

    tool_name = str(payload.get("tool_name") or "")
    key = tool_key(tool_name)
    engine = _engine(tool_name)
    if key is None or engine is None:
        return None

    query = _query_text(payload.get("tool_input"))
    if not query:
        return None

    normalized = normalize_response(engine, payload.get("tool_response"))
    issue_types = {issue["type"] for issue in detect_issues(engine, normalized)}
    if not issue_types & CONSOLIDATABLE_ISSUE_TYPES:
        return None

    search_text = truncate(
        f"{normalized.raw_text}\n{query}".strip(), MAX_SEARCH_TEXT
    )
    return {
        "tool_key": key,
        "engine": engine,
        "query": query,
        "search_text": search_text,
        "issue_types": sorted(issue_types),
        "source_event": str(
            payload.get("source_event") or payload.get("hook_event_name") or "PostToolUse"
        ),
        "session_id": str(payload.get("session_id") or "unknown"),
    }


def _query_records(result: Any) -> list[Any]:
    return list(getattr(result, "records", result) or [])


def fetch_matching_patterns(
    driver,
    database: str,
    *,
    project_id: str,
    tool: str,
    search_text: str,
    query_vector: list[float] | None,
    limit: int = MAX_PATTERNS,
) -> list[dict[str, Any]]:
    """Hybrid (vector + fulltext, RRF-fused) pattern lookup, scoped to one tool."""
    keyword_query = _hybrid_keyword_query(search_text)
    if not query_vector and not keyword_query:
        return []

    params = {
        "project_id": project_id,
        "tool_key": tool,
        "search_query": keyword_query,
        "query_vector": query_vector,
        "rank_limit": RANK_LIMIT,
        "rrf_k": RRF_K,
        "limit": limit,
    }

    if query_vector:
        branches = [_VECTOR_BRANCH]
        if keyword_query:
            branches.append(_FULLTEXT_BRANCH)
        hybrid_query = f"""
            CALL () {{
                {'UNION ALL'.join(branches)}
            }}
            WITH node, sum(1.0 / ($rrf_k + rank)) AS score
            {_PATTERN_RETURN}
            ORDER BY score DESC, coalesce(node.occurrence_count, 0) DESC
            LIMIT $limit
        """
        try:
            records = driver.execute_query(hybrid_query, database_=database, **params)
            return [dict(record) for record in _query_records(records)]
        except Exception:
            pass

    if keyword_query:
        fulltext_query = f"""
            CALL () {{
                {_FULLTEXT_BRANCH}
            }}
            WITH node, 1.0 / ($rrf_k + rank) AS score
            {_PATTERN_RETURN}
            ORDER BY score DESC, coalesce(node.occurrence_count, 0) DESC
            LIMIT $limit
        """
        try:
            records = driver.execute_query(fulltext_query, database_=database, **params)
            return [dict(record) for record in _query_records(records)]
        except Exception:
            return []

    return []


def mark_patterns_injected(
    driver,
    database: str,
    pattern_ids: list[str],
    session_id: str,
) -> None:
    """Usage stats plus a session link for observability. Unlike memory
    injection there is no per-session dedup: a recurring error should surface
    its fix every time it recurs."""
    if not pattern_ids:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    query = """
        UNWIND $pattern_ids AS pattern_id
        MATCH (e:QueryErrorPattern {id: pattern_id})
        SET e.last_used_at = datetime($timestamp),
            e.use_count = coalesce(e.use_count, 0) + 1
    """
    driver.execute_query(
        query, database_=database, pattern_ids=pattern_ids, timestamp=timestamp
    )
    if session_id and session_id != "unknown":
        driver.execute_query(
            """
            MERGE (s:Session {session_id: $session_id})
            ON CREATE SET s.created_at = datetime($timestamp)
            WITH s
            UNWIND $pattern_ids AS pattern_id
            MATCH (e:QueryErrorPattern {id: pattern_id})
            MERGE (e)-[r:INJECTED_IN]->(s)
            ON CREATE SET r.first_injected_at = datetime($timestamp)
            SET r.last_injected_at = datetime($timestamp),
                r.inject_count = coalesce(r.inject_count, 0) + 1
            """,
            database_=database,
            session_id=session_id,
            pattern_ids=pattern_ids,
            timestamp=timestamp,
        )


def format_pattern_context(tool: str, patterns: list[dict[str, Any]]) -> str:
    if not patterns:
        return ""
    lines = [
        f"Known {tool} failure patterns from prior sessions match this error:"
    ]
    for index, row in enumerate(patterns, start=1):
        lines.append("")
        lines.append(f"{index}. {truncate(str(row.get('title') or ''), 200)}")
        for label, field, limit in (
            ("Error", "error_signature", 300),
            ("Root cause", "root_cause", 400),
            ("Fix", "resolution", 700),
            ("Example failing query", "example_query", 400),
            ("Corrected query", "example_fix", 400),
        ):
            value = row.get(field)
            if value:
                lines.append(f"   {label}: {truncate(str(value), limit)}")
    lines.extend(
        [
            "",
            "If one of these patterns matches the failure, apply its fix "
            "instead of retrying small variations of the failed query.",
        ]
    )
    return "\n".join(lines)


def build_hook_output(source_event: str, context: str) -> dict[str, Any]:
    """Pick the delivery channel the firing event actually supports.

    ``PostToolUseFailure`` has no ``additionalContext``; its documented way to
    put text in front of the model is ``decision: "block"`` + ``reason`` (the
    tool already failed, so nothing is blocked). Plain ``PostToolUse`` gets the
    non-blocking ``additionalContext`` next to the tool result.
    """
    if source_event == "PostToolUseFailure":
        return {"decision": "block", "reason": context}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": context,
        }
    }


def recall(payload: dict[str, Any]) -> dict[str, Any] | None:
    request = build_recall_request(payload)
    if request is None:
        return None

    project_root = Path(__file__).resolve().parents[1]
    project = resolve_project(payload, project_root)
    if project is None:
        return None

    query_vector = embed_text(request["search_text"])

    from neo4j import GraphDatabase

    uri, user, password, database = neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        patterns = fetch_matching_patterns(
            driver,
            database,
            project_id=project.id,
            tool=request["tool_key"],
            search_text=request["search_text"],
            query_vector=query_vector,
        )
        if not patterns:
            return None
        mark_patterns_injected(
            driver,
            database,
            [str(row["id"]) for row in patterns if row.get("id")],
            request["session_id"],
        )

    context = format_pattern_context(request["tool_key"], patterns)
    if not context:
        return None
    return build_hook_output(request["source_event"], context)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        output = recall(payload)
        if output is not None:
            print(json.dumps(output))
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[inject_query_error_context] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
