#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0"]
# ///
"""Hook: capture failed or suspicious query tool results as graph artifacts.

Wired as a PostToolUse hook — and, on Claude Code, a PostToolUseFailure hook —
for the BigQuery ``execute_query`` and Neo4j ``read_cypher`` MCP tools. The
hook only writes query artifacts when it detects an issue in the tool output;
clean successful reads are ignored. Failure payloads are normalized into the
same ``{content, isError}`` response shape Codex failures use before
classification, so both clients converge on identical QueryExecution nodes.

Graph shape::

    (:Project)-[:HAS_QUERY_EXECUTION]->(:QueryExecution)
    (:Session)-[:HAS_QUERY_EXECUTION]->(:QueryExecution)
    (:QueryExecution)-[:HAS_ISSUE]->(:QueryIssue)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import (  # noqa: E402
    ensure_project_schema,
    load_mkg_env,
    merge_project_and_session,
    neo4j_config,
    normalize_tool_failure_payload,
    resolve_project,
)

MAX_TEXT = 4000
BIGQUERY_SUFFIX = "bigquery_execute_query"
NEO4J_SUFFIX = "neo4j_read_cypher"

PERMISSION_PATTERNS = (
    "permission denied",
    "access denied",
    "forbidden",
    "unauthorized",
    "unauthenticated",
    "credentials",
)
TIMEOUT_PATTERNS = (
    "timeout",
    "timed out",
    "deadline exceeded",
    "transaction has been terminated",
    "transaction timed out",
)
RESULT_SIZE_PATTERNS = (
    "result size",
    "result set too large",
    "response too large",
    "too many rows",
    "exceeds the maximum",
    "maximum response size",
    "allowlargeresults",
)
RESOURCE_PATTERNS = (
    "resources exceeded",
    "quota exceeded",
    "rate limit",
    "exceeded memory",
    "memory limit",
    "out of memory",
)
# Procedures/extensions that are genuinely absent from the instance.
CAPABILITY_PATTERNS = (
    "procedurenotfound",
    "there is no procedure",
    "no procedure with the name",
    "unknown procedure",
    "not installed",
    "not registered",
)
# A function call scoped to an optional plugin namespace; an unknown one means
# the plugin is missing (capability gap). A bare unknown function is a typo and
# is handled as a syntax error instead.
EXTENSION_NAMESPACES = ("apoc.", "gds.")
SCHEMA_PATTERNS = (
    "unrecognized name",
    "not found: table",
    "not found: dataset",
    "no such table",
    "unknown column",
    "unknown field",
    "cannot access field",
    "variable `",
    "not defined",
)
SYNTAX_PATTERNS = (
    "syntax error",
    "syntaxerror",
    "parse error",
    "parser exception",
    "invalid input",
    "unexpected keyword",
    "unexpected end",
    "select list must not be empty",
    "unknown function",
)


@dataclass(frozen=True)
class NormalizedResponse:
    status: str
    raw_text: str
    parsed: Any
    shape: str
    row_count: int | None
    metadata: dict[str, Any]
    is_error: bool


def _truncate(value: Any, limit: int = MAX_TEXT) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _stable_id(*parts: Any, prefix: str) -> str:
    digest = sha1(
        "\n".join(json.dumps(part, sort_keys=True, default=str) for part in parts).encode(
            "utf-8"
        )
    ).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _parse_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _response_texts(value: Any) -> tuple[list[str], bool, str]:
    if value is None:
        return [], False, "missing"
    if isinstance(value, str):
        return [value], False, "string"
    if isinstance(value, list):
        texts: list[str] = []
        is_error = False
        for item in value:
            child_texts, child_error, _ = _response_texts(item)
            texts.extend(child_texts)
            is_error = is_error or child_error
        return texts, is_error, "list"
    if isinstance(value, dict):
        is_error = value.get("isError") is True
        if isinstance(value.get("text"), str):
            return [value["text"]], is_error, "text_block"
        if isinstance(value.get("result"), str):
            return [value["result"]], is_error, "result_string"
        if "result" in value and not isinstance(value.get("result"), str):
            return _response_texts(value["result"])
        if isinstance(value.get("content"), list):
            texts: list[str] = []
            child_error = False
            for item in value["content"]:
                child_texts, item_error, _ = _response_texts(item)
                texts.extend(child_texts)
                child_error = child_error or item_error
            return texts, is_error or child_error, "content"
        message = value.get("error") or value.get("message")
        if isinstance(message, str):
            return [message], True, "error_object"
        return [json.dumps(value, default=str)], is_error, "object"
    return [str(value)], False, type(value).__name__


def _unwrap_text_payload(text: str) -> Any:
    parsed = _parse_json(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("result"), str):
        nested = _parse_json(parsed["result"])
        return nested if nested is not None else parsed["result"]
    return parsed if parsed is not None else text


def _row_count(engine: str, parsed: Any) -> int | None:
    if engine == "neo4j" and isinstance(parsed, list):
        return len(parsed)
    if engine == "bigquery" and isinstance(parsed, dict):
        rows = parsed.get("rows")
        if isinstance(rows, list):
            return len(rows)
        if "schema" in parsed or parsed.get("jobComplete") is True:
            return 0
    return None


def _metadata(engine: str, parsed: Any) -> dict[str, Any]:
    if engine != "bigquery" or not isinstance(parsed, dict):
        return {}
    keys = (
        "queryId",
        "jobComplete",
        "totalBytesBilled",
        "totalBytesProcessed",
        "totalSlotMs",
    )
    return {key: parsed[key] for key in keys if key in parsed}


def normalize_response(engine: str, tool_response: Any) -> NormalizedResponse:
    texts, is_error, shape = _response_texts(tool_response)
    raw_text = "\n".join(text for text in texts if text is not None).strip()
    parsed = _unwrap_text_payload(raw_text) if raw_text else None
    status = "error" if is_error or _looks_like_error(raw_text) else "success"
    return NormalizedResponse(
        status=status,
        raw_text=raw_text,
        parsed=parsed,
        shape=shape,
        row_count=_row_count(engine, parsed),
        metadata=_metadata(engine, parsed),
        is_error=is_error,
    )


def _looks_like_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        pattern in lowered
        for pattern in (
            *PERMISSION_PATTERNS,
            *TIMEOUT_PATTERNS,
            *RESULT_SIZE_PATTERNS,
            *RESOURCE_PATTERNS,
            *CAPABILITY_PATTERNS,
            *SCHEMA_PATTERNS,
            *SYNTAX_PATTERNS,
            "neo4jerror",
            "invalidquery",
        )
    )


def _is_capability_issue(lowered: str) -> bool:
    """A missing procedure, or an unknown function in an optional plugin namespace."""
    if any(pattern in lowered for pattern in CAPABILITY_PATTERNS):
        return True
    return "unknown function" in lowered and any(
        namespace in lowered for namespace in EXTENSION_NAMESPACES
    )


def _contains_empty_object_value(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_empty_object_value(item) for item in value)
    if isinstance(value, dict):
        for child in value.values():
            if child == {}:
                return True
            if _contains_empty_object_value(child):
                return True
    return False


def _issue(
    issue_type: str,
    message: str,
    *,
    severity: str = "medium",
    confidence: float = 0.9,
    evidence: dict[str, Any] | None = None,
    needs_optimization: bool = False,
) -> dict[str, Any]:
    return {
        "type": issue_type,
        "severity": severity,
        "confidence": confidence,
        "message": _truncate(message, 1000),
        "evidence": evidence or {},
        "needs_optimization": needs_optimization,
    }


def detect_issues(engine: str, normalized: NormalizedResponse) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    raw = normalized.raw_text
    lowered = raw.lower()

    if not raw:
        issues.append(
            _issue(
                "tool_output_shape_error",
                "Tool response did not include a readable output payload.",
                severity="high",
                confidence=0.95,
                evidence={"response_shape": normalized.shape},
            )
        )
        return issues

    if normalized.status == "error":
        if any(pattern in lowered for pattern in PERMISSION_PATTERNS):
            issues.append(
                _issue(
                    "permission_error",
                    raw,
                    severity="high",
                    confidence=0.95,
                    evidence={"status": normalized.status},
                )
            )
        elif any(pattern in lowered for pattern in TIMEOUT_PATTERNS):
            issues.append(
                _issue(
                    "timeout",
                    raw,
                    severity="high",
                    confidence=0.92,
                    evidence={"status": normalized.status},
                    needs_optimization=True,
                )
            )
        elif any(pattern in lowered for pattern in RESULT_SIZE_PATTERNS):
            issues.append(
                _issue(
                    "result_size_limit",
                    raw,
                    severity="high",
                    confidence=0.92,
                    evidence={"status": normalized.status},
                    needs_optimization=True,
                )
            )
        elif any(pattern in lowered for pattern in RESOURCE_PATTERNS):
            issues.append(
                _issue(
                    "resource_limit",
                    raw,
                    severity="high",
                    confidence=0.9,
                    evidence={"status": normalized.status},
                    needs_optimization=True,
                )
            )
        elif _is_capability_issue(lowered):
            issues.append(
                _issue(
                    "capability_unavailable",
                    raw,
                    severity="medium",
                    confidence=0.92,
                    evidence={"status": normalized.status},
                )
            )
        elif any(pattern in lowered for pattern in SCHEMA_PATTERNS):
            issues.append(
                _issue(
                    "schema_mismatch",
                    raw,
                    severity="high",
                    confidence=0.9,
                    evidence={"status": normalized.status},
                )
            )
        elif any(pattern in lowered for pattern in SYNTAX_PATTERNS):
            issues.append(
                _issue(
                    "syntax_error",
                    raw,
                    severity="high",
                    confidence=0.9,
                    evidence={"status": normalized.status},
                )
            )
        else:
            issues.append(
                _issue(
                    "query_execution_error",
                    raw,
                    severity="high",
                    confidence=0.75,
                    evidence={"status": normalized.status},
                )
            )

    if normalized.status == "success" and normalized.row_count == 0:
        issues.append(
            _issue(
                "empty_result",
                "Query succeeded but returned zero rows.",
                severity="medium",
                confidence=0.95,
                evidence={
                    "row_count": normalized.row_count,
                    "response_shape": normalized.shape,
                },
                needs_optimization=True,
            )
        )

    if (
        engine == "neo4j"
        and normalized.status == "success"
        and _contains_empty_object_value(normalized.parsed)
    ):
        issues.append(
            _issue(
                "serialization_issue",
                "Neo4j result included an empty object value; temporal or spatial values may need explicit string/property projection.",
                severity="medium",
                confidence=0.85,
                evidence={"response_shape": normalized.shape},
                needs_optimization=True,
            )
        )

    return issues


def _engine(tool_name: str) -> str | None:
    lowered = tool_name.lower()
    if lowered.endswith(BIGQUERY_SUFFIX) or BIGQUERY_SUFFIX in lowered:
        return "bigquery"
    if lowered.endswith(NEO4J_SUFFIX) or NEO4J_SUFFIX in lowered:
        return "neo4j"
    return None


def _query_text(tool_input: Any) -> str | None:
    if isinstance(tool_input, dict) and isinstance(tool_input.get("query"), str):
        return tool_input["query"]
    if isinstance(tool_input, str):
        parsed = _parse_json(tool_input)
        if isinstance(parsed, dict) and isinstance(parsed.get("query"), str):
            return parsed["query"]
    return None


def build_failure_projection(payload: dict[str, Any]) -> dict[str, Any] | None:
    tool_name = str(payload.get("tool_name") or "")
    engine = _engine(tool_name)
    if engine is None:
        return None

    query = _query_text(payload.get("tool_input"))
    if not query:
        return None

    normalized = normalize_response(engine, payload.get("tool_response"))
    issues = detect_issues(engine, normalized)
    if not issues:
        return None

    session_id = str(payload.get("session_id") or "unknown")
    tool_use_id = payload.get("tool_use_id")
    query_hash = sha1(query.encode("utf-8")).hexdigest()
    response_hash = sha1(normalized.raw_text.encode("utf-8")).hexdigest()
    query_id = (
        f"query-execution:{session_id}:{tool_use_id}"
        if tool_use_id
        else _stable_id(session_id, tool_name, query_hash, response_hash, prefix="query-execution")
    )

    issue_rows = []
    for issue in issues:
        issue_rows.append(
            {
                "id": f"query-issue:{query_id}:{issue['type']}",
                **issue,
                "evidence_json": json.dumps(issue["evidence"], sort_keys=True, default=str),
            }
        )

    return {
        "query": {
            "id": query_id,
            "engine": engine,
            "tool_name": tool_name,
            "tool_use_id": str(tool_use_id) if tool_use_id else None,
            "query_text": query,
            "query_hash": query_hash,
            "status": normalized.status,
            "row_count": normalized.row_count,
            "response_shape": normalized.shape,
            "response_excerpt": _truncate(normalized.raw_text),
            "issue_count": len(issue_rows),
            "issue_types": sorted({row["type"] for row in issue_rows}),
            "needs_review": True,
            "needs_optimization": any(row["needs_optimization"] for row in issue_rows),
            "metadata_json": json.dumps(normalized.metadata, sort_keys=True, default=str),
            "query_job_id": normalized.metadata.get("queryId"),
            "total_bytes_billed": _int_or_none(normalized.metadata.get("totalBytesBilled")),
            "total_bytes_processed": _int_or_none(
                normalized.metadata.get("totalBytesProcessed")
            ),
            "total_slot_ms": _int_or_none(normalized.metadata.get("totalSlotMs")),
        },
        "issues": issue_rows,
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ensure_query_failure_schema(tx) -> None:
    ensure_project_schema(tx)
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (q:QueryExecution) REQUIRE q.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:QueryIssue) REQUIRE i.id IS UNIQUE")
    tx.run(
        "CREATE FULLTEXT INDEX query_execution_fulltext IF NOT EXISTS "
        "FOR (q:QueryExecution) ON EACH [q.query_text, q.response_excerpt]"
    )
    tx.run(
        "CREATE FULLTEXT INDEX query_issue_fulltext IF NOT EXISTS "
        "FOR (i:QueryIssue) ON EACH [i.type, i.message]"
    )


def write_failure_projection(
    tx,
    project,
    session_id: str,
    projection: dict[str, Any],
    timestamp: str,
) -> None:
    merge_project_and_session(tx, project, session_id, timestamp)
    tx.run(
        """
        MATCH (p:Project {id: $project_id})
        MATCH (s:Session {session_id: $session_id})
        MERGE (q:QueryExecution {id: $query_row.id})
        ON CREATE SET q.created_at = datetime($timestamp)
        SET q += $query_row,
            q.project_id = $project_id,
            q.session_id = $session_id,
            q.updated_at = datetime($timestamp),
            q.last_seen_at = datetime($timestamp)
        MERGE (p)-[:HAS_QUERY_EXECUTION]->(q)
        MERGE (s)-[:HAS_QUERY_EXECUTION]->(q)
        WITH q
        UNWIND $issues AS row
        MERGE (issue:QueryIssue {id: row.id})
        ON CREATE SET issue.created_at = datetime($timestamp)
        SET issue.type = row.type,
            issue.severity = row.severity,
            issue.confidence = row.confidence,
            issue.message = row.message,
            issue.evidence_json = row.evidence_json,
            issue.needs_optimization = row.needs_optimization,
            issue.updated_at = datetime($timestamp),
            issue.last_seen_at = datetime($timestamp)
        MERGE (q)-[:HAS_ISSUE]->(issue)
        """,
        project_id=project.id,
        session_id=session_id,
        query_row=projection["query"],
        issues=projection["issues"],
        timestamp=timestamp,
    )


def capture(payload: dict[str, Any]) -> int:
    payload = normalize_tool_failure_payload(payload)
    # A user interrupt (Ctrl+C) aborts the call before the engine answers;
    # recording it as a query issue would be noise.
    if payload.get("is_interrupt") is True:
        return 0
    projection = build_failure_projection(payload)
    if projection is None:
        return 0

    from neo4j import GraphDatabase

    project_root = Path(__file__).resolve().parents[1]
    project = resolve_project(payload, project_root)
    if project is None:
        return 0

    session_id = str(payload.get("session_id") or "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()
    uri, user, password, database = neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_query_failure_schema)
            session.execute_write(
                write_failure_projection,
                project,
                session_id,
                projection,
                timestamp,
            )
    return len(projection["issues"])


def _iter_transcript_records(transcript_path: str):
    path = Path(transcript_path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "\n".join(parts)
    return json.dumps(content, default=str) if content is not None else ""


def _decode_arguments(value: Any) -> Any:
    if isinstance(value, str):
        parsed = _parse_json(value)
        return parsed if parsed is not None else value
    return value


def _codex_tool_name(namespace: Any, name: Any) -> str:
    tool_name = str(name or "")
    namespace_name = str(namespace or "")
    if namespace_name:
        return f"{namespace_name}__{tool_name}"
    return tool_name


def _codex_invocation_tool_name(invocation: dict[str, Any]) -> str:
    tool_name = str(invocation.get("tool") or "")
    server = str(invocation.get("server") or "")
    if server:
        namespace = "mcp__" + server.replace("-", "_")
        return _codex_tool_name(namespace, tool_name)
    return tool_name


def _codex_tool_response(result: Any) -> dict[str, Any]:
    if isinstance(result, dict) and "Ok" in result:
        ok = result.get("Ok")
        if isinstance(ok, dict):
            return ok
        return {"content": [{"type": "text", "text": _tool_result_text(ok)}], "isError": False}
    if isinstance(result, dict) and "Err" in result:
        return {
            "content": [{"type": "text", "text": _tool_result_text(result.get("Err"))}],
            "isError": True,
        }
    return {
        "content": [{"type": "text", "text": _tool_result_text(result)}],
        "isError": True,
    }


def extract_query_payloads(transcript_path: str, session_id: str) -> list[dict[str, Any]]:
    """Reconstruct PostToolUse-shaped payloads for query tools from a transcript.

    Claude Code now surfaces failures live through PostToolUseFailure, but Codex
    has no failure event and PostToolUse never fires for ``isError`` tool
    results there, so the transcript remains the only complete record of failed
    Codex queries — and a safety net for calls whose live hook never ran.
    Supports both Claude-style ``tool_use``/``tool_result`` message blocks and
    Codex ``response_item/function_call`` plus ``event_msg/mcp_tool_call_end``
    records, normalizing each response into the ``{content, isError}`` shape the
    classifier already understands.
    """
    pending: dict[Any, dict[str, Any]] = {}
    payloads: list[dict[str, Any]] = []
    for record in _iter_transcript_records(transcript_path):
        payload = record.get("payload") if isinstance(record, dict) else None
        if isinstance(payload, dict):
            payload_type = payload.get("type")
            if record.get("type") == "response_item" and payload_type == "function_call":
                tool_name = _codex_tool_name(payload.get("namespace"), payload.get("name"))
                if _engine(tool_name) is not None:
                    pending[payload.get("call_id")] = {
                        "tool_name": tool_name,
                        "tool_input": _decode_arguments(payload.get("arguments")),
                    }
                continue
            if record.get("type") == "event_msg" and payload_type == "mcp_tool_call_end":
                call_id = payload.get("call_id")
                invocation = payload.get("invocation")
                invocation = invocation if isinstance(invocation, dict) else {}
                meta = pending.pop(call_id, None)
                if meta is None:
                    tool_name = _codex_invocation_tool_name(invocation)
                    if _engine(tool_name) is None:
                        continue
                    meta = {
                        "tool_name": tool_name,
                        "tool_input": invocation.get("arguments"),
                    }
                payloads.append(
                    {
                        "session_id": session_id,
                        "tool_use_id": call_id,
                        "tool_name": meta["tool_name"],
                        "tool_input": meta["tool_input"],
                        "tool_response": _codex_tool_response(payload.get("result")),
                    }
                )
                continue

        message = record.get("message") if isinstance(record, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                if _engine(str(block.get("name") or "")) is None:
                    continue
                pending[block.get("id")] = {
                    "tool_name": str(block.get("name") or ""),
                    "tool_input": block.get("input"),
                }
            elif block_type == "tool_result":
                meta = pending.pop(block.get("tool_use_id"), None)
                if meta is None:
                    continue
                payloads.append(
                    {
                        "session_id": session_id,
                        "tool_use_id": block.get("tool_use_id"),
                        "tool_name": meta["tool_name"],
                        "tool_input": meta["tool_input"],
                        "tool_response": {
                            "content": [
                                {"type": "text", "text": _tool_result_text(block.get("content"))}
                            ],
                            "isError": bool(block.get("is_error")),
                        },
                    }
                )
    return payloads


def capture_transcript(
    transcript_path: str,
    session_id: str,
    project_payload: dict[str, Any],
    project_root: Path,
) -> int:
    """Scan a transcript and persist every detectable query issue. Idempotent.

    Reuses the same stable ``query-execution:{session}:{tool_use_id}`` ids as
    the live PostToolUse / PostToolUseFailure hooks, so issues already captured
    live converge on the same nodes instead of duplicating.
    """
    projections = [
        projection
        for payload in extract_query_payloads(transcript_path, session_id)
        if (projection := build_failure_projection(payload)) is not None
    ]
    if not projections:
        return 0

    from neo4j import GraphDatabase

    project = resolve_project(project_payload, project_root)
    if project is None:
        return 0

    timestamp = datetime.now(timezone.utc).isoformat()
    uri, user, password, database = neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_query_failure_schema)
            for projection in projections:
                session.execute_write(
                    write_failure_projection,
                    project,
                    session_id,
                    projection,
                    timestamp,
                )
    return len(projections)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        captured = capture(payload)
        if captured:
            print(f"[capture_query_failures] stored {captured} query issue(s)")
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[capture_query_failures] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
