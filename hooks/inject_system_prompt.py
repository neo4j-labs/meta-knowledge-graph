#!/usr/bin/env python3
"""SessionStart / UserPromptSubmit hook.

Injects a system prompt fetched from a Neo4j ``(:SystemPrompt {name})`` node into
the agent session as additional context. Falls back to a bootstrap prompt
if Neo4j is unreachable or the requested prompt does not exist.

Customize the active prompt by editing the node in Neo4j
(``MATCH (p:SystemPrompt {name: 'default'}) SET p.content = ...``).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path

DEFAULT_PROMPT = """You are the Intelligence Agent for the Meta Knowledge Graph (MKG).

The MKG is the enterprise intelligence layer for AI agents: it captures technical,
operational, business, and agentic metadata in a Neo4j graph and exposes it via
the ``meta-knowledge-graph`` MCP server.

Your runtime environment varies between projects and sessions — the exact set of
MCP tools mounted under ``meta-knowledge-graph`` is not fixed. Do not assume any
specific tool exists from prior context. Inspect the live tool list at session
start and use the tools you actually see. If a capability you expected is
missing, treat that as a fact about this environment, not a transient glitch.

Operating principles (tool-agnostic — apply them with whatever MKG tools are
available in this session):

1. Recall before asking. Pull project-scoped learnings and decisions before
   asking the user to recap context they have already given.
2. Capture durable signal. When the user corrects you or asserts a constraint
   that future sessions will need, store it as a learning right away rather
   than relying on end-of-session auto-capture to notice.
3. Trust the auto-capture for routine work. Stop / SessionEnd processing sends
   the session corpus to an LLM memory extractor that writes ``:Learning`` /
   ``:Decision`` candidates. Do not double-record what the pipeline will catch.
4. Keep stored items small, durable, and reusable across tasks. Avoid
   transcripts, ephemeral state, and project-internal trivia.
5. Improve the persisted system prompt itself when the environment teaches you
   something stable about how this project should be operated. Future sessions
   read that prompt on start, so it is the right place for durable operating
   guidance.
"""

FALLBACK_BOOTSTRAP_PROMPT = DEFAULT_PROMPT + """

Session bootstrap (run before doing other work, unless the answers are already
in project context):

1. Inspect the available ``meta-knowledge-graph`` MCP tools and confirm what is
   actually callable in this runtime. Treat the live tool list as authoritative,
   and note which capabilities (graph reads, BigQuery, neocarta, project memory,
   system-prompt management, etc.) are present.
2. Pull existing project-scoped learnings and decisions. If they already cover
   the active project context, do not ask the user to repeat themselves.
3. If the user's name, the active project, or its goals are still unknown, open
   with one friendly onboarding question: ask whether they already know what
   they want to work on or would like help getting started. If their name is
   unknown, you may ask for it in the same sentence while keeping the work goal
   primary.
4. If the user wants help getting started or context is still missing, gather
   concise answers only:
   - the user's name,
   - what project they are working on,
   - what goals or outcomes they want,
   - important constraints, relevant systems/assets,
   - what success criteria define "done".
   Keep project and goals first; name is secondary but worth having.
5. Store only concise, factual, user-provided answers as durable learnings:
   user name, project name, goals, constraints, success criteria. Do not store
   a raw transcript or a verbose biography.
6. Once bootstrap answers are gathered, persist a refined ``SystemPrompt`` back
   to Neo4j using whatever write capability this runtime offers, folding in the
   discovered tools, the user's name, and the project specifics, so the next
   session loads that prompt directly.
"""


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _neo4j_config() -> tuple[str, str, str, str]:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    return uri, user, password, database


def fetch_prompt_from_neo4j(name: str) -> str | None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return None

    uri, user, password, database = _neo4j_config()

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=database) as session:
                record = session.run(
                    "MATCH (p:SystemPrompt {name: $name}) "
                    "RETURN p.content AS content LIMIT 1",
                    name=name,
                ).single()
        if record and record.get("content"):
            content = str(record["content"])
            if content.strip():
                return content
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[inject_system_prompt] Neo4j lookup failed: {exc}", file=sys.stderr)
    return None


def summarize_injection_content(prompt_name: str, content: str, source: str) -> str:
    if source == "neo4j":
        return f"Injected SystemPrompt {prompt_name!r} from Neo4j ({len(content)} chars)."
    return (
        f"Injected default MKG SystemPrompt (no persisted node for {prompt_name!r}; "
        f"{len(content)} chars). Drives bootstrap: inspect tools, recall memory, "
        "ask whether the user knows what they want or wants help starting, gather "
        "name/project/goals when missing, then persist a refined SystemPrompt."
    )


def content_sha(content: str) -> str:
    return sha1(content.encode("utf-8")).hexdigest()[:16]


def record_injection(
    session_id: str,
    hook_event: str,
    target: str,
    prompt_name: str,
    content: str,
    source: str,
) -> bool:
    """Record the injection in the graph and report whether to inject.

    The prompt text is not copied onto the ``:SystemPromptInjection`` node; the
    node keeps a content hash plus summary, and links to the ``(:SystemPrompt)``
    it came from.
    Returns ``False`` when this exact prompt content was already injected into
    this session (the conversation already has it in context), ``True`` when the
    injection is new or dedup state is unavailable.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return True

    uri, user, password, database = _neo4j_config()
    timestamp = datetime.now(timezone.utc).isoformat()
    injection_id = f"{session_id}_{timestamp}_{hook_event}_{target}"
    content_summary = summarize_injection_content(prompt_name, content, source)

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=database) as session:
                record = session.run(
                    """
                    MERGE (s:Session {session_id: $session_id})
                    ON CREATE SET s.created_at = $timestamp
                    MERGE (s)-[:INJECTED]->(i:SystemPromptInjection {
                        prompt_name: $prompt_name,
                        content_sha: $content_sha,
                        target: $target
                    })
                    ON CREATE SET i.injection_id = $injection_id,
                                  i.hook_event = $hook_event,
                                  i.source = $source,
                                  i.content_summary = $content_summary,
                                  i.char_count = $char_count,
                                  i.summary_char_count = $summary_char_count,
                                  i.timestamp = $timestamp,
                                  i.injection_count = 1
                    ON MATCH SET i.last_seen_at = $timestamp,
                                 i.injection_count = i.injection_count + 1
                    WITH i, i.timestamp = $timestamp AS created
                    FOREACH (_ IN CASE WHEN created AND $source = 'neo4j' THEN [1] ELSE [] END |
                        MERGE (sp:SystemPrompt {name: $prompt_name})
                        MERGE (i)-[:OF_PROMPT]->(sp)
                    )
                    RETURN created
                    """,
                    session_id=session_id,
                    injection_id=injection_id,
                    hook_event=hook_event,
                    target=target,
                    prompt_name=prompt_name,
                    source=source,
                    content_sha=content_sha(content),
                    content_summary=content_summary,
                    char_count=len(content),
                    summary_char_count=len(content_summary),
                    timestamp=timestamp,
                )
                row = record.single()
        if row is not None:
            return bool(row["created"])
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[inject_system_prompt] injection log failed: {exc}", file=sys.stderr)
    return True


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}

    session_id = payload.get("session_id", "unknown")
    hook_event = payload.get("hook_event_name", "SessionStart")
    # On clear/compact the conversation context was wiped, so an earlier
    # injection in this session is gone and dedup must not suppress this one.
    context_wiped = payload.get("source") in {"clear", "compact"}

    prompt_name = "default"
    fetched = fetch_prompt_from_neo4j(prompt_name)
    prompt = fetched or FALLBACK_BOOTSTRAP_PROMPT
    source = "neo4j" if fetched else "default"

    is_new_injection = record_injection(
        session_id=session_id,
        hook_event=hook_event,
        target="additionalContext",
        prompt_name=prompt_name,
        content=prompt,
        source=source,
    )

    if not is_new_injection and not context_wiped:
        print(
            f"[inject_system_prompt] SystemPrompt {prompt_name!r} already injected "
            f"in session {session_id}; skipping duplicate.",
            file=sys.stderr,
        )
        return 0

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": prompt,
        }
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
