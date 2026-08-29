#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0"]
# ///
"""SessionStart / UserPromptSubmit hook.

Injects a system prompt fetched from a Neo4j ``(:SystemPrompt {name})`` node into
the agent session as additional context. Falls back to a bootstrap prompt
if Neo4j is unreachable or the requested prompt does not exist.

The base prompt is composed at injection time with the consolidated
``(:UserProfile)`` section — the compact "user adaptations" block the
consolidation service maintains from human-approved user facts. The base
persona itself is never rewritten by consolidation; only this appended
section evolves.

Customize the active prompt by editing the node in Neo4j
(``MATCH (p:SystemPrompt {name: 'default'}) SET p.content = ...``).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import load_mkg_env, neo4j_config  # noqa: E402

DEFAULT_PROMPT = """You are a general-purpose AI assistant with persistent memory.

Your memory lives in a knowledge graph exposed through the
``meta-knowledge-graph`` MCP server. It carries durable context across
sessions — facts about the user, the active project, and lessons from past
work — so you can pick up where earlier sessions left off instead of starting
cold. Depending on the environment, the same server may also expose domain
data and other capabilities beyond memory.

Help the user with whatever they bring: questions, code, analysis, research,
or planning. Memory is infrastructure, not the task — lead with the user's
goal and use the graph in service of it.

Your runtime environment varies between projects and sessions — the exact set of
MCP tools mounted under ``meta-knowledge-graph`` is not fixed. Do not assume any
specific tool exists from prior context. Inspect the live tool list at session
start and use the tools you actually see. If a capability you expected is
missing, treat that as a fact about this environment, not a transient glitch.

Operating principles (tool-agnostic — apply them with whatever memory tools are
available in this session):

1. Recall before asking. Use the user-scoped learnings injected at
   SessionStart, and project-scoped learnings injected on user
   prompts, before asking the user to recap context they have already given.
2. Capture durable signal. When the user corrects you or asserts a constraint
   that future sessions will need, store it as a learning right away rather
   than relying on end-of-session auto-capture to notice.
3. Trust the auto-capture for routine work. Stop processing sends the session
   corpus to an LLM memory extractor that writes ``:Learning`` candidates
   (durable facts, preferences, and decisions alike). Do not double-record what
   the pipeline will catch.
4. Keep stored items small, durable, and reusable across tasks. Avoid
   transcripts, ephemeral state, and project-internal trivia.
5. Separate user memory from project memory. A durable fact about the person
   you work with — their role, communication and workflow preferences, recurring
   constraints, or domain priorities — is a user-scoped learning that should
   follow them across projects. Repo- and domain-specific facts stay
   project-scoped. Lean toward capturing both as you learn them.
"""

FALLBACK_BOOTSTRAP_PROMPT = DEFAULT_PROMPT + """

Session bootstrap (run before doing other work, unless the answers are already
in project context):

1. Inspect the available ``meta-knowledge-graph`` MCP tools and confirm what is
   actually callable in this runtime. Treat the live tool list as authoritative,
   and note which capabilities (graph reads, data access, project memory,
   system-prompt management, etc.) are present.
2. Use existing user-scoped learnings from SessionStart, and pull
   project-scoped learnings for the active work. If they already
   cover the context, do not ask the user to repeat themselves.
3. If the user's name, the active project, or its goals are still unknown, open
   with one friendly onboarding question: ask whether they already know what
   they want to work on or would like help getting started. If they want help —
   or this looks like a fresh, unseeded environment — point them to the
   ``/mkg-start`` command, which checks the live MKG tools and graph state and
   walks them through getting set up, either from a demo dataset or by building
   a custom agent persona from their own memories. If their name is unknown, you
   may ask for it in the same sentence while keeping the work goal primary.
4. If the user wants help getting started or context is still missing, gather
   concise answers only:
   - the user's name,
   - what project they are working on,
   - what goals or outcomes they want,
   - important constraints, relevant systems/assets,
   - what success criteria define "done".
   Keep project and goals first; name is secondary but worth having.
5. Store only concise, factual, user-provided answers as durable learnings.
   Capture facts about the person (name, role, durable preferences) as
   user-scoped learnings, and facts about the work (project name, goals,
   constraints, success criteria) as project-scoped learnings. Do not store a
   raw transcript or a verbose biography.
"""


USER_PROFILE_NAME = "default"
USER_PROFILE_HEADER = "## User adaptations"


def _first_record(result):
    records = getattr(result, "records", result) or []
    return records[0] if records else None


def fetch_prompt_bundle_from_neo4j(
    name: str,
    profile_name: str = USER_PROFILE_NAME,
) -> tuple[str | None, str | None, bool]:
    """Fetch the base prompt and the consolidated user-profile section in one
    connection.

    Returns ``(base, profile, profile_needs_revision)``; each part is ``None``
    when its node is missing or empty, and the whole bundle degrades to
    ``(None, None, False)`` when Neo4j is unreachable.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return None, None, False

    uri, user, password, database = neo4j_config()

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            prompt_record = _first_record(
                driver.execute_query(
                    "MATCH (p:SystemPrompt {name: $name}) "
                    "RETURN p.content AS content LIMIT 1",
                    name=name,
                    database_=database,
                )
            )
            profile_record = _first_record(
                driver.execute_query(
                    "MATCH (up:UserProfile {name: $name}) "
                    "RETURN up.content AS content, "
                    "coalesce(up.needs_revision, false) AS needs_revision LIMIT 1",
                    name=profile_name,
                    database_=database,
                )
            )
        base = None
        if prompt_record and prompt_record.get("content"):
            content = str(prompt_record["content"])
            if content.strip():
                base = content
        profile = None
        needs_revision = False
        if profile_record and profile_record.get("content"):
            content = str(profile_record["content"])
            if content.strip():
                profile = content
                needs_revision = bool(profile_record.get("needs_revision"))
        return base, profile, needs_revision
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[inject_system_prompt] Neo4j lookup failed: {exc}", file=sys.stderr)
    return None, None, False


def compose_prompt(base: str, profile: str | None, needs_revision: bool = False) -> str:
    """Append the consolidated user-adaptations section to the frozen base.

    The base persona is never edited by consolidation; everything durable the
    graph has learned about the user arrives through this appended section, so
    the seeded prompt stays as-is while behaviour still adapts to the person.
    """
    section = (profile or "").strip()
    if not section:
        return base
    parts = [
        base.rstrip(),
        "",
        USER_PROFILE_HEADER,
        "",
        (
            "Consolidated from human-approved memory about this user. Apply "
            "these adaptations on top of the persona above; the user can "
            "revise them through the learning review queue."
        ),
        "",
        section,
    ]
    if needs_revision:
        parts.extend(
            [
                "",
                (
                    "(Note: part of this section traces to memory that was "
                    "retracted after consolidation; weigh it carefully until "
                    "the section is re-consolidated.)"
                ),
            ]
        )
    return "\n".join(parts)


def summarize_injection_content(prompt_name: str, content: str, source: str) -> str:
    if source == "neo4j":
        return f"Injected SystemPrompt {prompt_name!r} from Neo4j ({len(content)} chars)."
    return (
        f"Injected default MKG SystemPrompt (no persisted node for {prompt_name!r}; "
        f"{len(content)} chars). Drives bootstrap: inspect tools, recall memory, "
        "ask whether the user knows what they want or wants help starting, gather "
        "name/project/goals when missing, then capture them as user/project learnings."
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

    uri, user, password, database = neo4j_config()
    timestamp = datetime.now(timezone.utc).isoformat()
    injection_id = f"{session_id}_{timestamp}_{hook_event}_{target}"
    content_summary = summarize_injection_content(prompt_name, content, source)

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            row = _first_record(
                driver.execute_query(
                    """
                    MERGE (s:Session {session_id: $session_id})
                    ON CREATE SET s.created_at = datetime($timestamp)
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
                                  i.timestamp = datetime($timestamp),
                                  i.injection_count = 1
                    ON MATCH SET i.last_seen_at = datetime($timestamp),
                                 i.injection_count = i.injection_count + 1
                    WITH i, i.timestamp = datetime($timestamp) AS created
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
                    database_=database,
                )
            )
        if row is not None:
            return bool(row["created"])
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[inject_system_prompt] injection log failed: {exc}", file=sys.stderr)
    return True


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)

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
    fetched, profile, profile_needs_revision = fetch_prompt_bundle_from_neo4j(prompt_name)
    base = fetched or FALLBACK_BOOTSTRAP_PROMPT
    prompt = compose_prompt(base, profile, profile_needs_revision)
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
