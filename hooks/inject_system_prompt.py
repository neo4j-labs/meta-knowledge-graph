#!/usr/bin/env python3
"""SessionStart / UserPromptSubmit hook.

Injects a system prompt fetched from a Neo4j ``(:SystemPrompt {name})`` node into
the agent session as additional context. Falls back to a bootstrap prompt
if Neo4j is unreachable or the requested prompt does not exist.

Customize the active prompt by either:
  * editing the node in Neo4j (``MATCH (p:SystemPrompt {name: 'default'}) SET p.content = ...``)
  * adding a new node and exporting ``MKG_PROMPT_NAME=<name>`` before launching Claude Code.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PROMPT = """You are the Intelligence Agent for the Meta Knowledge Graph (MKG).

The MKG is the enterprise intelligence layer for AI agents: it captures technical,
operational, business, and agentic metadata in a Neo4j graph and exposes it via
the ``metagraph-mcp`` MCP server.

You have specialized tools available — prefer them over generic search:

- **Neocarta** (`mcp__metagraph-mcp__neocarta_*`)
  Search and read the data catalog. Use to discover schemas, tables, and columns
  before answering questions about enterprise data. Start with
  ``neocarta_list_schemas`` / ``neocarta_list_tables_by_schema`` for navigation,
  ``neocarta_get_context_by_column_hybrid_search`` /
  ``neocarta_get_context_by_table_hybrid_search`` /
  ``neocarta_get_context_by_schema_and_table_vector_search`` for semantic lookup
  (each takes ``text_content``),
  ``neocarta_get_full_metadata_schema`` when you need the complete picture.

- **Memory** (`mcp__metagraph-mcp__memory_*`)
  Persistent agent memory backed by Neo4j. ``memory_search`` / ``memory_get_context``
  to recall; ``memory_store_message``, ``memory_add_fact``, ``memory_add_entity``,
  ``memory_add_preference`` to persist. Use ``memory_start_trace`` /
  ``memory_record_step`` / ``memory_complete_trace`` to capture reasoning chains
  the GDS engine will later mine for patterns.

- **Knowledge graph** (`mcp__metagraph-mcp__neo4j_*`, ``import_text_to_kg``)
  ``neo4j_get_schema`` and ``neo4j_read_cypher`` for direct graph access;
  ``import_text_to_kg`` to extract entities and relationships from raw text.

- **BigQuery** (`mcp__metagraph-mcp__bigquery_execute_query`)
  For querying source warehouses when the metadata layer is not enough.

Operating principles:
1. Check memory before asking the user to repeat themselves.
2. Search neocarta before guessing about data shapes.
3. Record meaningful decisions and corrections back into memory so the graph
   compounds over time.
"""

FALLBACK_BOOTSTRAP_PROMPT = DEFAULT_PROMPT + """

Neo4j did not return a ``SystemPrompt`` for this session. Bootstrap the MKG
context before continuing:

1. Inspect the available ``metagraph-mcp`` MCP tools so you know which Neo4j,
   memory, Neocarta, and BigQuery capabilities are callable in this runtime.
2. If the user's name and interests are not already available from memory, ask
   the user to share their name and the topics or projects they care about.
3. After the user answers, store only a concise profile in Neo4j memory. Keep it
   short, factual, and user-provided, for example: "User is <name>; interests:
   <comma-separated interests>." Do not store a raw transcript or a verbose
   biography.
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
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
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
        "Injected fallback MKG bootstrap prompt because Neo4j did not return a "
        "SystemPrompt. It tells the agent to inspect metagraph-mcp tools, ask "
        "for the user's name and interests, and store a concise Neo4j memory "
        "profile."
    )


def log_injection(
    session_id: str,
    hook_event: str,
    target: str,
    prompt_name: str,
    content: str,
    source: str,
) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return

    uri, user, password, database = _neo4j_config()
    timestamp = datetime.now(timezone.utc).isoformat()
    injection_id = f"{session_id}_{timestamp}_{hook_event}_{target}"
    content_summary = summarize_injection_content(prompt_name, content, source)

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=database) as session:
                session.run(
                    """
                    MERGE (s:Session {session_id: $session_id})
                    ON CREATE SET s.created_at = $timestamp
                    CREATE (i:Injection {
                        injection_id: $injection_id,
                        hook_event: $hook_event,
                        target: $target,
                        prompt_name: $prompt_name,
                        source: $source,
                        content: $content_summary,
                        content_summary: $content_summary,
                        char_count: $stored_char_count,
                        original_char_count: $original_char_count,
                        stored_char_count: $stored_char_count,
                        timestamp: $timestamp
                    })
                    CREATE (s)-[:INJECTED]->(i)
                    """,
                    session_id=session_id,
                    injection_id=injection_id,
                    hook_event=hook_event,
                    target=target,
                    prompt_name=prompt_name,
                    source=source,
                    content_summary=content_summary,
                    original_char_count=len(content),
                    stored_char_count=len(content_summary),
                    timestamp=timestamp,
                )
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[inject_system_prompt] injection log failed: {exc}", file=sys.stderr)


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

    prompt_name = os.getenv("MKG_PROMPT_NAME", "default")
    fetched = fetch_prompt_from_neo4j(prompt_name)
    prompt = fetched or FALLBACK_BOOTSTRAP_PROMPT
    source = "neo4j" if fetched else "default"

    log_injection(
        session_id=session_id,
        hook_event=hook_event,
        target="additionalContext",
        prompt_name=prompt_name,
        content=prompt,
        source=source,
    )

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
