#!/usr/bin/env python3
"""SessionStart / UserPromptSubmit hook.

Injects a system prompt fetched from a Neo4j ``(:SystemPrompt {name})`` node into
the Claude Code session as additional context. Falls back to ``DEFAULT_PROMPT``
if Neo4j is unreachable or the requested prompt does not exist.

Customize the active prompt by either:
  * editing the node in Neo4j (``MATCH (p:SystemPrompt {name: 'default'}) SET p.content = ...``)
  * adding a new node and exporting ``MKG_PROMPT_NAME=<name>`` before launching Claude Code.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_PROMPT = """You are the Intelligence Agent for the Meta Knowledge Graph (MKG).

The MKG is the enterprise intelligence layer for AI agents: it captures technical,
operational, business, and agentic metadata in a Neo4j graph and exposes it via
the ``metagraph-mcp`` MCP server.

You have specialized tools available — prefer them over generic search:

- **Neocarta** (`mcp__metagraph-mcp__neocarta_*`)
  Search and read the data catalog. Use to discover schemas, tables, and columns
  before answering questions about enterprise data. Start with
  ``neocarta_list_schemas`` / ``neocarta_list_tables`` for navigation,
  ``neocarta_get_context_by_*_vector_search`` for semantic lookup,
  ``neocarta_full_schema`` when you need the complete picture.

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


def fetch_prompt_from_neo4j(name: str) -> str | None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return None

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=database) as session:
                record = session.run(
                    "MATCH (p:SystemPrompt {name: $name}) "
                    "RETURN p.content AS content LIMIT 1",
                    name=name,
                ).single()
        if record and record.get("content"):
            return str(record["content"])
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[inject_system_prompt] Neo4j lookup failed: {exc}", file=sys.stderr)
    return None


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    prompt_name = os.getenv("MKG_PROMPT_NAME", "default")
    prompt = fetch_prompt_from_neo4j(prompt_name) or DEFAULT_PROMPT

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
