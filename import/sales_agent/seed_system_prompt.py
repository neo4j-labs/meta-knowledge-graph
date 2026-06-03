"""Persist the sales-agent persona prompt to Neo4j as a ``(:SystemPrompt)`` node.

The ``inject_system_prompt`` SessionStart hook loads ``(:SystemPrompt {name})``
where ``name = $MKG_PROMPT_NAME`` (default ``"default"``) and otherwise falls
back to a generic, tool-agnostic bootstrap prompt baked into the hook. This
script seeds the sales persona from ``system_prompt.md``.

    # seed under its own name (select it via MKG_PROMPT_NAME=sales_agent)
    uv run python import/sales_agent/seed_system_prompt.py

    # ALSO make it the active prompt with no env change (writes 'default' too)
    uv run python import/sales_agent/seed_system_prompt.py --default

Other personas/datasets ship their own system_prompt.md and seed under their own
name; switch between them with MKG_PROMPT_NAME. The in-hook fallback stays
generic so an unseeded environment still bootstraps sensibly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROMPT_NAME = "sales_agent"
PROMPT_FILE = Path(__file__).resolve().parent / "system_prompt.md"

load_dotenv()


def _seed(session, name: str, content: str) -> None:
    session.run(
        "MERGE (p:SystemPrompt {name: $name}) "
        "SET p.content = $content, p.updated_at = datetime()",
        name=name,
        content=content,
    )
    print(f"  seeded (:SystemPrompt {{name: '{name}'}}) — {len(content)} chars")


def main(argv: list[str]) -> int:
    also_default = "--default" in argv
    content = PROMPT_FILE.read_text()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            _seed(session, PROMPT_NAME, content)
            if also_default:
                _seed(session, "default", content)

    if not also_default:
        print(f"\nTip: set MKG_PROMPT_NAME={PROMPT_NAME} to activate, "
              "or re-run with --default to make it the active prompt directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
