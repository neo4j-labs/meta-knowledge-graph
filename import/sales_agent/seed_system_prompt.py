"""Persist the sales-agent persona prompt to Neo4j as a ``(:SystemPrompt)`` node.

The ``inject_system_prompt`` SessionStart hook loads ``(:SystemPrompt
{name: 'default'})`` and otherwise falls back to a generic, tool-agnostic
bootstrap prompt baked into the hook. This script seeds the sales persona from
``system_prompt.md``:

    uv run python import/sales_agent/seed_system_prompt.py

The in-hook fallback stays generic so an unseeded environment still bootstraps
sensibly.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

HOOKS_DIR = Path(__file__).resolve().parents[2] / "hooks"
if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))

from apply_system_prompt import upsert_prompt  # noqa: E402
from project_common import ensure_project_schema  # noqa: E402

PROMPT_NAME = "default"
PROMPT_FILE = Path(__file__).resolve().parent / "system_prompt.md"

load_dotenv()


def _seed(session, name: str, content: str, now: str) -> None:
    result = session.execute_write(
        upsert_prompt,
        name=name,
        content=content,
        source="seed",
        now=now,
    )
    print(
        f"  seeded (:SystemPrompt {{name: '{name}'}}) — "
        f"{result['action']}, v{result['version']}, {len(content)} chars"
    )


def main(_argv: list[str]) -> int:
    content = PROMPT_FILE.read_text()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    now = datetime.now(timezone.utc).isoformat()

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_project_schema)
            _seed(session, PROMPT_NAME, content, now)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
