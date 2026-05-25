#!/usr/bin/env python3
"""Seed or update a ``(:SystemPrompt {name})`` node in Neo4j.

Usage:
    python .claude/hooks/seed_system_prompt.py            # seed 'default' from DEFAULT_PROMPT
    python .claude/hooks/seed_system_prompt.py NAME       # seed NAME from DEFAULT_PROMPT
    python .claude/hooks/seed_system_prompt.py NAME FILE  # seed NAME from FILE
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from inject_system_prompt import DEFAULT_PROMPT, load_dotenv


def main(argv: list[str]) -> int:
    name = argv[1] if len(argv) > 1 else "default"
    if len(argv) > 2:
        content = Path(argv[2]).read_text()
    else:
        content = DEFAULT_PROMPT

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.run(
                "MERGE (p:SystemPrompt {name: $name}) "
                "SET p.content = $content, p.updated_at = datetime()",
                name=name,
                content=content,
            )
    print(f"Seeded (:SystemPrompt {{name: '{name}'}}) — {len(content)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
