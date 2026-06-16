#!/usr/bin/env python3
"""Seed or update a ``(:SystemPrompt {name})`` node in Neo4j.

The prompt is frozen at runtime — the SessionStart hook only reads it — so this
seed script and the consolidation service (``consolidate_system_prompt.py``) are
its writers. Re-seeding identical content is a no-op; a content change bumps the
version counter.

Usage:
    python hooks/seed_system_prompt.py            # seed 'default' from DEFAULT_PROMPT
    python hooks/seed_system_prompt.py NAME       # seed NAME from DEFAULT_PROMPT
    python hooks/seed_system_prompt.py NAME FILE  # seed NAME from FILE
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from inject_system_prompt import DEFAULT_PROMPT, load_dotenv  # noqa: E402
from project_common import (  # noqa: E402
    ensure_project_schema,
    neo4j_config,
    upsert_prompt_node,
)


def main(argv: list[str]) -> int:
    name = argv[1] if len(argv) > 1 else "default"
    if len(argv) > 2:
        content = Path(argv[2]).read_text()
    else:
        content = DEFAULT_PROMPT

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    from neo4j import GraphDatabase

    uri, user, password, database = neo4j_config()
    now = datetime.now(timezone.utc).isoformat()

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_project_schema)
            result = session.execute_write(
                upsert_prompt_node,
                label="SystemPrompt",
                name=name,
                content=content,
                now=now,
            )
    print(
        f"Seeded (:SystemPrompt {{name: '{name}'}}) — "
        f"{result['action']}, v{result['version']}, {len(content)} chars"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
