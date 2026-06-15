#!/usr/bin/env python3
"""Seed or update a ``(:MemoryExtractionPrompt {name})`` node in Neo4j.

The template is frozen at runtime — ``process_project.py`` only reads it — so
this seed script (and the future consolidation service) are the writers.
Re-seeding identical content is a no-op; a content change bumps the version
counter.

Usage:
    python hooks/seed_memory_extraction_prompt.py            # seed 'default'
    python hooks/seed_memory_extraction_prompt.py NAME       # seed NAME
    python hooks/seed_memory_extraction_prompt.py NAME FILE  # seed NAME from FILE
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from process_project import (  # noqa: E402
    DEFAULT_MEMORY_EXTRACTION_PROMPT,
    DEFAULT_MEMORY_EXTRACTION_PROMPT_NAME,
    memory_extraction_prompt_is_valid,
)
from project_common import (  # noqa: E402
    ensure_project_schema,
    load_dotenv,
    neo4j_config,
    upsert_prompt_node,
)


def main(argv: list[str]) -> int:
    name = argv[1] if len(argv) > 1 else DEFAULT_MEMORY_EXTRACTION_PROMPT_NAME
    if len(argv) > 2:
        content = Path(argv[2]).read_text()
    else:
        content = DEFAULT_MEMORY_EXTRACTION_PROMPT

    if not memory_extraction_prompt_is_valid(content):
        print(
            "Memory extraction prompt is missing required runtime tokens.",
            file=sys.stderr,
        )
        return 2

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
                label="MemoryExtractionPrompt",
                name=name,
                content=content,
                now=now,
            )
    print(
        f"Seeded (:MemoryExtractionPrompt {{name: '{name}'}}) - "
        f"{result['action']}, v{result['version']}, {len(content)} chars"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
