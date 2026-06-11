"""Seed bootstrap ``(:Learning)`` / ``(:Decision)`` nodes for the sales persona.

A fresh environment starts with an empty project memory, so the first session
re-discovers facts the demo already knows: where the warehouse lives, how to
pick an engine per question, how to discover schema. This script seeds those
as curated, approved learnings and decisions attached to the ``(:Project)``,
using the same ids (``learning_id`` / ``decision_id``) and node shapes the
Stop-hook memory extraction writes, so runtime captures dedupe against them.

    uv run python import/sales_agent/seed_learnings.py
    uv run python import/sales_agent/seed_learnings.py --project my-project-id

The project id defaults to the repo folder name (mirroring how the hooks
resolve the active project from the session CWD).

Idempotent: MERGEs on the content-hash ids; re-runs refresh text/confidence
without inflating support counts.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / "hooks"
if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))

from project_common import (  # noqa: E402
    MAX_LEARNING_TEXT,
    decision_id,
    ensure_project_schema,
    learning_id,
    slugify,
)

LEARNINGS: list[dict[str, str | float]] = [
    {
        "text": (
            "Users are sales / customer-success people (CSMs and AEs) working "
            "RoadFlex's B2B enterprise rental book; sessions revolve around "
            "account health, renewals, expansion plays, stakeholder mapping, "
            "and external news monitoring."
        ),
        "summary": "Who the users are and what sessions cover",
        "task_pattern": "session bootstrap / user role",
        "confidence": 0.9,
    },
    {
        "text": (
            "BigQuery dataset acme_corp is the sales system of record "
            "(48 accounts; query via bigquery_execute_query). accounts: "
            "health_score, trajectory steady/expanding/at_risk/new, arr_usd, "
            "renewal_date, owner_csm. account_contacts: role champion/"
            "economic_buyer/technical/executive, is_decision_maker. "
            "account_renewals: term_months, auto_renew, status. "
            "account_product_usage: monthly mau/revenue/contracted_seats per "
            "sku. products: category program/addon/services/support, tier, "
            "list_price_usd."
        ),
        "summary": "BigQuery acme_corp warehouse map with observed value domains",
        "task_pattern": "sales data lookup via BigQuery",
        "confidence": 0.9,
    },
    {
        "text": (
            "Resolve unfamiliar warehouse tables through the neocarta catalog "
            "before writing SQL: neocarta_list_schemas / "
            "neocarta_list_tables_by_schema to enumerate, table-level hybrid "
            "search when a query mixes concepts with literal tokens, "
            "column-level hybrid search for field-level questions. Returned "
            "context includes column types, example values, and foreign keys."
        ),
        "summary": "Use the neocarta catalog for table/column discovery",
        "task_pattern": "table discovery / metadata search",
        "confidence": 0.85,
    },
]

DECISIONS: list[dict[str, str | float]] = [
    {
        "text": (
            "Route questions by engine: BigQuery (acme_corp) for time-series, "
            "aggregates, and cohort SQL; the Neo4j graph for multi-hop "
            "relationship questions (contacts, CSM ownership, product "
            "footprint); Diffbot enhance_entity / search_news for external "
            "signal."
        ),
        "rationale": (
            "The same accounts are mirrored across all three planes and join "
            "on account slug and domain, so the engine should match the "
            "question shape instead of forcing one store to answer everything."
        ),
        "summary": "Match the engine to the question shape",
        "task_pattern": "choosing query engine",
        "confidence": 0.85,
    },
    {
        "text": (
            "Confirm warehouse schema through the neocarta catalog before "
            "writing BigQuery SQL instead of guessing table or column names."
        ),
        "rationale": (
            "Catalog context carries column types, example values, and "
            "foreign-key references, which prevents failed queries and silent "
            "wrong-column joins."
        ),
        "summary": "Catalog lookup before SQL",
        "task_pattern": "SQL authoring",
        "confidence": 0.8,
    },
    {
        "text": (
            "During session bootstrap, persist durable user-stated facts "
            "(role, project, goals, constraints) as learnings immediately "
            "instead of waiting for end-of-session auto-capture."
        ),
        "rationale": (
            "Bootstrap facts are needed by the very next session, and the "
            "Stop-hook memory extraction only runs if the session ends cleanly."
        ),
        "summary": "Capture bootstrap facts immediately",
        "task_pattern": "session bootstrap",
        "confidence": 0.8,
    },
]

MERGE_PROJECT = """
MERGE (p:Project {id: $project_id})
ON CREATE SET p.created_at = $now,
              p.name = $project_name,
              p.status = 'active',
              p.source = 'seed'
SET p.updated_at = $now
"""

MERGE_LEARNINGS = """
MATCH (p:Project {id: $project_id})
UNWIND $rows AS row
MERGE (l:Learning {id: row.id})
ON CREATE SET l.created_at = $now,
              l.use_count = 0,
              l.support_count = 1
SET l.text = row.text,
    l.summary = row.summary,
    l.task_pattern = row.task_pattern,
    l.status = 'approved',
    l.scope = 'project',
    l.source = 'seed',
    l.last_source = 'seed',
    l.project_id = $project_id,
    l.updated_at = $now,
    l.confidence = CASE
        WHEN coalesce(l.confidence, 0.0) < row.confidence THEN row.confidence
        ELSE l.confidence
    END
MERGE (p)-[:HAS_LEARNING]->(l)
"""

MERGE_DECISIONS = """
MATCH (p:Project {id: $project_id})
UNWIND $rows AS row
MERGE (d:Decision {id: row.id})
ON CREATE SET d.created_at = $now,
              d.support_count = 1
SET d.text = row.text,
    d.rationale = row.rationale,
    d.summary = row.summary,
    d.task_pattern = row.task_pattern,
    d.source = 'seed',
    d.last_source = 'seed',
    d.project_id = $project_id,
    d.updated_at = $now,
    d.confidence = CASE
        WHEN coalesce(d.confidence, 0.0) < row.confidence THEN row.confidence
        ELSE d.confidence
    END
MERGE (p)-[:HAS_DECISION]->(d)
"""

load_dotenv()


def _resolve_project_id(argv: list[str]) -> str:
    if "--project" in argv:
        idx = argv.index("--project")
        if idx + 1 >= len(argv):
            raise SystemExit("--project requires a value")
        return slugify(argv[idx + 1])
    return slugify(REPO_ROOT.name)


def _rows(items: list[dict[str, str | float]], id_fn, project_id: str) -> list[dict]:
    rows = []
    for item in items:
        text = str(item["text"])
        if len(text) > MAX_LEARNING_TEXT:
            raise SystemExit(
                f"seed text exceeds {MAX_LEARNING_TEXT} chars ({len(text)}): "
                f"{text[:60]}..."
            )
        rows.append({**item, "id": id_fn(project_id, text)})
    return rows


def main(argv: list[str]) -> int:
    project_id = _resolve_project_id(argv)
    project_name = REPO_ROOT.name.replace("-", " ").replace("_", " ").title()
    learning_rows = _rows(LEARNINGS, learning_id, project_id)
    decision_rows = _rows(DECISIONS, decision_id, project_id)

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    now = datetime.now(timezone.utc).isoformat()

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_project_schema)
            session.run(
                MERGE_PROJECT,
                project_id=project_id,
                project_name=project_name,
                now=now,
            )
            session.run(
                MERGE_LEARNINGS, project_id=project_id, rows=learning_rows, now=now
            )
            session.run(
                MERGE_DECISIONS, project_id=project_id, rows=decision_rows, now=now
            )

    print(f"  seeded {len(learning_rows)} learnings, {len(decision_rows)} decisions "
          f"on (:Project {{id: '{project_id}'}})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
