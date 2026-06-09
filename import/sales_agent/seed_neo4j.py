"""Seed Neo4j with the RoadFlex enterprise car-rental sales-demo graph.

Unlike a flat warehouse load, this materializes the *relationships* that make
the graph useful to an enterprise mobility sales assistant:

    (:CSM)-[:OWNS]->(:Account)
    (:Account)-[:USES_PRODUCT {mau, contracted_seats, utilization, ...}]->(:Product)
    (:Account)-[:HAS_CONTACT]->(:Contact)

Renewal/health fields are denormalized onto :Account for fast filtering. The
:Product catalog and :Account roster mirror the BigQuery tables (same ids), so
an account can be pivoted across Neo4j, BigQuery, and Diffbot on its slug/domain.

Idempotent: MERGEs on natural keys, and refreshes seed-owned USES_PRODUCT /
HAS_CONTACT edges so a re-run exactly reflects the current dataset.

Run:
    uv run python import/sales_agent/seed_neo4j.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_data import build_seed_dataset, latest_usage_by_account  # noqa: E402

load_dotenv()


SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT account_id_unique  IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT product_sku_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.sku IS UNIQUE",
    "CREATE CONSTRAINT contact_id_unique  IF NOT EXISTS FOR (c:Contact) REQUIRE c.contact_id IS UNIQUE",
    "CREATE CONSTRAINT csm_name_unique     IF NOT EXISTS FOR (c:CSM) REQUIRE c.name IS UNIQUE",
    "CREATE FULLTEXT INDEX account_fulltext IF NOT EXISTS "
    "FOR (a:Account) ON EACH [a.name, a.domain, a.industry]",
]

MERGE_ACCOUNTS = """
UNWIND $accounts AS row
MERGE (a:Account {id: row.account_id})
ON CREATE SET a.created_at = datetime()
SET a.name = row.name,
    a.domain = row.domain,
    a.industry = row.industry,
    a.region = row.region,
    a.size_class = row.size_class,
    a.employee_count_band = row.employee_count_band,
    a.arr_band_usd = row.arr_band_usd,
    a.arr_usd = row.arr_usd,
    a.seats_total = row.seats_total,
    a.avg_utilization = row.avg_utilization,
    a.health_score = row.health_score,
    a.trajectory = row.trajectory,
    a.signed_at = date(row.signed_at),
    a.renewal_date = date(row.renewal_date),
    a.owner_csm = row.owner_csm,
    a.source = 'seed',
    a.updated_at = datetime()
"""

MERGE_PRODUCTS = """
UNWIND $products AS row
MERGE (p:Product {sku: row.sku})
ON CREATE SET p.created_at = datetime()
SET p.name = row.name,
    p.category = row.category,
    p.tier = row.tier,
    p.list_price_usd = row.list_price_usd,
    p.launched_at = date(row.launched_at),
    p.source = 'seed',
    p.updated_at = datetime()
"""

MERGE_OWNERSHIP = """
UNWIND $accounts AS row
MATCH (a:Account {id: row.account_id})
MERGE (c:CSM {name: row.owner_csm})
MERGE (c)-[:OWNS]->(a)
"""

REFRESH_CONTACT_EDGES = """
MATCH (a:Account {source: 'seed'})-[r:HAS_CONTACT]->()
DELETE r
"""

MERGE_CONTACTS = """
UNWIND $contacts AS row
MATCH (a:Account {id: row.account_id})
MERGE (k:Contact {contact_id: row.contact_id})
SET k.first_name = row.first_name,
    k.last_name = row.last_name,
    k.email = row.email,
    k.title = row.title,
    k.role = row.role,
    k.is_decision_maker = row.is_decision_maker,
    k.is_champion = row.is_champion,
    k.source = 'seed'
MERGE (a)-[:HAS_CONTACT]->(k)
"""

REFRESH_USAGE_EDGES = """
MATCH (:Account {source: 'seed'})-[r:USES_PRODUCT]->(:Product)
DELETE r
"""

MERGE_USAGE_EDGES = """
UNWIND $edges AS row
MATCH (a:Account {id: row.account_id})
MATCH (p:Product {sku: row.sku})
MERGE (a)-[r:USES_PRODUCT]->(p)
SET r.mau = row.mau,
    r.contracted_seats = row.contracted_seats,
    r.utilization = row.utilization,
    r.monthly_revenue_usd = row.monthly_revenue_usd,
    r.month = date(row.month),
    r.last_active_at = datetime(row.last_active_at)
"""


def _driver():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


def _usage_edges(usage: list[dict]) -> list[dict]:
    edges: list[dict] = []
    for lines in latest_usage_by_account(usage).values():
        edges.extend(lines)
    return edges


def main() -> None:
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    print(f"seeding Neo4j database {database!r}")

    data = build_seed_dataset()
    edges = _usage_edges(data["usage"])

    with _driver() as driver:
        with driver.session(database=database) as session:
            for stmt in SCHEMA_STATEMENTS:
                session.run(stmt)
            session.run(MERGE_ACCOUNTS, accounts=data["accounts"])
            session.run(MERGE_PRODUCTS, products=data["products"])
            session.run(MERGE_OWNERSHIP, accounts=data["accounts"])
            session.run(REFRESH_CONTACT_EDGES)
            session.run(MERGE_CONTACTS, contacts=data["contacts"])
            session.run(REFRESH_USAGE_EDGES)
            session.run(MERGE_USAGE_EDGES, edges=edges)

    print(f"  accounts merged      : {len(data['accounts'])}")
    print(f"  products merged      : {len(data['products'])}")
    print(f"  contacts merged      : {len(data['contacts'])}")
    print(f"  USES_PRODUCT edges   : {len(edges)}")
    print("done.")


if __name__ == "__main__":
    main()
