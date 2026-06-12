"""Build the Neocarta catalog for the RoadFlex sales-agent BigQuery dataset.

Run after ``seed_bigquery.py`` so the warehouse tables and column descriptions
exist before Neocarta introspects BigQuery, writes catalog metadata to Neo4j,
and then populates vector embeddings.

Run:
    uv run python import/sales_agent/run_neocarta.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery
from neo4j import GraphDatabase

from neocarta import NodeLabel as nl
from neocarta.connectors.bigquery import BigQuerySchemaConnector
from neocarta.enrichment.embeddings import LiteLLMEmbeddingsConnector


REPO_ROOT = Path(__file__).resolve().parents[2]


async def populate_embeddings(neo4j_db: str) -> None:
    # Give the async embedding connector a dedicated driver lifetime.
    embed_driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        connector = LiteLLMEmbeddingsConnector(
            embedding_model="text-embedding-3-small",
            neo4j_driver=embed_driver,
            database_name=neo4j_db,
        )
        await connector.arun(node_labels=[nl.DATABASE, nl.TABLE, nl.COLUMN])
    finally:
        embed_driver.close()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
    bq_project_id = os.environ["GCP_PROJECT_ID"]
    dataset_id = os.environ["BIGQUERY_DATASET_ID"]
    billing_project_id = os.environ.get("GCP_BILLING_PROJECT_ID") or bq_project_id

    print(f"building Neocarta catalog for {bq_project_id}.{dataset_id}")

    neo4j_driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        bq_client = bigquery.Client(project=billing_project_id)

        BigQuerySchemaConnector(
            client=bq_client,
            project_id=bq_project_id,
            dataset_id=dataset_id,
            neo4j_driver=neo4j_driver,
            database_name=neo4j_db,
        ).run()

        print("populating Neocarta embeddings")
        asyncio.run(populate_embeddings(neo4j_db))
    finally:
        neo4j_driver.close()

    print("done.")


if __name__ == "__main__":
    main()
