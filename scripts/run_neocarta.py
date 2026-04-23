import os
import asyncio
from dotenv import load_dotenv
from neo4j import GraphDatabase
from google.cloud import bigquery
from openai import AsyncOpenAI

from neocarta import NodeLabel as nl
from neocarta.connectors.bigquery import BigQuerySchemaConnector
from neocarta.enrichment.embeddings import OpenAIEmbeddingsConnector

load_dotenv()

neo4j_driver = GraphDatabase.driver(
    uri=os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
)
neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")

# 1. Extract BigQuery INFORMATION_SCHEMA -> load into Neo4j
bq_client = bigquery.Client(
    project=os.environ.get("GCP_BILLING_PROJECT_ID") or os.environ["GCP_PROJECT_ID"]
)

schema_connector = BigQuerySchemaConnector(
    client=bq_client,
    project_id=os.environ["GCP_PROJECT_ID"],
    dataset_id=os.environ["BIGQUERY_DATASET_ID"],
    neo4j_driver=neo4j_driver,
    database_name=neo4j_db,
)
schema_connector.run()

# 2. Create vector indexes + backfill embeddings.
# The embedder closes its driver when done, so give it a dedicated one.
async def embed():
    embed_driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    embedder = OpenAIEmbeddingsConnector(
        neo4j_driver=embed_driver,
        async_client=AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        embedding_model="text-embedding-3-small",
        dimensions=768,
        database_name=neo4j_db,
    )
    await embedder.arun(node_labels=[nl.DATABASE, nl.SCHEMA, nl.TABLE, nl.COLUMN])

asyncio.run(embed())

from neocarta.connectors.bigquery import BigQueryLogsConnector

BigQueryLogsConnector(
    client=bq_client,
    project_id=os.environ["GCP_PROJECT_ID"],
    neo4j_driver=neo4j_driver,
    database_name=neo4j_db,
).run(
    dataset_id=os.environ["BIGQUERY_DATASET_ID"],
    region="region-us",
    limit=1000,
    drop_failed_queries=True,
)

neo4j_driver.close()