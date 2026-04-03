"""
Graph document import queries and utilities.

Adapted from langchain-neo4j Neo4jGraph.add_graph_documents to work with
the async Neo4j driver directly. Uses Neo4j 5 dynamic labels and
relationship types instead of APOC procedures.
"""

from hashlib import md5
from typing import Any, Dict, List, Optional

from neo4j import AsyncDriver

BASE_ENTITY_LABEL = "__Entity__"

include_docs_query = (
    "MERGE (d:Document {id:$document.metadata.id}) "
    "ON CREATE SET d.created_at = datetime(), d.source_count = 1 "
    "ON MATCH SET d.updated_at = datetime(), d.source_count = coalesce(d.source_count, 0) + 1 "
    "SET d.text = $document.page_content "
    "SET d += $document.metadata "
    "WITH d "
)


def _get_node_import_query(baseEntityLabel: bool, include_source: bool) -> str:
    if baseEntityLabel:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.id}}) "
            "ON CREATE SET source.created_at = datetime(), source.source_count = 1 "
            "ON MATCH SET source.updated_at = datetime(), source.source_count = coalesce(source.source_count, 0) + 1 "
            "SET source += row.properties "
            "SET source:$(row.type) "
            f"{'MERGE (d)-[:MENTIONS]->(source) ' if include_source else ''}"
            "RETURN distinct 'done' AS result"
        )
    else:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            "MERGE (source:$(row.type) {id: row.id}) "
            "ON CREATE SET source.created_at = datetime(), source.source_count = 1 "
            "ON MATCH SET source.updated_at = datetime(), source.source_count = coalesce(source.source_count, 0) + 1 "
            "SET source += row.properties "
            f"{'MERGE (d)-[:MENTIONS]->(source) ' if include_source else ''}"
            "RETURN distinct 'done' AS result"
        )


def _get_rel_import_query(baseEntityLabel: bool) -> str:
    if baseEntityLabel:
        return (
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.source}}) "
            f"MERGE (target:`{BASE_ENTITY_LABEL}` {{id: row.target}}) "
            "WITH source, target, row "
            "MERGE (source)-[r:$(row.type)]->(target) "
            "ON CREATE SET r.created_at = datetime(), r.source_count = 1 "
            "ON MATCH SET r.updated_at = datetime(), r.source_count = coalesce(r.source_count, 0) + 1 "
            "SET r += row.properties "
            "RETURN distinct 'done'"
        )
    else:
        return (
            "UNWIND $data AS row "
            "MERGE (source:$(row.source_label) {id: row.source}) "
            "MERGE (target:$(row.target_label) {id: row.target}) "
            "WITH source, target, row "
            "MERGE (source)-[r:$(row.type)]->(target) "
            "ON CREATE SET r.created_at = datetime(), r.source_count = 1 "
            "ON MATCH SET r.updated_at = datetime(), r.source_count = coalesce(r.source_count, 0) + 1 "
            "SET r += row.properties "
            "RETURN distinct 'done'"
        )


def _remove_backticks(text: str) -> str:
    return text.replace("`", "")


async def add_graph_documents(
    driver: AsyncDriver,
    graph_documents: List[Any],
    database: str = "neo4j",
    include_source: bool = False,
    baseEntityLabel: bool = False,
) -> None:
    """
    Import graph documents into Neo4j.

    Args:
        driver: Async Neo4j driver instance.
        graph_documents: List of GraphDocument objects with nodes, relationships,
            and optionally source document info.
        database: Neo4j database name.
        include_source: If True, stores the source document and links it to
            nodes via MENTIONS relationships. Merges on metadata.id or
            MD5 hash of page_content.
        baseEntityLabel: If True, all nodes get a secondary __Entity__ label
            with a unique constraint on id, improving import speed.
    """
    if baseEntityLabel:
        await driver.execute_query(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (b:{BASE_ENTITY_LABEL}) "
            "REQUIRE b.id IS UNIQUE;",
            database_=database,
        )

    if include_source:
        for doc in graph_documents:
            if doc.source is None:
                raise TypeError(
                    "include_source is set to True, "
                    "but at least one document has no `source`."
                )

    node_import_query = _get_node_import_query(baseEntityLabel, include_source)
    rel_import_query = _get_rel_import_query(baseEntityLabel)

    for document in graph_documents:
        # Prepare node data
        node_import_query_params: Dict[str, Any] = {
            "data": [el.__dict__ for el in document.nodes]
        }
        if include_source and document.source:
            if not document.source.metadata.get("id"):
                document.source.metadata["id"] = md5(
                    document.source.page_content.encode("utf-8")
                ).hexdigest()
            node_import_query_params["document"] = document.source.__dict__

        # Remove backticks from node types
        for node in document.nodes:
            node.type = _remove_backticks(node.type)

        # Import nodes
        await driver.execute_query(
            node_import_query,
            parameters_=node_import_query_params,
            database_=database,
        )

        # Import relationships
        await driver.execute_query(
            rel_import_query,
            parameters_={
                "data": [
                    {
                        "source": el.source.id,
                        "source_label": _remove_backticks(el.source.type),
                        "target": el.target.id,
                        "target_label": _remove_backticks(el.target.type),
                        "type": _remove_backticks(
                            el.type.replace(" ", "_").upper()
                        ),
                        "properties": el.properties,
                    }
                    for el in document.relationships
                ]
            },
            database_=database,
        )
