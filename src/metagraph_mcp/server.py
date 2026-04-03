import json
import logging
import os
import time
import uuid
from typing import List, Literal, Optional, Union

from fastmcp import FastMCP
from fastmcp.client.transports import StdioTransport
from neo4j import AsyncGraphDatabase, AsyncDriver, RoutingControl
from neo4j.exceptions import Neo4jError
from pydantic import Field

from metagraph_mcp.graph_import import add_graph_documents
from metagraph_mcp.memory import (
    memory_delete,
    memory_list,
    memory_read,
    memory_write,
)

logger = logging.getLogger(__name__)


def create_mcp_server(
    neo4j_driver: AsyncDriver,
    database: str = "neo4j",
    db_url: str = "",
    username: str = "",
    password: str = "",
) -> FastMCP:
    mcp = FastMCP("metagraph-mcp")

    # Mount official Neo4j MCP server (read-only: excludes write-cypher)
    neo4j_mcp_proxy = FastMCP.as_proxy(
        StdioTransport(
            command="neo4j-mcp-server",
            args=[],
            env={
                "NEO4J_URI": db_url,
                "NEO4J_USERNAME": username,
                "NEO4J_PASSWORD": password,
                "NEO4J_DATABASE": database,
                "NEO4J_READ_ONLY": "true",
            },
        )
    )
    mcp.mount(
        neo4j_mcp_proxy,
        prefix="neo4j",
        tool_names={
            "get-schema": "neo4j_get_schema",
            "read-cypher": "neo4j_read_cypher",
        },
    )

    @mcp.tool(name="import_text_to_kg")
    async def import_text_to_kg(
        text: str = Field(..., description="Text to extract entities and relationships from"),
        model_name: Optional[str] = Field(None, description="LLM model to use for extraction (overrides LLM_MODEL env var, defaults to gpt-5.4-mini)"),
        allowed_nodes: Optional[List[str]] = Field(None, description="Node labels to extract, e.g. ['Person', 'Organization']"),
        allowed_relationships: Optional[List[Union[str, List[str]]]] = Field(
            None,
            description="Relationship constraints. Either simple strings like ['WORKS_AT'] "
            "or 3-element lists like [['Person', 'WORKS_AT', 'Organization']]",
        ),
        document_id: Optional[str] = Field(None, description="Identifier for the source document"),
        source_uri: Optional[str] = Field(None, description="Source URI of the text (file path, URL, etc.)"),
        chunk_index: Optional[int] = Field(None, description="Index of this chunk if text was split"),
        total_chunks: Optional[int] = Field(None, description="Total number of chunks if text was split"),
        chunk_of: Optional[str] = Field(None, description="Document ID of the parent document if this is a chunk"),
        parent_import_id: Optional[str] = Field(None, description="Import ID of a previous import this is a re-run or continuation of"),
        caller: Optional[str] = Field(None, description="Identifier for the agent or tool that triggered this import"),
    ) -> str:
        """Extract entities and relationships from text using an LLM and import them as a knowledge graph into Neo4j."""
        from hashlib import md5

        from langchain_core.documents import Document
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        from langchain_openai import ChatOpenAI

        import_id = str(uuid.uuid4())
        
        # Use the provided model, fallback to env var, fallback to default
        model = model_name or os.environ.get("LLM_MODEL", "gpt-5.4-mini")
        llm = ChatOpenAI(model=model)

        # Convert 3-element lists to tuples for LLMGraphTransformer
        parsed_rels = None
        if allowed_relationships:
            parsed_rels = []
            for r in allowed_relationships:
                if isinstance(r, list) and len(r) == 3:
                    parsed_rels.append(tuple(r))
                else:
                    parsed_rels.append(r)

        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=allowed_nodes or [],
            allowed_relationships=parsed_rels or [],
            node_properties=True,
        )

        # Build document metadata
        content_hash = md5(text.encode("utf-8")).hexdigest()
        doc_metadata = {"import_id": import_id, "content_hash": content_hash}
        if document_id:
            doc_metadata["id"] = document_id
        if source_uri:
            doc_metadata["source_uri"] = source_uri
        if chunk_of:
            doc_metadata["chunk_of"] = chunk_of
        if chunk_index is not None:
            doc_metadata["chunk_index"] = chunk_index
        if total_chunks is not None:
            doc_metadata["total_chunks"] = total_chunks
        docs = [Document(page_content=text, metadata=doc_metadata)]

        status = "success"
        error_message = None
        total_nodes = 0
        total_rels = 0
        node_types = set()
        rel_types = set()

        start_time = time.time()
        try:
            graph_docs = await transformer.aconvert_to_graph_documents(docs)
            duration_ms = int((time.time() - start_time) * 1000)

            # Stamp import_id on all extracted nodes and relationships
            for d in graph_docs:
                for n in d.nodes:
                    n.properties["import_id"] = import_id
                for r in d.relationships:
                    r.properties["import_id"] = import_id

            # Import into Neo4j
            await add_graph_documents(
                driver=neo4j_driver,
                graph_documents=graph_docs,
                database=database,
                include_source=True,
                baseEntityLabel=True,
            )

            # Build summary
            total_nodes = sum(len(d.nodes) for d in graph_docs)
            total_rels = sum(len(d.relationships) for d in graph_docs)
            for d in graph_docs:
                for n in d.nodes:
                    node_types.add(n.type)
                for r in d.relationships:
                    rel_types.add(r.type)

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            status = "failed"
            error_message = str(e)
            logger.error(f"Import failed: {e}")

        # Create ImportEvent metadata node
        await neo4j_driver.execute_query(
            "CREATE (e:ImportEvent {"
            "  id: $id,"
            "  created_at: datetime(),"
            "  status: $status,"
            "  error_message: $error_message,"
            "  model: $model,"
            "  duration_ms: $duration_ms,"
            "  text_length: $text_length,"
            "  text_preview: $text_preview,"
            "  content_hash: $content_hash,"
            "  document_id: $document_id,"
            "  source_uri: $source_uri,"
            "  chunk_index: $chunk_index,"
            "  total_chunks: $total_chunks,"
            "  chunk_of: $chunk_of,"
            "  parent_import_id: $parent_import_id,"
            "  caller: $caller,"
            "  allowed_nodes: $allowed_nodes,"
            "  allowed_relationships: $allowed_relationships,"
            "  nodes_created: $nodes_created,"
            "  relationships_created: $relationships_created,"
            "  node_types: $node_types,"
            "  relationship_types: $relationship_types"
            "})",
            parameters_={
                "id": import_id,
                "status": status,
                "error_message": error_message,
                "model": model,
                "duration_ms": duration_ms,
                "text_length": len(text),
                "text_preview": text[:200],
                "content_hash": content_hash,
                "document_id": document_id,
                "source_uri": source_uri,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_of": chunk_of,
                "parent_import_id": parent_import_id,
                "caller": caller,
                "allowed_nodes": allowed_nodes or [],
                "allowed_relationships": json.dumps(allowed_relationships) if allowed_relationships else "[]",
                "nodes_created": total_nodes,
                "relationships_created": total_rels,
                "node_types": sorted(node_types),
                "relationship_types": sorted(rel_types),
            },
            database_=database,
        )

        result = {
            "import_id": import_id,
            "status": status,
            "nodes_created": total_nodes,
            "relationships_created": total_rels,
            "node_types": sorted(node_types),
            "relationship_types": sorted(rel_types),
            "duration_ms": duration_ms,
        }
        if error_message:
            result["error_message"] = error_message

        return json.dumps(result, default=str)

    # ── Memory tools ────────────────────────────────────────────────

    @mcp.tool(name="memory_write")
    async def _memory_write(
        category: str = Field(
            ...,
            description="Memory category: 'tools' (per-tool learnings), 'user' (persona & preferences), or 'general' (interesting facts)",
        ),
        key: str = Field(
            ...,
            description="Identifier for this memory entry, e.g. tool name ('import_text_to_kg') or topic ('persona')",
        ),
        content: str = Field(
            ...,
            description="Full markdown content to store. When updating, provide the complete updated text.",
        ),
    ) -> str:
        """Write or update a markdown memory entry. Use this to persist learnings, user info, or interesting facts across conversations."""
        return await memory_write(neo4j_driver, database, category, key, content)

    @mcp.tool(name="memory_read")
    async def _memory_read(
        category: str = Field(
            ...,
            description="Memory category: 'tools', 'user', or 'general'",
        ),
        key: str = Field(
            ...,
            description="Identifier for the memory entry to read",
        ),
    ) -> str:
        """Read a markdown memory entry by category and key."""
        return await memory_read(neo4j_driver, database, category, key)

    @mcp.tool(name="memory_list")
    async def _memory_list(
        category: Optional[str] = Field(
            None,
            description="Optional category filter: 'tools', 'user', or 'general'. Omit to list all.",
        ),
    ) -> str:
        """List all stored memory entries, optionally filtered by category."""
        result = await memory_list(neo4j_driver, database, category)
        if not result:
            return "No memories stored yet."
        lines = []
        for cat, keys in result.items():
            lines.append(f"## {cat}")
            for k in keys:
                lines.append(f"- {k}")
        return "\n".join(lines)

    @mcp.tool(name="memory_delete")
    async def _memory_delete(
        category: str = Field(
            ...,
            description="Memory category: 'tools', 'user', or 'general'",
        ),
        key: str = Field(
            ...,
            description="Identifier for the memory entry to delete",
        ),
    ) -> str:
        """Delete a memory entry by category and key."""
        return await memory_delete(neo4j_driver, database, category, key)

    return mcp


async def main(
    db_url: str,
    username: str,
    password: str,
    database: str,
    transport: Literal["stdio", "sse", "http"] = "stdio",
) -> None:
    neo4j_driver = AsyncGraphDatabase.driver(db_url, auth=(username, password))

    try:
        await neo4j_driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {db_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise

    mcp = create_mcp_server(
        neo4j_driver,
        database,
        db_url=db_url,
        username=username,
        password=password,
    )

    try:
        match transport:
            case "stdio":
                await mcp.run_stdio_async()
            case "sse" | "http":
                await mcp.run_http_async(host="127.0.0.1", port=8000, transport=transport)
    finally:
        await neo4j_driver.close()