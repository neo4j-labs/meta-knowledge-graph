import json
import logging
import os
import tempfile
import time
import uuid
from typing import List, Literal, Optional, Union

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from neo4j import AsyncGraphDatabase, AsyncDriver, RoutingControl
from neo4j.exceptions import Neo4jError
from pydantic import Field

from metagraph_mcp.graph_import import add_graph_documents

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

    # Mount Neo4j Agent Memory MCP server (https://github.com/neo4j-labs/agent-memory)
    # if os.environ.get("OPENAI_API_KEY"):
    #     agent_memory_proxy = FastMCP.as_proxy(
    #         StdioTransport(
    #             command="uvx",
    #             args=["--from", "neo4j-agent-memory[mcp,openai]", "neo4j-agent-memory", "mcp", "serve"],
    #             env={
    #                 "NEO4J_URI": db_url,
    #                 "NEO4J_USERNAME": username,
    #                 "NEO4J_PASSWORD": password,
    #                 "NEO4J_DATABASE": database,
    #                 "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
    #             },
    #         )
    #     )
    #     mcp.mount(agent_memory_proxy)
    #     logger.info("Mounted Neo4j Agent Memory MCP proxy")
    # else:
    #     logger.info("Neo4j Agent Memory MCP proxy not mounted (OPENAI_API_KEY unset)")

    # Mount Neocarta MCP server when required env vars are present
    neocarta_required = ("GCP_PROJECT_ID", "BIGQUERY_DATASET_ID", "OPENAI_API_KEY")
    if all(os.environ.get(v) for v in neocarta_required):
        neocarta_env = {
            "NEO4J_URI": db_url,
            "NEO4J_USERNAME": username,
            "NEO4J_PASSWORD": password,
            "NEO4J_DATABASE": database,
            "GCP_PROJECT_ID": os.environ["GCP_PROJECT_ID"],
            "BIGQUERY_DATASET_ID": os.environ["BIGQUERY_DATASET_ID"],
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        }
        optional_vars = (
            "BIGQUERY_REGION",
            "EMBEDDING_MODEL",
            "EMBEDDING_DIMENSIONS",
            # GCP auth — forwarded so ADC / service-account creds reach the subprocess
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_QUOTA_PROJECT",
            "GOOGLE_CLOUD_PROJECT",
            "CLOUDSDK_CONFIG",
            "HOME",
            "PATH",
        )
        for optional in optional_vars:
            if os.environ.get(optional):
                neocarta_env[optional] = os.environ[optional]

        # Allow pasting the service account JSON inline instead of a file path.
        # Write it to a temp file and point GOOGLE_APPLICATION_CREDENTIALS at it.
        sa_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
        if sa_json and "GOOGLE_APPLICATION_CREDENTIALS" not in neocarta_env:
            try:
                json.loads(sa_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"GCP_SERVICE_ACCOUNT_JSON is not valid JSON: {e}") from e
            fd, sa_path = tempfile.mkstemp(prefix="neocarta-sa-", suffix=".json")
            try:
                os.write(fd, sa_json.encode("utf-8"))
            finally:
                os.close(fd)
            os.chmod(sa_path, 0o600)
            neocarta_env["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
            logger.info(f"Wrote GCP_SERVICE_ACCOUNT_JSON to {sa_path}")

        neocarta_proxy = FastMCP.as_proxy(
            StdioTransport(
                command="uvx",
                args=["--from", "neocarta[mcp]", "neocarta-mcp"],
                env=neocarta_env,
            )
        )
        mcp.mount(neocarta_proxy, prefix="neocarta")
        logger.info("Mounted Neocarta MCP proxy")
    else:
        missing = [v for v in neocarta_required if not os.environ.get(v)]
        logger.info(f"Neocarta MCP proxy not mounted (missing env vars: {missing})")

    # Mount BigQuery remote MCP server when URL is configured.
    # BIGQUERY_MCP_URL: streamable-http endpoint, e.g. https://bigquery.googleapis.com/mcp
    # BIGQUERY_MCP_AUTH: optional bearer token. If unset and the URL points at
    #   bigquery.googleapis.com, we auto-fetch a Google ADC token (refreshing on expiry).
    # BIGQUERY_MCP_HEADERS: optional JSON dict of extra headers.
    bigquery_mcp_url = os.environ.get("BIGQUERY_MCP_URL")
    if bigquery_mcp_url:
        bq_headers: dict[str, str] = {}
        raw_headers = os.environ.get("BIGQUERY_MCP_HEADERS")
        if raw_headers:
            try:
                bq_headers.update(json.loads(raw_headers))
            except json.JSONDecodeError as e:
                raise ValueError(f"BIGQUERY_MCP_HEADERS is not valid JSON: {e}") from e

        bq_auth: object | None = os.environ.get("BIGQUERY_MCP_AUTH") or None
        if bq_auth is None and "googleapis.com" in bigquery_mcp_url:
            import google.auth
            import google.auth.transport.requests
            import httpx

            adc_creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )

            class _GoogleADCAuth(httpx.Auth):
                def __init__(self, creds):
                    self._creds = creds
                    self._request = google.auth.transport.requests.Request()

                def auth_flow(self, request):
                    if not self._creds.valid:
                        self._creds.refresh(self._request)
                    request.headers["Authorization"] = f"Bearer {self._creds.token}"
                    yield request

            bq_auth = _GoogleADCAuth(adc_creds)

        bigquery_client = Client(
            StreamableHttpTransport(
                url=bigquery_mcp_url,
                headers=bq_headers or None,
                auth=bq_auth,
            )
        )
        bigquery_project_id = os.environ.get("GCP_PROJECT_ID") or ""

        @mcp.tool(name="bigquery_execute_query")
        async def bigquery_execute_query(
            query: str = Field(
                ...,
                description="Read-only BigQuery SQL (SELECT / WITH). The GCP project is configured server-side.",
            ),
        ) -> str:
            """Execute a read-only SQL query against the configured BigQuery project."""
            async with bigquery_client:
                result = await bigquery_client.call_tool(
                    "execute_sql_readonly",
                    {"projectId": bigquery_project_id, "query": query},
                )
            if result.content:
                text_parts = [
                    block.text for block in result.content if hasattr(block, "text")
                ]
                if text_parts:
                    return "\n".join(text_parts)
            return json.dumps(result.structured_content or {}, default=str)

        logger.info(
            f"Registered bigquery_execute_query tool -> {bigquery_mcp_url} "
            f"(project={bigquery_project_id or 'unset'})"
        )
    else:
        logger.info("BigQuery remote MCP proxy not mounted (BIGQUERY_MCP_URL unset)")

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
        doc_metadata = {"content_hash": content_hash}
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

            # Import into Neo4j (import_ids accumulated as list by the import queries)
            await add_graph_documents(
                driver=neo4j_driver,
                graph_documents=graph_docs,
                database=database,
                include_source=True,
                baseEntityLabel=True,
                import_id=import_id,
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
