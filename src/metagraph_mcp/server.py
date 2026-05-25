import json
import logging
import os
import re
import tempfile
import time
import uuid
from hashlib import sha1
from pathlib import Path
from typing import List, Literal, Optional, Union

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from neo4j import AsyncGraphDatabase, AsyncDriver, RoutingControl
from neo4j.exceptions import Neo4jError
from pydantic import Field

from metagraph_mcp.graph_import import add_graph_documents


MAX_LEARNING_TEXT = 500


def _slugify_project_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "default"


def _resolve_project_id(explicit: Optional[str]) -> str:
    if explicit and explicit.strip():
        return _slugify_project_id(explicit)
    return _slugify_project_id(Path(os.getcwd()).name or "default")


def _learning_id(project_id: str, text: str) -> str:
    digest = sha1(f"{project_id}\n{text.strip()}".encode("utf-8")).hexdigest()[:16]
    return f"learning:{project_id}:{digest}"


def _truncate(value: str, limit: int = MAX_LEARNING_TEXT) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."

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

    @mcp.tool(name="project_get_context")
    async def project_get_context(
        project_id: Optional[str] = Field(
            None,
            description="Project id (slug). Defaults to the MCP server's CWD folder name.",
        ),
        query: Optional[str] = Field(
            None,
            description="Optional free-text query for fulltext ranking of learnings and decisions.",
        ),
        statuses: Optional[List[str]] = Field(
            None,
            description="Learning statuses to include. Defaults to ['approved', 'candidate'].",
        ),
        limit: int = Field(5, description="Max learnings AND max decisions to return."),
    ) -> str:
        """Return scoped project learnings and decisions for the given project.

        Use this to self-bootstrap before answering questions about the active project,
        or to recall what the agent has learned across sessions. Mirrors the data the
        UserPromptSubmit hook injects, but on demand.
        """
        resolved_pid = _resolve_project_id(project_id)
        resolved_statuses = statuses or ["approved", "candidate"]
        normalized_query = (query or "").strip()

        async with neo4j_driver.session(database=database) as session:
            project_record = await (
                await session.run(
                    "MATCH (p:Project {id: $project_id}) "
                    "RETURN p.id AS id, p.name AS name, p.status AS status, "
                    "p.last_activity_at AS last_activity_at",
                    project_id=resolved_pid,
                )
            ).single()

            learnings: list[dict] = []
            if normalized_query:
                try:
                    records = await session.run(
                        """
                        CALL db.index.fulltext.queryNodes('project_learning_fulltext', $q)
                        YIELD node, score
                        MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(node)
                        WHERE node.status IN $statuses
                        RETURN node.id AS id, node.text AS text, node.status AS status,
                               node.confidence AS confidence, node.task_pattern AS task_pattern,
                               score
                        ORDER BY CASE node.status WHEN 'approved' THEN 0 ELSE 1 END,
                                 score DESC, coalesce(node.confidence, 0.0) DESC
                        LIMIT $limit
                        """,
                        project_id=resolved_pid,
                        q=normalized_query,
                        statuses=resolved_statuses,
                        limit=limit,
                    )
                    learnings = [dict(r) async for r in records]
                except Neo4jError:
                    learnings = []

            if not learnings:
                records = await session.run(
                    """
                    MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(l:Learning)
                    WHERE l.status IN $statuses
                    RETURN l.id AS id, l.text AS text, l.status AS status,
                           l.confidence AS confidence, l.task_pattern AS task_pattern,
                           0.0 AS score
                    ORDER BY CASE l.status WHEN 'approved' THEN 0 ELSE 1 END,
                             coalesce(l.last_used_at, l.updated_at, l.created_at) DESC
                    LIMIT $limit
                    """,
                    project_id=resolved_pid,
                    statuses=resolved_statuses,
                    limit=limit,
                )
                learnings = [dict(r) async for r in records]

            decisions: list[dict] = []
            if normalized_query:
                try:
                    records = await session.run(
                        """
                        CALL db.index.fulltext.queryNodes('project_decision_fulltext', $q)
                        YIELD node, score
                        MATCH (:Project {id: $project_id})-[:HAS_DECISION]->(node)
                        RETURN node.id AS id, node.text AS text, node.rationale AS rationale,
                               node.confidence AS confidence, node.task_pattern AS task_pattern,
                               score
                        ORDER BY score DESC, coalesce(node.confidence, 0.0) DESC
                        LIMIT $limit
                        """,
                        project_id=resolved_pid,
                        q=normalized_query,
                        limit=limit,
                    )
                    decisions = [dict(r) async for r in records]
                except Neo4jError:
                    decisions = []

            if not decisions:
                records = await session.run(
                    """
                    MATCH (:Project {id: $project_id})-[:HAS_DECISION]->(d:Decision)
                    RETURN d.id AS id, d.text AS text, d.rationale AS rationale,
                           d.confidence AS confidence, d.task_pattern AS task_pattern,
                           0.0 AS score
                    ORDER BY coalesce(d.updated_at, d.created_at) DESC
                    LIMIT $limit
                    """,
                    project_id=resolved_pid,
                    limit=limit,
                )
                decisions = [dict(r) async for r in records]

            if learnings:
                ids = [item["id"] for item in learnings if item.get("id")]
                if ids:
                    await session.run(
                        """
                        MATCH (l:Learning) WHERE l.id IN $ids
                        SET l.last_used_at = datetime(),
                            l.use_count = coalesce(l.use_count, 0) + 1
                        """,
                        ids=ids,
                    )

        payload = {
            "project": dict(project_record) if project_record else {"id": resolved_pid},
            "query": normalized_query or None,
            "statuses": resolved_statuses,
            "learnings": learnings,
            "decisions": decisions,
        }
        return json.dumps(payload, default=str)

    @mcp.tool(name="project_add_learning")
    async def project_add_learning(
        text: str = Field(..., description="Durable, reusable learning text (<=500 chars)."),
        project_id: Optional[str] = Field(
            None,
            description="Project id (slug). Defaults to the MCP server's CWD folder name.",
        ),
        task_pattern: Optional[str] = Field(
            None,
            description="Short reusable task pattern this learning applies to.",
        ),
        confidence: float = Field(
            0.6,
            description="Confidence 0.0-1.0. Existing higher confidence is preserved.",
        ),
        status: str = Field(
            "candidate",
            description="'candidate' (default) or 'approved'. Approved skips the review queue.",
        ),
        source: str = Field(
            "agent",
            description="Provenance tag for the writer (e.g. 'agent', 'user', '<tool>_llm').",
        ),
    ) -> str:
        """Persist a durable project learning. Idempotent on (project_id, text)."""
        clean_text = _truncate((text or "").strip())
        if not clean_text:
            return json.dumps({"status": "error", "error": "text is required"})
        normalized_status = status if status in {"candidate", "approved"} else "candidate"
        clamped_confidence = max(0.0, min(1.0, float(confidence)))
        resolved_pid = _resolve_project_id(project_id)
        row_id = _learning_id(resolved_pid, clean_text)

        async with neo4j_driver.session(database=database) as session:
            for stmt in (
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Learning) REQUIRE l.id IS UNIQUE",
                "CREATE FULLTEXT INDEX project_learning_fulltext IF NOT EXISTS "
                "FOR (l:Learning) ON EACH [l.text, l.task_pattern, l.summary]",
            ):
                await session.run(stmt)

            record = await (
                await session.run(
                    """
                    MERGE (p:Project {id: $project_id})
                    ON CREATE SET p.created_at = datetime(),
                                  p.name = $project_id,
                                  p.source = 'agent'
                    SET p.updated_at = datetime(),
                        p.last_activity_at = datetime()
                    MERGE (l:Learning {id: $row_id})
                    ON CREATE SET l.created_at = datetime(),
                                  l.use_count = 0,
                                  l.support_count = 0
                    SET l.text = $text,
                        l.summary = $text,
                        l.task_pattern = coalesce($task_pattern, l.task_pattern),
                        l.status = CASE
                            WHEN l.status = 'approved' THEN l.status
                            ELSE $status
                        END,
                        l.scope = coalesce(l.scope, 'project'),
                        l.source = coalesce(l.source, $source),
                        l.last_source = $source,
                        l.project_id = $project_id,
                        l.updated_at = datetime(),
                        l.support_count = coalesce(l.support_count, 0) + 1,
                        l.confidence = CASE
                            WHEN coalesce(l.confidence, 0.0) < $confidence THEN $confidence
                            ELSE l.confidence
                        END
                    MERGE (p)-[:HAS_LEARNING]->(l)
                    RETURN l.id AS id, l.text AS text, l.status AS status,
                           l.confidence AS confidence, l.task_pattern AS task_pattern,
                           l.support_count AS support_count,
                           CASE WHEN l.created_at = l.updated_at THEN 'created' ELSE 'updated' END AS action
                    """,
                    project_id=resolved_pid,
                    row_id=row_id,
                    text=clean_text,
                    task_pattern=task_pattern,
                    status=normalized_status,
                    source=source,
                    confidence=clamped_confidence,
                )
            ).single()

        return json.dumps(dict(record) if record else {}, default=str)

    @mcp.tool(name="system_prompt_list")
    async def system_prompt_list(
        prompt_name: Optional[str] = Field(
            None,
            description="Filter both prompts and suggestions to this prompt_name (e.g. 'default'). None returns all.",
        ),
        project_id: Optional[str] = Field(
            None,
            description="Filter suggestions to this project_id. None returns suggestions from all projects.",
        ),
        statuses: Optional[List[str]] = Field(
            None,
            description="Suggestion statuses to include. Defaults to ['candidate'].",
        ),
    ) -> str:
        """List live :SystemPrompt nodes and :SystemPromptSuggestion candidates for review."""
        resolved_statuses = statuses or ["candidate"]
        async with neo4j_driver.session(database=database) as session:
            prompt_records = await session.run(
                """
                MATCH (p:SystemPrompt)
                WHERE $prompt_name IS NULL OR p.name = $prompt_name
                RETURN p.name AS name,
                       p.content AS content,
                       size(coalesce(p.content, '')) AS char_count,
                       p.updated_at AS updated_at
                ORDER BY p.name
                """,
                prompt_name=prompt_name,
            )
            prompts = [dict(r) async for r in prompt_records]

            suggestion_records = await session.run(
                """
                MATCH (s:SystemPromptSuggestion)
                WHERE s.status IN $statuses
                  AND ($prompt_name IS NULL OR s.prompt_name = $prompt_name)
                  AND ($project_id IS NULL OR s.project_id = $project_id)
                RETURN s.id AS id,
                       s.prompt_name AS prompt_name,
                       s.project_id AS project_id,
                       s.status AS status,
                       s.instruction AS instruction,
                       s.rationale AS rationale,
                       s.confidence AS confidence,
                       s.support_count AS support_count,
                       s.source AS source,
                       s.updated_at AS updated_at
                ORDER BY coalesce(s.confidence, 0.0) DESC,
                         coalesce(s.support_count, 0) DESC,
                         s.updated_at DESC
                """,
                statuses=resolved_statuses,
                prompt_name=prompt_name,
                project_id=project_id,
            )
            suggestions = [dict(r) async for r in suggestion_records]

        return json.dumps({"prompts": prompts, "suggestions": suggestions}, default=str)

    @mcp.tool(name="system_prompt_replace")
    async def system_prompt_replace(
        prompt_name: str = Field(..., description="Name of the :SystemPrompt to write (e.g. 'default')."),
        content: str = Field(..., description="Full prompt content. Replaces whatever was there."),
    ) -> str:
        """Replace the content of a :SystemPrompt node. Creates the node if it doesn't exist.

        Also flips every :SystemPromptSuggestion with the same prompt_name from
        'candidate' to 'applied' and links it to the prompt, so the listing query
        stops surfacing them.
        """
        if not (prompt_name or "").strip():
            return json.dumps({"status": "error", "error": "prompt_name is required"})
        if not (content or "").strip():
            return json.dumps({"status": "error", "error": "content is required"})

        async with neo4j_driver.session(database=database) as session:
            prompt_record = await (
                await session.run(
                    """
                    MERGE (p:SystemPrompt {name: $name})
                    ON CREATE SET p.created_at = datetime()
                    SET p.content = $content,
                        p.updated_at = datetime()
                    RETURN p.name AS name,
                           size(p.content) AS char_count,
                           CASE WHEN p.created_at = p.updated_at THEN 'created' ELSE 'updated' END AS action
                    """,
                    name=prompt_name,
                    content=content,
                )
            ).single()

            applied_records = await session.run(
                """
                MATCH (p:SystemPrompt {name: $name})
                MATCH (s:SystemPromptSuggestion {prompt_name: $name, status: 'candidate'})
                SET s.status = 'applied',
                    s.applied_at = datetime()
                MERGE (s)-[:APPLIED_TO]->(p)
                RETURN s.id AS id
                """,
                name=prompt_name,
            )
            applied_ids = [r["id"] async for r in applied_records]

        result = dict(prompt_record) if prompt_record else {}
        result["applied_suggestion_ids"] = applied_ids
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
