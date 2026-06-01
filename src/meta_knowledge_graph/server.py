import json
import logging
import os
import re
import tempfile
import time
import uuid
from hashlib import sha1
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import httpx
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError
from pydantic import Field

from meta_knowledge_graph.graph_import import add_graph_documents


MAX_LEARNING_TEXT = 500
DIFFBOT_API_BASE_URL = "https://kg.diffbot.com/kg/v3"
DIFFBOT_TIMEOUT_SECONDS = 30.0
DIFFBOT_NEWS_FILTER = (
    "title pageUrl siteName date author sentiment tags.label publisherCountry"
)
DIFFBOT_ORGANIZATION_ENHANCE_FILTER = (
    "name allNames homepageUri linkedInUri twitterUri description summary "
    "industries categories nbEmployees revenue locations ceo"
)
DIFFBOT_PERSON_ENHANCE_FILTER = (
    "name allNames linkedInUri twitterUri description summary location "
    "employments skills"
)
DIFFBOT_PERSON_ONLY_ENHANCE_FIELDS = {"email", "employer", "title"}
DIFFBOT_ENHANCE_IDENTIFIER_FIELDS = {
    "id",
    "name",
    "url",
    "email",
    "phone",
    "location",
    "employer",
    "title",
}


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


def _json_error(error: str, **extra: Any) -> str:
    return json.dumps({"status": "error", "error": error, **extra}, default=str)


def _diffbot_payload_error_message(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    error = payload.get("error")
    message = payload.get("message")
    if isinstance(error, str) and error:
        return error
    if isinstance(message, str) and message:
        return message
    if error:
        return str(error)
    return None


def _diffbot_filter_parse_failed(response: httpx.Response) -> bool:
    if response.status_code != 422:
        return False

    try:
        payload: Any = response.json()
    except ValueError:
        payload = {}

    message = _diffbot_payload_error_message(payload) or response.text
    return "parse filter failed" in message.lower()


def _diffbot_token() -> Optional[str]:
    for env_var in ("DIFFBOT_TOKEN", "DIFFBOT_API_TOKEN", "DIFFBOT_API_KEY"):
        token = os.environ.get(env_var)
        if token and token.strip():
            return token.strip()
    return None


def _has_diffbot_token() -> bool:
    return _diffbot_token() is not None


def _compact_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value not in (None, "")}


def _bounded_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _build_diffbot_enhance_params(
    entity_type: str,
    values: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    normalized_type = (entity_type or "").strip()
    if normalized_type not in {"Organization", "Person"}:
        return None, "entity_type must be either 'Organization' or 'Person'"

    compact_values = _compact_params(values)
    identifiers = DIFFBOT_ENHANCE_IDENTIFIER_FIELDS & compact_values.keys()
    if not identifiers:
        return (
            None,
            "At least one identifier is required, such as name, url, id, email, or phone",
        )

    person_only = DIFFBOT_PERSON_ONLY_ENHANCE_FIELDS & compact_values.keys()
    if normalized_type == "Organization" and person_only:
        return (
            None,
            "These fields are only valid for Person enhance requests: "
            + ", ".join(sorted(person_only)),
        )

    params = {"type": normalized_type}
    params.update(compact_values)
    return params, None


def _diffbot_enhance_filter(entity_type: str) -> str:
    if entity_type == "Person":
        return DIFFBOT_PERSON_ENHANCE_FILTER
    return DIFFBOT_ORGANIZATION_ENHANCE_FILTER


def _diffbot_response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload: Any = response.json()
    except ValueError:
        payload = {"raw": response.text}

    if response.is_error:
        message = _diffbot_payload_error_message(payload)
        return {
            "status": "error",
            "status_code": response.status_code,
            "error": message or response.reason_phrase,
            "response": payload,
        }

    return payload if isinstance(payload, dict) else {"data": payload}


async def _diffbot_get_json(path: str, params: dict[str, Any]) -> str:
    token = _diffbot_token()
    if not token:
        return _json_error(
            "Diffbot token is required. Set DIFFBOT_TOKEN, DIFFBOT_API_TOKEN, "
            "or DIFFBOT_API_KEY."
        )

    request_params = {"token": token, **_compact_params(params)}
    try:
        async with httpx.AsyncClient(timeout=DIFFBOT_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f"{DIFFBOT_API_BASE_URL}/{path.lstrip('/')}",
                params=request_params,
                headers={"Accept": "application/json"},
            )
            if "filter" in request_params and _diffbot_filter_parse_failed(response):
                retry_params = {
                    key: value
                    for key, value in request_params.items()
                    if key != "filter"
                }
                response = await client.get(
                    f"{DIFFBOT_API_BASE_URL}/{path.lstrip('/')}",
                    params=retry_params,
                    headers={"Accept": "application/json"},
                )
    except httpx.HTTPError as e:
        return _json_error(f"Diffbot request failed: {e}")

    return json.dumps(_diffbot_response_payload(response), default=str)


logger = logging.getLogger(__name__)


def create_mcp_server(
    neo4j_driver: AsyncDriver,
    database: str = "neo4j",
    db_url: str = "",
    username: str = "",
    password: str = "",
) -> FastMCP:
    mcp = FastMCP("meta-knowledge-graph")

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

    if _has_diffbot_token():

        @mcp.tool(name="search_news")
        async def search_news(
            dql: str = Field(
                ...,
                description=(
                    "Diffbot DQL Article query for sales news monitoring. Examples: "
                    'company news from the last 3 days = `type:Article tags.label:"Acme Corp" '
                    'date<3d language:"en" sortBy:date`; topic news from the last 3 days = '
                    '`type:Article tags.label:"supply chain disruption" date<3d '
                    'language:"en" sortBy:date`.'
                ),
            ),
            max_results: int = Field(
                10,
                description="Maximum news articles to return, from 1 to 25.",
            ),
        ) -> str:
            """Search recent Diffbot news/articles with a concise sales-friendly payload."""
            clean_dql = (dql or "").strip()
            if not clean_dql:
                return _json_error("dql is required")
            if "type:article" not in clean_dql.lower():
                return _json_error("search_news requires a DQL query with type:Article")

            params = {
                "query": clean_dql,
                "size": _bounded_int(max_results, 1, 25),
                "format": "json",
                "cluster": "dedupe",
                "filter": DIFFBOT_NEWS_FILTER,
            }

            return await _diffbot_get_json("dql", params)

        @mcp.tool(name="enhance_entity")
        async def enhance_entity(
            entity_type: Literal["Organization", "Person"] = Field(
                ...,
                description="Entity type to enrich. Allowed values: Organization or Person.",
            ),
            name: Optional[str] = Field(
                None,
                description="Organization or person name to enhance.",
            ),
            url: Optional[str] = Field(
                None,
                description="Origin or homepage URL for the entity.",
            ),
            id: Optional[str] = Field(
                None,
                description="Diffbot ID of the entity to enhance.",
            ),
            location: Optional[str] = Field(
                None,
                description="Location hint for either an Organization or Person.",
            ),
            phone: Optional[str] = Field(
                None,
                description="Phone hint for either an Organization or Person.",
            ),
            email: Optional[str] = Field(
                None,
                description="Person email hint. Diffbot Enhance accepts email only for Person.",
            ),
            employer: Optional[str] = Field(
                None,
                description="Person employer hint.",
            ),
            title: Optional[str] = Field(
                None,
                description="Person title hint.",
            ),
            max_results: int = Field(
                1,
                description="Maximum enriched entity matches to return, from 1 to 5.",
            ),
        ) -> str:
            """Enrich a Person or Organization with concise sales-relevant fields."""
            params, error = _build_diffbot_enhance_params(
                entity_type,
                {
                    "id": id,
                    "name": name,
                    "url": url,
                    "location": location,
                    "phone": phone,
                    "email": email,
                    "employer": employer,
                    "title": title,
                },
            )
            if error:
                return _json_error(error)

            assert params is not None
            params["size"] = _bounded_int(max_results, 1, 5)
            params["filter"] = _diffbot_enhance_filter(entity_type)

            return await _diffbot_get_json("enhance", params)

        logger.info("Registered Diffbot search_news and enhance_entity tools")
    else:
        logger.info(
            "Diffbot tools not registered (DIFFBOT_TOKEN / DIFFBOT_API_TOKEN unset)"
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
