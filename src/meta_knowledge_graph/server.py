import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, List, Literal, Optional

import httpx
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError
from pydantic import Field


MAX_LEARNING_TEXT = 500
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DIFFBOT_API_BASE_URL = "https://kg.diffbot.com/kg/v3"
DIFFBOT_TIMEOUT_SECONDS = 30.0
MAX_DIFFBOT_LOCATIONS = 3
MAX_DIFFBOT_LIST_ITEMS = 6
MAX_DIFFBOT_RESPONSE_CHARS = 20_000
# Nested Diffbot entity references carry diffbotUri/image/types baggage that
# dominated oversized enhance responses. Place refs collapse to their name;
# person/org refs keep name + Diffbot id so they can be written back to Neo4j.
DIFFBOT_PLACE_REF_KEYS = {"city", "region", "country"}
DIFFBOT_AGENT_REF_KEYS = {"ceo", "employer", "parentCompany"}
# Dropped in order, payload-wide, while a response still exceeds the char budget.
DIFFBOT_HEAVY_FIELD_DROP_ORDER = (
    "locations",
    "categories",
    "allNames",
    "description",
    "summary",
)
DIFFBOT_NEWS_FILTER = (
    "title pageUrl siteName date author sentiment tags.label publisherCountry "
    "summary text"
)
DIFFBOT_ORGANIZATION_ENHANCE_FILTER = (
    "name allNames diffbotUri homepageUri linkedInUri twitterUri description "
    "summary industries categories nbEmployees revenue locations ceo"
)
DIFFBOT_PERSON_ENHANCE_FILTER = (
    "name allNames diffbotUri linkedInUri twitterUri description summary "
    "location employments skills"
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


USER_LEARNING_NAMESPACE = "user"
LEARNING_SCOPES = ("project", "user")


def _normalize_scope(value: Optional[str]) -> str:
    scope = (value or "").strip().lower()
    return scope if scope in LEARNING_SCOPES else "project"


def _learning_namespace(project_id: str, scope: str) -> str:
    return USER_LEARNING_NAMESPACE if _normalize_scope(scope) == "user" else project_id


def _learning_id(namespace: str, text: str) -> str:
    digest = sha1(f"{namespace}\n{text.strip()}".encode("utf-8")).hexdigest()[:16]
    return f"learning:{namespace}:{digest}"


def _truncate(value: str, limit: int = MAX_LEARNING_TEXT) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    token = os.environ.get("DIFFBOT_TOKEN")
    if token and token.strip():
        return token.strip()
    return None


def _has_diffbot_token() -> bool:
    return _diffbot_token() is not None


def _llm_credentials_present(model: str) -> bool:
    """True when the environment holds the credentials litellm needs for ``model``.

    Lets credential gating follow the configured provider instead of assuming
    OpenAI, mirroring how the memory hooks gate their LLM calls."""
    # litellm auto-loads a .env on its first import in the default DEV mode,
    # which would mask a deliberately unset key; pin PRODUCTION so the check
    # reflects the process environment.
    os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
    try:
        import litellm
    except Exception:
        return False
    # validate_environment checks key *presence*; a present-but-empty var means
    # "no credentials", so hide blanks to match plain truthiness.
    blanked = {k: v for k, v in os.environ.items() if not v.strip()}
    for key in blanked:
        del os.environ[key]
    try:
        result = litellm.validate_environment(model=model)
    except Exception:
        return False
    finally:
        os.environ.update(blanked)
    return bool(result.get("keys_in_environment"))


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


def _entity_ref_name(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("name", value)
    return value


def _compact_diffbot_ref(value: Any) -> Any:
    """Keep name + Diffbot id for a person/organization reference."""
    if not isinstance(value, dict):
        return value
    compact = {
        "name": value.get("name"),
        "diffbotUri": value.get("diffbotUri") or value.get("targetDiffbotId"),
    }
    compact = {key: child for key, child in compact.items() if child is not None}
    return compact or value


def _compact_diffbot_location(location: Any) -> Any:
    if not isinstance(location, dict):
        return location
    compact = {
        "address": location.get("address"),
        "city": _entity_ref_name(location.get("city")),
        "region": _entity_ref_name(location.get("region")),
        "country": _entity_ref_name(location.get("country")),
        "isPrimary": location.get("isPrimary"),
    }
    return {key: value for key, value in compact.items() if value is not None}


def _compact_diffbot_payload(value: Any) -> Any:
    """Slim nested entity references and cap list fields.

    Applied to every response, so the no-filter retry path is covered too: a
    single unfiltered Organization can blow the MCP output limit on its
    locations alone.
    """
    if isinstance(value, list):
        return [_compact_diffbot_payload(item) for item in value]
    if not isinstance(value, dict):
        return value

    compacted: dict[str, Any] = {}
    for key, child in value.items():
        if key == "locations" and isinstance(child, list):
            primary_first = sorted(
                child,
                key=lambda loc: not (isinstance(loc, dict) and loc.get("isPrimary")),
            )
            compacted[key] = [
                _compact_diffbot_location(item)
                for item in primary_first[:MAX_DIFFBOT_LOCATIONS]
            ]
        elif key == "location" and isinstance(child, dict):
            compacted[key] = _compact_diffbot_location(child)
        elif key == "categories" and isinstance(child, list):
            compacted[key] = [
                _entity_ref_name(item) for item in child[:MAX_DIFFBOT_LIST_ITEMS]
            ]
        elif key == "allNames" and isinstance(child, list):
            compacted[key] = child[:MAX_DIFFBOT_LIST_ITEMS]
        elif key in DIFFBOT_PLACE_REF_KEYS:
            compacted[key] = _entity_ref_name(child)
        elif key in DIFFBOT_AGENT_REF_KEYS:
            compacted[key] = _compact_diffbot_ref(child)
        else:
            compacted[key] = _compact_diffbot_payload(child)
    return compacted


def _bounded_diffbot_json(payload: Any) -> str:
    """Serialize a Diffbot payload, shedding heavy fields while over budget."""
    text = json.dumps(payload, default=str)
    if len(text) <= MAX_DIFFBOT_RESPONSE_CHARS or not isinstance(payload, dict):
        return text

    data = payload.get("data")
    if not isinstance(data, list):
        return text

    entities = [
        item["entity"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("entity"), dict)
    ]
    dropped: list[str] = []
    for field in DIFFBOT_HEAVY_FIELD_DROP_ORDER:
        if not any(field in entity for entity in entities):
            continue
        for entity in entities:
            entity.pop(field, None)
        dropped.append(field)
        payload["truncated"] = (
            "dropped " + ", ".join(dropped) + " to fit the response budget"
        )
        text = json.dumps(payload, default=str)
        if len(text) <= MAX_DIFFBOT_RESPONSE_CHARS:
            return text

    if len(data) > 1:
        payload["data"] = data[:1]
        payload["truncated"] = (
            payload["truncated"] + "; kept only the first match"
            if dropped
            else "kept only the first match to fit the response budget"
        )
        text = json.dumps(payload, default=str)
    return text


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
            "response": _compact_diffbot_payload(payload),
        }

    payload = payload if isinstance(payload, dict) else {"data": payload}
    return _compact_diffbot_payload(payload)


async def _diffbot_get_json(path: str, params: dict[str, Any]) -> str:
    token = _diffbot_token()
    if not token:
        return _json_error(
            "Diffbot token is required. Set DIFFBOT_TOKEN."
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

    return _bounded_diffbot_json(_diffbot_response_payload(response))


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

    async def _execute_query(query: str, **params: Any) -> list[Any]:
        result = await neo4j_driver.execute_query(
            query,
            database_=database,
            **params,
        )
        return list(getattr(result, "records", result) or [])

    async def _execute_query_single(query: str, **params: Any):
        records = await _execute_query(query, **params)
        return records[0] if records else None

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

    # Mount Neocarta MCP server when the warehouse settings and the embedding
    # model's credentials are present. EMBEDDING_MODEL is a litellm model string,
    # so the provider — and thus the required key — follows the configured model
    # rather than assuming OpenAI, the same way the memory hooks pick their LLM.
    embedding_model = os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    neocarta_required = ("GCP_PROJECT_ID", "BIGQUERY_DATASET_ID")
    have_warehouse = all(os.environ.get(v) for v in neocarta_required)
    have_embedding_creds = _llm_credentials_present(embedding_model)
    if have_warehouse and have_embedding_creds:
        neocarta_env = {
            "NEO4J_URI": db_url,
            "NEO4J_USERNAME": username,
            "NEO4J_PASSWORD": password,
            "NEO4J_DATABASE": database,
            "GCP_PROJECT_ID": os.environ["GCP_PROJECT_ID"],
            "BIGQUERY_DATASET_ID": os.environ["BIGQUERY_DATASET_ID"],
        }
        # Forward whichever provider credentials are set so the embedding model
        # can authenticate regardless of provider (OpenAI, Cohere, Bedrock, ...).
        for key, value in os.environ.items():
            if value and (key.endswith("_API_KEY") or key.endswith("_AUTH_TOKEN")):
                neocarta_env[key] = value
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
        if not have_embedding_creds:
            missing.append(f"credentials for embedding model '{embedding_model}'")
        logger.info(f"Neocarta MCP proxy not mounted (missing: {missing})")

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
                    'language:"en" sortBy:date`. For company/news trigger searches, '
                    'try `tags.label:"Company Name"` first for precise entity-tagged '
                    'matches; if that returns no useful results, retry with '
                    '`text:"Company Name"` before concluding there is no recent signal.'
                ),
            ),
            max_results: int = Field(
                10,
                description="Maximum news articles to return, from 1 to 25.",
            ),
        ) -> str:
            """Search recent Diffbot news/articles; use tags.label first, then text fallback."""
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
            "Diffbot tools not registered (DIFFBOT_TOKEN unset)"
        )

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

        async with neo4j_driver.session(database=database):
            project_record = await _execute_query_single(
                "MATCH (p:Project {id: $project_id}) "
                "RETURN p.id AS id, p.name AS name, p.status AS status, "
                "p.last_activity_at AS last_activity_at",
                project_id=resolved_pid,
            )

            learnings: list[dict] = []
            if normalized_query:
                try:
                    records = await _execute_query(
                        """
                        CALL db.index.fulltext.queryNodes('project_learning_fulltext', $q)
                        YIELD node, score
                        MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(node)
                        WHERE node.status IN $statuses
                          AND coalesce(node.scope, 'project') = 'project'
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
                    learnings = [dict(r) for r in records]
                except Neo4jError:
                    learnings = []

            if not learnings:
                records = await _execute_query(
                    """
                    MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(l:Learning)
                    WHERE l.status IN $statuses
                      AND coalesce(l.scope, 'project') = 'project'
                    RETURN l.id AS id, l.text AS text, l.status AS status,
                           l.confidence AS confidence, l.task_pattern AS task_pattern,
                           0.0 AS score
                    ORDER BY CASE l.status WHEN 'approved' THEN 0 ELSE 1 END,
                             toString(coalesce(l.last_used_at, l.updated_at, l.created_at)) DESC
                    LIMIT $limit
                    """,
                    project_id=resolved_pid,
                    statuses=resolved_statuses,
                    limit=limit,
                )
                learnings = [dict(r) for r in records]

            user_learnings: list[dict] = []
            if normalized_query:
                try:
                    records = await _execute_query(
                        """
                        CALL db.index.fulltext.queryNodes('project_learning_fulltext', $q)
                        YIELD node, score
                        WHERE node.scope = 'user'
                          AND node.status IN $statuses
                          AND (node.consolidated_at IS NULL
                               OR toString(coalesce(node.updated_at, node.created_at)) > node.consolidated_at)
                        RETURN node.id AS id, node.text AS text, node.status AS status,
                               node.confidence AS confidence, node.task_pattern AS task_pattern,
                               score
                        ORDER BY CASE node.status WHEN 'approved' THEN 0 ELSE 1 END,
                                 score DESC, coalesce(node.confidence, 0.0) DESC
                        LIMIT $limit
                        """,
                        q=normalized_query,
                        statuses=resolved_statuses,
                        limit=limit,
                    )
                    user_learnings = [dict(r) for r in records]
                except Neo4jError:
                    user_learnings = []

            if not user_learnings:
                records = await _execute_query(
                    """
                    MATCH (l:Learning {scope: 'user'})
                    WHERE l.status IN $statuses
                      AND (l.consolidated_at IS NULL
                           OR toString(coalesce(l.updated_at, l.created_at)) > l.consolidated_at)
                    RETURN l.id AS id, l.text AS text, l.status AS status,
                           l.confidence AS confidence, l.task_pattern AS task_pattern,
                           0.0 AS score
                    ORDER BY CASE l.status WHEN 'approved' THEN 0 ELSE 1 END,
                             toString(coalesce(l.last_used_at, l.updated_at, l.created_at)) DESC
                    LIMIT $limit
                    """,
                    statuses=resolved_statuses,
                    limit=limit,
                )
                user_learnings = [dict(r) for r in records]

            decisions: list[dict] = []
            if normalized_query:
                try:
                    records = await _execute_query(
                        """
                        CALL db.index.fulltext.queryNodes('project_decision_fulltext', $q)
                        YIELD node, score
                        MATCH (:Project {id: $project_id})-[:HAS_DECISION]->(node)
                        WHERE coalesce(node.scope, 'project') = 'project'
                        RETURN node.id AS id, node.text AS text, node.rationale AS rationale,
                               node.confidence AS confidence, node.task_pattern AS task_pattern,
                               coalesce(node.scope, 'project') AS scope,
                               score
                        ORDER BY score DESC, coalesce(node.confidence, 0.0) DESC
                        LIMIT $limit
                        """,
                        project_id=resolved_pid,
                        q=normalized_query,
                        limit=limit,
                    )
                    decisions = [dict(r) for r in records]
                except Neo4jError:
                    decisions = []

            if not decisions:
                records = await _execute_query(
                    """
                    MATCH (:Project {id: $project_id})-[:HAS_DECISION]->(d:Decision)
                    WHERE coalesce(d.scope, 'project') = 'project'
                    RETURN d.id AS id, d.text AS text, d.rationale AS rationale,
                           d.confidence AS confidence, d.task_pattern AS task_pattern,
                           coalesce(d.scope, 'project') AS scope,
                           0.0 AS score
                    ORDER BY coalesce(d.updated_at, d.created_at) DESC
                    LIMIT $limit
                    """,
                    project_id=resolved_pid,
                    limit=limit,
                )
                decisions = [dict(r) for r in records]

            ids = [
                item["id"]
                for item in (*user_learnings, *learnings)
                if item.get("id")
            ]
            if ids:
                timestamp = _now_iso()
                await _execute_query(
                    """
                    MATCH (l:Learning) WHERE l.id IN $ids
                    SET l.last_used_at = $timestamp,
                        l.use_count = coalesce(l.use_count, 0) + 1
                    """,
                    ids=ids,
                    timestamp=timestamp,
                )

        payload = {
            "project": dict(project_record) if project_record else {"id": resolved_pid},
            "query": normalized_query or None,
            "statuses": resolved_statuses,
            "user_learnings": user_learnings,
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
        scope: str = Field(
            "project",
            description=(
                "'project' (default) for a fact about this project/environment, or "
                "'user' for a durable fact about the person that holds across "
                "projects (role, preferences, recurring constraints)."
            ),
        ),
        confidence: float = Field(
            0.6,
            description="Confidence 0.0-1.0. Existing higher confidence is preserved.",
        ),
        source: str = Field(
            "agent",
            description="Provenance tag for the writer (e.g. 'agent', 'user', '<tool>_llm').",
        ),
    ) -> str:
        """Persist a durable learning. Idempotent on (scope namespace, text)."""
        clean_text = _truncate((text or "").strip())
        if not clean_text:
            return json.dumps({"status": "error", "error": "text is required"})
        normalized_scope = _normalize_scope(scope)
        # The tool always writes candidates; promotion to 'approved' is owned by
        # the human review gate (and the seed scripts). This keeps user-scoped
        # facts flowing through the queue the prompt-consolidation service reads.
        normalized_status = "candidate"
        clamped_confidence = max(0.0, min(1.0, float(confidence)))
        resolved_pid = _resolve_project_id(project_id)
        row_id = _learning_id(
            _learning_namespace(resolved_pid, normalized_scope), clean_text
        )
        timestamp = _now_iso()

        async with neo4j_driver.session(database=database):
            for stmt in (
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Learning) REQUIRE l.id IS UNIQUE",
                "CREATE FULLTEXT INDEX project_learning_fulltext IF NOT EXISTS "
                "FOR (l:Learning) ON EACH [l.text, l.task_pattern, l.summary]",
            ):
                await _execute_query(stmt)

            record = await _execute_query_single(
                """
                MERGE (p:Project {id: $project_id})
                ON CREATE SET p.created_at = $timestamp,
                              p.name = $project_id,
                              p.source = 'agent'
                SET p.updated_at = $timestamp,
                    p.last_activity_at = $timestamp
                MERGE (l:Learning {id: $row_id})
                ON CREATE SET l.created_at = $timestamp,
                              l.use_count = 0,
                              l.support_count = 0
                SET l.text = $text,
                    l.summary = $text,
                    l.task_pattern = coalesce($task_pattern, l.task_pattern),
                    l.status = CASE
                        WHEN l.status = 'approved' THEN l.status
                        ELSE $status
                    END,
                    l.scope = $scope,
                    l.source = coalesce(l.source, $source),
                    l.last_source = $source,
                    l.project_id = $project_id,
                    l.updated_at = $timestamp,
                    l.support_count = coalesce(l.support_count, 0) + 1,
                    l.confidence = CASE
                        WHEN coalesce(l.confidence, 0.0) < $confidence THEN $confidence
                        ELSE l.confidence
                    END
                MERGE (p)-[:HAS_LEARNING]->(l)
                RETURN l.id AS id, l.text AS text, l.status AS status,
                       l.scope AS scope,
                       l.confidence AS confidence, l.task_pattern AS task_pattern,
                       l.support_count AS support_count,
                       CASE WHEN l.created_at = l.updated_at THEN 'created' ELSE 'updated' END AS action
                """,
                project_id=resolved_pid,
                row_id=row_id,
                text=clean_text,
                task_pattern=task_pattern,
                status=normalized_status,
                scope=normalized_scope,
                source=source,
                confidence=clamped_confidence,
                timestamp=timestamp,
            )

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
