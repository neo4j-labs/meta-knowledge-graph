#!/usr/bin/env python3
"""Hook: build Diffbot tool results back into the Neo4j graph.

Wired in .claude/settings.json as a PostToolUse hook matched on the
meta-knowledge-graph MCP Diffbot tools (``enhance_entity`` / ``search_news``).
Reads the hook payload from stdin, unwraps the Diffbot JSON the tool returned,
and merges it into the graph so external signal stays queryable next to the
internal account record instead of evaporating with the conversation.

Graph shape::

    (Account)-[:HAS_ENRICHMENT]->(DiffbotEntity)   # firmographics, enhance_entity
    (NewsArticle)-[:MENTIONS]->(Account)           # trigger events, search_news
    (NewsArticle)-[:TAGGED]->(NewsTag)
    (DiffbotEntity|NewsArticle)-[:CAPTURED_IN]->(Session)

Accounts are matched on domain (enhance) and on lowercased account name
against article tag labels / quoted DQL terms (news); payloads that match no
account are still stored.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import load_dotenv, neo4j_config  # noqa: E402

ENHANCE_SUFFIX = "enhance_entity"
NEWS_SUFFIX = "search_news"
DQL_TERM_PATTERN = re.compile(r'(?:tags\.label|text):"([^"]+)"')
SAVED_OUTPUT_PATTERN = re.compile(r"saved to (.+?\.txt)\.?(?:\n|$)", re.IGNORECASE)

ENHANCE_CYPHER = """
UNWIND $rows AS row
MERGE (e:DiffbotEntity {id: row.id})
ON CREATE SET e.first_seen_at = $timestamp
SET e += row.props,
    e.source = 'diffbot',
    e.last_enriched_at = $timestamp
WITH e, row
CALL (e, row) {
    MATCH (a:Account)
    WHERE (row.domain IS NOT NULL AND a.domain = row.domain)
       OR toLower(a.name) IN row.name_keys
    MERGE (a)-[r:HAS_ENRICHMENT]->(e)
    SET r.updated_at = $timestamp
    RETURN count(a) AS linked
}
RETURN count(e) AS stored, sum(linked) AS links
"""

NEWS_CYPHER = """
UNWIND $rows AS row
MERGE (n:NewsArticle {url: row.url})
ON CREATE SET n.first_seen_at = $timestamp
SET n += row.props,
    n.source = 'diffbot',
    n.last_seen_at = $timestamp
WITH n, row
FOREACH (label IN row.tags |
    MERGE (t:NewsTag {label: label})
    MERGE (n)-[:TAGGED]->(t)
)
WITH n, row
CALL (n, row) {
    MATCH (a:Account)
    WHERE toLower(a.name) IN row.account_keys
       OR toLower(coalesce(a.domain, '')) IN row.account_keys
    MERGE (n)-[r:MENTIONS]->(a)
    ON CREATE SET r.created_at = $timestamp
    RETURN count(a) AS linked
}
RETURN count(n) AS stored, sum(linked) AS links
"""

SESSION_LINK_ENHANCE = """
MERGE (s:Session {session_id: $session_id})
ON CREATE SET s.created_at = $timestamp
WITH s
MATCH (e:DiffbotEntity)
WHERE e.id IN $keys
MERGE (e)-[:CAPTURED_IN]->(s)
"""

SESSION_LINK_NEWS = """
MERGE (s:Session {session_id: $session_id})
ON CREATE SET s.created_at = $timestamp
WITH s
MATCH (n:NewsArticle)
WHERE n.url IN $keys
MERGE (n)-[:CAPTURED_IN]->(s)
"""


def _extract_payload(tool_response: Any) -> dict[str, Any] | None:
    """Unwrap the Diffbot JSON from however the harness shaped tool_response."""
    if isinstance(tool_response, str):
        stripped = tool_response.strip()
        if not stripped.startswith(("{", "[")):
            # Oversized results get file-redirected by the harness; the hook
            # receives the notice text with the saved path instead of the JSON.
            match = SAVED_OUTPUT_PATTERN.search(stripped)
            if match:
                return _extract_payload(Path(match.group(1)).read_text())
            return None
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) and "result" not in parsed else _extract_payload(parsed)
    if isinstance(tool_response, dict):
        # FastMCP structured content wraps a string tool return as {"result": "<json>"};
        # the hook payload may carry that dict itself JSON-encoded once more.
        if isinstance(tool_response.get("result"), str):
            return _extract_payload(tool_response["result"])
        content = tool_response.get("content")
        if isinstance(content, list):
            return _extract_payload(content)
        return tool_response
    if isinstance(tool_response, list):
        for block in tool_response:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                return json.loads(block["text"])
    return None


def _domain(url: Any) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    candidate = url.strip()
    if "//" not in candidate:
        candidate = f"https://{candidate}"
    host = urlparse(candidate).netloc.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _diffbot_date(value: Any) -> str | None:
    """Diffbot dates arrive as {"str": "d2026-...", "timestamp": ms} or a string."""
    if isinstance(value, dict):
        timestamp = value.get("timestamp")
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat()
        value = value.get("str")
    if isinstance(value, str):
        return value[1:] if value.startswith("d") else value
    return None


def _labels(value: Any) -> list[str]:
    """Normalize Diffbot list fields whose items are strings or {name|label} objects."""
    labels: list[str] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            labels.append(item)
        elif isinstance(item, dict):
            label = item.get("name") or item.get("label")
            if isinstance(label, str) and label:
                labels.append(label)
    return labels


def _location_summary(value: Any) -> str | None:
    for location in value if isinstance(value, list) else []:
        if not isinstance(location, dict):
            continue
        parts = [
            part["name"]
            for part in (location.get("city"), location.get("region"), location.get("country"))
            if isinstance(part, dict) and part.get("name")
        ]
        if parts:
            return ", ".join(parts)
    return None


def _compact(props: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in props.items() if v not in (None, "", [])}


def _enhance_rows(payload: dict[str, Any], tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    input_name = tool_input.get("name")
    input_domain = _domain(tool_input.get("url"))
    rows = []
    for item in payload.get("data") or []:
        entity = item.get("entity") if isinstance(item, dict) else None
        if not isinstance(entity, dict):
            continue
        name = entity.get("name")
        domain = _domain(entity.get("homepageUri")) or input_domain
        entity_id = entity.get("id") or entity.get("diffbotUri")
        if not entity_id:
            entity_id = "sha1:" + sha1(f"{name}|{domain}".encode()).hexdigest()[:16]
        revenue = entity.get("revenue") if isinstance(entity.get("revenue"), dict) else {}
        ceo = entity.get("ceo") if isinstance(entity.get("ceo"), dict) else {}
        props = _compact(
            {
                "name": name,
                "entity_type": entity.get("type") or tool_input.get("entity_type"),
                "domain": domain,
                "description": entity.get("description"),
                "summary": entity.get("summary"),
                "homepage_uri": entity.get("homepageUri"),
                "linkedin_uri": entity.get("linkedInUri"),
                "twitter_uri": entity.get("twitterUri"),
                "nb_employees": entity.get("nbEmployees"),
                "revenue_value": revenue.get("value"),
                "revenue_currency": revenue.get("currency"),
                "ceo": ceo.get("name"),
                "industries": _labels(entity.get("industries")),
                "categories": _labels(entity.get("categories")),
                "location": _location_summary(entity.get("locations")),
            }
        )
        name_keys = {
            candidate.strip().lower()
            for candidate in [name, input_name, *_labels(entity.get("allNames"))]
            if isinstance(candidate, str) and candidate.strip()
        }
        rows.append(
            {
                "id": entity_id,
                "domain": domain,
                "name_keys": sorted(name_keys),
                "props": props,
            }
        )
    return rows


def _news_rows(payload: dict[str, Any], tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    query_terms = [
        term.lower() for term in DQL_TERM_PATTERN.findall(str(tool_input.get("dql") or ""))
    ]
    rows = []
    for item in payload.get("data") or []:
        entity = item.get("entity") if isinstance(item, dict) else None
        if not isinstance(entity, dict) and isinstance(item, dict) and item.get("pageUrl"):
            entity = item
        if not isinstance(entity, dict):
            continue
        url = entity.get("pageUrl")
        if not isinstance(url, str) or not url:
            continue
        tags = _labels(entity.get("tags"))
        props = _compact(
            {
                "title": entity.get("title"),
                "site_name": entity.get("siteName"),
                "author": entity.get("author"),
                "date": _diffbot_date(entity.get("date")),
                "sentiment": entity.get("sentiment"),
                "publisher_country": entity.get("publisherCountry"),
            }
        )
        rows.append(
            {
                "url": url,
                "tags": tags,
                "account_keys": sorted({*(tag.lower() for tag in tags), *query_terms}),
                "props": props,
            }
        )
    return rows


def _ensure_constraints(tx) -> None:
    tx.run(
        "CREATE CONSTRAINT diffbot_entity_id_unique IF NOT EXISTS "
        "FOR (e:DiffbotEntity) REQUIRE e.id IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT news_article_url_unique IF NOT EXISTS "
        "FOR (n:NewsArticle) REQUIRE n.url IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT news_tag_label_unique IF NOT EXISTS "
        "FOR (t:NewsTag) REQUIRE t.label IS UNIQUE"
    )


def _run_single(tx, cypher: str, **params):
    return tx.run(cypher, **params).single()


def ingest(data: dict[str, Any]) -> None:
    tool_name = str(data.get("tool_name") or "")
    if tool_name.endswith(ENHANCE_SUFFIX):
        kind = "enhance"
    elif tool_name.endswith(NEWS_SUFFIX):
        kind = "news"
    else:
        return

    payload = _extract_payload(data.get("tool_response"))
    if not isinstance(payload, dict) or payload.get("status") == "error":
        return
    tool_input = data.get("tool_input") if isinstance(data.get("tool_input"), dict) else {}

    if kind == "enhance":
        rows = _enhance_rows(payload, tool_input)
        cypher, link_cypher, key_field, noun = (
            ENHANCE_CYPHER,
            SESSION_LINK_ENHANCE,
            "id",
            "entities",
        )
    else:
        rows = _news_rows(payload, tool_input)
        cypher, link_cypher, key_field, noun = (
            NEWS_CYPHER,
            SESSION_LINK_NEWS,
            "url",
            "articles",
        )
    if not rows:
        return

    from neo4j import GraphDatabase

    timestamp = datetime.now(timezone.utc).isoformat()
    session_id = data.get("session_id")
    uri, user, password, database = neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as db:
            db.execute_write(_ensure_constraints)
            record = db.execute_write(_run_single, cypher, rows=rows, timestamp=timestamp)
            if session_id and session_id != "unknown":
                db.execute_write(
                    _run_single,
                    link_cypher,
                    session_id=session_id,
                    timestamp=timestamp,
                    keys=[row[key_field] for row in rows],
                )
    stored = record["stored"] if record else 0
    links = record["links"] if record else 0
    print(f"[ingest_diffbot] stored {stored} {noun}, {links} account link(s)")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        ingest(data)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[ingest_diffbot] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
