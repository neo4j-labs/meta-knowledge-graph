#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0"]
# ///
"""Hook: build Diffbot tool results back into the Neo4j graph.

Wired in .claude/settings.json as a PostToolUse hook matched on the
meta-knowledge-graph MCP Diffbot tools (``enhance_entity`` / ``search_news``).
Reads the hook payload from stdin, unwraps the Diffbot JSON the tool returned,
and merges it into the graph so external signal stays queryable next to the
internal account record instead of evaporating with the conversation.

Graph shape::

    (Account)-[:HAS_ENRICHMENT]->(DiffbotOrganization)
    (Account)-[:HAS_ENRICHMENT]->(DiffbotPerson)
    (DiffbotOrganization)-[:HAS_CEO]->(DiffbotPerson)
    (DiffbotPerson)-[:EMPLOYED_BY]->(DiffbotOrganization)
    (NewsArticle)-[:MENTIONS]->(Account)           # matched internal account
    (NewsArticle)-[:MENTIONS]->(DiffbotOrganization)
    (NewsArticle)-[:TAGGED]->(NewsTag)
    (DiffbotPerson|DiffbotOrganization|NewsArticle)-[:CAPTURED_IN]->(Session)

Only entities that arrive with a real Diffbot id are written back; references
without one (malformed employer strings, unresolved news tags) are dropped
rather than keyed on synthetic hashes. diffbot.com entity URIs are normalized
to one scheme so the enhance and news paths converge on the same node.

Accounts are matched on domain (enhance) and on lowercased account or employer
name against article tag labels / quoted DQL terms (news); payloads that match
no account are still stored.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import load_mkg_env, neo4j_config  # noqa: E402

ENHANCE_SUFFIX = "enhance_entity"
NEWS_SUFFIX = "search_news"
DQL_TERM_PATTERN = re.compile(r'(?:tags\.label|text):"([^"]+)"')
SAVED_OUTPUT_PATTERN = re.compile(r"saved to (.+?\.txt)\.?(?:\n|$)", re.IGNORECASE)

KIND_LABELS = {"organization": "DiffbotOrganization", "person": "DiffbotPerson"}

ENHANCE_CYPHER_TEMPLATE = """
UNWIND $rows AS row
MERGE (e:__LABEL__ {id: row.id})
ON CREATE SET e.first_seen_at = datetime($timestamp)
SET e += row.props,
    e.source = 'diffbot',
    e.last_enriched_at = datetime($timestamp)
WITH e, row
FOREACH (ceo IN CASE WHEN row.ceo IS NULL THEN [] ELSE [row.ceo] END |
    MERGE (p:DiffbotPerson {id: ceo.id})
    ON CREATE SET p.first_seen_at = datetime($timestamp)
    SET p.name = coalesce(ceo.name, p.name),
        p.source = 'diffbot',
        p.last_seen_at = datetime($timestamp)
    MERGE (e)-[cr:HAS_CEO]->(p)
    ON CREATE SET cr.created_at = datetime($timestamp)
)
FOREACH (employer IN row.employer_refs |
    MERGE (o:DiffbotOrganization {id: employer.id})
    ON CREATE SET o.first_seen_at = datetime($timestamp)
    SET o.name = coalesce(employer.name, o.name),
        o.source = 'diffbot',
        o.last_seen_at = datetime($timestamp)
    MERGE (e)-[er:EMPLOYED_BY]->(o)
    ON CREATE SET er.created_at = datetime($timestamp)
)
WITH e, row
CALL (e, row) {
    MATCH (a:Account)
    WHERE (row.domain IS NOT NULL AND a.domain = row.domain)
       OR toLower(a.name) IN row.name_keys
    MERGE (a)-[r:HAS_ENRICHMENT]->(e)
    SET r.updated_at = datetime($timestamp)
    RETURN count(a) AS linked
}
RETURN count(e) AS stored, sum(linked) AS links
"""


def _enhance_cypher(kind: str) -> str:
    return ENHANCE_CYPHER_TEMPLATE.replace("__LABEL__", KIND_LABELS[kind])

NEWS_CYPHER = """
UNWIND $rows AS row
MERGE (n:NewsArticle {url: row.url})
ON CREATE SET n.first_seen_at = datetime($timestamp)
SET n += row.props,
    n.source = 'diffbot',
    n.last_seen_at = datetime($timestamp)
WITH n, row
FOREACH (label IN row.tags |
    MERGE (t:NewsTag {label: label})
    MERGE (n)-[:TAGGED]->(t)
)
WITH n, row
FOREACH (company IN row.companies |
    MERGE (c:DiffbotOrganization {id: company.id})
    ON CREATE SET c.first_seen_at = datetime($timestamp)
    SET c += company.props,
        c.source = 'diffbot',
        c.last_seen_at = datetime($timestamp)
    MERGE (n)-[cr:MENTIONS]->(c)
    ON CREATE SET cr.created_at = datetime($timestamp)
)
WITH n, row
CALL (n, row) {
    MATCH (a:Account)
    WHERE toLower(a.name) IN row.account_keys
       OR toLower(coalesce(a.domain, '')) IN row.account_keys
    MERGE (n)-[r:MENTIONS]->(a)
    ON CREATE SET r.created_at = datetime($timestamp)
    WITH n, a
    OPTIONAL MATCH (existing:DiffbotOrganization)
    WHERE (a.domain IS NOT NULL AND existing.domain = a.domain)
       OR toLower(existing.name) = toLower(a.name)
    WITH n, a, collect(existing)[0] AS existing
    FOREACH (c IN CASE WHEN existing IS NULL THEN [] ELSE [existing] END |
        MERGE (n)-[cr:MENTIONS]->(c)
        ON CREATE SET cr.created_at = datetime($timestamp)
    )
    RETURN count(a) AS linked
}
RETURN count(n) AS stored, sum(linked + size(row.companies)) AS links
"""

SESSION_LINK_ENHANCE = """
MERGE (s:Session {session_id: $session_id})
ON CREATE SET s.created_at = datetime($timestamp)
WITH s
MATCH (e)
WHERE (e:DiffbotPerson OR e:DiffbotOrganization) AND e.id IN $keys
MERGE (e)-[:CAPTURED_IN]->(s)
"""

SESSION_LINK_NEWS = """
MERGE (s:Session {session_id: $session_id})
ON CREATE SET s.created_at = datetime($timestamp)
WITH s
MATCH (n:NewsArticle)
WHERE n.url IN $keys
MERGE (n)-[:CAPTURED_IN]->(s)
WITH s
MATCH (e:DiffbotOrganization)
WHERE e.id IN $company_ids
MERGE (e)-[:CAPTURED_IN]->(s)
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
        label = _label(item)
        if label:
            labels.append(label)
    return labels


def _label(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        label = value.get("name") or value.get("label")
        if isinstance(label, str) and label:
            return label
    return None


def _type_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: list[str] = []
        for key in ("type", "name", "label", "entityType"):
            item = value.get(key)
            if isinstance(item, str):
                texts.append(item)
        for key in ("types", "entityTypes"):
            texts.extend(_type_texts(value.get(key)))
        return texts
    if isinstance(value, list):
        texts = []
        for item in value:
            texts.extend(_type_texts(item))
        return texts
    return []


def _entity_kind(*values: Any) -> str | None:
    texts = [text.lower() for value in values for text in _type_texts(value)]
    if any("person" in text for text in texts):
        return "person"
    if any(
        token in text
        for text in texts
        for token in ("organization", "organisation", "company")
    ):
        return "organization"
    return None


DIFFBOT_URI_PATTERN = re.compile(r"^https?://diffbot\.com/entity/([\w-]+)/?$")


def _canonical_diffbot_id(*values: Any) -> str | None:
    """First usable Diffbot id, with diffbot.com entity URIs normalized to one scheme."""
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = value.strip()
        match = DIFFBOT_URI_PATTERN.match(candidate)
        return f"http://diffbot.com/entity/{match.group(1)}" if match else candidate
    return None


def _entity_ref(value: Any) -> dict[str, Any] | None:
    """Normalize a compacted person/org reference to {id, name}; no Diffbot id, no ref."""
    if not isinstance(value, dict):
        return None
    ref_id = _canonical_diffbot_id(value.get("diffbotUri"))
    if not ref_id:
        return None
    return _compact({"id": ref_id, "name": _label(value)})


def _employment_refs(value: Any) -> list[dict[str, Any]]:
    """Unique employer references; entries without a Diffbot id are dropped."""
    refs: dict[str, dict[str, Any]] = {}
    for employment in value if isinstance(value, list) else []:
        if not isinstance(employment, dict):
            continue
        ref = _entity_ref(employment.get("employer")) or _entity_ref(
            employment.get("organization")
        )
        if ref:
            refs.setdefault(ref["id"], ref)
    return list(refs.values())


def _news_companies(tags: Any) -> list[dict[str, Any]]:
    companies: dict[str, dict[str, Any]] = {}
    for tag in tags if isinstance(tags, list) else []:
        if not isinstance(tag, dict):
            continue
        tag_kind = _entity_kind(
            tag.get("type"),
            tag.get("types"),
            tag.get("entityType"),
            tag.get("entityTypes"),
        )
        if tag_kind != "organization":
            continue
        name = _label(tag)
        if not name:
            continue
        raw_id = _canonical_diffbot_id(tag.get("id"), tag.get("uri"), tag.get("diffbotUri"))
        if not raw_id:
            continue
        company = {
            "id": raw_id,
            "props": _compact(
                {
                    "name": name,
                    "diffbot_uri": tag.get("uri") if isinstance(tag.get("uri"), str) else None,
                }
            ),
        }
        companies[company["id"]] = company
    return list(companies.values())


def _location_summary(value: Any) -> str | None:
    for location in value if isinstance(value, list) else []:
        if not isinstance(location, dict):
            continue
        parts = [
            label
            for label in (
                _label(location.get("city")),
                _label(location.get("region")),
                _label(location.get("country")),
            )
            if label
        ]
        if parts:
            return ", ".join(parts)
    return None


def _compact(props: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in props.items() if v not in (None, "", [])}


def _text_prop(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _article_text_props(entity: dict[str, Any]) -> dict[str, Any]:
    text = _text_prop(entity.get("text"))
    if not text:
        return {}
    return {
        "text": text,
        "text_chars": len(text),
        "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
    }


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
        entity_id = _canonical_diffbot_id(entity.get("id"), entity.get("diffbotUri"))
        kind = _entity_kind(entity.get("type"), tool_input.get("entity_type"))
        if not entity_id or kind is None:
            # Without a real Diffbot id there is nothing stable to key the node
            # on, and without a kind we cannot label it person vs organization.
            continue
        revenue = entity.get("revenue") if isinstance(entity.get("revenue"), dict) else {}
        ceo_ref = _entity_ref(entity.get("ceo"))
        employer_refs = _employment_refs(entity.get("employments"))
        employer_hint = tool_input.get("employer")
        employer_names = [ref["name"] for ref in employer_refs if ref.get("name")]
        props = _compact(
            {
                "name": name,
                "domain": domain,
                "description": entity.get("description"),
                "summary": entity.get("summary"),
                "homepage_uri": entity.get("homepageUri"),
                "linkedin_uri": entity.get("linkedInUri"),
                "twitter_uri": entity.get("twitterUri"),
                "title": entity.get("title") or tool_input.get("title"),
                "email": entity.get("email") or tool_input.get("email"),
                "employers": sorted(set(employer_names)),
                "nb_employees": entity.get("nbEmployees"),
                "revenue_value": revenue.get("value"),
                "revenue_currency": revenue.get("currency"),
                "ceo": _label(entity.get("ceo")),
                "industries": _labels(entity.get("industries")),
                "categories": _labels(entity.get("categories")),
                "location": _location_summary(entity.get("locations")),
            }
        )
        name_keys = {
            candidate.strip().lower()
            for candidate in [
                name,
                input_name,
                employer_hint,
                *_labels(entity.get("allNames")),
                *employer_names,
            ]
            if isinstance(candidate, str) and candidate.strip()
        }
        rows.append(
            {
                "id": entity_id,
                "entity_kind": kind,
                "domain": domain,
                "name_keys": sorted(name_keys),
                "props": props,
                "ceo": ceo_ref,
                "employer_refs": employer_refs,
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
                "summary": _text_prop(entity.get("summary")),
                "site_name": entity.get("siteName"),
                "author": entity.get("author"),
                "date": _diffbot_date(entity.get("date")),
                "sentiment": entity.get("sentiment"),
                "publisher_country": entity.get("publisherCountry"),
                **_article_text_props(entity),
            }
        )
        rows.append(
            {
                "url": url,
                "tags": tags,
                "companies": _news_companies(entity.get("tags")),
                "account_keys": sorted({*(tag.lower() for tag in tags), *query_terms}),
                "props": props,
            }
        )
    return rows


def _ensure_constraints(tx) -> None:
    tx.run(
        "CREATE CONSTRAINT diffbot_person_id_unique IF NOT EXISTS "
        "FOR (e:DiffbotPerson) REQUIRE e.id IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT diffbot_organization_id_unique IF NOT EXISTS "
        "FOR (e:DiffbotOrganization) REQUIRE e.id IS UNIQUE"
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
        statements = [
            (_enhance_cypher(entity_kind), batch)
            for entity_kind in KIND_LABELS
            if (batch := [row for row in rows if row["entity_kind"] == entity_kind])
        ]
        link_cypher, key_field, noun = SESSION_LINK_ENHANCE, "id", "entities"
    else:
        rows = _news_rows(payload, tool_input)
        statements = [(NEWS_CYPHER, rows)]
        link_cypher, key_field, noun = SESSION_LINK_NEWS, "url", "articles"
    if not rows:
        return

    from neo4j import GraphDatabase

    timestamp = datetime.now(timezone.utc).isoformat()
    session_id = data.get("session_id")
    uri, user, password, database = neo4j_config()
    stored = links = 0
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as db:
            db.execute_write(_ensure_constraints)
            for statement, batch in statements:
                record = db.execute_write(_run_single, statement, rows=batch, timestamp=timestamp)
                if record:
                    stored += record["stored"] or 0
                    links += record["links"] or 0
            if session_id and session_id != "unknown":
                db.execute_write(
                    _run_single,
                    link_cypher,
                    session_id=session_id,
                    timestamp=timestamp,
                    keys=[row[key_field] for row in rows],
                    company_ids=sorted(
                        {
                            company["id"]
                            for row in rows
                            for company in row.get("companies", [])
                        }
                    ),
                )
    print(f"[ingest_diffbot] stored {stored} {noun}, {links} account link(s)")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)

    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        ingest(data)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[ingest_diffbot] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
