from __future__ import annotations

import os
import re
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any


MAX_LEARNING_TEXT = 500


@dataclass(frozen=True)
class ProjectRef:
    id: str
    name: str
    description: str | None = None
    status: str = "active"
    repo_root: str | None = None
    source: str = "auto"


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def neo4j_config() -> tuple[str, str, str, str]:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    return uri, user, password, database


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "default"


def resolve_project(payload: dict[str, Any], project_root: Path) -> ProjectRef | None:
    del payload

    folder_name = project_root.name if project_root.name else "default"
    project_id = slugify(folder_name)

    return ProjectRef(
        id=project_id,
        name=folder_name.replace("-", " ").replace("_", " ").title() or "Default",
        description=None,
        status="active",
        repo_root=str(project_root),
        source="folder",
    )


def project_props(project: ProjectRef) -> dict[str, Any]:
    props = {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "status": project.status,
        "repo_root": project.repo_root,
        "source": project.source,
    }
    return {k: v for k, v in props.items() if v is not None}


def ensure_project_schema(tx) -> None:
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Learning) REQUIRE l.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE")
    tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SystemPromptSuggestion) "
        "REQUIRE s.id IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (sp:SystemPrompt) "
        "REQUIRE sp.name IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (v:SystemPromptVersion) "
        "REQUIRE v.id IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:ProjectProcessing) "
        "REQUIRE p.id IS UNIQUE"
    )
    tx.run(
        "CREATE FULLTEXT INDEX project_learning_fulltext IF NOT EXISTS "
        "FOR (l:Learning) ON EACH [l.text, l.task_pattern, l.summary]"
    )
    tx.run(
        "CREATE FULLTEXT INDEX project_decision_fulltext IF NOT EXISTS "
        "FOR (d:Decision) ON EACH [d.text, d.rationale, d.task_pattern, d.summary]"
    )


def merge_project_and_session(
    tx,
    project: ProjectRef,
    session_id: str,
    timestamp: str,
) -> None:
    tx.run(
        """
        MERGE (p:Project {id: $project_id})
        ON CREATE SET p.created_at = $timestamp
        SET p += $project_props,
            p.updated_at = $timestamp,
            p.last_activity_at = $timestamp
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET s.created_at = $timestamp
        MERGE (p)-[:HAS_SESSION]->(s)
        """,
        project_id=project.id,
        project_props=project_props(project),
        session_id=session_id,
        timestamp=timestamp,
    )


def link_event_to_project(
    tx,
    project: ProjectRef,
    session_id: str,
    event_id: str,
    timestamp: str,
) -> None:
    del event_id  # Events are reachable via Project-[:HAS_SESSION]->Session-[:HAS_EVENT]->Event
    tx.run(
        """
        MERGE (p:Project {id: $project_id})
        ON CREATE SET p.created_at = $timestamp
        SET p += $project_props,
            p.updated_at = $timestamp,
            p.last_activity_at = $timestamp
        MATCH (s:Session {session_id: $session_id})
        MERGE (p)-[:HAS_SESSION]->(s)
        """,
        project_id=project.id,
        project_props=project_props(project),
        session_id=session_id,
        timestamp=timestamp,
    )


def learning_id(project_id: str, text: str) -> str:
    digest = sha1(f"{project_id}\n{text.strip()}".encode("utf-8")).hexdigest()[:16]
    return f"learning:{project_id}:{digest}"


def decision_id(project_id: str, text: str) -> str:
    digest = sha1(f"{project_id}\n{text.strip()}".encode("utf-8")).hexdigest()[:16]
    return f"decision:{project_id}:{digest}"


def system_prompt_suggestion_id(project_id: str, instruction: str) -> str:
    digest = sha1(f"{project_id}\n{instruction.strip()}".encode("utf-8")).hexdigest()[:16]
    return f"system-prompt-suggestion:{project_id}:{digest}"


def truncate(value: str, limit: int = MAX_LEARNING_TEXT) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def fetch_project_learnings(
    session,
    project_id: str,
    query: str | None,
    statuses: list[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    statuses = statuses or ["approved", "candidate"]
    if query and query.strip():
        try:
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('project_learning_fulltext', $search_query)
                YIELD node, score
                MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(node)
                WHERE node.status IN $statuses
                RETURN node.id AS id,
                       node.text AS text,
                       node.status AS status,
                       node.confidence AS confidence,
                       node.task_pattern AS task_pattern,
                       score
                ORDER BY CASE node.status WHEN 'approved' THEN 0 ELSE 1 END,
                         score DESC,
                         coalesce(node.confidence, 0.0) DESC
                LIMIT $limit
                """,
                project_id=project_id,
                search_query=query,
                statuses=statuses,
                limit=limit,
            )
            rows = [dict(record) for record in records]
            if rows:
                return rows
        except Exception:
            pass

    records = session.run(
        """
        MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(l:Learning)
        WHERE l.status IN $statuses
        RETURN l.id AS id,
               l.text AS text,
               l.status AS status,
               l.confidence AS confidence,
               l.task_pattern AS task_pattern,
               0.0 AS score
        ORDER BY CASE l.status WHEN 'approved' THEN 0 ELSE 1 END,
                 coalesce(l.last_used_at, l.updated_at, l.created_at) DESC
        LIMIT $limit
        """,
        project_id=project_id,
        statuses=statuses,
        limit=limit,
    )
    return [dict(record) for record in records]


def fetch_project_decisions(
    session,
    project_id: str,
    query: str | None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if query and query.strip():
        try:
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('project_decision_fulltext', $search_query)
                YIELD node, score
                MATCH (:Project {id: $project_id})-[:HAS_DECISION]->(node)
                RETURN node.id AS id,
                       node.text AS text,
                       node.rationale AS rationale,
                       node.confidence AS confidence,
                       node.task_pattern AS task_pattern,
                       score
                ORDER BY score DESC,
                         coalesce(node.confidence, 0.0) DESC
                LIMIT $limit
                """,
                project_id=project_id,
                search_query=query,
                limit=limit,
            )
            rows = [dict(record) for record in records]
            if rows:
                return rows
        except Exception:
            pass

    records = session.run(
        """
        MATCH (:Project {id: $project_id})-[:HAS_DECISION]->(d:Decision)
        RETURN d.id AS id,
               d.text AS text,
               d.rationale AS rationale,
               d.confidence AS confidence,
               d.task_pattern AS task_pattern,
               0.0 AS score
        ORDER BY coalesce(d.updated_at, d.created_at) DESC
        LIMIT $limit
        """,
        project_id=project_id,
        limit=limit,
    )
    return [dict(record) for record in records]


def mark_learnings_used(session, learning_ids: list[str]) -> None:
    if not learning_ids:
        return
    session.run(
        """
        MATCH (l:Learning)
        WHERE l.id IN $learning_ids
        SET l.last_used_at = datetime(),
            l.use_count = coalesce(l.use_count, 0) + 1
        """,
        learning_ids=learning_ids,
    )


def format_learning_context(
    project: ProjectRef,
    learnings: list[dict[str, Any]],
    decisions: list[dict[str, Any]] | None = None,
) -> str:
    decisions = decisions or []
    if not learnings and not decisions:
        return ""

    lines = [
        f"Project context for {project.name} ({project.id}):",
    ]
    if learnings:
        lines.extend(["", "Relevant project learnings:"])
        for learning in learnings:
            status = learning.get("status") or "candidate"
            confidence = learning.get("confidence")
            confidence_text = (
                f", confidence {float(confidence):.2f}" if confidence is not None else ""
            )
            lines.append(
                f"- [{status}{confidence_text}] "
                f"{truncate(str(learning.get('text') or ''), 240)}"
            )
    if decisions:
        lines.extend(["", "Relevant project decisions:"])
        for decision in decisions:
            confidence = decision.get("confidence")
            confidence_text = (
                f", confidence {float(confidence):.2f}" if confidence is not None else ""
            )
            lines.append(
                f"- [decision{confidence_text}] "
                f"{truncate(str(decision.get('text') or ''), 240)}"
            )
    lines.extend(
        [
            "",
            "Use approved learnings as scoped project memory. Treat candidate learnings as hints and decisions as context, not policy.",
        ]
    )
    return "\n".join(lines)
