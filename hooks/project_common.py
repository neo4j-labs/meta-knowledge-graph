from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any


MAX_LEARNING_TEXT = 500
DEFAULT_LLM_MODEL = "gpt-5.4-mini"
# How far back an injection and its hook SessionEvent may be apart and still be
# considered the same hook firing. Inject and log hooks run in parallel, so the
# INJECTED_AT link is attempted from both sides within this window.
INJECTION_EVENT_WINDOW_SECONDS = 120


def injection_window_start() -> str:
    return (
        datetime.now(timezone.utc) - timedelta(seconds=INJECTION_EVENT_WINDOW_SECONDS)
    ).isoformat()


def llm_model() -> str:
    """Single model knob for every LLM call made by the hooks.

    The value is a litellm model string, so any provider works: ``gpt-5.4-mini``
    routes to OpenAI, ``anthropic/claude-...`` to Anthropic, etc."""
    return os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL


def llm_ready() -> bool:
    """True when the environment holds the credentials litellm needs for the
    configured model, so hooks gate on whatever provider is in use rather than
    assuming OpenAI."""
    # litellm auto-loads a .env on its first import in the default DEV mode,
    # which would silently repopulate a key the caller deliberately unset; pin
    # PRODUCTION so the gate reflects the environment the hooks already loaded.
    os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
    try:
        import litellm
    except Exception:
        return False
    # litellm.validate_environment checks key *presence*, but a present-but-empty
    # var (e.g. ``OPENAI_API_KEY=``) means "no credentials" — hide those so the
    # gate matches plain truthiness for whichever provider's key is required.
    blanked = {k: v for k, v in os.environ.items() if not v.strip()}
    for key in blanked:
        del os.environ[key]
    try:
        result = litellm.validate_environment(model=llm_model())
    except Exception:
        return False
    finally:
        os.environ.update(blanked)
    return bool(result.get("keys_in_environment"))


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
    user = os.getenv("NEO4J_USERNAME", "neo4j")
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
        "CREATE CONSTRAINT IF NOT EXISTS FOR (sp:SystemPrompt) "
        "REQUIRE sp.name IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:MemoryExtractionPrompt) "
        "REQUIRE p.name IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:ProjectProcessing) "
        "REQUIRE p.id IS UNIQUE"
    )
    tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (v:SystemPromptVersion) "
        "REQUIRE v.id IS UNIQUE"
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
    del event_id  # Events are reachable via Project-[:HAS_SESSION]->Session-[:HAS_EVENT]->SessionEvent
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


# User-scoped learnings live above any single project, so they are keyed on this
# fixed namespace instead of a project id. The same durable fact about the person
# then collapses to one node no matter which project surfaced it.
USER_LEARNING_NAMESPACE = "user"
USER_DECISION_NAMESPACE = "user"
LEARNING_SCOPES = ("project", "user")


def normalize_scope(value: object) -> str:
    scope = str(value or "").strip().lower()
    return scope if scope in LEARNING_SCOPES else "project"


def learning_namespace(project_id: str, scope: str) -> str:
    return USER_LEARNING_NAMESPACE if normalize_scope(scope) == "user" else project_id


def decision_namespace(project_id: str, scope: str) -> str:
    return USER_DECISION_NAMESPACE if normalize_scope(scope) == "user" else project_id


def learning_id(namespace: str, text: str) -> str:
    digest = sha1(f"{namespace}\n{text.strip()}".encode("utf-8")).hexdigest()[:16]
    return f"learning:{namespace}:{digest}"


def decision_id(namespace: str, text: str) -> str:
    digest = sha1(f"{namespace}\n{text.strip()}".encode("utf-8")).hexdigest()[:16]
    return f"decision:{namespace}:{digest}"


PROMPT_LABELS = ("SystemPrompt", "MemoryExtractionPrompt")


def upsert_prompt_node(tx, label: str, name: str, content: str, now: str) -> dict[str, Any]:
    """MERGE a frozen prompt node (``SystemPrompt`` / ``MemoryExtractionPrompt``).

    The prompts no longer rewrite themselves at runtime, so this only sets the
    content and bumps a version counter when it actually changes; it keeps no
    version-history snapshot. The seed scripts use this for plain (re)seeds; the
    system-prompt consolidation service instead goes through
    ``snapshot_and_update_system_prompt`` so it can preserve history. Returns the
    action taken and the resulting version.
    """
    if label not in PROMPT_LABELS:
        raise ValueError(f"unknown prompt label: {label!r}")
    record = tx.run(
        f"""
        MERGE (p:{label} {{name: $name}})
        ON CREATE SET p.created_at = datetime($now), p.version = 1
        WITH p, p.content AS old_content
        SET p.content = $content,
            p.updated_at = datetime($now),
            p.version = CASE
                WHEN old_content IS NULL OR old_content = $content
                    THEN coalesce(p.version, 1)
                ELSE coalesce(p.version, 1) + 1
            END
        RETURN
            CASE
                WHEN old_content IS NULL THEN 'created'
                WHEN old_content = $content THEN 'unchanged'
                ELSE 'updated'
            END AS action,
            p.version AS version
        """,
        name=name,
        content=content,
        now=now,
    ).single()
    if not record:
        return {"action": "created", "version": 1}
    return {"action": str(record["action"]), "version": int(record["version"])}


# --- System-prompt consolidation -------------------------------------------
#
# A rate-limited service (hooks/consolidate_system_prompt.py) folds durable
# user-profile facts into the persisted (:SystemPrompt) when enough of them have
# piled up unreviewed. "In need of review" means a user-scoped :Learning still
# sitting in the candidate queue that has not yet been folded into the prompt.
# Default threshold is "more than 5"; the cooldown keeps it from re-firing on
# every Stop/SessionEnd.
USER_PROFILE_REVIEW_THRESHOLD = 5
PROMPT_CONSOLIDATION_INTERVAL_HOURS = 24.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def consolidation_threshold() -> int:
    return int(_env_float("MKG_PROMPT_CONSOLIDATION_THRESHOLD", USER_PROFILE_REVIEW_THRESHOLD))


def consolidation_interval_hours() -> float:
    return _env_float(
        "MKG_PROMPT_CONSOLIDATION_INTERVAL_HOURS", PROMPT_CONSOLIDATION_INTERVAL_HOURS
    )


def count_user_profile_memories_pending(session) -> int:
    """Count user-scoped candidate learnings still awaiting consolidation.

    A learning counts as pending when it has never been folded into the prompt
    (``consolidated_at`` unset) or has been edited since it last was, so a fact
    that changes after consolidation re-enters the review backlog.
    """
    record = session.run(
        """
        MATCH (l:Learning {scope: 'user', status: 'candidate'})
        WHERE l.consolidated_at IS NULL
           OR coalesce(l.updated_at, l.created_at) > l.consolidated_at
        RETURN count(l) AS pending
        """
    ).single()
    return int(record["pending"]) if record else 0


def fetch_user_profile_memories_pending(session, limit: int = 40) -> list[dict[str, Any]]:
    """Fetch the user-scoped candidate learnings the consolidation should fold in."""
    records = session.run(
        """
        MATCH (l:Learning {scope: 'user', status: 'candidate'})
        WHERE l.consolidated_at IS NULL
           OR coalesce(l.updated_at, l.created_at) > l.consolidated_at
        RETURN l.id AS id,
               l.text AS text,
               l.confidence AS confidence,
               l.task_pattern AS task_pattern,
               coalesce(l.updated_at, l.created_at) AS updated_at
        ORDER BY coalesce(l.confidence, 0.0) DESC,
                 coalesce(l.updated_at, l.created_at) DESC
        LIMIT $limit
        """,
        limit=limit,
    )
    return [dict(record) for record in records]


def read_system_prompt_state(session, name: str) -> dict[str, Any]:
    """Read the active prompt's content, version, and last consolidation time.

    ``last_consolidated_at`` is stored as an ISO string (unlike the node's
    datetime-typed created_at/updated_at) so the rate-limit math can stay in
    Python on the hook side."""
    record = session.run(
        """
        MATCH (sp:SystemPrompt {name: $name})
        RETURN sp.content AS content,
               coalesce(sp.version, 1) AS version,
               sp.last_consolidated_at AS last_consolidated_at
        """,
        name=name,
    ).single()
    if not record:
        return {"content": None, "version": 0, "last_consolidated_at": None}
    return {
        "content": record["content"],
        "version": int(record["version"]),
        "last_consolidated_at": record["last_consolidated_at"],
    }


def snapshot_and_update_system_prompt(
    tx,
    name: str,
    new_content: str,
    folded_learning_ids: list[str],
    model: str,
    session_id: str | None,
    now: str,
) -> dict[str, Any]:
    """Archive the outgoing prompt as a ``:SystemPromptVersion`` and write the new one.

    Unlike ``upsert_prompt_node`` (which only bumps a counter), this keeps a
    full history snapshot: every superseded prompt is preserved as its own
    ``:SystemPromptVersion`` node, and the new active content is mirrored onto a
    fresh version node flagged ``is_current``. The folded learnings are stamped
    ``consolidated_at`` so they drop out of the review backlog; their status is
    left untouched, so the human promotion gate still owns ``candidate ->
    approved``.
    """
    record = tx.run(
        """
        MERGE (sp:SystemPrompt {name: $name})
        ON CREATE SET sp.created_at = datetime($now), sp.version = 0
        WITH sp, sp.content AS old_content, coalesce(sp.version, 0) AS old_version
        // Archive the outgoing version as history (skip when there was no content).
        FOREACH (_ IN CASE WHEN old_content IS NULL THEN [] ELSE [1] END |
            MERGE (ov:SystemPromptVersion {id: $name + ':v' + toString(old_version)})
            ON CREATE SET ov.name = $name,
                          ov.version = old_version,
                          ov.content = old_content,
                          ov.source = coalesce(sp.last_source, 'seed'),
                          ov.created_at = coalesce(sp.updated_at, sp.created_at, datetime($now))
            SET ov.is_current = false,
                ov.archived_at = datetime($now)
            MERGE (sp)-[:HAS_VERSION]->(ov)
        )
        WITH sp, old_version
        SET sp.content = $new_content,
            sp.version = old_version + 1,
            sp.updated_at = datetime($now),
            sp.last_consolidated_at = $now,
            sp.last_source = 'consolidation',
            sp.last_consolidation_model = $model
        MERGE (nv:SystemPromptVersion {id: $name + ':v' + toString(old_version + 1)})
        ON CREATE SET nv.created_at = datetime($now)
        SET nv.name = $name,
            nv.version = old_version + 1,
            nv.content = $new_content,
            nv.source = 'consolidation',
            nv.model = $model,
            nv.session_id = $session_id,
            nv.folded_learning_count = size($folded_ids),
            nv.supersedes_version = old_version,
            nv.is_current = true
        MERGE (sp)-[:HAS_VERSION]->(nv)
        RETURN old_version AS old_version, old_version + 1 AS new_version
        """,
        name=name,
        new_content=new_content,
        model=model,
        session_id=session_id,
        folded_ids=folded_learning_ids,
        now=now,
    ).single()

    old_version = int(record["old_version"]) if record else 0
    new_version = int(record["new_version"]) if record else 1

    if folded_learning_ids:
        tx.run(
            """
            MATCH (sp:SystemPrompt {name: $name})
            MATCH (nv:SystemPromptVersion {id: $name + ':v' + toString($new_version)})
            UNWIND $folded_ids AS lid
            MATCH (l:Learning {id: lid})
            SET l.consolidated_at = $now,
                l.consolidated_prompt_version = $new_version
            MERGE (nv)-[:FOLDED_LEARNING]->(l)
            MERGE (sp)-[:CONSOLIDATED]->(l)
            """,
            name=name,
            new_version=new_version,
            folded_ids=folded_learning_ids,
            now=now,
        )

    return {"old_version": old_version, "new_version": new_version}


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
    exclude_session_id: str | None = None,
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
                  AND coalesce(node.scope, 'project') = 'project'
                  AND ($session_id IS NULL OR (
                       NOT (node)-[:INJECTED_IN]->(:Session {session_id: $session_id})
                       AND NOT (node)-[:FROM_SESSION]->(:Session {session_id: $session_id})))
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
                session_id=exclude_session_id,
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
          AND coalesce(l.scope, 'project') = 'project'
          AND ($session_id IS NULL OR (
               NOT (l)-[:INJECTED_IN]->(:Session {session_id: $session_id})
               AND NOT (l)-[:FROM_SESSION]->(:Session {session_id: $session_id})))
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
        session_id=exclude_session_id,
    )
    return [dict(record) for record in records]


def fetch_user_learnings(
    session,
    query: str | None,
    statuses: list[str] | None = None,
    limit: int = 5,
    exclude_session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch durable, cross-project facts about the user (``scope = 'user'``).

    Unlike project learnings these are not bound to a single ``(:Project)``: a
    user fact applies in every project, so the query spans all of them. Dedup
    still skips anything already injected into or first produced during the
    active session.
    """
    statuses = statuses or ["approved", "candidate"]
    if query and query.strip():
        try:
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('project_learning_fulltext', $search_query)
                YIELD node, score
                WHERE node.scope = 'user'
                  AND node.status IN $statuses
                  AND (node.consolidated_at IS NULL
                       OR coalesce(node.updated_at, node.created_at) > node.consolidated_at)
                  AND ($session_id IS NULL OR (
                       NOT (node)-[:INJECTED_IN]->(:Session {session_id: $session_id})
                       AND NOT (node)-[:FROM_SESSION]->(:Session {session_id: $session_id})))
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
                search_query=query,
                statuses=statuses,
                limit=limit,
                session_id=exclude_session_id,
            )
            rows = [dict(record) for record in records]
            if rows:
                return rows
        except Exception:
            pass

    records = session.run(
        """
        MATCH (l:Learning {scope: 'user'})
        WHERE l.status IN $statuses
          AND (l.consolidated_at IS NULL
               OR coalesce(l.updated_at, l.created_at) > l.consolidated_at)
          AND ($session_id IS NULL OR (
               NOT (l)-[:INJECTED_IN]->(:Session {session_id: $session_id})
               AND NOT (l)-[:FROM_SESSION]->(:Session {session_id: $session_id})))
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
        statuses=statuses,
        limit=limit,
        session_id=exclude_session_id,
    )
    return [dict(record) for record in records]


def fetch_project_decisions(
    session,
    project_id: str,
    query: str | None,
    limit: int = 3,
    exclude_session_id: str | None = None,
) -> list[dict[str, Any]]:
    if query and query.strip():
        try:
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('project_decision_fulltext', $search_query)
                YIELD node, score
                MATCH (:Project {id: $project_id})-[:HAS_DECISION]->(node)
                WHERE ($session_id IS NULL OR (
                      NOT (node)-[:INJECTED_IN]->(:Session {session_id: $session_id})
                      AND NOT (node)-[:FROM_SESSION]->(:Session {session_id: $session_id})))
                  AND coalesce(node.scope, 'project') = 'project'
                RETURN node.id AS id,
                       node.text AS text,
                       node.rationale AS rationale,
                       node.confidence AS confidence,
                       node.task_pattern AS task_pattern,
                       coalesce(node.scope, 'project') AS scope,
                       score
                ORDER BY score DESC,
                         coalesce(node.confidence, 0.0) DESC
                LIMIT $limit
                """,
                project_id=project_id,
                search_query=query,
                limit=limit,
                session_id=exclude_session_id,
            )
            rows = [dict(record) for record in records]
            if rows:
                return rows
        except Exception:
            pass

    records = session.run(
        """
        MATCH (:Project {id: $project_id})-[:HAS_DECISION]->(d:Decision)
        WHERE ($session_id IS NULL OR (
              NOT (d)-[:INJECTED_IN]->(:Session {session_id: $session_id})
              AND NOT (d)-[:FROM_SESSION]->(:Session {session_id: $session_id})))
          AND coalesce(d.scope, 'project') = 'project'
        RETURN d.id AS id,
               d.text AS text,
               d.rationale AS rationale,
               d.confidence AS confidence,
               d.task_pattern AS task_pattern,
               coalesce(d.scope, 'project') AS scope,
               0.0 AS score
        ORDER BY coalesce(d.updated_at, d.created_at) DESC
        LIMIT $limit
        """,
        project_id=project_id,
        limit=limit,
        session_id=exclude_session_id,
    )
    return [dict(record) for record in records]


def fetch_user_decisions(
    session,
    query: str | None,
    limit: int = 3,
    exclude_session_id: str | None = None,
) -> list[dict[str, Any]]:
    if query and query.strip():
        try:
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('project_decision_fulltext', $search_query)
                YIELD node, score
                WHERE node.scope = 'user'
                  AND ($session_id IS NULL OR (
                       NOT (node)-[:INJECTED_IN]->(:Session {session_id: $session_id})
                       AND NOT (node)-[:FROM_SESSION]->(:Session {session_id: $session_id})))
                RETURN node.id AS id,
                       node.text AS text,
                       node.rationale AS rationale,
                       node.confidence AS confidence,
                       node.task_pattern AS task_pattern,
                       node.scope AS scope,
                       score
                ORDER BY score DESC,
                         coalesce(node.confidence, 0.0) DESC
                LIMIT $limit
                """,
                search_query=query,
                limit=limit,
                session_id=exclude_session_id,
            )
            rows = [dict(record) for record in records]
            if rows:
                return rows
        except Exception:
            pass

    records = session.run(
        """
        MATCH (d:Decision {scope: 'user'})
        WHERE $session_id IS NULL OR (
              NOT (d)-[:INJECTED_IN]->(:Session {session_id: $session_id})
              AND NOT (d)-[:FROM_SESSION]->(:Session {session_id: $session_id}))
        RETURN d.id AS id,
               d.text AS text,
               d.rationale AS rationale,
               d.confidence AS confidence,
               d.task_pattern AS task_pattern,
               d.scope AS scope,
               0.0 AS score
        ORDER BY coalesce(d.updated_at, d.created_at) DESC
        LIMIT $limit
        """,
        limit=limit,
        session_id=exclude_session_id,
    )
    return [dict(record) for record in records]


def mark_injected_in_session(
    session,
    session_id: str | None,
    learning_ids: list[str],
    decision_ids: list[str],
    hook_event: str,
    source: str | None = None,
    prompt: str | None = None,
) -> None:
    """Link injected memory to the session so the same conversation never
    receives the same learning/decision twice, and to the specific hook
    ``SessionEvent`` that carried the injection.

    ``(m)-[:INJECTED_IN]->(:Session)`` powers per-session deduplication.
    ``(m)-[:INJECTED_AT]->(:SessionEvent)`` records *where* the memory entered
    context: the SessionStart or UserPromptSubmit event of this hook firing.
    The log_event hook runs in parallel and may not have written that event
    yet, so the link is matched on event name plus the shared payload
    discriminators (``source`` for SessionStart, ``prompt`` for
    UserPromptSubmit); log_event back-fills the link when it runs second."""
    if not session_id or session_id == "unknown":
        return
    for label, ids in (("Learning", learning_ids), ("Decision", decision_ids)):
        if not ids:
            continue
        session.run(
            f"""
            MERGE (s:Session {{session_id: $session_id}})
            ON CREATE SET s.created_at = datetime()
            WITH s
            UNWIND $ids AS memory_id
            MATCH (m:{label} {{id: memory_id}})
            MERGE (m)-[r:INJECTED_IN]->(s)
            ON CREATE SET r.first_injected_at = datetime(),
                          r.hook_event = $hook_event
            SET r.last_injected_at = datetime()
            """,
            session_id=session_id,
            ids=ids,
            hook_event=hook_event,
        )
        session.run(
            f"""
            MATCH (s:Session {{session_id: $session_id}})
                  -[:HAS_EVENT]->(e:SessionEvent {{event_name: $hook_event}})
            WHERE e.timestamp >= $since
              AND ($prompt IS NULL OR e.prompt = $prompt)
              AND ($source IS NULL OR e.source = $source)
            WITH e
            ORDER BY e.timestamp DESC
            LIMIT 1
            UNWIND $ids AS memory_id
            MATCH (m:{label} {{id: memory_id}})
            MERGE (m)-[r:INJECTED_AT]->(e)
            ON CREATE SET r.injected_at = datetime()
            """,
            session_id=session_id,
            ids=ids,
            hook_event=hook_event,
            since=injection_window_start(),
            prompt=prompt,
            source=source,
        )


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
    user_learnings: list[dict[str, Any]] | None = None,
    user_decisions: list[dict[str, Any]] | None = None,
) -> str:
    decisions = decisions or []
    user_learnings = user_learnings or []
    user_decisions = user_decisions or []
    if not learnings and not decisions and not user_learnings and not user_decisions:
        return ""

    lines = [
        f"Project context for {project.name} ({project.id}):",
    ]
    if user_learnings:
        lines.extend(["", "What we know about the user:"])
        for learning in user_learnings:
            status = learning.get("status") or "candidate"
            confidence = learning.get("confidence")
            confidence_text = (
                f", confidence {float(confidence):.2f}" if confidence is not None else ""
            )
            lines.append(
                f"- [{status}{confidence_text}] "
                f"{truncate(str(learning.get('text') or ''), 240)}"
            )
    if user_decisions:
        lines.extend(["", "User-scoped decisions:"])
        for decision in user_decisions:
            confidence = decision.get("confidence")
            confidence_text = (
                f", confidence {float(confidence):.2f}" if confidence is not None else ""
            )
            lines.append(
                f"- [decision{confidence_text}] "
                f"{truncate(str(decision.get('text') or ''), 240)}"
            )
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
            "Treat user facts and user-scoped decisions as durable context about the person. Use approved learnings as scoped project memory; treat candidate learnings as hints and decisions as context, not policy.",
        ]
    )
    return "\n".join(lines)
