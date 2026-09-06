#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0", "litellm>=1.40.0"]
# ///
"""Inject scoped memory for the current prompt/session."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import (  # noqa: E402
    count_gate_blocked,
    count_pending_skill_proposals,
    count_project_observations,
    ensure_project_schema,
    embed_text,
    fetch_approved_skill_slugs,
    fetch_project_learnings,
    fetch_recent_observations,
    fetch_user_learnings,
    format_learning_context,
    inject_min_similarity,
    load_mkg_env,
    mark_injected_in_session,
    mark_learnings_used,
    neo4j_config,
    project_git_root,
    resolve_project,
    resolve_user,
    skill_catalog_inject_enabled,
    skill_review_required,
    truncate,
)


USER_SCOPED_CONTEXT_EVENTS = {"SessionStart"}
MAX_DEGRADED_REASON = 200


def _read_payload() -> dict:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def context_scope_for_hook(hook_event: str) -> str:
    return "user" if hook_event in USER_SCOPED_CONTEXT_EVENTS else "project"


def format_degraded_context(hook_event: str, exc: BaseException) -> str:
    """One line for the transcript when the memory pipeline fails.

    A hook must never crash the session, so failures are caught and the hook
    exits 0. But exit 0 with nothing on stdout looks exactly like a project
    with no memory, and stderr is not shown in a normal session: the hook
    once ran dead for weeks that way. Saying so in the injected context puts
    the failure where the agent and the user will see it."""
    reason = truncate(" ".join(f"{type(exc).__name__}: {exc}".split()), MAX_DEGRADED_REASON)
    unit = "session" if hook_event == "SessionStart" else "prompt"
    return (
        f"Persistent memory could not be loaded for this {unit}: the "
        f"inject_project_context hook failed ({reason}). Memory recall is "
        "degraded until the store is reachable."
    )


def _emit(hook_event: str, context: str) -> None:
    output = {
        "hookSpecificOutput": {
            "hookEventName": hook_event,
            "additionalContext": context,
        }
    }
    print(json.dumps(output))


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)

    payload = _read_payload()
    hook_event = payload.get("hook_event_name", "UserPromptSubmit")
    session_id = payload.get("session_id") or None
    # On clear/compact the conversation context was wiped, so earlier
    # injections in this session are gone and must not be deduplicated away.
    context_wiped = payload.get("source") in {"clear", "compact"}
    exclude_session_id = None if context_wiped else session_id
    project = resolve_project(payload, project_root)
    if not project:
        return 0
    # User facts are recalled per person: only this user's memory, this
    # user's gate record, and this user's name on the injected block.
    user = resolve_user(project_git_root(project))

    prompt = payload.get("prompt") or payload.get("last_assistant_message") or ""
    query_vector = embed_text(prompt) if prompt else None
    context_scope = context_scope_for_hook(hook_event)
    if prompt and query_vector is None:
        # Prompt-time recall is vector-only and gated on cosine similarity,
        # so with no embedding there is nothing to rank against the floor.
        # Say so on stderr instead of silently degrading to a keyword match.
        print(
            "[inject_project_context] prompt embedding unavailable; "
            "relevance-gated recall skipped for this prompt",
            file=sys.stderr,
        )
    user_learnings: list[dict] = []
    learnings: list[dict] = []
    observations: list[dict] = []
    observation_total = 0
    gate_blocked = 0
    pending_skills = 0
    skill_slugs: list[str] = []

    try:
        from neo4j import GraphDatabase

        uri, db_user, password, database = neo4j_config()
        with GraphDatabase.driver(uri, auth=(db_user, password)) as driver:
            with driver.session(database=database) as session:
                session.execute_write(ensure_project_schema)
                if context_scope == "user":
                    # SessionStart also carries the episodic "previously on this
                    # project" block: latest observations by recency, no query.
                    # Headlines only — bodies stay behind episode_fetch/search.
                    observations = fetch_recent_observations(
                        driver,
                        database,
                        project_id=project.id,
                        limit=3,
                    )
                    if observations:
                        observation_total = count_project_observations(
                            driver, database, project_id=project.id
                        )
                    user_learnings = fetch_user_learnings(
                        driver,
                        database,
                        query=prompt,
                        statuses=["approved", "candidate"],
                        limit=5,
                        exclude_session_id=exclude_session_id,
                        query_vector=query_vector,
                        min_similarity=inject_min_similarity(),
                        user_id=user.id,
                    )
                    # Accountability, not review: the gate runs autonomously,
                    # so session start surfaces what it recently blocked and
                    # points at the audit tool instead of asking for decisions.
                    gate_blocked = count_gate_blocked(
                        driver, database, project_id=project.id, user_id=user.id
                    )
                    # One line of approved-skill slugs so every session knows
                    # the skill_search / skill_fetch surface exists; anything
                    # deeper is pull-based through those tools.
                    if skill_catalog_inject_enabled():
                        skill_slugs = fetch_approved_skill_slugs(
                            driver, database, project_id=project.id
                        )
                    # Human-in-the-loop publishing: the one place a person is
                    # a dependency, so session start says how long the queue
                    # is and where to review it. Silent in auto mode.
                    if skill_review_required():
                        pending_skills = count_pending_skill_proposals(
                            driver, database, project_id=project.id
                        )
                else:
                    # Relevance-gated and vector-only: only memory clearing the
                    # cosine floor (MKG_INJECT_MIN_SIMILARITY) rides along with
                    # a prompt; no keyword or recency padding.
                    learnings = fetch_project_learnings(
                        driver,
                        database,
                        project_id=project.id,
                        query=prompt,
                        statuses=["approved", "candidate"],
                        limit=5,
                        exclude_session_id=exclude_session_id,
                        query_vector=query_vector,
                        min_similarity=inject_min_similarity(),
                    )
                learning_ids = [
                    learning["id"]
                    for learning in (*user_learnings, *learnings)
                    if learning.get("id")
                ]
                mark_learnings_used(driver, database, learning_ids)
                mark_injected_in_session(
                    driver,
                    database,
                    session_id,
                    learning_ids,
                    hook_event,
                    source=payload.get("source"),
                    prompt=payload.get("prompt"),
                )
    except Exception as exc:
        # Never crash the session, never fail silently either.
        print(f"[inject_project_context] error: {exc}", file=sys.stderr)
        _emit(hook_event, format_degraded_context(hook_event, exc))
        return 0

    context = format_learning_context(
        project,
        learnings,
        user_learnings,
        observations=observations,
        skill_slugs=skill_slugs,
        gate_blocked=gate_blocked,
        observation_total=observation_total,
        user_id=user.id,
        pending_skills=pending_skills,
    )
    if not context:
        return 0

    _emit(hook_event, context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
