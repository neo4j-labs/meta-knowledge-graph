#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0"]
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
    ensure_project_schema,
    fetch_project_decisions,
    fetch_project_learnings,
    fetch_user_decisions,
    fetch_user_learnings,
    format_learning_context,
    load_mkg_env,
    mark_injected_in_session,
    mark_learnings_used,
    neo4j_config,
    resolve_project,
)


USER_SCOPED_CONTEXT_EVENTS = {"SessionStart"}


def _read_payload() -> dict:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def context_scope_for_hook(hook_event: str) -> str:
    return "user" if hook_event in USER_SCOPED_CONTEXT_EVENTS else "project"


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

    prompt = payload.get("prompt") or payload.get("last_assistant_message") or ""
    context_scope = context_scope_for_hook(hook_event)
    user_learnings: list[dict] = []
    user_decisions: list[dict] = []
    learnings: list[dict] = []
    decisions: list[dict] = []

    try:
        from neo4j import GraphDatabase

        uri, user, password, database = neo4j_config()
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=database) as session:
                session.execute_write(ensure_project_schema)
                if context_scope == "user":
                    user_learnings = fetch_user_learnings(
                        driver,
                        database,
                        query=prompt,
                        statuses=["approved", "candidate"],
                        limit=5,
                        exclude_session_id=exclude_session_id,
                    )
                    user_decisions = fetch_user_decisions(
                        driver,
                        database,
                        query=prompt,
                        limit=5,
                        exclude_session_id=exclude_session_id,
                    )
                else:
                    learnings = fetch_project_learnings(
                        driver,
                        database,
                        project_id=project.id,
                        query=prompt,
                        statuses=["approved", "candidate"],
                        limit=5,
                        exclude_session_id=exclude_session_id,
                    )
                    decisions = fetch_project_decisions(
                        driver,
                        database,
                        project_id=project.id,
                        query=prompt,
                        limit=3,
                        exclude_session_id=exclude_session_id,
                    )
                learning_ids = [
                    learning["id"]
                    for learning in (*user_learnings, *learnings)
                    if learning.get("id")
                ]
                mark_learnings_used(driver, database, learning_ids)
                decision_ids = [
                    decision["id"]
                    for decision in (*user_decisions, *decisions)
                    if decision.get("id")
                ]
                mark_injected_in_session(
                    driver,
                    database,
                    session_id,
                    learning_ids,
                    decision_ids,
                    hook_event,
                    source=payload.get("source"),
                    prompt=payload.get("prompt"),
                )
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[inject_project_context] error: {exc}", file=sys.stderr)
        return 0

    context = format_learning_context(
        project,
        learnings,
        decisions,
        user_learnings,
        user_decisions,
    )
    if not context:
        return 0

    output = {
        "hookSpecificOutput": {
            "hookEventName": hook_event,
            "additionalContext": context,
        }
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
