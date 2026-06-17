#!/usr/bin/env python3
"""Stop / SessionEnd hook: rate-limited system-prompt consolidation service.

This is the consolidation service the README and the Part 2 write-up leave as a
TODO: the only writer of ``(:SystemPrompt)`` besides the seed scripts. It folds
durable user-profile facts into the persona once enough of them have piled up
unreviewed.

It runs in the background on every Stop / SessionEnd, but does real work rarely:

1. Rate limit. A cooldown window (``MKG_PROMPT_CONSOLIDATION_INTERVAL_HOURS``,
   default 24h) keeps it from re-running on every turn. The last run time lives
   on the ``:SystemPrompt`` node.
2. Threshold gate. It only consolidates when *more than*
   ``MKG_PROMPT_CONSOLIDATION_THRESHOLD`` (default 5) user-profile memories are
   in need of review — user-scoped ``candidate`` learnings not yet folded into
   the prompt.
3. Consolidate. The current prompt plus the pending user facts go to the LLM,
   which returns a revised prompt that folds those facts into the persona.
4. Keep history. The outgoing prompt is archived as a ``:SystemPromptVersion``
   before the new content overwrites the active node, and the folded learnings
   are stamped so they drop out of the review backlog.

Like ``process_project.py`` it swallows its own errors so a Neo4j or LLM outage
never blocks the session.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from inject_system_prompt import DEFAULT_PROMPT  # noqa: E402
from project_common import (  # noqa: E402
    consolidation_interval_hours,
    consolidation_threshold,
    count_user_profile_memories_pending,
    ensure_project_schema,
    extraction_model_label,
    fetch_user_profile_memories_pending,
    in_extraction_subprocess,
    llm_complete,
    llm_ready,
    load_mkg_env,
    neo4j_config,
    read_system_prompt_state,
    snapshot_and_update_system_prompt,
    truncate,
)

PROMPT_NAME = "default"
MIN_CONSOLIDATED_PROMPT_CHARS = 200

CONSOLIDATION_INSTRUCTION = """\
You maintain the system prompt (the persona and operating policy) for an AI
agent. Below is the current system prompt, followed by durable, repeatedly
observed facts about the user that have accumulated in the agent's memory and
should now be reflected in the persona itself.

Produce a single revised system prompt that folds these durable user facts into
the current one. Requirements:
- This is an edit, not a rewrite. Preserve the existing structure, voice,
  section headings, and any runtime- or tool-agnostic operating principles.
- Integrate only durable, broadly applicable user facts (role, preferences,
  recurring constraints, domain priorities). Drop anything transient, narrow, or
  one-off. Merge overlapping facts instead of listing them.
- Keep it tight. Do not let the prompt bloat; tighten existing wording if facts
  overlap with what is already there.
- Output only the finished system prompt the agent will run on. Do not mention
  memory, consolidation, candidates, reviews, or that the prompt was generated.
  No preamble, no commentary, no code fences.

CURRENT SYSTEM PROMPT:
[[CURRENT_PROMPT]]

DURABLE USER-PROFILE FACTS TO FOLD IN:
[[USER_FACTS]]
"""


def consolidation_gate(
    pending_count: int,
    threshold: int,
    last_consolidated_at: str | None,
    interval_hours: float,
    now: datetime,
) -> tuple[bool, str]:
    """Decide whether to run, applying the threshold then the cooldown.

    Pure so the rate-limit / threshold logic is testable without Neo4j.
    """
    if pending_count <= threshold:
        return False, (
            f"{pending_count} user-profile memories in review "
            f"(need more than {threshold}); skipping"
        )
    if last_consolidated_at:
        last = _parse_iso(last_consolidated_at)
        if last is not None:
            elapsed_hours = (now - last).total_seconds() / 3600.0
            if elapsed_hours < interval_hours:
                return False, (
                    f"rate-limited: last consolidation {elapsed_hours:.1f}h ago "
                    f"(< {interval_hours:.0f}h cooldown)"
                )
    return True, (
        f"{pending_count} user-profile memories in review "
        f"(> {threshold}); consolidating"
    )


def _parse_iso(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def build_consolidation_prompt(current_prompt: str, memories: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for memory in memories:
        confidence = memory.get("confidence")
        confidence_text = (
            f" (confidence {float(confidence):.2f})" if confidence is not None else ""
        )
        lines.append(f"- {truncate(str(memory.get('text') or ''), 300)}{confidence_text}")
    facts = "\n".join(lines) if lines else "- (none)"
    return CONSOLIDATION_INSTRUCTION.replace(
        "[[CURRENT_PROMPT]]", current_prompt
    ).replace("[[USER_FACTS]]", facts)


def _clean_llm_prompt(text: str) -> str:
    """Strip stray code fences a model may wrap the prompt in."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def ask_llm_for_consolidated_prompt(prompt: str) -> str | None:
    if not llm_ready():
        return None

    content = llm_complete(
        [
            {
                "role": "system",
                "content": (
                    "You revise an AI agent's system prompt. Return only the "
                    "finished prompt text, with no commentary or fences."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    cleaned = _clean_llm_prompt(content)
    if len(cleaned) < MIN_CONSOLIDATED_PROMPT_CHARS:
        return None
    return cleaned


def consolidate(payload: dict[str, Any]) -> None:
    session_id = str(payload.get("session_id") or "unknown")

    from neo4j import GraphDatabase

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    uri, user, password, database = neo4j_config()
    threshold = consolidation_threshold()
    interval_hours = consolidation_interval_hours()

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_project_schema)

            pending_count = count_user_profile_memories_pending(driver, database)
            state = read_system_prompt_state(driver, database, PROMPT_NAME)

            proceed, reason = consolidation_gate(
                pending_count=pending_count,
                threshold=threshold,
                last_consolidated_at=state.get("last_consolidated_at"),
                interval_hours=interval_hours,
                now=now,
            )
            print(f"[consolidate_system_prompt] {reason}")
            if not proceed:
                return

            if not llm_ready():
                print(
                    "[consolidate_system_prompt] LLM credentials unavailable; "
                    "leaving the prompt untouched.",
                    file=sys.stderr,
                )
                return

            memories = fetch_user_profile_memories_pending(driver, database, limit=40)
            if not memories:
                return

            base_prompt = state.get("content") or DEFAULT_PROMPT
            prompt = build_consolidation_prompt(base_prompt, memories)
            new_content = ask_llm_for_consolidated_prompt(prompt)
            if not new_content:
                print(
                    "[consolidate_system_prompt] LLM returned no usable prompt; "
                    "leaving the prompt untouched.",
                    file=sys.stderr,
                )
                return

            folded_ids = [str(m["id"]) for m in memories if m.get("id")]
            result = session.execute_write(
                snapshot_and_update_system_prompt,
                name=PROMPT_NAME,
                new_content=new_content,
                folded_learning_ids=folded_ids,
                model=extraction_model_label(),
                session_id=session_id,
                now=timestamp,
            )
            print(
                f"[consolidate_system_prompt] consolidated {len(folded_ids)} "
                f"user-profile memories into (:SystemPrompt {{name: '{PROMPT_NAME}'}}) "
                f"v{result['old_version']} -> v{result['new_version']} "
                f"(history kept as :SystemPromptVersion); {len(new_content)} chars"
            )


def _read_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def _spawn_background(session_id: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--session-id",
        session_id,
    ]
    with open(os.devnull, "rb") as stdin, open(os.devnull, "wb") as output:
        subprocess.Popen(
            command,
            cwd=str(project_root),
            env=os.environ.copy(),
            stdin=stdin,
            stdout=output,
            stderr=output,
            start_new_session=True,
            close_fds=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id")
    parser.add_argument(
        "--background",
        action="store_true",
        help="Spawn the consolidation in the background and return immediately.",
    )
    args = parser.parse_args()

    # No-op inside a claude_cli extraction subprocess (see process_project).
    if in_extraction_subprocess():
        return 0

    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)
    payload = _read_payload()
    if args.session_id:
        payload["session_id"] = args.session_id

    if args.background:
        _spawn_background(str(payload.get("session_id") or "unknown"))
        return 0

    try:
        consolidate(payload)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[consolidate_system_prompt] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
