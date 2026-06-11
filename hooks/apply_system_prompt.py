#!/usr/bin/env python3
"""Stop hook: fold accumulated system prompt suggestions into the live prompt.

The adjudicator (process_project.py) queues ``(:SystemPromptSuggestion
{status: 'candidate'})`` nodes but never touches the live ``(:SystemPrompt)``.
This hook closes that loop on Stop, rate-limited so the prompt is not churned
on every turn. A rebuild only runs when both gates pass:

  * at least ``MKG_PROMPT_REBUILD_MIN_HOURS`` (default 8) since the prompt's
    last rebuild, and
  * at least ``MKG_PROMPT_REBUILD_MIN_SUGGESTIONS`` (default 2) pending
    candidate suggestions.

Gate check and claim happen in one conditional write so concurrent Stop events
cannot double-rebuild. The previous prompt content is snapshotted as a
``(:SystemPromptVersion)`` node and each consumed suggestion is marked
``applied`` or ``rejected``. With ``OPENAI_API_KEY`` set the rewrite is an LLM
fold-in; without it the instructions are appended verbatim under a
"Learned operating notes" section so the loop still closes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from inject_system_prompt import DEFAULT_PROMPT  # noqa: E402
from project_common import (  # noqa: E402
    ensure_project_schema,
    load_dotenv,
    neo4j_config,
    truncate,
)

LEARNED_NOTES_HEADER = "## Learned operating notes"
CLAIM_TIMEOUT_MINUTES = 15
SUGGESTION_BATCH_LIMIT = 8


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def claim_rebuild(
    tx,
    name: str,
    default_content: str,
    min_suggestions: int,
    rebuild_cutoff: str,
    claim_cutoff: str,
    now: str,
) -> dict[str, Any] | None:
    """Atomically check both gates and claim the rebuild.

    Returns the current prompt content plus the pending suggestion batch, or
    None when a gate failed (not enough suggestions, rebuilt too recently, or
    another process holds a fresh claim). The prompt node is seeded from the
    default content if it does not exist yet, so the loop works on an empty
    graph without a manual seeding step.
    """
    record = tx.run(
        """
        MATCH (s:SystemPromptSuggestion {prompt_name: $name, status: 'candidate'})
        WITH s ORDER BY coalesce(s.confidence, 0.0) DESC, s.updated_at
        WITH collect(s) AS pending
        WHERE size(pending) >= $min_suggestions
        MERGE (p:SystemPrompt {name: $name})
        ON CREATE SET p.content = $default_content,
                      p.created_at = datetime($now),
                      p.version = 1
        WITH p, pending
        WHERE (p.last_rebuilt_at IS NULL OR p.last_rebuilt_at <= datetime($rebuild_cutoff))
          AND (p.rebuild_claimed_at IS NULL OR p.rebuild_claimed_at <= datetime($claim_cutoff))
        SET p.rebuild_claimed_at = datetime($now)
        RETURN p.content AS content,
               [s IN pending[..$batch_limit] | {
                   id: s.id,
                   instruction: s.instruction,
                   rationale: s.rationale,
                   confidence: s.confidence,
                   support_count: s.support_count
               }] AS suggestions
        """,
        name=name,
        default_content=default_content,
        min_suggestions=min_suggestions,
        rebuild_cutoff=rebuild_cutoff,
        claim_cutoff=claim_cutoff,
        now=now,
        batch_limit=SUGGESTION_BATCH_LIMIT,
    ).single()
    return dict(record) if record else None


def build_rebuild_prompt(
    current: str,
    suggestions: list[dict[str, Any]],
    max_chars: int,
) -> str:
    lines = []
    for item in suggestions:
        lines.append(
            "- "
            f"id={item.get('id')}; "
            f"confidence={item.get('confidence')}; "
            f"support_count={item.get('support_count')}; "
            f"instruction={truncate(str(item.get('instruction') or ''), 700)}; "
            f"rationale={truncate(str(item.get('rationale') or ''), 400)}"
        )
    pending = "\n".join(lines) if lines else "- none"
    return f"""You maintain the persisted system prompt for an agent. Fold the
suggestions worth keeping into the prompt and reject the rest.

Current system prompt:
---
{current}
---

Pending suggestions:
{pending}

Rules:
- Preserve the prompt's identity, intent, and overall structure.
- Integrate accepted instructions where they belong instead of appending blindly,
  and consolidate any existing learned notes that say the same thing.
- Reject suggestions that duplicate guidance already in the prompt, are
  project-specific trivia, are transient, or contradict the prompt.
- Keep the full prompt under {max_chars} characters.

Return JSON only with this shape:
{{"content": "the complete rewritten prompt", "applied_ids": ["..."], "rejected": [{{"id": "...", "reason": "..."}}]}}
"""


def _json_from_llm_text(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def ask_llm_for_rewrite(prompt: str) -> dict[str, Any]:
    from openai import OpenAI

    model = (
        os.environ.get("MKG_PROMPT_REBUILD_MODEL")
        or os.environ.get("MKG_LEARNING_MODEL")
        or os.environ.get("LLM_MODEL", "gpt-5.4-mini")
    )
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You maintain an agent's persisted system prompt. Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return _json_from_llm_text(content)


def validate_llm_rewrite(
    parsed: dict[str, Any],
    pending_ids: set[str],
    max_chars: int,
) -> tuple[str, list[str], list[dict[str, Any]]] | None:
    content = parsed.get("content")
    if not isinstance(content, str) or not content.strip() or len(content) > max_chars:
        return None
    applied = [
        sid
        for sid in (parsed.get("applied_ids") or [])
        if isinstance(sid, str) and sid in pending_ids
    ]
    rejected: list[dict[str, Any]] = []
    for row in parsed.get("rejected") or []:
        if not isinstance(row, dict):
            continue
        sid = row.get("id")
        if not isinstance(sid, str) or sid not in pending_ids or sid in applied:
            continue
        rejected.append({"id": sid, "reason": truncate(str(row.get("reason") or ""), 300) or None})
    if not applied and not rejected:
        return None
    return content, applied, rejected


def fallback_rewrite(
    current: str,
    suggestions: list[dict[str, Any]],
    max_chars: int,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Deterministic rewrite used when no LLM is available.

    Appends instructions under a learned-notes section, treats instructions
    already present verbatim as applied, and leaves anything over the length
    budget as a candidate for a future rebuild.
    """
    content = current.rstrip("\n")
    applied: list[str] = []
    for item in suggestions:
        sid = str(item.get("id") or "")
        instruction = str(item.get("instruction") or "").strip()
        if not sid or not instruction:
            continue
        if instruction in content:
            applied.append(sid)
            continue
        addition = f"- {instruction}"
        if LEARNED_NOTES_HEADER not in content:
            addition = f"\n{LEARNED_NOTES_HEADER}\n{addition}"
        if len(content) + len(addition) + 2 > max_chars:
            continue
        content = f"{content}\n{addition}"
        applied.append(sid)
    return f"{content}\n", applied, []


def rewrite_prompt(
    current: str,
    suggestions: list[dict[str, Any]],
    max_chars: int,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    pending_ids = {str(item.get("id")) for item in suggestions if item.get("id")}
    if os.environ.get("OPENAI_API_KEY"):
        try:
            parsed = ask_llm_for_rewrite(build_rebuild_prompt(current, suggestions, max_chars))
            validated = validate_llm_rewrite(parsed, pending_ids, max_chars)
            if validated:
                return validated
            print(
                "[apply_system_prompt] LLM rewrite invalid, falling back to verbatim append",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"[apply_system_prompt] LLM rewrite failed, falling back to verbatim append: {exc}",
                file=sys.stderr,
            )
    return fallback_rewrite(current, suggestions, max_chars)


def version_node_id(name: str, version: int, content: str) -> str:
    digest = sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"promptversion:{name}:v{version}:{digest}"


def upsert_prompt(tx, name: str, content: str, source: str, now: str) -> dict[str, Any]:
    """Create or replace a ``(:SystemPrompt)`` with rebuild-identical versioning.

    Replaced content is snapshotted as a ``(:SystemPromptVersion)`` and the
    version counter bumped, so manual seeds and Stop-event rebuilds share one
    history chain. Unchanged content is a no-op. Returns the action taken.
    """
    record = tx.run(
        "MATCH (p:SystemPrompt {name: $name}) "
        "RETURN p.content AS content, coalesce(p.version, 1) AS version",
        name=name,
    ).single()
    if record is None:
        tx.run(
            """
            MERGE (p:SystemPrompt {name: $name})
            SET p.content = $content,
                p.version = 1,
                p.created_at = datetime($now),
                p.updated_at = datetime($now)
            """,
            name=name,
            content=content,
            now=now,
        )
        return {"action": "created", "version": 1}

    old_content = str(record["content"] or "")
    old_version = int(record["version"])
    if old_content.strip() == content.strip():
        return {"action": "unchanged", "version": old_version}

    tx.run(
        """
        MATCH (p:SystemPrompt {name: $name})
        MERGE (v:SystemPromptVersion {id: $version_id})
        ON CREATE SET v.created_at = datetime($now)
        SET v.prompt_name = $name,
            v.version = $old_version,
            v.content = $old_content,
            v.source = $source
        MERGE (p)-[:HAS_VERSION]->(v)
        SET p.content = $content,
            p.version = $old_version + 1,
            p.updated_at = datetime($now)
        """,
        name=name,
        version_id=version_node_id(name, old_version, old_content),
        old_version=old_version,
        old_content=old_content,
        content=content,
        source=source,
        now=now,
    )
    return {"action": "updated", "version": old_version + 1}


def write_rebuild(
    tx,
    name: str,
    content: str,
    applied_ids: list[str],
    rejected_rows: list[dict[str, Any]],
    now: str,
) -> None:
    upsert_prompt(tx, name=name, content=content, source="stop_rebuild", now=now)
    tx.run(
        """
        MATCH (p:SystemPrompt {name: $name})
        SET p.last_rebuilt_at = datetime($now),
            p.rebuild_claimed_at = null
        """,
        name=name,
        now=now,
    )
    if applied_ids:
        tx.run(
            """
            MATCH (p:SystemPrompt {name: $name})
            UNWIND $ids AS sid
            MATCH (s:SystemPromptSuggestion {id: sid})
            SET s.status = 'applied',
                s.applied_at = datetime($now),
                s.applied_version = p.version
            MERGE (s)-[:APPLIED_TO]->(p)
            """,
            name=name,
            ids=applied_ids,
            now=now,
        )
    if rejected_rows:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (s:SystemPromptSuggestion {id: row.id})
            SET s.status = 'rejected',
                s.rejected_at = datetime($now),
                s.rejected_reason = row.reason
            """,
            rows=rejected_rows,
            now=now,
        )


def apply_system_prompt(prompt_name: str | None = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    name = prompt_name or os.getenv("MKG_PROMPT_NAME", "default")
    min_hours = _float_env("MKG_PROMPT_REBUILD_MIN_HOURS", 8.0)
    min_suggestions = _int_env("MKG_PROMPT_REBUILD_MIN_SUGGESTIONS", 2)
    max_chars = _int_env("MKG_PROMPT_MAX_CHARS", 12000)

    from neo4j import GraphDatabase

    now = datetime.now(timezone.utc)
    rebuild_cutoff = (now - timedelta(hours=min_hours)).isoformat()
    claim_cutoff = (now - timedelta(minutes=CLAIM_TIMEOUT_MINUTES)).isoformat()
    uri, user, password, database = neo4j_config()

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_project_schema)
            claim = session.execute_write(
                claim_rebuild,
                name=name,
                default_content=DEFAULT_PROMPT,
                min_suggestions=min_suggestions,
                rebuild_cutoff=rebuild_cutoff,
                claim_cutoff=claim_cutoff,
                now=now.isoformat(),
            )
            if not claim:
                return
            try:
                old_content = str(claim["content"] or "")
                suggestions = [dict(item) for item in claim["suggestions"] or []]
                content, applied_ids, rejected_rows = rewrite_prompt(
                    old_content,
                    suggestions,
                    max_chars,
                )
                session.execute_write(
                    write_rebuild,
                    name=name,
                    content=content,
                    applied_ids=applied_ids,
                    rejected_rows=rejected_rows,
                    now=now.isoformat(),
                )
            except Exception:
                try:
                    session.run(
                        "MATCH (p:SystemPrompt {name: $name}) SET p.rebuild_claimed_at = null",
                        name=name,
                    )
                except Exception:
                    pass
                raise


def _spawn_background(prompt_name: str | None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(Path(__file__).resolve())]
    if prompt_name:
        command += ["--prompt-name", prompt_name]
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
    parser.add_argument("--prompt-name")
    parser.add_argument(
        "--background",
        action="store_true",
        help="Spawn the rebuild in the background and return immediately.",
    )
    args = parser.parse_args()

    if args.background:
        _spawn_background(args.prompt_name)
        return 0

    try:
        apply_system_prompt(args.prompt_name)
    except Exception as exc:  # pragma: no cover - hook must never crash the session
        print(f"[apply_system_prompt] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
