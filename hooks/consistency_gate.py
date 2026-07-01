"""Consistency gate for freshly-extracted :Learning / :Decision candidates.

The Stop extractor (``--mode turn``) writes candidates with
``status = 'candidate'``. This module turns that status into an *automated
approval gate*. It runs only at Stop, never at SessionEnd:

1. Embed each new candidate (litellm, ``EMBEDDING_MODEL``) when the embedding
   model is available.
2. Retrieve the top-K closest **approved** items in the *same project and scope*.
   With an embedding this is a semantic vector-index search; without one (no
   embedding model configured) it falls back to the lexical fulltext index.
   Either way retrieval only nominates candidates for the judge — it is not the
   decision step.
3. An LLM judge decides whether the candidate genuinely *contradicts* any of
   those neighbours (as opposed to merely resembling them) and, per conflict,
   which side is more likely correct. It also flags whether the candidate is
   simply *already learned* — a restatement fully covered by one existing item.
4. Resolve: newer information is preferred but not absolute.
   - no contradiction .......... candidate -> ``approved``
   - already learned ........... candidate -> ``already_learned`` (``ALREADY_LEARNED_FROM``);
     the canonical item's ``support_count`` is reinforced (+1) and its confidence raised
   - candidate wins a conflict .. candidate -> ``approved``; loser -> ``rejected`` (``SUPERSEDES``)
   - an existing item vetoes .... candidate -> ``rejected`` (``CONTRADICTED_BY``); existing stays approved
   - only ambiguous conflicts ... candidate stays ``candidate`` (``CONTRADICTS``) for the human gate

Only the LLM judge is required. Without an embedding model the gate uses fulltext
retrieval; without the judge (or with the gate disabled) candidates keep
``status = 'candidate'``.

Vector retrieval uses Neo4j's native **pre-filtered** vector search: the index
stores ``project_id`` / ``scope`` / ``status`` as metadata (``WITH [...]``) and
the ``SEARCH`` clause applies them as an in-index ``WHERE`` so the walk only
visits approved items in the right project. This requires **Neo4j 2026.02+**
(the ``SEARCH`` clause / filtered vector search).
"""

from __future__ import annotations

import json
import os
from typing import Any

from project_common import (
    ProjectRef,
    embed_texts,
    embedding_dimensions,
    embeddings_ready,
    extraction_model_label,
    llm_complete,
    llm_readiness_status,
)

_VECTOR_INDEXES: list[tuple[str, str]] = [
    ("Learning", "project_learning_vector"),
    ("Decision", "project_decision_vector"),
]

_FALSEY = {"0", "false", "off", "no", ""}


def consistency_gate_enabled() -> bool:
    """The gate is on by default; set ``MKG_CONSISTENCY_GATE=0`` to disable."""
    return os.environ.get("MKG_CONSISTENCY_GATE", "1").strip().lower() not in _FALSEY


def _topk() -> int:
    try:
        return max(1, int(os.environ.get("MKG_CONSISTENCY_TOPK", "15")))
    except ValueError:
        return 15


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def ensure_memory_vector_indexes(driver, database: str) -> None:
    """Create cosine vector indexes on :Learning / :Decision ``embedding``.

    ``project_id`` / ``scope`` / ``status`` are declared as index metadata
    (``WITH [...]``) so the ``SEARCH`` clause can pre-filter on them in-index.
    Each ``CREATE`` runs in its own session so one schema statement never blocks
    the next.
    """
    dims = embedding_dimensions()
    for label, name in _VECTOR_INDEXES:
        stmt = (
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON n.embedding "
            f"WITH [n.project_id, n.scope, n.status] "
            f"OPTIONS {{indexConfig: {{"
            f"`vector.dimensions`: {dims}, "
            f"`vector.similarity_function`: 'cosine'}}}}"
        )
        with driver.session(database=database) as session:
            session.run(stmt).consume()


# --------------------------------------------------------------------------- #
# Candidate embeddings (written with the candidate so it is itself searchable)
# --------------------------------------------------------------------------- #
def _embedding_input(row: dict[str, Any]) -> str | None:
    text = (row.get("text") or "").strip()
    if not text:
        return None
    rationale = (row.get("rationale") or "").strip()
    return f"{text}\n{rationale}".strip() if rationale else text


def attach_candidate_embeddings(*row_groups: list[dict[str, Any]]) -> bool:
    """Populate ``row['embedding']`` in place for every row with text.

    Returns ``True`` if at least one embedding was produced. A single batched
    embedding call covers all groups; failures leave ``embedding`` unset so the
    write and the gate degrade gracefully.
    """
    indexed: list[tuple[dict[str, Any], str]] = []
    for group in row_groups:
        for row in group:
            text = _embedding_input(row)
            if text:
                indexed.append((row, text))
    if not indexed:
        return False
    vectors = embed_texts([text for _, text in indexed])
    produced = False
    for (row, _), vector in zip(indexed, vectors):
        if vector:
            row["embedding"] = vector
            produced = True
    return produced


# --------------------------------------------------------------------------- #
# Retrieval
#
# Two interchangeable paths return the same neighbour shape: semantic (vector
# index) when the candidate has an embedding, lexical (fulltext index) otherwise.
# A run therefore uses embeddings when the embedding model is available and
# falls back to fulltext when it is not, with no other behaviour change.
# --------------------------------------------------------------------------- #
import re as _re

_NEIGHBOUR_RETURN = """
        RETURN node.id AS id,
               node.text AS text,
               node.rationale AS rationale,
               node.confidence AS confidence,
               node.support_count AS support_count,
               toString(node.updated_at) AS updated_at,
               score
"""


def _lucene_query(*texts: str, limit: int = 16) -> str:
    """Build a Lucene-safe OR query from candidate text: alphanumeric tokens of
    3+ chars, space-joined (the fulltext index ORs terms by default)."""
    words: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for word in _re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower()):
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
            if len(words) >= limit:
                return " ".join(words)
    return " ".join(words)


def _fetch_neighbours_vector(
    driver,
    database: str,
    *,
    label: str,
    index_name: str,
    project_id: str,
    scope: str,
    vector: list[float],
    topk: int,
) -> list[dict[str, Any]]:
    # In-index pre-filter (Neo4j 2026.02+): the WHERE lives inside the SEARCH so
    # the walk only visits approved items in this project/scope. The candidate
    # itself is 'candidate', so status = 'approved' already excludes it.
    query = f"""
        MATCH (node:{label})
        SEARCH node IN (
            VECTOR INDEX {index_name}
            FOR $vector
            WHERE node.project_id = $project_id
              AND node.scope = $scope
              AND node.status = 'approved'
            LIMIT $limit
        ) SCORE AS score
        {_NEIGHBOUR_RETURN}
        ORDER BY score DESC
    """
    with driver.session(database=database) as session:
        result = session.run(
            query,
            vector=vector,
            project_id=project_id,
            scope=scope,
            limit=topk,
        )
        return [dict(record) for record in result]


def _fetch_neighbours_fulltext(
    driver,
    database: str,
    *,
    label: str,
    fulltext_index: str,
    project_id: str,
    scope: str,
    text: str,
    topk: int,
) -> list[dict[str, Any]]:
    lucene = _lucene_query(text)
    if not lucene:
        return []
    query = f"""
        CALL db.index.fulltext.queryNodes($index, $lucene) YIELD node, score
        WHERE node:{label}
          AND node.project_id = $project_id
          AND node.scope = $scope
          AND node.status = 'approved'
        {_NEIGHBOUR_RETURN}
        ORDER BY score DESC
        LIMIT $limit
    """
    with driver.session(database=database) as session:
        result = session.run(
            query,
            index=fulltext_index,
            lucene=lucene,
            project_id=project_id,
            scope=scope,
            limit=topk,
        )
        return [dict(record) for record in result]


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM = (
    "You audit an AI agent's durable project memory for contradictions. "
    "Return strict JSON only, no prose."
)


def _format_neighbour(idx: int, item: dict[str, Any]) -> str:
    parts = [f"[{idx}] id={item['id']}", f"text={item.get('text')!r}"]
    if item.get("rationale"):
        parts.append(f"rationale={item['rationale']!r}")
    if item.get("confidence") is not None:
        parts.append(f"confidence={item['confidence']}")
    if item.get("support_count") is not None:
        parts.append(f"support_count={item['support_count']}")
    if item.get("updated_at"):
        parts.append(f"last_updated={item['updated_at']}")
    return "; ".join(parts)


def _build_judge_prompt(
    kind: str,
    candidate: dict[str, Any],
    neighbours: list[dict[str, Any]],
) -> str:
    candidate_lines = [f"text={candidate.get('text')!r}"]
    if candidate.get("rationale"):
        candidate_lines.append(f"rationale={candidate['rationale']!r}")
    if candidate.get("confidence") is not None:
        candidate_lines.append(f"confidence={candidate['confidence']}")
    neighbour_block = "\n".join(
        _format_neighbour(i, item) for i, item in enumerate(neighbours)
    )
    return f"""A new {kind} candidate was just extracted for this project. Make TWO \
independent judgements about it against the existing approved {kind}s below.

(1) CONTRADICTION: does the candidate GENUINELY CONTRADICT any existing item — \
the two statements cannot both be true at the same time (e.g. "we use REST" vs \
"we use GraphQL", "deploy on Fridays" vs "never deploy on Fridays")?

(2) ALREADY LEARNED: is the candidate simply a RESTATEMENT of one existing item — \
the same fact/decision in different words, adding no materially new constraint or \
detail (a paraphrase, or a strict subset of what the existing item already says)? \
If it refines or adds a genuinely new constraint, it is NOT already learned and \
should be treated as new.

A single existing item is never both contradicted and already-learned, and \
merely sharing a topic is neither.

NEW {kind.upper()} CANDIDATE:
{chr(10).join(candidate_lines)}

EXISTING APPROVED {kind.upper()}S:
{neighbour_block if neighbour_block else "(none)"}

For each genuine contradiction, decide which side is more likely correct now.
Prefer the NEW candidate (memory should track the latest reality), but do not
treat that preference as absolute: choose "existing" when the existing item is
clearly more reliable — much higher support_count/confidence, or the new
candidate looks mistaken or speculative. Use "unclear" only when you truly
cannot tell which is right.

For "already learned", return the id of the single existing item the candidate
restates, or null when the candidate carries new information.

Return strict JSON of exactly this shape:
{{"contradictions": [{{"existing_id": "<id from the list>", "winner": "new|existing|unclear", "reason": "<short>"}}], "already_learned_of": "<id from the list or null>", "already_learned_reason": "<short or empty>"}}
Return {{"contradictions": [], "already_learned_of": null}} when there is no contradiction and the candidate is new."""


def _parse_judge(text: str) -> dict[str, Any]:
    """Parse the judge reply into both dimensions: a cleaned ``contradictions``
    list and a single ``already_learned_of`` id (or ``None``)."""
    empty: dict[str, Any] = {"contradictions": [], "already_learned_of": None}
    if not text:
        return empty
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```", 2)[1] if "```" in body[3:] else body.strip("`")
        if body.lstrip().startswith("json"):
            body = body.lstrip()[4:]
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end == -1:
        return empty
    try:
        parsed = json.loads(body[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return empty
    if not isinstance(parsed, dict):
        return empty
    cleaned: list[dict[str, Any]] = []
    items = parsed.get("contradictions")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            existing_id = str(item.get("existing_id") or "").strip()
            if not existing_id:
                continue
            winner = str(item.get("winner") or "new").strip().lower()
            if winner not in {"new", "existing", "unclear"}:
                winner = "unclear"
            cleaned.append(
                {"existing_id": existing_id, "winner": winner, "reason": str(item.get("reason") or "")[:280]}
            )
    already = str(parsed.get("already_learned_of") or "").strip()
    return {"contradictions": cleaned, "already_learned_of": already or None}


def _resolve(
    candidate_id: str,
    neighbours: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    already_learned_of: str | None = None,
) -> dict[str, Any]:
    """Map judged contradictions + already-learned onto a status transition.

    Precedence (conservative): an "existing" veto beats an already-learned merge,
    which beats a "new" win, which beats an "unclear". So a candidate is only
    rejected when the judge is confident an existing item is more reliable; only
    folded in as already-learned when nothing vetoes it; and only auto-approved as
    a supersede when it is neither vetoed nor a restatement.
    """
    valid_ids = {item["id"] for item in neighbours}
    superseded, vetoed, unclear = [], [], []
    for conflict in contradictions:
        existing_id = conflict["existing_id"]
        if existing_id not in valid_ids or existing_id == candidate_id:
            continue
        if conflict["winner"] == "existing":
            vetoed.append(existing_id)
        elif conflict["winner"] == "new":
            superseded.append(existing_id)
        else:
            unclear.append(existing_id)

    already_id = (
        already_learned_of
        if already_learned_of in valid_ids and already_learned_of != candidate_id
        else None
    )

    if vetoed:
        outcome, consistency = "rejected", "vetoed"
    elif already_id:
        outcome, consistency = "already_learned", "already_learned"
    elif superseded:
        outcome, consistency = "approved", "superseded_conflicts"
    elif unclear:
        outcome, consistency = "candidate", "ambiguous"
    else:
        outcome, consistency = "approved", "clean"

    return {
        "id": candidate_id,
        "outcome": outcome,
        "consistency": consistency,
        "superseded_ids": sorted(set(superseded)) if outcome == "approved" else [],
        "contradicted_by_ids": sorted(set(vetoed)) if outcome == "rejected" else [],
        "unclear_ids": sorted(set(unclear)) if outcome == "candidate" else [],
        "already_learned_ids": [already_id] if outcome == "already_learned" else [],
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _apply_resolutions(tx, *, label: str, rows: list[dict[str, Any]], model: str | None, timestamp: str) -> None:
    tx.run(
        f"""
        UNWIND $rows AS row
        MATCH (c:{label} {{id: row.id}})
        SET c.status = row.outcome,
            c.consistency_status = row.consistency,
            c.consistency_checked_at = datetime($timestamp),
            c.consistency_model = $model
        WITH c, row
        CALL {{
            WITH c, row
            UNWIND row.superseded_ids AS sid
            MATCH (old:{label} {{id: sid}})
            SET old.status = 'rejected',
                old.rejected_at = datetime($timestamp),
                old.rejected_reason = 'superseded_by:' + c.id
            MERGE (c)-[r:SUPERSEDES]->(old)
            SET r.created_at = datetime($timestamp)
            RETURN count(*) AS superseded
        }}
        CALL {{
            WITH c, row
            UNWIND row.contradicted_by_ids AS xid
            MATCH (other:{label} {{id: xid}})
            MERGE (c)-[r:CONTRADICTED_BY]->(other)
            SET r.created_at = datetime($timestamp)
            RETURN count(*) AS vetoed
        }}
        CALL {{
            WITH c, row
            UNWIND row.unclear_ids AS uid
            MATCH (other:{label} {{id: uid}})
            MERGE (c)-[r:CONTRADICTS]->(other)
            SET r.created_at = datetime($timestamp)
            RETURN count(*) AS unclear
        }}
        CALL {{
            WITH c, row
            UNWIND row.already_learned_ids AS aid
            MATCH (canon:{label} {{id: aid}})
            SET canon.support_count = coalesce(canon.support_count, 0) + 1,
                canon.confidence = CASE
                    WHEN coalesce(canon.confidence, 0.0) < coalesce(c.confidence, 0.0)
                    THEN c.confidence ELSE canon.confidence END,
                canon.last_reinforced_at = datetime($timestamp),
                c.already_learned_from = aid,
                c.already_learned_at = datetime($timestamp)
            MERGE (c)-[r:ALREADY_LEARNED_FROM]->(canon)
            SET r.created_at = datetime($timestamp)
            RETURN count(*) AS already_learned
        }}
        RETURN count(*) AS updated
        """,
        rows=rows,
        model=model,
        timestamp=timestamp,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _gate_one_label(
    driver,
    database: str,
    *,
    project: ProjectRef,
    label: str,
    index_name: str,
    fulltext_index: str,
    kind: str,
    rows: list[dict[str, Any]],
    model: str | None,
    timestamp: str,
) -> int:
    topk = _topk()
    resolutions: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = row.get("id")
        if not candidate_id:
            continue
        vector = row.get("embedding")
        scope = str(row.get("scope") or "project")
        try:
            if vector:
                neighbours = _fetch_neighbours_vector(
                    driver,
                    database,
                    label=label,
                    index_name=index_name,
                    project_id=project.id,
                    scope=scope,
                    vector=vector,
                    topk=topk,
                )
            else:
                neighbours = _fetch_neighbours_fulltext(
                    driver,
                    database,
                    label=label,
                    fulltext_index=fulltext_index,
                    project_id=project.id,
                    scope=scope,
                    text=row.get("text") or "",
                    topk=topk,
                )
        except Exception as exc:
            print(
                f"[consistency_gate] neighbour search failed for {label} {candidate_id} "
                f"({type(exc).__name__}: {str(exc)[:140]}); leaving as candidate",
                flush=True,
            )
            continue
        contradictions: list[dict[str, Any]] = []
        already_learned_of: str | None = None
        if neighbours:
            try:
                judgement = llm_complete(
                    [
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user", "content": _build_judge_prompt(kind, row, neighbours)},
                    ],
                    model=model,
                )
                verdict = _parse_judge(judgement)
                contradictions = verdict["contradictions"]
                already_learned_of = verdict["already_learned_of"]
            except Exception as exc:
                print(
                    f"[consistency_gate] judge failed for {label} {candidate_id} "
                    f"({type(exc).__name__}: {str(exc)[:140]}); leaving as candidate",
                    flush=True,
                )
                continue
        resolutions.append(
            _resolve(candidate_id, neighbours, contradictions, already_learned_of)
        )

    if not resolutions:
        return 0
    with driver.session(database=database) as session:
        session.execute_write(
            _apply_resolutions,
            label=label,
            rows=resolutions,
            model=extraction_model_label(model),
            timestamp=timestamp,
        )
    return len(resolutions)


def run_consistency_gate(
    driver,
    database: str,
    *,
    project: ProjectRef,
    learning_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    model: str | None,
    timestamp: str,
) -> dict[str, int]:
    """Gate newly-created candidates. Returns per-label counts of items checked.

    Invoked only from the Stop pipeline (``--mode turn``); the caller does not
    run it at SessionEnd. Only new, **project-scoped** candidates are gated;
    ``update`` rows reinforce
    nodes that already passed, and ``user``-scoped candidates flow through the
    separate ``consolidate_system_prompt.py`` gate (which folds them into the
    persona and owns their ``candidate → approved`` transition) — auto-approving
    them here would pull them out of that backlog. No-ops (returning zeros) when
    the gate is disabled or the LLM judge is unavailable, so candidates simply
    keep ``status = 'candidate'``.
    """
    if not consistency_gate_enabled():
        return {"learnings": 0, "decisions": 0}
    ready, reason = llm_readiness_status(model)
    if not ready:
        print(f"[consistency_gate] judge unavailable ({reason}); leaving candidates", flush=True)
        return {"learnings": 0, "decisions": 0}

    def _gatable(row: dict[str, Any]) -> bool:
        return (
            row.get("action") == "create"
            and bool(row.get("text"))
            and str(row.get("scope") or "project") == "project"
        )

    learning_creates = [r for r in learning_rows if _gatable(r)]
    decision_creates = [r for r in decision_rows if _gatable(r)]

    checked_learnings = _gate_one_label(
        driver,
        database,
        project=project,
        label="Learning",
        index_name="project_learning_vector",
        fulltext_index="project_learning_fulltext",
        kind="learning",
        rows=learning_creates,
        model=model,
        timestamp=timestamp,
    )
    checked_decisions = _gate_one_label(
        driver,
        database,
        project=project,
        label="Decision",
        index_name="project_decision_vector",
        fulltext_index="project_decision_fulltext",
        kind="decision",
        rows=decision_creates,
        model=model,
        timestamp=timestamp,
    )
    return {"learnings": checked_learnings, "decisions": checked_decisions}
