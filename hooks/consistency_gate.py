"""Consistency gate for freshly-extracted :Learning candidates.

The Stop extractor (``--mode turn``) writes candidates with
``status = 'candidate'``. This module turns that status into an *automated
approval gate*. It runs only at Stop, never at SessionEnd:

1. Embed each new candidate (litellm, ``EMBEDDING_MODEL``) when the embedding
   model is available.
2. Retrieve the top-K closest stored items — **approved** plus other pending
   **candidates** — in the *same project and scope*, excluding the candidate
   itself. Including candidates is what lets restatements collapse instead of
   piling up side by side in the review queue. With an embedding this uses a
   hybrid vector + keyword search and reciprocal-rank fusion (RRF); without an
   embedding, or if the vector path is unavailable, it falls back to the lexical
   fulltext index. Either way retrieval only nominates candidates for the judge
   — it is not the decision step.
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
   - only ambiguous conflicts ... candidate stays ``candidate`` (``CONTRADICTS``) for the human gate;
     the judge's stated reason for punting is stamped on the edge so the
     reviewer sees why the pair needs a person

Resolutions are applied per candidate, immediately after judging it, so later
candidates in the same batch retrieve the updated statuses — two restatements
extracted together collapse onto one canonical item instead of mutually folding
into each other.

A companion sweep (:func:`sweep_ungated_candidates`) runs after the batch gate
and pushes through project-scoped candidates that entered the graph without a
gate run — MCP memory tool writes, and rows left behind by an
earlier judge or retrieval failure — backfilling and persisting their
embeddings first.

Only the LLM judge is required. Without an embedding model the gate uses
fulltext-only retrieval; without the judge (or with the gate disabled)
candidates keep ``status = 'candidate'``.

Vector retrieval uses Neo4j's native **pre-filtered** vector search: the
``SEARCH`` clause applies ``project_id`` / ``scope`` as an in-index ``WHERE`` so
the walk only visits the right project and scope. This requires **Neo4j
2026.02+** (the ``SEARCH`` clause / filtered vector search). There is no status
predicate because the vector index only ever contains live memory: when the
gate rejects an item or folds it as already-learned, it removes the item's
``embedding``, which drops it out of the index. Dead items keep their text and
provenance edges — they are tombstones, not searchable memory.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from project_common import (
    ProjectRef,
    embed_texts,
    embedding_dimensions,
    extraction_model_label,
    llm_complete,
    llm_readiness_status,
)

_VECTOR_INDEXES: list[tuple[str, str]] = [
    ("Learning", "project_learning_vector"),
    # Episodic observations are indexed for search but never gated: the gate
    # queries :Learning explicitly and ignores this label.
    ("Observation", "project_observation_vector"),
]

_HYBRID_RRF_K = 60.0
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
    """Create cosine vector indexes on :Learning / :Observation ``embedding``.

    ``project_id`` / ``scope`` / ``status`` are declared as index metadata
    (``WITH [...]``) so the ``SEARCH`` clause can pre-filter on them in-index.
    Retrieval only filters on ``project_id`` / ``scope``: dead items lose their
    embedding (and thus leave the index), so no status predicate is needed.
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
    return text or None


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
# Retrieval returns the same neighbour shape for hybrid and fallback paths. When
# the candidate has an embedding, vector and keyword hits are unioned and ranked
# with RRF. Without an embedding, keyword/fulltext is the fallback.
_NEIGHBOUR_RETURN = """
        RETURN node.id AS id,
               node.text AS text,
               node.rationale AS rationale,
               node.status AS status,
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
        for word in re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower()):
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
    candidate_id: str,
) -> list[dict[str, Any]]:
    # In-index pre-filter (Neo4j 2026.02+): the WHERE lives inside the SEARCH so
    # the walk only visits live memory in this project/scope. The candidate
    # finds itself (its embedding is already written), so it is dropped by id
    # after the search; the index LIMIT is one higher to keep topk real
    # neighbours.
    query = f"""
        MATCH (node:{label})
        SEARCH node IN (
            VECTOR INDEX {index_name}
            FOR $vector
            WHERE node.project_id = $project_id
              AND node.scope = $scope
            LIMIT $limit
        ) SCORE AS score
        WITH node, score
        WHERE node.id <> $candidate_id
          AND node.status IN ['approved', 'candidate']
        {_NEIGHBOUR_RETURN}
        ORDER BY score DESC
    """
    with driver.session(database=database) as session:
        result = session.run(
            query,
            vector=vector,
            project_id=project_id,
            scope=scope,
            limit=topk + 1,
            candidate_id=candidate_id,
        )
        return [dict(record) for record in result]


def _fetch_neighbours_hybrid_vector_keyword(
    driver,
    database: str,
    *,
    label: str,
    index_name: str,
    fulltext_index: str,
    project_id: str,
    scope: str,
    vector: list[float],
    text: str,
    topk: int,
    candidate_id: str,
) -> list[dict[str, Any]]:
    lucene = _lucene_query(text)
    if not lucene:
        return _fetch_neighbours_vector(
            driver,
            database,
            label=label,
            index_name=index_name,
            project_id=project_id,
            scope=scope,
            vector=vector,
            topk=topk,
            candidate_id=candidate_id,
        )
    query = f"""
        CALL () {{
            MATCH (node:{label})
            SEARCH node IN (
                VECTOR INDEX {index_name}
                FOR $vector
                WHERE node.project_id = $project_id
                  AND node.scope = $scope
                LIMIT $vector_limit
            ) SCORE AS raw_score
            WITH node, raw_score
            WHERE node.id <> $candidate_id
              AND node.status IN ['approved', 'candidate']
            ORDER BY raw_score DESC
            WITH collect({{node: node, raw_score: raw_score}}) AS rows
            UNWIND range(0, size(rows) - 1) AS idx
            WITH rows[idx] AS row, idx + 1 AS rank
            RETURN row.node AS node,
                   rank,
                   row.raw_score AS raw_score,
                   'vector' AS source

            UNION ALL

            CALL db.index.fulltext.queryNodes($index, $lucene)
            YIELD node, score AS raw_score
            WHERE node:{label}
              AND node.project_id = $project_id
              AND node.scope = $scope
              AND node.status IN ['approved', 'candidate']
              AND node.id <> $candidate_id
            WITH node, raw_score
            ORDER BY raw_score DESC
            LIMIT $keyword_limit
            WITH collect({{node: node, raw_score: raw_score}}) AS rows
            UNWIND range(0, size(rows) - 1) AS idx
            WITH rows[idx] AS row, idx + 1 AS rank
            RETURN row.node AS node,
                   rank,
                   row.raw_score AS raw_score,
                   'keyword' AS source
        }}
        WITH node,
             sum(1.0 / ($rrf_k + rank)) AS score,
             collect(source) AS sources
        {_NEIGHBOUR_RETURN}
        ORDER BY score DESC
        LIMIT $limit
    """
    with driver.session(database=database) as session:
        result = session.run(
            query,
            vector=vector,
            index=fulltext_index,
            lucene=lucene,
            project_id=project_id,
            scope=scope,
            vector_limit=topk + 1,
            keyword_limit=topk,
            limit=topk,
            rrf_k=_HYBRID_RRF_K,
            candidate_id=candidate_id,
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
    candidate_id: str,
) -> list[dict[str, Any]]:
    lucene = _lucene_query(text)
    if not lucene:
        return []
    query = f"""
        CALL db.index.fulltext.queryNodes($index, $lucene) YIELD node, score
        WHERE node:{label}
          AND node.project_id = $project_id
          AND node.scope = $scope
          AND node.status IN ['approved', 'candidate']
          AND node.id <> $candidate_id
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
            candidate_id=candidate_id,
        )
        return [dict(record) for record in result]


def _fetch_neighbours_hybrid(
    driver,
    database: str,
    *,
    label: str,
    index_name: str,
    fulltext_index: str,
    project_id: str,
    scope: str,
    vector: list[float] | None,
    text: str,
    topk: int,
    candidate_id: str,
) -> list[dict[str, Any]]:
    if vector:
        try:
            return _fetch_neighbours_hybrid_vector_keyword(
                driver,
                database,
                label=label,
                index_name=index_name,
                fulltext_index=fulltext_index,
                project_id=project_id,
                scope=scope,
                vector=vector,
                text=text,
                topk=topk,
                candidate_id=candidate_id,
            )
        except Exception:
            pass
    return _fetch_neighbours_fulltext(
        driver,
        database,
        label=label,
        fulltext_index=fulltext_index,
        project_id=project_id,
        scope=scope,
        text=text,
        topk=topk,
        candidate_id=candidate_id,
    )


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM = (
    "You audit an AI agent's durable project memory for contradictions. "
    "Return strict JSON only, no prose."
)


def _format_neighbour(idx: int, item: dict[str, Any]) -> str:
    parts = [
        f"[{idx}] id={item['id']}",
        f"status={item.get('status')}",
        f"text={item.get('text')!r}",
    ]
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
independent judgements about it against the existing stored {kind}s below. Each \
existing item carries a status: "approved" items passed review; "candidate" \
items are unreviewed peers still awaiting this same gate.

(1) CONTRADICTION: does the candidate GENUINELY CONTRADICT any existing item — \
the two statements cannot both be true at the same time (e.g. "we use REST" vs \
"we use GraphQL", "deploy on Fridays" vs "never deploy on Fridays")?

(2) ALREADY LEARNED: is the candidate simply a RESTATEMENT of one existing item — \
the same fact in different words, adding no materially new constraint or \
detail (a paraphrase, or a strict subset of what the existing item already says)? \
This applies whether the existing item is approved or itself a candidate. \
If it refines or adds a genuinely new constraint, it is NOT already learned and \
should be treated as new.

A single existing item is never both contradicted and already-learned, and \
merely sharing a topic is neither.

NEW {kind.upper()} CANDIDATE:
{chr(10).join(candidate_lines)}

EXISTING STORED {kind.upper()}S:
{neighbour_block if neighbour_block else "(none)"}

For each genuine contradiction, decide which side is more likely correct now.
Prefer the NEW candidate (memory should track the latest reality), but do not
treat that preference as absolute: choose "existing" when the existing item is
clearly more reliable — much higher support_count/confidence, or the new
candidate looks mistaken or speculative. An approved item outweighs an
unreviewed candidate of similar support. Use "unclear" only when you truly
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
    unclear_reasons: dict[str, str] = {}
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
            unclear_reasons.setdefault(existing_id, str(conflict.get("reason") or ""))

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
        "unclear_conflicts": (
            [
                {"id": uid, "reason": unclear_reasons.get(uid, "")}
                for uid in sorted(set(unclear))
            ]
            if outcome == "candidate"
            else []
        ),
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
            c.consistency_model = $model,
            // Dead memory leaves the vector index: dropping the embedding is
            // what lets neighbour retrieval skip status filtering entirely.
            c.embedding = CASE
                WHEN row.outcome IN ['rejected', 'already_learned'] THEN null
                ELSE c.embedding
            END
        WITH c, row
        CALL (c, row) {{
            UNWIND row.superseded_ids AS sid
            MATCH (old:{label} {{id: sid}})
            SET old.status = 'rejected',
                old.embedding = null,
                old.rejected_at = datetime($timestamp),
                old.rejected_reason = 'superseded_by:' + c.id
            MERGE (c)-[r:SUPERSEDES]->(old)
            SET r.created_at = datetime($timestamp)
            RETURN count(*) AS superseded
        }}
        CALL (c, row) {{
            UNWIND row.contradicted_by_ids AS xid
            MATCH (other:{label} {{id: xid}})
            MERGE (c)-[r:CONTRADICTED_BY]->(other)
            SET r.created_at = datetime($timestamp)
            RETURN count(*) AS vetoed
        }}
        CALL (c, row) {{
            UNWIND row.unclear_conflicts AS u
            MATCH (other:{label} {{id: u.id}})
            MERGE (c)-[r:CONTRADICTS]->(other)
            // The judge's rationale rides on the edge: it is the review
            // queue's explanation of why this pair needs a human.
            SET r.created_at = datetime($timestamp),
                r.reason = u.reason
            RETURN count(*) AS unclear
        }}
        CALL (c, row) {{
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
    applied = 0
    for row in rows:
        candidate_id = row.get("id")
        if not candidate_id:
            continue
        vector = row.get("embedding")
        scope = str(row.get("scope") or "project")
        try:
            neighbours = _fetch_neighbours_hybrid(
                driver,
                database,
                label=label,
                index_name=index_name,
                fulltext_index=fulltext_index,
                project_id=project.id,
                scope=scope,
                vector=vector,
                text=row.get("text") or "",
                topk=topk,
                candidate_id=candidate_id,
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
        resolution = _resolve(candidate_id, neighbours, contradictions, already_learned_of)
        # Apply immediately, not after the loop: later rows in this batch then
        # retrieve the updated statuses, so two restatements extracted together
        # collapse onto one canonical item instead of mutually folding into
        # each other (A already_learned_from B *and* B already_learned_from A).
        try:
            with driver.session(database=database) as session:
                session.execute_write(
                    _apply_resolutions,
                    label=label,
                    rows=[resolution],
                    model=extraction_model_label(model),
                    timestamp=timestamp,
                )
        except Exception as exc:
            print(
                f"[consistency_gate] applying resolution failed for {label} {candidate_id} "
                f"({type(exc).__name__}: {str(exc)[:140]}); leaving as candidate",
                flush=True,
            )
            continue
        applied += 1

    return applied


def run_consistency_gate(
    driver,
    database: str,
    *,
    project: ProjectRef,
    learning_rows: list[dict[str, Any]],
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
        return {"learnings": 0}
    ready, reason = llm_readiness_status(model)
    if not ready:
        print(f"[consistency_gate] judge unavailable ({reason}); leaving candidates", flush=True)
        return {"learnings": 0}

    def _gatable(row: dict[str, Any]) -> bool:
        return (
            row.get("action") == "create"
            and bool(row.get("text"))
            and str(row.get("scope") or "project") == "project"
        )

    learning_creates = [r for r in learning_rows if _gatable(r)]

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
    return {"learnings": checked_learnings}


# --------------------------------------------------------------------------- #
# Sweep: gate candidates that entered the graph without a gate run
# --------------------------------------------------------------------------- #
DEFAULT_SWEEP_LIMIT = 8

_SWEEP_LABELS = (
    ("learnings", "Learning", "project_learning_vector", "project_learning_fulltext", "learning"),
)


def sweep_limit() -> int:
    """Max ungated candidates swept per label per run (bounds judge LLM calls).
    ``MKG_CONSISTENCY_SWEEP_LIMIT`` overrides; 0 disables the sweep."""
    try:
        return max(0, int(os.environ.get("MKG_CONSISTENCY_SWEEP_LIMIT", str(DEFAULT_SWEEP_LIMIT))))
    except ValueError:
        return DEFAULT_SWEEP_LIMIT


def _fetch_ungated_candidates(
    driver,
    database: str,
    *,
    label: str,
    project_id: str,
    exclude_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Project-scoped candidates the gate has never resolved, oldest first.

    ``consistency_checked_at`` is stamped by ``_apply_resolutions`` for every
    gated row (including ambiguous keep-as-candidate outcomes), so its absence
    means the item bypassed the gate entirely: an MCP tool write, or a row an
    earlier judge/retrieval failure skipped. Oldest-first drains the backlog
    without starvation under the per-run limit.
    """
    query = f"""
        MATCH (n:{label} {{status: 'candidate'}})
        WHERE n.project_id = $project_id
          AND n.scope = 'project'
          AND n.consistency_checked_at IS NULL
          AND n.text IS NOT NULL
          AND NOT n.id IN $exclude_ids
        RETURN n.id AS id,
               n.text AS text,
               n.confidence AS confidence,
               n.scope AS scope,
               n.embedding AS embedding
        ORDER BY coalesce(n.updated_at, n.created_at)
        LIMIT $limit
    """
    with driver.session(database=database) as session:
        result = session.run(
            query,
            project_id=project_id,
            exclude_ids=list(exclude_ids),
            limit=limit,
        )
        return [dict(record) for record in result]


def _persist_embeddings(
    driver,
    database: str,
    *,
    label: str,
    rows: list[dict[str, Any]],
    timestamp: str,
) -> int:
    """Write freshly computed embeddings back onto swept nodes so they are
    permanently vector-searchable, not just embedded for this gate run."""
    payload = [
        {"id": row["id"], "embedding": row["embedding"]}
        for row in rows
        if row.get("id") and row.get("embedding")
    ]
    if not payload:
        return 0
    with driver.session(database=database) as session:
        session.run(
            f"""
            UNWIND $rows AS row
            MATCH (n:{label} {{id: row.id}})
            SET n.embedding = row.embedding,
                n.embedding_updated_at = datetime($timestamp)
            """,
            rows=payload,
            timestamp=timestamp,
        ).consume()
    return len(payload)


def sweep_ungated_candidates(
    driver,
    database: str,
    *,
    project: ProjectRef,
    model: str | None,
    timestamp: str,
    exclude_ids: list[str] | None = None,
) -> dict[str, int]:
    """Gate project-scoped candidates that never went through the gate.

    The batch gate only sees rows the Stop extractor just produced, so
    candidates written through the MCP memory tools — and rows
    an earlier judge failure left unresolved — would otherwise stay ungated
    (and un-embedded) forever. This sweep picks them up, embeds and persists
    missing embeddings, and runs the exact same retrieve-judge-resolve pipeline
    per item. ``exclude_ids`` should carry the ids the caller just attempted so
    a row that failed the judge moments ago is not immediately retried.

    Bounded by :func:`sweep_limit` per label per run so a large backlog drains
    across turns instead of stalling one Stop. User-scoped candidates are
    excluded for the same reason as in :func:`run_consistency_gate`.
    """
    if not consistency_gate_enabled():
        return {"learnings": 0}
    limit = sweep_limit()
    if limit <= 0:
        return {"learnings": 0}
    ready, reason = llm_readiness_status(model)
    if not ready:
        print(f"[consistency_gate] sweep skipped: judge unavailable ({reason})", flush=True)
        return {"learnings": 0}

    counts: dict[str, int] = {}
    for key, label, index_name, fulltext_index, kind in _SWEEP_LABELS:
        counts[key] = 0
        try:
            rows = _fetch_ungated_candidates(
                driver,
                database,
                label=label,
                project_id=project.id,
                exclude_ids=exclude_ids or [],
                limit=limit,
            )
        except Exception as exc:
            print(
                f"[consistency_gate] sweep fetch failed for {label} "
                f"({type(exc).__name__}: {str(exc)[:140]}); skipping",
                flush=True,
            )
            continue
        if not rows:
            continue
        missing = [row for row in rows if not row.get("embedding")]
        if missing and attach_candidate_embeddings(missing):
            try:
                _persist_embeddings(
                    driver, database, label=label, rows=missing, timestamp=timestamp
                )
            except Exception as exc:
                print(
                    f"[consistency_gate] persisting swept embeddings failed for {label} "
                    f"({type(exc).__name__}: {str(exc)[:140]}); "
                    "gating with the in-memory vector",
                    flush=True,
                )
        counts[key] = _gate_one_label(
            driver,
            database,
            project=project,
            label=label,
            index_name=index_name,
            fulltext_index=fulltext_index,
            kind=kind,
            rows=rows,
            model=model,
            timestamp=timestamp,
        )
        if counts[key]:
            print(
                f"[consistency_gate] swept {counts[key]} ungated {kind} candidate(s)",
                flush=True,
            )
    return counts
