#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["neo4j>=5.26.0", "litellm>=1.40.0"]
# ///
"""Learning salience: how close each learning is to where the project has
been working lately, spread one or two hops through related learnings.

One Personalized PageRank (PPR) per project. The skill service
(``consolidate_skills.py``) runs it in the background at most once per
``MKG_SALIENCE_INTERVAL_HOURS`` (default 4h); run a pass by hand with
``uv run --script hooks/learning_salience.py --force [--dry-run]``.

Graph. The nodes are the project's approved, project-scoped learnings.
Undirected, weighted edges join learnings that are related in one of four
ways:

- co-origin: first recorded in the same root session (a subagent session
  folds into its parent);
- same skill: both ``DERIVED_FROM`` by one approved skill;
- same error: error learnings that share an ``error_signature``;
- similar: embedding nearest neighbours (the top ``SIMILAR_TOP_K`` at or
  above ``SIMILAR_MIN_COSINE`` raw cosine), at ``SIMILAR_EDGE_WEIGHT``.

The first three are group edges. Each pair in a group of n weighs
``1 / (n - 1)``, so one busy session or a large skill does not become a dense
hub, and a group larger than ``MAX_GROUP_SIZE`` (a bulk import) adds no edges.
Edges the injector creates (``INJECTED_IN``, ``INJECTED_AT``, use and
retrieval counts) never enter the graph: a score built on them would measure
what recall already chose to show, and would reinforce itself.

Seeds. The project's recent real work, passed to PageRank as node-bias pairs.
Each creation of a learning (``PRODUCED_LEARNING``, or ``created_at`` for one
added outside the extractor) adds ``CREATE_TOUCH_WEIGHT``. Each non-echo
update, and each non-echo restatement the gate folded into it
(``ALREADY_LEARNED_FROM``), adds ``UPDATE_TOUCH_WEIGHT``. A touch decays by
``MKG_SALIENCE_DECAY`` (default 0.9) for every root project session started
after it, so recency is counted in project activity, not days, and a dormant
project does not fade. Echoes (a session restating a learning it was injected
with) are not work and are excluded.

Score. ``dampingFactor`` 0.5 keeps the walk within one or two hops of the
seeds. Scores are max-normalized per project and written as ``l.salience`` in
[0, 1] together with ``l.salience_run_id``; a learning that has left the graph
loses both. Each approved skill gets ``sk.salience``: the sum of its source
learnings' salience, max-normalized across the project's skills.

Backend. GDS ``gds.pageRank.stream`` on a Cypher projection that is dropped
afterwards, when the plugin is installed. Otherwise, or when GDS fails, the
same PPR runs as a power iteration in Python, which is fast at a project's
scale. The two can differ slightly in absolute values; after normalization
the ordering is what recall uses.

Use. Recall multiplies a learning's or skill's relevance by
``1 + MKG_SALIENCE_WEIGHT * salience``. Salience never makes anything
eligible: it reorders memory that already cleared relevance. ``MKG_SALIENCE=0``
turns off both the job and the boost.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import (  # noqa: E402
    _execute_query,
    _execute_query_single,
    ensure_project_schema,
    gds_available,
    in_extraction_subprocess,
    learning_salience_decay,
    learning_salience_enabled,
    learning_salience_interval_hours,
    load_mkg_env,
    neo4j_config,
    resolve_project,
    use_utf8_output,
)


METHOD_GDS = "gds_pagerank"
METHOD_PYTHON = "python_power_iteration"

# Probability of following an edge rather than restarting at a seed. 0.5 keeps
# most of the mass within two hops, as in HippoRAG 2.
DAMPING = 0.5
MAX_ITERATIONS = 50
TOLERANCE = 1e-7

CREATE_TOUCH_WEIGHT = 2.0
UPDATE_TOUCH_WEIGHT = 1.0

GROUP_EDGE_WEIGHT = 1.0
MAX_GROUP_SIZE = 50
SIMILAR_TOP_K = 5
# Raw cosine. "Related", not "duplicate": with text-embedding-3-small,
# distinct learnings on one topic sit around 0.6-0.75 and almost none reach
# 0.80. Same floor as MKG_TASK_PATTERN_EDGE_MIN_SIMILARITY; recalibrate when
# changing EMBEDDING_MODEL.
SIMILAR_MIN_COSINE = 0.65
SIMILAR_EDGE_WEIGHT = 0.5

TOP_REPORT = 10


def _parse_iso(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_native"):  # neo4j.time.DateTime
        parsed = value.to_native()
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def salience_gate(
    last_run_at: Any, interval_hours: float, now: datetime
) -> tuple[bool, str]:
    """Cooldown only. A pass is cheap, but session ages move with every new
    session, so it reruns on a clock rather than on new learnings."""
    last = _parse_iso(last_run_at)
    if last is not None:
        elapsed_hours = (now - last).total_seconds() / 3600.0
        if elapsed_hours < interval_hours:
            return False, (
                f"rate-limited: last salience pass {elapsed_hours:.1f}h ago "
                f"(< {interval_hours:g}h cooldown)"
            )
    return True, "salience pass due"


def read_last_salience_run(driver, database: str, project_id: str) -> Any:
    record = _execute_query_single(
        driver,
        database,
        "MATCH (p:Project {id: $project_id}) "
        "RETURN p.salience_last_run_at AS last_run_at",
        project_id=project_id,
    )
    return record["last_run_at"] if record else None


def mark_salience_run(
    driver, database: str, project_id: str, now: str, run_id: str, method: str | None
) -> None:
    _execute_query(
        driver,
        database,
        """
        MERGE (p:Project {id: $project_id})
        SET p.salience_last_run_at = datetime($now),
            p.salience_run_id = $run_id,
            p.salience_method = $method
        """,
        project_id=project_id,
        now=now,
        run_id=run_id,
        method=method,
    )


# --------------------------------------------------------------------------- #
# Graph reads
# --------------------------------------------------------------------------- #
def fetch_learning_rows(driver, database: str, project_id: str) -> list[dict[str, Any]]:
    """Every graph node with what seeding and grouping need: its non-echo
    writes, its non-echo restatements, and its root origin session."""
    records = _execute_query(
        driver,
        database,
        """
        MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(l:Learning)
        WHERE l.status = 'approved' AND l.scope = 'project'
        RETURN l.id AS id,
               l.created_at AS created_at,
               l.kind AS kind,
               l.error_signature AS error_signature,
               COLLECT {
                   MATCH (:ProjectProcessing)-[r:PRODUCED_LEARNING|UPDATED_LEARNING]->(l)
                   WHERE NOT coalesce(r.echo, false)
                   RETURN {kind: type(r), at: coalesce(r.updated_at, r.created_at)}
               } AS edits,
               COLLECT {
                   MATCH (:Learning)-[r:ALREADY_LEARNED_FROM]->(l)
                   WHERE NOT coalesce(r.echo, false)
                   RETURN r.created_at
               } AS restatements,
               COLLECT {
                   MATCH (l)-[:FROM_SESSION]->(s:Session)
                   OPTIONAL MATCH (root:Session)-[:HAS_SUBAGENT*1..]->(s)
                   WHERE NOT ()-[:HAS_SUBAGENT]->(root)
                   WITH coalesce(root, s) AS origin, s
                   ORDER BY s.created_at ASC
                   LIMIT 1
                   RETURN origin.session_id
               } AS origin
        ORDER BY id
        """,
        project_id=project_id,
    )
    return [dict(record) for record in records]


def fetch_root_session_times(driver, database: str, project_id: str) -> list[datetime]:
    records = _execute_query(
        driver,
        database,
        """
        MATCH (:Project {id: $project_id})-[:HAS_SESSION]->(s:Session)
        WHERE s.created_at IS NOT NULL
          AND NOT ()-[:HAS_SUBAGENT]->(s)
        RETURN s.created_at AS created_at
        ORDER BY created_at
        """,
        project_id=project_id,
    )
    times = [_parse_iso(record["created_at"]) for record in records]
    return sorted(t for t in times if t is not None)


def fetch_skill_groups(driver, database: str, project_id: str) -> list[dict[str, Any]]:
    records = _execute_query(
        driver,
        database,
        """
        MATCH (:Project {id: $project_id})-[:HAS_SKILL]->(sk:Skill {status: 'approved'})
              -[:DERIVED_FROM]->(l:Learning)
        WHERE l.status = 'approved' AND l.scope = 'project'
        RETURN sk.id AS skill_id, collect(DISTINCT l.id) AS members
        ORDER BY skill_id
        """,
        project_id=project_id,
    )
    return [dict(record) for record in records]


_SIMILAR_BY_INDEX = """
    CYPHER 25
    MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(a:Learning)
    WHERE a.status = 'approved' AND a.scope = 'project' AND a.embedding IS NOT NULL
    CALL (a) {
        MATCH (b:Learning)
        SEARCH b IN (
            VECTOR INDEX project_learning_vector
            FOR a.embedding
            WHERE b.project_id = $project_id
              AND b.scope = 'project'
              AND b.status = 'approved'
            LIMIT $index_limit
        ) SCORE AS score
        WITH b, score
        WHERE b.id <> a.id AND score >= $min_score
        RETURN b.id AS target, score
        ORDER BY score DESC
        LIMIT $top_k
    }
    RETURN a.id AS source, target, score
"""

# Exact pairwise cosine: the fallback for a database whose vector index is
# missing or predates the filter properties. Quadratic, fine at project scale.
_SIMILAR_EXACT = """
    CYPHER 25
    MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(a:Learning)
    WHERE a.status = 'approved' AND a.scope = 'project' AND a.embedding IS NOT NULL
    CALL (a) {
        MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(b:Learning)
        WHERE b <> a
          AND b.status = 'approved'
          AND b.scope = 'project'
          AND b.embedding IS NOT NULL
          AND size(b.embedding) = size(a.embedding)
        WITH b, vector.similarity.cosine(a.embedding, b.embedding) AS score
        WHERE score >= $min_score
        RETURN b.id AS target, score
        ORDER BY score DESC
        LIMIT $top_k
    }
    RETURN a.id AS source, target, score
"""


def fetch_similarity_pairs(driver, database: str, project_id: str) -> list[dict[str, Any]]:
    params = {
        "project_id": project_id,
        "top_k": SIMILAR_TOP_K,
        "index_limit": SIMILAR_TOP_K + 1,  # each learning finds itself first
        # Neo4j reports cosine normalized to (1 + cos) / 2.
        "min_score": (1.0 + SIMILAR_MIN_COSINE) / 2.0,
    }
    try:
        return [dict(r) for r in _execute_query(driver, database, _SIMILAR_BY_INDEX, **params)]
    except Exception as exc:
        print(
            f"[learning_salience] vector-index neighbours failed ({exc}); "
            "using exact cosine",
            file=sys.stderr,
        )
    try:
        return [dict(r) for r in _execute_query(driver, database, _SIMILAR_EXACT, **params)]
    except Exception as exc:
        print(
            f"[learning_salience] similarity edges skipped: {exc}",
            file=sys.stderr,
        )
        return []


# --------------------------------------------------------------------------- #
# Seeds and edges (pure)
# --------------------------------------------------------------------------- #
def sessions_since(at: datetime, root_times: list[datetime]) -> int:
    """Root project sessions started after ``at``; 0 for the current one."""
    return len(root_times) - bisect_right(root_times, at)


def seed_biases(
    rows: list[dict[str, Any]], root_times: list[datetime], decay: float
) -> dict[str, float]:
    biases: dict[str, float] = {}
    for row in rows:
        touches: list[tuple[float, datetime]] = []
        produced = False
        for edit in row.get("edits") or []:
            at = _parse_iso((edit or {}).get("at"))
            if at is None:
                continue
            if edit.get("kind") == "PRODUCED_LEARNING":
                produced = True
                touches.append((CREATE_TOUCH_WEIGHT, at))
            else:
                touches.append((UPDATE_TOUCH_WEIGHT, at))
        if not produced:
            # Added outside the extractor (project_add_learning, a seed
            # script): its creation is still a touch.
            created = _parse_iso(row.get("created_at"))
            if created is not None:
                touches.append((CREATE_TOUCH_WEIGHT, created))
        for value in row.get("restatements") or []:
            at = _parse_iso(value)
            if at is not None:
                touches.append((UPDATE_TOUCH_WEIGHT, at))
        bias = sum(weight * decay ** sessions_since(at, root_times) for weight, at in touches)
        if bias > 0:
            biases[str(row["id"])] = bias
    return biases


def group_edges(
    groups: Iterable[Iterable[str]],
    nodes: set[str],
    weight: float = GROUP_EDGE_WEIGHT,
    max_size: int = MAX_GROUP_SIZE,
) -> dict[tuple[str, str], float]:
    """Pair every two members of a group at ``weight / (n - 1)``."""
    edges: dict[tuple[str, str], float] = defaultdict(float)
    for group in groups:
        members = sorted({str(member) for member in group if str(member) in nodes})
        size = len(members)
        if size < 2 or size > max_size:
            continue
        pair_weight = weight / (size - 1)
        for i in range(size):
            for j in range(i + 1, size):
                edges[(members[i], members[j])] += pair_weight
    return dict(edges)


def similarity_edges(
    pairs: list[dict[str, Any]], nodes: set[str]
) -> dict[tuple[str, str], float]:
    edges: dict[tuple[str, str], float] = {}
    for pair in pairs:
        a, b = str(pair["source"]), str(pair["target"])
        if a == b or a not in nodes or b not in nodes:
            continue
        edges[(min(a, b), max(a, b))] = SIMILAR_EDGE_WEIGHT
    return edges


def merge_edges(*edge_maps: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    merged: dict[tuple[str, str], float] = defaultdict(float)
    for edge_map in edge_maps:
        for pair, weight in edge_map.items():
            merged[pair] += weight
    return dict(merged)


def build_edges(
    rows: list[dict[str, Any]],
    skill_groups: list[dict[str, Any]],
    similar_pairs: list[dict[str, Any]],
) -> dict[tuple[str, str], float]:
    nodes = {str(row["id"]) for row in rows}
    by_origin: dict[str, list[str]] = defaultdict(list)
    by_error: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        origin = (row.get("origin") or [None])[0]
        if origin:
            by_origin[str(origin)].append(str(row["id"]))
        if row.get("kind") == "error" and row.get("error_signature"):
            by_error[str(row["error_signature"])].append(str(row["id"]))
    return merge_edges(
        group_edges(by_origin.values(), nodes),
        group_edges((group["members"] for group in skill_groups), nodes),
        group_edges(by_error.values(), nodes),
        similarity_edges(similar_pairs, nodes),
    )


# --------------------------------------------------------------------------- #
# PageRank
# --------------------------------------------------------------------------- #
def personalized_pagerank(
    nodes: list[str],
    edges: dict[tuple[str, str], float],
    seeds: dict[str, float],
    damping: float = DAMPING,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
) -> dict[str, float]:
    """Weighted, undirected PPR by power iteration. Like GDS, a node with no
    edges keeps only its restart share (its walk mass is not redistributed)."""
    total = sum(bias for bias in seeds.values() if bias > 0)
    if total <= 0:
        return {node: 0.0 for node in nodes}
    restart = {node: max(seeds.get(node, 0.0), 0.0) / total for node in nodes}
    neighbours: dict[str, list[tuple[str, float]]] = defaultdict(list)
    strength: dict[str, float] = defaultdict(float)
    for (a, b), weight in edges.items():
        if weight <= 0 or a not in restart or b not in restart:
            continue
        neighbours[a].append((b, weight))
        neighbours[b].append((a, weight))
        strength[a] += weight
        strength[b] += weight

    scores = dict(restart)
    for _ in range(max_iterations):
        updated = {node: (1.0 - damping) * restart[node] for node in nodes}
        for node in nodes:
            mass = scores[node]
            if mass <= 0 or strength[node] <= 0:
                continue
            share = damping * mass / strength[node]
            for other, weight in neighbours[node]:
                updated[other] += share * weight
        delta = sum(abs(updated[node] - scores[node]) for node in nodes)
        scores = updated
        if delta < tolerance:
            break
    return scores


def gds_personalized_pagerank(
    driver,
    database: str,
    project_id: str,
    nodes: list[str],
    edges: dict[tuple[str, str], float],
    seeds: dict[str, float],
) -> dict[str, float]:
    """Project the learning graph into GDS, stream PPR, drop the graph."""
    digest = sha1(project_id.encode("utf-8")).hexdigest()[:10]
    graph_name = f"mkg-salience-{digest}-{uuid.uuid4().hex[:8]}"
    connected = {node for pair in edges for node in pair}
    rows = [
        {"source": a, "target": b, "weight": weight} for (a, b), weight in edges.items()
    ]
    # A learning with no related learning still needs to be in the graph, as
    # a seed or to receive its zero.
    rows.extend(
        {"source": node, "target": None, "weight": 0.0}
        for node in nodes
        if node not in connected
    )
    _execute_query(
        driver,
        database,
        """
        UNWIND $rows AS row
        MATCH (a:Learning {id: row.source})
        OPTIONAL MATCH (b:Learning {id: row.target})
        WITH gds.graph.project(
          $graph_name, a, b,
          {relationshipProperties: {weight: row.weight}},
          {undirectedRelationshipTypes: ['*']}
        ) AS graph
        RETURN graph.nodeCount AS node_count,
               graph.relationshipCount AS relationship_count
        """,
        graph_name=graph_name,
        rows=rows,
    )
    try:
        records = _execute_query(
            driver,
            database,
            """
            UNWIND $seeds AS seed
            MATCH (l:Learning {id: seed.id})
            WITH collect([l, seed.bias]) AS source_nodes
            CALL gds.pageRank.stream($graph_name, {
              sourceNodes: source_nodes,
              dampingFactor: $damping,
              maxIterations: $max_iterations,
              tolerance: $tolerance,
              relationshipWeightProperty: 'weight',
              concurrency: 1
            })
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS id, score
            """,
            graph_name=graph_name,
            seeds=[{"id": node, "bias": bias} for node, bias in sorted(seeds.items())],
            damping=DAMPING,
            max_iterations=MAX_ITERATIONS,
            tolerance=TOLERANCE,
        )
        return {str(record["id"]): float(record["score"]) for record in records}
    finally:
        try:
            _execute_query(
                driver,
                database,
                "CALL gds.graph.drop($graph_name, false) YIELD graphName "
                "RETURN graphName",
                graph_name=graph_name,
            )
        except Exception as exc:
            print(
                f"[learning_salience] could not drop GDS graph {graph_name}: {exc}",
                file=sys.stderr,
            )


def compute_scores(
    driver,
    database: str,
    project_id: str,
    nodes: list[str],
    edges: dict[tuple[str, str], float],
    seeds: dict[str, float],
) -> tuple[dict[str, float], str]:
    if gds_available(driver, database):
        try:
            scores = gds_personalized_pagerank(
                driver, database, project_id, nodes, edges, seeds
            )
            return {node: scores.get(node, 0.0) for node in nodes}, METHOD_GDS
        except Exception as exc:
            print(
                f"[learning_salience] GDS PageRank failed ({exc}); "
                "computing it in Python",
                file=sys.stderr,
            )
    else:
        print("[learning_salience] GDS is not installed; computing PageRank in Python")
    return personalized_pagerank(nodes, edges, seeds), METHOD_PYTHON


def normalize(scores: dict[str, float]) -> dict[str, float]:
    top = max(scores.values(), default=0.0)
    if top <= 0:
        return {key: 0.0 for key in scores}
    return {key: round(max(value, 0.0) / top, 6) for key, value in scores.items()}


def skill_salience(
    skill_groups: list[dict[str, Any]], learning_salience: dict[str, float]
) -> dict[str, float]:
    totals = {
        str(group["skill_id"]): sum(
            learning_salience.get(str(member), 0.0) for member in group["members"]
        )
        for group in skill_groups
    }
    return normalize(totals)


# --------------------------------------------------------------------------- #
# Write-back
# --------------------------------------------------------------------------- #
def write_salience(
    driver,
    database: str,
    project_id: str,
    learning_scores: dict[str, float],
    skill_scores: dict[str, float],
    run_id: str,
) -> None:
    _execute_query(
        driver,
        database,
        """
        UNWIND $rows AS row
        MATCH (l:Learning {id: row.id})
        SET l.salience = row.salience,
            l.salience_run_id = $run_id
        """,
        rows=[{"id": key, "salience": value} for key, value in learning_scores.items()],
        run_id=run_id,
    )
    # A learning that left the graph (superseded, demoted) keeps no score.
    _execute_query(
        driver,
        database,
        """
        MATCH (:Project {id: $project_id})-[:HAS_LEARNING]->(l:Learning)
        WHERE l.salience_run_id IS NOT NULL AND l.salience_run_id <> $run_id
        REMOVE l.salience, l.salience_run_id
        """,
        project_id=project_id,
        run_id=run_id,
    )
    _execute_query(
        driver,
        database,
        """
        UNWIND $rows AS row
        MATCH (sk:Skill {id: row.id})
        SET sk.salience = row.salience,
            sk.salience_run_id = $run_id
        """,
        rows=[{"id": key, "salience": value} for key, value in skill_scores.items()],
        run_id=run_id,
    )
    _execute_query(
        driver,
        database,
        """
        MATCH (:Project {id: $project_id})-[:HAS_SKILL]->(sk:Skill)
        WHERE sk.salience_run_id IS NOT NULL AND sk.salience_run_id <> $run_id
        REMOVE sk.salience, sk.salience_run_id
        """,
        project_id=project_id,
        run_id=run_id,
    )


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #
def run_learning_salience(
    driver,
    database: str,
    project,
    now: datetime,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One rate-limited salience pass for ``project``. Returns a summary.

    ``force`` ignores the cooldown. ``dry_run`` computes and reports the
    scores without writing them or stamping the run."""
    summary: dict[str, Any] = {
        "ran": False,
        "method": None,
        "nodes": 0,
        "edges": 0,
        "seeds": 0,
        "top": [],
    }
    if not learning_salience_enabled():
        print("[learning_salience] disabled via MKG_SALIENCE; skipping")
        return summary
    if not force:
        proceed, reason = salience_gate(
            read_last_salience_run(driver, database, project.id),
            learning_salience_interval_hours(),
            now,
        )
        print(f"[learning_salience] {reason}")
        if not proceed:
            return summary

    summary["ran"] = True
    timestamp = now.isoformat()
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"

    rows = fetch_learning_rows(driver, database, project.id)
    nodes = [str(row["id"]) for row in rows]
    seeds = seed_biases(
        rows,
        fetch_root_session_times(driver, database, project.id),
        learning_salience_decay(),
    )
    skill_groups = fetch_skill_groups(driver, database, project.id)
    edges = (
        build_edges(rows, skill_groups, fetch_similarity_pairs(driver, database, project.id))
        if nodes
        else {}
    )
    summary.update(nodes=len(nodes), edges=len(edges), seeds=len(seeds))

    learning_scores: dict[str, float] = {}
    if nodes and seeds:
        raw, method = compute_scores(driver, database, project.id, nodes, edges, seeds)
        learning_scores = normalize(raw)
        summary["method"] = method
    skill_scores = skill_salience(skill_groups, learning_scores)
    summary["top"] = sorted(learning_scores.items(), key=lambda item: (-item[1], item[0]))[
        :TOP_REPORT
    ]
    print(
        f"[learning_salience] {len(nodes)} learnings, {len(edges)} edges, "
        f"{len(seeds)} seeds, {len(skill_scores)} skills "
        f"({summary['method'] or 'nothing to score'})"
    )

    if dry_run:
        return summary
    write_salience(driver, database, project.id, learning_scores, skill_scores, run_id)
    mark_salience_run(driver, database, project.id, timestamp, run_id, summary["method"])
    return summary


# --------------------------------------------------------------------------- #
# Manual entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(
        description="Run one learning-salience pass for the current project."
    )
    parser.add_argument("--force", action="store_true", help="Ignore the cooldown.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the scores without writing them or stamping the run.",
    )
    args = parser.parse_args()
    if in_extraction_subprocess():
        return 0

    project_root = Path(__file__).resolve().parents[1]
    load_mkg_env(project_root)
    project = resolve_project({"cwd": os.getcwd()}, project_root)
    if not project:
        print("[learning_salience] no resolvable project", file=sys.stderr)
        return 1

    from neo4j import GraphDatabase

    uri, db_user, password, database = neo4j_config()
    with GraphDatabase.driver(uri, auth=(db_user, password)) as driver:
        with driver.session(database=database) as session:
            session.execute_write(ensure_project_schema)
        summary = run_learning_salience(
            driver,
            database,
            project,
            datetime.now(timezone.utc),
            force=args.force,
            dry_run=args.dry_run,
        )
    print(
        f"[learning_salience] project={project.id} ran={summary['ran']} "
        f"method={summary['method']} learnings={summary['nodes']} "
        f"edges={summary['edges']} seeds={summary['seeds']}"
    )
    for learning_id, score in summary["top"]:
        print(f"  {score:.3f}  {learning_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
