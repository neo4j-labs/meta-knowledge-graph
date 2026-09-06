#!/usr/bin/env python3
"""Backfill embeddings on existing :Learning nodes.

The consistency gate retrieves prior *approved* neighbours through a vector
index, so items that predate the gate need an ``embedding`` property before they
can be found. Run this once after deploying the gate (and any time you change
``EMBEDDING_MODEL``):

    python scripts/backfill_memory_embeddings.py            # embed missing only
    python scripts/backfill_memory_embeddings.py --all      # re-embed everything
    python scripts/backfill_memory_embeddings.py --dry-run  # count, embed nothing

Idempotent: by default it only touches nodes without an ``embedding``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parents[1] / "hooks"
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from project_common import embed_texts, embeddings_ready, load_mkg_env, neo4j_config  # noqa: E402
from consistency_gate import ensure_memory_vector_indexes  # noqa: E402

BATCH = 64


def _embedding_input(record: dict) -> str:
    text = (record.get("text") or "").strip()
    rationale = (record.get("rationale") or "").strip()
    return f"{text}\n{rationale}".strip() if rationale else text


def _fetch(session, label: str, only_missing: bool) -> list[dict]:
    # Only live memory belongs in the vector index: the gate removes the
    # embedding when it rejects or folds an item, and re-embedding those
    # tombstones here would put them back into neighbour retrieval. Learnings
    # folded into the user profile or compiled into a skill are live: they
    # keep their embedding for deduplication and stay out of recall through
    # the ``consolidated`` flag, so they are embedded here like any other.
    predicate = "n.embedding IS NULL" if only_missing else "true"
    result = session.run(
        f"""
        MATCH (n:{label})
        WHERE n.text IS NOT NULL
          AND n.status IN ['approved', 'candidate']
          AND ({predicate})
        RETURN n.id AS id, n.text AS text, n.rationale AS rationale
        """
    )
    return [dict(r) for r in result]


def _write(session, label: str, rows: list[dict]) -> None:
    session.run(
        f"""
        UNWIND $rows AS row
        MATCH (n:{label} {{id: row.id}})
        SET n.embedding = row.embedding
        """,
        rows=rows,
    )


def backfill(label: str, session, *, only_missing: bool, dry_run: bool) -> tuple[int, int]:
    records = _fetch(session, label, only_missing)
    if dry_run or not records:
        return len(records), 0
    embedded = 0
    for start in range(0, len(records), BATCH):
        chunk = records[start : start + BATCH]
        vectors = embed_texts([_embedding_input(r) for r in chunk])
        payload = [
            {"id": r["id"], "embedding": v}
            for r, v in zip(chunk, vectors)
            if v
        ]
        if payload:
            _write(session, label, payload)
            embedded += len(payload)
    return len(records), embedded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="re-embed every node, not just missing")
    parser.add_argument("--dry-run", action="store_true", help="report counts without embedding")
    args = parser.parse_args()

    load_mkg_env()
    ready, reason = embeddings_ready()
    if not ready and not args.dry_run:
        print(f"Embeddings unavailable: {reason}", file=sys.stderr)
        return 1

    from neo4j import GraphDatabase

    uri, user, password, database = neo4j_config()
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        if not args.dry_run:
            ensure_memory_vector_indexes(driver, database)
        with driver.session(database=database) as session:
            for label in ("Learning",):
                total, embedded = backfill(
                    label, session, only_missing=not args.all, dry_run=args.dry_run
                )
                verb = "would embed" if args.dry_run else "embedded"
                print(f"{label}: {verb} {embedded if not args.dry_run else total} / {total} candidate node(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
