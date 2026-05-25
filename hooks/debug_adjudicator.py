#!/usr/bin/env python3
"""One-shot diagnostic: rebuild and inspect a process_project LLM call.

Usage: python hooks/debug_adjudicator.py --mode session --session-id <id>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from process_project import (  # noqa: E402
    _event_corpus,
    _search_query,
    build_memory_adjudication_prompt,
)
from project_common import (  # noqa: E402
    ProjectRef,
    fetch_project_decisions,
    fetch_project_learnings,
    load_dotenv,
    neo4j_config,
)


def fetch_all_session_events(session, session_id: str):
    records = session.run(
        """
        MATCH (s:Session {session_id: $session_id})-[:HAS_EVENT]->(e:Event)
        RETURN properties(e) AS event
        ORDER BY e.timestamp
        """,
        session_id=session_id,
    )
    return [dict(r["event"]) for r in records]


def fetch_events_for_processing(session, processing_id: str):
    records = session.run(
        """
        MATCH (:ProjectProcessing {id: $processing_id})-[:PROCESSED_EVENT]->(e:Event)
        RETURN properties(e) AS event
        ORDER BY e.timestamp
        """,
        processing_id=processing_id,
    )
    return [dict(r["event"]) for r in records]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["turn", "session"], required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--processing-id")
    parser.add_argument("--project-id", default="meta-knowledge-graph")
    args = parser.parse_args()
    if not args.session_id and not args.processing_id:
        parser.error("provide --session-id or --processing-id")

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    from neo4j import GraphDatabase

    project = ProjectRef(id=args.project_id, name="Meta Knowledge Graph")
    uri, user, password, database = neo4j_config()

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as sess:
            if args.processing_id:
                events = fetch_events_for_processing(sess, args.processing_id)
            else:
                events = fetch_all_session_events(sess, args.session_id)
            print(f"[debug] events: {len(events)}", file=sys.stderr)
            corpus = _event_corpus(events)
            print(f"[debug] corpus chars: {len(corpus)}", file=sys.stderr)
            search_query = _search_query(corpus)
            similar_learnings = fetch_project_learnings(
                sess, project_id=project.id, query=search_query,
                statuses=["approved", "candidate"], limit=8,
            )
            similar_decisions = fetch_project_decisions(
                sess, project_id=project.id, query=search_query, limit=8,
            )
            print(f"[debug] similar_learnings: {len(similar_learnings)}", file=sys.stderr)
            print(f"[debug] similar_decisions: {len(similar_decisions)}", file=sys.stderr)

    prompt = build_memory_adjudication_prompt(
        project, args.mode, events, similar_learnings, similar_decisions,
    )

    out_dir = Path("/tmp/mkg-debug")
    out_dir.mkdir(exist_ok=True)
    prompt_path = out_dir / f"prompt_{args.mode}.txt"
    prompt_path.write_text(prompt)
    print(f"[debug] prompt saved: {prompt_path} ({len(prompt)} chars)", file=sys.stderr)

    if not os.environ.get("OPENAI_API_KEY"):
        print("[debug] OPENAI_API_KEY unset; skipping LLM call", file=sys.stderr)
        return 0

    from openai import OpenAI

    model = os.environ.get("MKG_LEARNING_MODEL") or os.environ.get(
        "LLM_MODEL", "gpt-5.4-mini",
    )
    print(f"[debug] calling model: {model}", file=sys.stderr)
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You curate project memory for an agent. Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    finish = response.choices[0].finish_reason
    usage = response.usage

    response_path = out_dir / f"response_{args.mode}.txt"
    response_path.write_text(content)
    print(f"[debug] response saved: {response_path} ({len(content)} chars)", file=sys.stderr)
    print(f"[debug] finish_reason: {finish}", file=sys.stderr)
    print(
        f"[debug] tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}",
        file=sys.stderr,
    )

    print("\n=== RAW RESPONSE ===")
    print(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
