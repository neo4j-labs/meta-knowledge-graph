"""Sales-assistant demo dataset + seeders for the Meta Knowledge Graph.

This package is self-contained so that other datasets/personas can live beside
it under ``import/`` as sibling folders (e.g. ``import/support_agent/``)
following the same layout:

    seed_data.py          canonical, deterministic dataset
    seed_bigquery.py      load the warehouse tables
    seed_neo4j.py         load nodes + relationships (the graph layer)
    seed_system_prompt.py persist the persona's (:SystemPrompt) to Neo4j
    system_prompt.md       the persona prompt itself
    seed_all.py           one-shot orchestrator
"""

from __future__ import annotations

from .seed_data import build_seed_dataset, latest_usage_by_account

__all__ = ["build_seed_dataset", "latest_usage_by_account"]
