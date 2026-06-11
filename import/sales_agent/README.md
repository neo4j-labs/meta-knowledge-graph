# sales_agent

A self-contained demo **dataset + persona** for the Meta Knowledge Graph: a
sales / customer-success intelligence assistant working a book of ~48
enterprise car-rental customer accounts for RoadFlex (a corporate mobility and
rental-car provider). It seeds a BigQuery warehouse, a Neocarta catalog in
Neo4j, a Neo4j relationship graph, and a `(:SystemPrompt)` persona that the
SessionStart hook injects.

## Layout

| File | Purpose |
|---|---|
| `seed_data.py` | Canonical, deterministic dataset (one source of truth for both stores). |
| `seed_bigquery.py` | Loads `accounts`, `products`, `account_product_usage`, `account_contacts`, `account_renewals`. |
| `run_neocarta.py` | Builds the Neocarta catalog from the seeded BigQuery dataset, then backfills LiteLLM embeddings. |
| `seed_neo4j.py` | Loads `:Account` / `:Product` / `:Contact` / `:CSM` nodes **and the relationships** (`USES_PRODUCT`, `HAS_CONTACT`, `OWNS`). |
| `seed_learnings.py` | Seeds bootstrap `:Learning` / `:Decision` nodes so the first session starts with scoped project memory. |
| `seed_system_prompt.py` | Persists `system_prompt.md` to a `(:SystemPrompt)` node. |
| `system_prompt.md` | The sales-assistant persona prompt. |
| `seed_all.py` | Runs all of the above, including Neocarta. |

## Seed it

Requires the repo `.env` (Neo4j + GCP credentials). From the repo root:

```bash
# everything (data + catalog + active default persona)
uv run python import/sales_agent/seed_all.py

# or run pieces individually
uv run python import/sales_agent/seed_bigquery.py
uv run python import/sales_agent/run_neocarta.py
uv run python import/sales_agent/seed_neo4j.py
uv run python import/sales_agent/seed_learnings.py
uv run python import/sales_agent/seed_system_prompt.py
```

Re-running is safe: the data is generated from a fixed seed, BigQuery tables are
dropped/recreated, Neocarta refreshes the catalog from the current warehouse
metadata, and Neo4j MERGEs on natural keys (seed-owned `USES_PRODUCT` and
`HAS_CONTACT` edges are refreshed so a re-run reflects the current dataset).

Preview the generated data without touching any database:

```bash
uv run python import/sales_agent/seed_data.py
```

## What makes the data useful

- **Internally consistent firmographics.** `employee_count_band`, `seats_total`,
  `arr_band_usd`, and `arr_usd` all derive from one `size_class` + real usage, so
  they reconcile instead of contradicting each other.
- **Trajectories.** Every account is `expanding` / `steady` / `at_risk` / `new`,
  which shapes its monthly rental usage — so expansion, churn/renewal-risk, and
  ramping new logos are all detectable (not "everything grows forever").
- **Actionable.** Named contacts (champion / economic buyer / technical / exec)
  and per-account renewal dates power "who do I email" and "what renews in 90 days".
- **Graph-native.** `USES_PRODUCT` edges (with utilization/revenue) and `OWNS`
  ownership enable multi-hop questions that are awkward in SQL.
- **Live external signal.** Real company names + domains, so Diffbot
  `enhance_entity` / `search_news` return real hits.

## System prompt

`inject_system_prompt` loads `(:SystemPrompt {name: 'default'})` and, only if
that node is missing, falls back to a generic tool-agnostic bootstrap prompt
baked into the hook. `seed_system_prompt.py` writes the sales persona under
`default`, making it active immediately; the generic fallback stays in the
hook for unseeded environments.
