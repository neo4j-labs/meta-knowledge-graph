# sales_agent

A self-contained demo **dataset + persona** for the Meta Knowledge Graph: a
sales / customer-success intelligence assistant working a book of 50 Atlas
accounts. It seeds a BigQuery warehouse, a Neo4j relationship graph, and a
`(:SystemPrompt)` persona that the SessionStart hook injects.

## Layout

| File | Purpose |
|---|---|
| `seed_data.py` | Canonical, deterministic dataset (one source of truth for both stores). |
| `seed_bigquery.py` | Loads `accounts`, `products`, `account_product_usage`, `account_contacts`, `account_renewals`. |
| `seed_neo4j.py` | Loads `:Account` / `:Product` / `:Contact` / `:CSM` nodes **and the relationships** (`USES_PRODUCT`, `HAS_CONTACT`, `OWNS`). |
| `seed_system_prompt.py` | Persists `system_prompt.md` to a `(:SystemPrompt)` node. |
| `system_prompt.md` | The sales-assistant persona prompt. |
| `seed_all.py` | Runs all of the above. |

## Seed it

Requires the repo `.env` (Neo4j + GCP credentials). From the repo root:

```bash
# everything (data + persona under name 'sales_agent')
uv run python import/sales_agent/seed_all.py

# everything, and make the persona the ACTIVE prompt (writes the 'default' node too)
uv run python import/sales_agent/seed_all.py --default

# or run pieces individually
uv run python import/sales_agent/seed_bigquery.py
uv run python import/sales_agent/seed_neo4j.py
uv run python import/sales_agent/seed_system_prompt.py --default
```

Re-running is safe: the data is generated from a fixed seed, BigQuery tables are
dropped/recreated, and Neo4j MERGEs on natural keys (seed-owned `USES_PRODUCT`
and `HAS_CONTACT` edges are refreshed so a re-run reflects the current dataset).

Preview the generated data without touching any database:

```bash
uv run python import/sales_agent/seed_data.py
```

## What makes the data useful

- **Internally consistent firmographics.** `employee_count_band`, `seats_total`,
  `arr_band_usd`, and `arr_usd` all derive from one `size_class` + real usage, so
  they reconcile instead of contradicting each other.
- **Trajectories.** Every account is `expanding` / `steady` / `at_risk` / `new`,
  which shapes its monthly usage — so expansion, churn/renewal-risk, and ramping
  new logos are all detectable (not "everything grows forever").
- **Actionable.** Named contacts (champion / economic buyer / technical / exec)
  and per-account renewal dates power "who do I email" and "what renews in 90 days".
- **Graph-native.** `USES_PRODUCT` edges (with utilization/revenue) and `OWNS`
  ownership enable multi-hop questions that are awkward in SQL.
- **Live external signal.** Real company names + domains, so Diffbot
  `enhance_entity` / `search_news` return real hits.

## System prompt

`inject_system_prompt` loads `(:SystemPrompt {name: $MKG_PROMPT_NAME})` (default
name `default`) and, only if that node is missing, falls back to a generic
tool-agnostic bootstrap prompt baked into the hook. So:

- The generic fallback stays in the hook for unseeded environments.
- `seed_system_prompt.py` writes the sales persona under `sales_agent`; pass
  `--default` to also write `default` and make it active with no env change.
- To run several personas, seed each under its own name and switch with
  `MKG_PROMPT_NAME`.

## Adding another dataset / persona

Create a sibling folder under `import/` (e.g. `import/support_agent/`) with the
same files. Keep each dataset self-contained; if shared helpers accumulate,
factor them into a small common module then.
