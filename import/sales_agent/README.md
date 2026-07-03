# sales_agent

A self-contained demo **dataset + persona** for the Meta Knowledge Graph: a
sales / customer-success intelligence assistant working a book of ~48
enterprise car-rental customer accounts for RoadFlex (a corporate mobility and
rental-car provider). The minimum setup seeds a blank Neo4j database with the
RoadFlex relationship graph, bootstrap project memory, and a `(:SystemPrompt)`
persona that the SessionStart hook injects. BigQuery/Neocarta and Diffbot are
optional add-ons.

## Layout

| File | Purpose |
|---|---|
| `seed_data.py` | Canonical, deterministic dataset (one source of truth for both stores). |
| `seed_bigquery.py` | Loads `accounts`, `products`, `account_product_usage`, `account_contacts`, `account_renewals`. |
| `run_neocarta.py` | Builds the Neocarta catalog from the seeded BigQuery dataset, then backfills LiteLLM embeddings. |
| `seed_neo4j.py` | Loads `:Account` / `:Product` / `:Contact` / `:CSM` nodes **and the relationships** (`USES_PRODUCT`, `HAS_CONTACT`, `OWNS`). |
| `seed_learnings.py` | Seeds bootstrap `:Learning` nodes (durable facts and decisions alike) so the first session starts with scoped project memory. |
| `seed_system_prompt.py` | Persists `system_prompt.md` to a `(:SystemPrompt)` node. |
| `system_prompt.md` | The sales-assistant persona prompt. |
| `seed_all.py` | Runs the mandatory Neo4j-backed seeders, plus BigQuery/Neocarta when configured and reachable. |

## Seed it

The seeders resolve env the same way the MCP server does: the current
directory's `.env` first, then `~/.config/meta-knowledge-graph/.env`
authoritatively (`override=True`). So either works — a repo-root `./.env` for
checkout/dev, or the user-global config env for an installed plugin. From a
checkout, do not edit `.env.example` as the active config; copy it first if
needed:

```bash
cp .env.example .env
chmod 600 .env
```

When installed as a Claude Code plugin (no checkout), put the same vars in
`~/.config/meta-knowledge-graph/.env` and run the seeders from the plugin cache
dir via `uv run --project "$ROOT" python "$ROOT/import/sales_agent/<seeder>.py"`
(see the `/sales_agent_demo` command for resolving `$ROOT`).

Neo4j and OpenAI are the only mandatory settings for the minimum sales-agent
experience:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=<your-openai-api-key>
```

From the repo root, seed the minimum Neo4j-backed demo:

```bash
uv run python import/sales_agent/seed_neo4j.py
uv run python import/sales_agent/seed_learnings.py
uv run python import/sales_agent/seed_system_prompt.py
```

Diffbot is optional and has no seed step. Add this to the repo-root `./.env` and
restart the MCP server to enable `enhance_entity` and `search_news`:

```bash
DIFFBOT_TOKEN=<your-diffbot-token>
```

BigQuery needs Google auth plus a warehouse seed. Configure application-default
credentials, `GOOGLE_APPLICATION_CREDENTIALS`, or `GCP_SERVICE_ACCOUNT_JSON`,
then set these in the repo-root `./.env`:

```bash
GCP_PROJECT_ID=<your-gcp-project>
BIGQUERY_DATASET_ID=acme_corp
BIGQUERY_MCP_URL=https://bigquery.googleapis.com/mcp
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
# or:
# GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account", ...}'
```

Neocarta also needs the embedding provider key. With the default
`EMBEDDING_MODEL=text-embedding-3-small`, the `OPENAI_API_KEY` above is enough.
If you change `EMBEDDING_MODEL`, add that provider's key to `./.env`.

Then seed the warehouse and catalog:

```bash
uv run python import/sales_agent/seed_bigquery.py
uv run python import/sales_agent/run_neocarta.py
```

This convenience command verifies the mandatory Neo4j connection, runs the
minimum Neo4j-backed seeders, and runs the optional warehouse/catalog seeders
only when GCP env/auth is available:

```bash
uv run python import/sales_agent/seed_all.py
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
