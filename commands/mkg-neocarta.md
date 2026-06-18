---
description: Set up and seed the Neocarta data catalog (BigQuery semantic layer) for MKG — configure the GCP/embedding env, authenticate to Google, build the catalog into Neo4j with embeddings, and verify the neocarta_* MCP tools mount.
argument-hint: [optional BigQuery dataset, e.g. my_project.my_dataset]
---

# MKG Neocarta Setup

Help the user stand up **Neocarta**, MKG's optional data-catalog / semantic
layer over BigQuery. Walk them through the config, GCP auth, and the seed, then
verify the tools mount. Be concrete; never echo secret values back in plaintext.

Optional argument — a BigQuery dataset to target as `project.dataset`: **$ARGUMENTS**

## What Neocarta is, and what MKG supports today

Neocarta introspects a **BigQuery** dataset, writes catalog metadata into the
**same Neo4j** graph MKG already uses (`:Database` / `:Schema` / `:Table` /
`:Column`, plus optional `:BusinessTerm` nodes), populates **vector embeddings**
over those nodes, and exposes the result as a semantic-retrieval MCP server. MKG
mounts it as a proxy under the `neocarta` prefix, so the agent can discover and
search warehouse structure for query generation, query routing, and data
discovery — without dumping the whole schema into context.

MKG mounts these tools (full names are `mcp__meta-knowledge-graph__<tool>`):

- **Catalog (always mounted):** `neocarta_list_schemas`,
  `neocarta_list_tables_by_schema`, `neocarta_get_full_metadata_schema`.
- **Retrieval (one strategy per label, chosen by which indexes exist):** a Table
  tool and a Column tool. The priority is **business-term hybrid → hybrid →
  vector → full-text**, so e.g. a vector-seeded catalog mounts
  `neocarta_get_context_by_table_vector_search` and
  `neocarta_get_context_by_column_vector_search`; a catalog that also has
  `:BusinessTerm` nodes + their full-text index gets the
  `..._business_term_hybrid_search` variants instead.
- **Schema vector search:** `neocarta_get_context_by_schema_and_table_vector_search`
  when a `Schema` vector index is present.

The exact retrieval tool set therefore depends on what the seed created — that's
expected, not a bug.

## Two gates before the tools appear

The MCP server mounts Neocarta only when **both** hold (otherwise it logs
`Neocarta MCP proxy not mounted (missing: ...)` and the `neocarta_*` tools never
show up):

1. **Warehouse settings present:** `GCP_PROJECT_ID` **and** `BIGQUERY_DATASET_ID`.
2. **Embedding-model credentials present:** the provider key for `EMBEDDING_MODEL`
   (default `text-embedding-3-small` → `OPENAI_API_KEY`). The required key follows
   the configured model — point `EMBEDDING_MODEL` at a Cohere/Gemini/etc. model
   and the matching provider key is what's checked.

Mounting the tools and **seeding** the catalog are separate: the tools can mount
against an empty catalog and return nothing useful until the seed has run.

## Step 1 — Configure the environment

Add to the same credentials file MKG already uses,
`~/.config/meta-knowledge-graph/.env` (or the repo-root `.env` in a checkout),
the user editing it in their own terminal for secrets:

```bash
# Required to mount Neocarta
GCP_PROJECT_ID=<your-gcp-project>          # also the project bigquery_execute_query targets
BIGQUERY_DATASET_ID=<your-dataset>         # e.g. acme_corp
OPENAI_API_KEY=<key for EMBEDDING_MODEL's provider>   # default embedder is OpenAI

# Optional
# EMBEDDING_MODEL=text-embedding-3-small   # any litellm embedding model; sets which key is required
# EMBEDDING_DIMENSIONS=1536
# BIGQUERY_REGION=US                       # also the dataset location for the demo seeder
# GCP_BILLING_PROJECT_ID=<billing-project> # defaults to GCP_PROJECT_ID
```

If `$ARGUMENTS` named a dataset as `project.dataset`, map the left side to
`GCP_PROJECT_ID` and the right side to `BIGQUERY_DATASET_ID`. Offer to write these
**non-secret** lines into the env file directly; leave the API key for the user.

## Step 2 — Authenticate to Google

The seeder and the runtime proxy both need Google credentials. Pick one (the user
runs auth commands in their own terminal):

- **Application Default Credentials (simplest for local):**
  `gcloud auth application-default login`
- **Service account file:** set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json`.
- **Inline service account JSON:** set `GCP_SERVICE_ACCOUNT_JSON='{...}'` — MKG and
  the demo seeder write it to a temp `600` file and point Google libs at it.

The service account / user needs BigQuery read + job-run on the dataset (the demo
seeder also creates tables, so it needs write there).

## Step 3 — Seed the catalog

Seeding = introspect BigQuery → write catalog nodes to Neo4j → populate
embeddings. It **must run after** the BigQuery tables and column descriptions
exist, because Neocarta reads them.

- **Demo (RoadFlex sales agent):** from the repo, seed the warehouse first, then
  build the catalog:

  ```bash
  uv run python import/sales_agent/seed_bigquery.py   # creates the demo BQ tables
  uv run python import/sales_agent/run_neocarta.py     # introspect + embed into Neo4j
  ```

  Or let the orchestrator do both — it preflights GCP env/auth and only runs the
  BigQuery + Neocarta steps when they're reachable, skipping them with a clear
  message otherwise:

  ```bash
  uv run python import/sales_agent/seed_all.py
  ```

- **Your own existing dataset:** `run_neocarta.py` reads `GCP_PROJECT_ID` /
  `BIGQUERY_DATASET_ID` from the env, so once those point at your dataset it
  introspects and embeds **that** dataset — no demo `seed_bigquery.py` needed,
  since your tables already exist. (`run_neocarta.py` lives under
  `import/sales_agent/` today; that's the only catalog seeder the repo ships.)

You may run these `uv run` seeders on the user's behalf and report the output;
don't run the interactive `gcloud`/secret steps for them.

## Step 4 — Verify

1. **Restart the session** so the MCP server re-reads the env and re-evaluates the
   mount gates (env changes don't apply mid-session).
2. Confirm the `mcp__meta-knowledge-graph__neocarta_*` tools are now available
   (at minimum the three catalog tools).
3. Smoke-test: call `neocarta_list_schemas`, then a context search such as
   `neocarta_get_context_by_table_vector_search` with a natural-language query and
   confirm it returns ranked tables/columns from the seeded dataset.

## Troubleshooting

- **No `neocarta_*` tools after restart** → one of the two gates failed. Check the
  server log line `Neocarta MCP proxy not mounted (missing: ...)`: it names the
  missing warehouse vars and/or `credentials for embedding model '<model>'`.
- **Tools mount but searches return nothing** → the catalog hasn't been seeded (or
  was seeded for a different dataset). Re-run Step 3 against the right
  `GCP_PROJECT_ID` / `BIGQUERY_DATASET_ID`.
- **Only catalog tools, no retrieval tools** → no search index was created for
  Table/Column; the embedding step didn't run or failed. Re-run `run_neocarta.py`
  and confirm the `populating Neocarta embeddings (...)` step completes.
- **Auth errors during seed** → ADC/service-account isn't reachable or lacks
  BigQuery permission on the dataset; re-do Step 2.
