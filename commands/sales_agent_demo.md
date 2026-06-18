---
description: Set up the RoadFlex sales-agent demo for MKG - configure the repo .env, seed Neo4j, optionally enable Diffbot and BigQuery/Neocarta, and verify the MCP tools mount.
argument-hint: [optional BigQuery dataset, e.g. my_project.my_dataset]
---

# Sales Agent Demo Setup

Help the user stand up the **RoadFlex sales-agent demo** in MKG: a
sales/customer-success assistant backed by Neo4j memory and graph data, with
optional **Diffbot** live enrichment/news and **BigQuery + Neocarta** warehouse
catalog search. Be concrete; never echo secret values back in plaintext.

Optional argument - a BigQuery dataset to target as `project.dataset`:
**$ARGUMENTS**

## What this demo includes

- **Required minimum:** Neo4j graph seed, bootstrap learnings, and the RoadFlex
  sales persona system prompt.
- **Optional Diffbot:** live firmographic enrichment and recent news through
  `enhance_entity` and `search_news`.
- **Optional BigQuery:** seeded RoadFlex warehouse tables queried through
  `bigquery_execute_query`.
- **Optional Neocarta:** semantic catalog over the BigQuery dataset, exposed as
  `neocarta_*` tools so the agent can discover tables/columns before writing
  SQL.

## The exact env file to update

For this repo checkout, update the **repo-root `.env` file**:

```bash
./.env
```

Do **not** edit `.env.example` as the active config. Do **not** put these demo
settings only in a shell profile. The sales-agent seeders and MKG hooks load the
repo-root `.env`, and in a repo checkout it takes precedence over the
user-global MKG env file.

If `./.env` does not exist yet, create it from the template:

```bash
cp .env.example .env
chmod 600 .env
```

You may offer to write non-secret lines into `./.env` for the user, such as
`GCP_PROJECT_ID` or `BIGQUERY_DATASET_ID`. Leave secret values for the user to
fill in, and never print existing secrets.

## Step 1 - Configure the required minimum

Add or update these lines in `./.env`:

```bash
# Required: Neo4j graph used by MKG and the sales-agent demo.
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-neo4j-password>
NEO4J_DATABASE=neo4j

# Required: LLM calls through LiteLLM.
# The default local/Codex path uses OpenAI unless LLM_MODEL is changed.
OPENAI_API_KEY=<your-openai-api-key>

# Optional model override for memory extraction / prompt consolidation.
# LLM_MODEL=gpt-5.4-mini
```

## Step 2 - Configure optional Diffbot

To enable live company enrichment and recent-news research, add this to
`./.env`:

```bash
# Optional: Diffbot live news / firmographic enrichment.
DIFFBOT_TOKEN=<your-diffbot-token>
```

Diffbot has **no seed step**. After restarting the MCP server/session,
`DIFFBOT_TOKEN` makes the Diffbot tools available:

- `enhance_entity` for company/person firmographics.
- `search_news` for recent company and trigger-event articles.

## Step 3 - Configure optional BigQuery + Neocarta

To seed the RoadFlex warehouse and build the Neocarta semantic catalog, add or
update these lines in `./.env`:

```bash
# Optional: BigQuery warehouse + Neocarta catalog.
GCP_PROJECT_ID=<your-gcp-project>
BIGQUERY_DATASET_ID=acme_corp
BIGQUERY_MCP_URL=https://bigquery.googleapis.com/mcp

# Choose one Google auth method:
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
# or:
# GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account", ...}'

# Optional BigQuery / embedding knobs:
# GCP_BILLING_PROJECT_ID=<billing-project>  # defaults to GCP_PROJECT_ID
# BIGQUERY_REGION=US
# EMBEDDING_MODEL=text-embedding-3-small
# EMBEDDING_DIMENSIONS=1536
```

If `$ARGUMENTS` is `project.dataset`, map it exactly:

- `GCP_PROJECT_ID=project`
- `BIGQUERY_DATASET_ID=dataset`

Neocarta also needs credentials for the embedding model provider. With the
default `EMBEDDING_MODEL=text-embedding-3-small`, the `OPENAI_API_KEY` from the
required section is enough. If the user changes `EMBEDDING_MODEL` to another
provider, add that provider's key to `./.env` too.

Google auth must be reachable by both the seed scripts and runtime tools. The
user can use either:

- Application Default Credentials: `gcloud auth application-default login`
- `GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json`
- `GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account", ...}'`

The Google principal needs BigQuery read + job-run permissions. The demo
warehouse seeder also creates/replaces tables in `GCP_PROJECT_ID.BIGQUERY_DATASET_ID`.

## Step 4 - Seed the demo

Minimum Neo4j-backed demo:

```bash
uv run python import/sales_agent/seed_neo4j.py
uv run python import/sales_agent/seed_learnings.py
uv run python import/sales_agent/seed_system_prompt.py
```

Optional BigQuery + Neocarta seed, after the BigQuery env/auth above is set:

```bash
uv run python import/sales_agent/seed_bigquery.py
uv run python import/sales_agent/run_neocarta.py
```

Or run the orchestrator, which always runs the mandatory Neo4j seeders and only
runs the optional warehouse/catalog seeders when GCP env/auth is available:

```bash
uv run python import/sales_agent/seed_all.py
```

## Step 5 - Restart and verify tools

Restart the MCP server/session after changing `./.env`; env changes do not apply
to already-running servers.

Expected tools by configuration:

- Minimum Neo4j setup: graph/memory tools such as schema/read Cypher and project
  memory.
- `DIFFBOT_TOKEN` set: `enhance_entity`, `search_news`.
- `BIGQUERY_MCP_URL` set with Google auth: `bigquery_execute_query`.
- `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID`, and embedding provider key set:
  `neocarta_*` catalog/search tools.

Smoke tests:

1. Ask for the current project memory or Neo4j schema.
2. If Diffbot is enabled, call `enhance_entity` for a known account domain or
   `search_news` for a company.
3. If BigQuery is enabled, call `bigquery_execute_query` against the configured
   dataset.
4. If Neocarta is enabled, call `neocarta_list_schemas`, then a table/column
   context search for a business phrase such as "renewal risk".

## Troubleshooting

- **No Diffbot tools:** `DIFFBOT_TOKEN` is missing or the MCP server was not
  restarted after editing `./.env`.
- **No BigQuery tool:** `BIGQUERY_MCP_URL` is missing, Google auth is unavailable,
  or the server needs a restart.
- **No `neocarta_*` tools:** one of the Neocarta mount gates failed:
  `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID`, or the embedding provider key is
  missing.
- **Neocarta tools mount but searches return nothing:** the catalog was not
  seeded, or it was seeded for a different dataset. Re-run
  `uv run python import/sales_agent/run_neocarta.py` after confirming `./.env`.
- **Auth errors during seed:** ADC/service-account credentials are not reachable
  or lack permission on the dataset.
