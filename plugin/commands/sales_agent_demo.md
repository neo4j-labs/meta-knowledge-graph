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

## Mutation confirmation guardrail

This command prepares configuration and can run seed/import scripts that write to
Neo4j, BigQuery, and the Neocarta catalog. Before making any change that updates
files or external systems, pause and ask for explicit confirmation. This includes
editing `~/.config/meta-knowledge-graph/.env`, editing the repo `.env`, syncing
env values, running `seed_all.py`, running individual seeders, or re-running a
catalog import.

When confirmation is needed, first summarize:

- Which targets will be changed, e.g. global env, repo `.env`, Neo4j,
  BigQuery dataset, Neocarta catalog.
- Which command(s) you intend to run.
- Whether the action creates/replaces demo data or only reads/verifies state.

Do not proceed until the user replies with an explicit approval such as "yes",
"approved", or "run it". If the user only asks a question or gives an ambiguous
answer, clarify before making changes.

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

## How env resolution works

There are **two separate env files** that serve different purposes:

| File | Read by | When it matters |
|------|---------|-----------------|
| `~/.config/meta-knowledge-graph/.env` | MCP server (plugin mode) | Always — controls which tools mount |
| `<repo-root>/.env` | Seed scripts | Only when running seeds from the checkout |

When MKG is installed as a Claude Code plugin, the MCP server resolves env as:
`MKG_ENV_FILE` → `<active-project>/.env` → `~/.config/meta-knowledge-graph/.env`
(first existing wins). The active project is whichever directory Claude Code is
open in — **not** the MKG repo checkout. So the repo `.env` is irrelevant to
the running MCP server unless `MKG_ENV_FILE` points to it.

**Practical consequence:** if Diffbot, BigQuery, or Neocarta vars are only in
the repo `.env` and not in `~/.config/meta-knowledge-graph/.env`, those tools
will not mount in the MCP server even after seeding.

## Step 1 - Ensure the global config env exists

Create `~/.config/meta-knowledge-graph/.env` if it does not exist:

```bash
mkdir -p ~/.config/meta-knowledge-graph
touch ~/.config/meta-knowledge-graph/.env
chmod 600 ~/.config/meta-knowledge-graph/.env
```

## Step 2 - Configure required vars (Neo4j + OpenAI)

Add or update these lines in `~/.config/meta-knowledge-graph/.env`:

```bash
# Required: Neo4j graph used by MKG and the sales-agent demo.
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-neo4j-password>
NEO4J_DATABASE=neo4j

# Required: LLM calls and embeddings (text-embedding-3-small by default).
OPENAI_API_KEY=<your-openai-api-key>

# Optional model override for memory extraction / prompt consolidation.
# LLM_MODEL=gpt-5.4-mini
```

## Step 3 - Configure optional Diffbot

To enable live company enrichment and recent-news research, add to
`~/.config/meta-knowledge-graph/.env`:

```bash
# Optional: Diffbot live news / firmographic enrichment.
DIFFBOT_TOKEN=<your-diffbot-token>
```

Diffbot has **no seed step**. After restarting the MCP server/session,
`DIFFBOT_TOKEN` makes the Diffbot tools available:

- `enhance_entity` for company/person firmographics.
- `search_news` for recent company and trigger-event articles.

## Step 4 - Configure optional BigQuery + Neocarta

To seed the RoadFlex warehouse and build the Neocarta semantic catalog, add to
`~/.config/meta-knowledge-graph/.env`:

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

Google auth must be reachable by both the seed scripts and the runtime MCP
tools. The user can use either ADC (`gcloud auth application-default login`),
`GOOGLE_APPLICATION_CREDENTIALS`, or `GCP_SERVICE_ACCOUNT_JSON`. The principal
needs BigQuery read + job-run permissions; the warehouse seeder also
creates/replaces tables in `GCP_PROJECT_ID.BIGQUERY_DATASET_ID`.

## Step 5 - Confirm, then seed the demo

The seed scripts and their dependencies ship inside the plugin, so seeding works
from an installed plugin with **no repo checkout**. They read env the same way
the MCP server does: the current directory's `.env` first, then
`~/.config/meta-knowledge-graph/.env` authoritatively. So if you configured the
global config env in Steps 2-4, the seeders pick it up automatically — no repo
`.env` and no syncing required.

Resolve the plugin install dir (the same fallback chain the MCP server uses),
then run every `uv`/seed command against it so the bundled scripts and venv
resolve regardless of your current directory:

```bash
ROOT="${CLAUDE_PLUGIN_ROOT}"
[ -f "$ROOT/pyproject.toml" ] || ROOT="$(ls -d "${CLAUDE_CONFIG_DIR:-$HOME/.claude}"/plugins/cache/*/meta-knowledge-graph/*/ 2>/dev/null | sort -V | tail -1)"
[ -f "$ROOT/pyproject.toml" ] || ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"
```

Before running any seed/import command, ask for confirmation and name the exact
targets. The orchestrator always runs the mandatory Neo4j seeders and only runs
the optional warehouse/catalog seeders when GCP env/auth is available, so phrase
the confirmation in concrete terms, for example:

> This will seed/update the RoadFlex demo in Neo4j and, if GCP env/auth is
> available, create or replace the BigQuery demo tables and rebuild the Neocarta
> catalog. Should I run `uv run --project "$ROOT" python "$ROOT/import/sales_agent/seed_all.py"` now?

```bash
uv run --project "$ROOT" python "$ROOT/import/sales_agent/seed_all.py"
```

Or run individual seeders:

```bash
# Minimum Neo4j-backed demo:
uv run --project "$ROOT" python "$ROOT/import/sales_agent/seed_neo4j.py"
uv run --project "$ROOT" python "$ROOT/import/sales_agent/seed_learnings.py"
uv run --project "$ROOT" python "$ROOT/import/sales_agent/seed_system_prompt.py"

# Optional BigQuery + Neocarta (requires GCP env/auth above):
uv run --project "$ROOT" python "$ROOT/import/sales_agent/seed_bigquery.py"
uv run --project "$ROOT" python "$ROOT/import/sales_agent/run_neocarta.py"
```

> **Repo checkout?** If you're developing from a git clone, `$ROOT` resolves to
> the checkout and these commands still work; `uv run python import/sales_agent/seed_all.py`
> from the repo root remains equivalent.

### Project scoping of bootstrap memory

`seed_learnings.py` attaches its `:Learning` nodes to the **same
project the SessionStart hook will scope recall to**, resolved the same way the
hook does: it honors `MKG_PROJECT_ID` if the hooks have pinned it, otherwise
derives the project from your active project dir (`CLAUDE_PROJECT_DIR` → nearest
repo-root name), and falls back to the MKG checkout folder name. Because the
command runs the seeder with `CLAUDE_PROJECT_DIR` set to your active project,
the bootstrap memory lands on the project the running agent actually queries —
not the plugin cache dir.

To target a specific project explicitly, pass `--project`:

```bash
uv run --project "$ROOT" python "$ROOT/import/sales_agent/seed_learnings.py" --project "my-sales-workspace"
```

The seeder prints the resolved `(:Project {id: ...})` so you can confirm the
scope before relying on it.

## Step 6 - Restart and verify tools

Restart the MCP server/session after changing `~/.config/meta-knowledge-graph/.env`;
env changes do not apply to already-running servers.

Expected tools by configuration:

- Minimum Neo4j setup: graph/memory tools such as schema/read Cypher and project
  memory.
- `DIFFBOT_TOKEN` set: `enhance_entity`, `search_news`.
- `BIGQUERY_MCP_URL` set with Google auth: `bigquery_execute_query`.
- `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID`, and `OPENAI_API_KEY` set:
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

- **No Diffbot tools:** `DIFFBOT_TOKEN` is missing from
  `~/.config/meta-knowledge-graph/.env`, or the MCP server was not restarted.
- **No BigQuery tool:** `BIGQUERY_MCP_URL` is missing from the global config
  env, Google auth is unavailable, or the server needs a restart.
- **No `neocarta_*` tools:** one of the Neocarta mount gates failed —
  `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID`, or `OPENAI_API_KEY` is missing from
  `~/.config/meta-knowledge-graph/.env`.
- **Tools mount but optional vars are in repo `.env` only:** the MCP server
  reads the global config env, not the repo `.env`. Copy the missing vars to
  `~/.config/meta-knowledge-graph/.env` and restart.
- **Neocarta tools mount but searches return nothing:** the catalog was not
  seeded, or it was seeded for a different dataset. Re-run
  `uv run --project "$ROOT" python "$ROOT/import/sales_agent/run_neocarta.py"`
  after confirming the env.
- **Auth errors during seed:** ADC/service-account credentials are not reachable
  or lack permission on the dataset.
- **Seeder can't find env / missing-var errors:** the seeders read the current
  directory's `.env` then `~/.config/meta-knowledge-graph/.env`. Put the required
  vars in the global config env (Steps 2-4); a repo `.env` is optional and only
  used when present in the directory you run from.
