# Meta Knowledge Graph: Self-Learning Memory for AI Agents

The **Meta Knowledge Graph** (MKG) is a self-improving, graph-structured memory
layer for AI agents, backed by Neo4j. It is **harness-agnostic**: the Neo4j
store and the MCP server plug into any MCP-capable harness, and the
capture/injection scripts ride on whatever lifecycle hooks the harness exposes.
This repo ships ready-made wiring for **Claude Code**
([`.claude/settings.json`](.claude/settings.json)) and **Codex**
([`.codex/config.toml`](.codex/config.toml) plus
[`.codex/hooks.json`](.codex/hooks.json)); plugging a custom harness in means
pointing its lifecycle events at the same scripts.

It ships as two halves that form a closed capture-and-recall loop:

- **MCP server (`meta-knowledge-graph`)** — surfaces project memory, the underlying
  graph, the persisted system prompt, and (optionally) a data catalog and
  warehouse to the agent as tools.
- **Lifecycle hooks** — plain Python scripts that log every session event,
  inject scoped project context on prompt submit, and run an LLM memory extraction processor at Stop
  (and SessionEnd when the harness exposes it)
  that distills durable `:Learning` / `:Decision` / `:SystemPromptSuggestion` /
  `:MemoryExtractionPromptSuggestion`
  candidates from what just happened.

The hooks write to the same graph the MCP tools read from, so each new session
starts with the most relevant prior learnings already injected. Two
self-improvement loops run on top: candidate suggestions are periodically folded
into the live persisted **system prompt** and into the live **memory extraction
prompt**, both versioned in the graph with rollback snapshots.

A complete end-to-end demo — a B2B sales / customer-success assistant for an
enterprise car-rental provider — ships in the repo; see
[Sales agent use case](#sales-agent-use-case) for setup.

## Architecture

### MCP server

Mounted under the `meta-knowledge-graph` prefix, in four groups:

| Group | Tools | Mounted when |
|---|---|---|
| Project memory & graph | `project_get_context`, `project_add_learning`, `neo4j_get_schema`, `neo4j_read_cypher` | Always. |
| Diffbot research | `search_news`, `enhance_entity` | `DIFFBOT_TOKEN` is set. |
| BigQuery warehouse | `bigquery_execute_query` | `BIGQUERY_MCP_URL` is set. |
| Neocarta data catalog | `neocarta_*` | `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID`, and `OPENAI_API_KEY` are set. |

### Hooks

The hook scripts are harness-agnostic: each reads the lifecycle event's JSON
payload from stdin, and `log_event.py` tags events with `--client <name>`
(default `claude_code`), so sessions record which harness produced them. Any
harness that exposes lifecycle hooks — Claude Code, Codex, Cursor, or a custom
loop — can drive the same scripts. The table below shows the Claude Code
wiring from this repo's own `.claude/settings.json`. All hooks swallow their
own exceptions so a Neo4j outage never blocks the session. A few scripts
(`inject_system_prompt.py`, `log_event.py`, `seed_system_prompt.py`) are
mirrored as identical copies under `.claude/hooks/`; the canonical versions
live in `hooks/`.

| Hook event | Script | Behavior |
|---|---|---|
| `SessionStart` (`startup\|resume\|clear`) | `hooks/inject_system_prompt.py` | Loads `(:SystemPrompt {name: 'default'})` from Neo4j and injects it. If the node is missing, injects a tool-agnostic bootstrap prompt telling the agent to discover its tools, recall project memory, and persist a refined system prompt back to Neo4j so the next session skips the fallback. The injection log keeps only a content hash + summary on the `:SystemPromptInjection` node and links it to its source via `[:OF_PROMPT]→(:SystemPrompt)` instead of copying the prompt text. If the same prompt content was already injected into the same session, the hook skips the duplicate (unless the context was wiped by `clear`/`compact`). |
| `SessionStart` (`startup\|resume\|clear`), `UserPromptSubmit` | `hooks/inject_project_context.py` | Fulltext-ranks `:Learning` and `:Decision` against the new prompt and injects the top hits scoped to the current project. Every injected item is linked to the session via `[:INJECTED_IN]`, and items already injected earlier in the same session are excluded from retrieval, so a conversation never receives the same learning or decision twice. Marks served learnings as used. |
| `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Notification`, `Stop`, `SubagentStop`, `PreCompact`, `SessionEnd` | `hooks/log_event.py` | Persists each event as a `:SessionEvent` node under the current `:Session`, threaded by `:NEXT`. This is the corpus the memory extraction processor later reads. |
| `Stop` (`--mode turn`), `SessionEnd` (`--mode session`) | `hooks/process_project.py` | Runs in the background. Pulls the session's unprocessed events, loads the persisted `(:MemoryExtractionPrompt {name: 'default'})` template from Neo4j (seeding the default if needed), builds a tail-preserving corpus, fetches the closest existing learnings/decisions, and asks an LLM to return create/update/ignore actions per category. System-prompt suggestions are reserved for rare operating-principle changes or explicit/reinforced high-level user preferences and interests; memory-extraction-prompt suggestions are reserved for improving future extraction quality. Writes new `:Learning` / `:Decision` / `:SystemPromptSuggestion` / `:MemoryExtractionPromptSuggestion` nodes with status `candidate`. |
| `Stop` | `hooks/apply_system_prompt.py` | Rate-limited rebuild of the live `(:SystemPrompt)`. Runs in the background on every Stop but only acts when at least `MKG_PROMPT_REBUILD_MIN_SUGGESTIONS` candidate suggestions are pending **and** `MKG_PROMPT_REBUILD_MIN_HOURS` have passed since the last rebuild (gate + claim in one conditional write, so concurrent Stops can't double-rebuild). Seeds the prompt node from the default if it doesn't exist, snapshots the previous content as `(:SystemPromptVersion)`, folds suggestions in via LLM (verbatim append under a learned-notes section when `OPENAI_API_KEY` is absent), and marks each suggestion `applied` or `rejected`. |
| `Stop` | `hooks/apply_memory_extraction_prompt.py` | Rate-limited rebuild of the live `(:MemoryExtractionPrompt)`. It mirrors `apply_system_prompt.py` and shares the same `MKG_PROMPT_REBUILD_*` / `MKG_PROMPT_MAX_CHARS` knobs: gates on pending `:MemoryExtractionPromptSuggestion` nodes, snapshots prior content as `(:MemoryExtractionPromptVersion)`, preserves required runtime tokens, and marks suggestions `applied` or `rejected`. |
| `PostToolUse` (matcher on the Diffbot tools `enhance_entity` / `search_news`) | `hooks/ingest_diffbot.py` | Builds Diffbot tool results back into the graph instead of letting them evaporate with the conversation. `enhance_entity` firmographics become `(:Account)-[:HAS_ENRICHMENT]→(:DiffbotEntity:DiffbotOrganization)` or `(:DiffbotEntity:DiffbotPerson)` (matched on domain, name/`allNames`, and employer hints); `search_news` articles become `(:NewsArticle)-[:MENTIONS]→(:Account)` plus `[:MENTIONS]→(:DiffbotOrganization)` when organization tags or account matches identify companies, and `[:TAGGED]→(:NewsTag)`. Diffbot entities and articles link `[:CAPTURED_IN]→(:Session)` for provenance. Handles the harness's response wrappers, including oversized results that arrive as a saved-to-file notice. |
| `PostToolUse` (matcher on the query tools `bigquery_execute_query` / `neo4j_read_cypher`) | `hooks/capture_query_failures.py` | Captures failed or suspicious query outputs as structured `(:QueryExecution)-[:HAS_ISSUE]→(:QueryIssue)` artifacts. The first pass records issues visible in PostToolUse payloads: empty result sets, parser/schema/permission/resource/capability errors, malformed outputs, and Neo4j serialization cases such as temporal values returned as `{}`. Clean successful query results are ignored. |

### Graph model

```
(:Project {id})
   ├─[:HAS_SESSION]→ (:Session)─[:HAS_EVENT]→ (:SessionEvent)─[:NEXT]→ ...
   │                            ─[:INJECTED]→ (:SystemPromptInjection {content_sha})─[:OF_PROMPT]→ (:SystemPrompt)
   ├─[:HAS_LEARNING]→ (:Learning {status: 'candidate'|'approved', confidence})─[:INJECTED_IN]→ (:Session)
   ├─[:HAS_DECISION]→ (:Decision)─[:INJECTED_IN]→ (:Session)
   ├─[:HAS_SYSTEM_PROMPT_SUGGESTION]→ (:SystemPromptSuggestion {status})─[:APPLIED_TO]→ (:SystemPrompt)
   ├─[:HAS_MEMORY_EXTRACTION_PROMPT_SUGGESTION]→ (:MemoryExtractionPromptSuggestion {status})─[:APPLIED_TO]→ (:MemoryExtractionPrompt)
   └─[:HAS_PROCESSING]→ (:ProjectProcessing)─[:PROCESSED_EVENT]→ (:SessionEvent)
                                            ─[:PRODUCED_LEARNING]→ (:Learning)
                                            ─[:UPDATED_LEARNING]→ (:Learning)
                                            ─[:PRODUCED_DECISION]→ (:Decision)

(:SystemPrompt {name, version})─[:HAS_VERSION]→ (:SystemPromptVersion)
(:MemoryExtractionPrompt {name, version})─[:HAS_VERSION]→ (:MemoryExtractionPromptVersion)

# Produced by hooks/enrich_events.py (on demand):
(:Session)─[:HAS_TURN]→ (:Turn)─[:ISSUED]→ (:ToolCall)─[:USES_TOOL]→ (:Tool)
                                          ─[:RETURNED]→ (:ToolResult)
                                          ─[:HAS_RATIONALE]→ (:ToolRationale)
                                          ─[:TARGETS]→ (:Resource)

# Produced by hooks/ingest_diffbot.py (PostToolUse on the Diffbot tools):
(:Account)─[:HAS_ENRICHMENT]→ (:DiffbotEntity:DiffbotOrganization)─[:CAPTURED_IN]→ (:Session)
(:Account)─[:HAS_ENRICHMENT]→ (:DiffbotEntity:DiffbotPerson)─[:CAPTURED_IN]→ (:Session)
(:NewsArticle)─[:MENTIONS]→ (:Account)
(:NewsArticle)─[:MENTIONS]→ (:DiffbotOrganization)
(:NewsArticle)─[:TAGGED]→ (:NewsTag)
(:NewsArticle)─[:CAPTURED_IN]→ (:Session)

# Produced by hooks/capture_query_failures.py (PostToolUse on query tools):
(:Project)─[:HAS_QUERY_EXECUTION]→ (:QueryExecution) ←[:HAS_QUERY_EXECUTION]─(:Session)
(:QueryExecution)─[:HAS_ISSUE]→ (:QueryIssue)
```

Candidate learnings flow through retrieval but are review-gated — they stay
`candidate` until promoted to `approved` (currently a manual Cypher update; see
TODOs). Fulltext indexes
`project_learning_fulltext` and `project_decision_fulltext` back the retrieval
path. System-prompt suggestions follow `candidate → applied | rejected`: the
Stop-event rebuild consumes them in batches once the time and count gates pass,
and every rebuild snapshots the prior prompt as a `(:SystemPromptVersion)` for
rollback. Memory-extraction-prompt suggestions follow the same
`candidate → applied | rejected` path into `(:MemoryExtractionPrompt)` while
preserving the runtime tokens used to render project/event context.

## Quick Start

### Claude Code / Claude Desktop

Add to your Claude Desktop `claude_desktop_config.json` or `.claude/settings.json`:

```json
{
  "mcpServers": {
    "meta-knowledge-graph": {
      "command": "uv",
      "args": ["--directory", "/path/to/meta-knowledge-graph", "run", "meta-knowledge-graph"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "<your-password>",
        "NEO4J_DATABASE": "neo4j",
        "OPENAI_API_KEY": "<your-openai-api-key>"
      }
    }
  }
}
```

Then register the hook scripts in `.claude/settings.json` under their
corresponding `hooks.*` events — this repo's own
[`.claude/settings.json`](.claude/settings.json) shows the full wiring.

### Codex

[`.codex/config.toml`](.codex/config.toml) wires the same MCP server into
Codex, with approval gates on the query and write tools:

```toml
[mcp_servers.meta-knowledge-graph]
command = "uv"
args = ["run", "--no-sync", "meta-knowledge-graph"]

[mcp_servers.meta-knowledge-graph.tools.bigquery_execute_query]
approval_mode = "approve"

[mcp_servers.meta-knowledge-graph.tools.project_add_learning]
approval_mode = "approve"
```

The hook scripts work under any harness with lifecycle events; pass
`--client codex` (or your harness's name) to `log_event.py` so captured
sessions are tagged with their origin.
This repo's [`.codex/hooks.json`](.codex/hooks.json) wires the documented Codex
events for recall, capture, Stop-time extraction, and prompt rebuilds. Codex
does not currently document a `SessionEnd` hook, so that session-level processor
is only configured for harnesses that expose it.

### Other harnesses

Point the harness at the same two surfaces: spawn `uv run meta-knowledge-graph`
as an MCP server, and call the `hooks/` scripts from the harness's lifecycle
events (JSON payload on stdin, `--client <name>` for attribution). The graph,
memory extraction, and prompt rebuild loops are identical regardless of which
harness produced the events.

## Sales agent use case

The repo ships a complete demo persona in [`import/sales_agent/`](import/sales_agent/):
a sales / customer-success intelligence assistant working a book of ~48
enterprise car-rental customer accounts for **RoadFlex** (a corporate mobility
provider). It exercises every layer of the system: BigQuery as the system of
record, Neo4j as the relationship graph, Diffbot for external signal, a
persisted `(:SystemPrompt)` persona, and bootstrap learnings so the very first
session starts with scoped project memory.

### 1. Configure `.env`

Create `.env` at the repo root (both the seeders and the hooks load it):

```bash
# Neo4j — required
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=neo4j

# LLM — required for memory extraction and prompt rebuilds
OPENAI_API_KEY=<your-openai-api-key>

# BigQuery warehouse + Neocarta catalog — required for the warehouse half
GCP_PROJECT_ID=<your-gcp-project>
BIGQUERY_DATASET_ID=acme_corp
BIGQUERY_MCP_URL=https://bigquery.googleapis.com/mcp
GOOGLE_APPLICATION_CREDENTIALS=<path-to-service-account.json>  # or GCP_SERVICE_ACCOUNT_JSON inline

# Diffbot — required for live news / firmographic enrichment
DIFFBOT_TOKEN=<your-diffbot-token>
```

### 2. Seed everything

```bash
uv run python import/sales_agent/seed_all.py
```

This runs five seeders in order (each independently re-runnable; re-running is
safe — BigQuery tables are dropped/recreated and Neo4j MERGEs on natural keys):

| Seeder | What it loads |
|---|---|
| `seed_bigquery.py` | `accounts`, `products`, `account_product_usage` (monthly time series), `account_contacts`, `account_renewals` under `$GCP_PROJECT_ID.$BIGQUERY_DATASET_ID`. |
| `run_neocarta.py` | Neocarta catalog metadata, query-log context, then LiteLLM-powered embeddings for the seeded BigQuery dataset. Run after `seed_bigquery.py` so the warehouse tables and descriptions exist. |
| `seed_neo4j.py` | `:Account` / `:Product` / `:Contact` / `:CSM` nodes plus `USES_PRODUCT` (utilization, revenue), `HAS_CONTACT`, and `OWNS` relationships. |
| `seed_learnings.py` | Bootstrap `:Learning` / `:Decision` nodes so the first session already has scoped project memory. |
| `seed_system_prompt.py` | Persists `system_prompt.md` (the RoadFlex sales persona) as `(:SystemPrompt {name: 'default'})`. |

To preview the generated dataset without touching any database:
`uv run python import/sales_agent/seed_data.py`. Dataset design notes live in
[`import/sales_agent/README.md`](import/sales_agent/README.md).

### 3. Register the MCP server and hooks

Follow the Quick Start above for your harness — Claude Code via
[`.claude/settings.json`](.claude/settings.json), Codex via
[`.codex/config.toml`](.codex/config.toml) and
[`.codex/hooks.json`](.codex/hooks.json), or any custom harness that can spawn
an MCP server and fire lifecycle hooks.
With `GCP_PROJECT_ID` / `BIGQUERY_DATASET_ID` / `OPENAI_API_KEY` set the server
mounts the `neocarta_*` catalog tools, with `BIGQUERY_MCP_URL` set it mounts
`bigquery_execute_query`, and with `DIFFBOT_TOKEN` set it mounts `search_news`
and `enhance_entity`.

### 4. Start a session

On SessionStart the hook injects the persisted RoadFlex persona, and prompt
submits inject the most relevant seeded learnings. Try:

- *"Which accounts renew in the next 90 days and which of them are at risk?"*
- *"Where do we have expansion room — accounts running over contracted capacity?"*
- *"Build me a brief on Accenture: footprint, contacts, and recent news."*
- *"Roll up the book of business by CSM."*

From there the loop takes over: every session's events are logged, memory
extraction distills new learnings/decisions on Stop, plus SessionEnd for
harnesses that emit it, and the
persona prompt keeps improving itself via `:SystemPromptSuggestion` rebuilds.

## Configuration

| Env Variable | CLI Flag | Default | Notes |
|---|---|---|---|
| `NEO4J_URI` | `--db-url` | `bolt://localhost:7687` | |
| `NEO4J_USERNAME` | `--username` | `neo4j` | |
| `NEO4J_PASSWORD` | `--password` | `password` | |
| `NEO4J_DATABASE` | `--database` | `neo4j` | |
| `NEO4J_TRANSPORT` | `--transport` | `stdio` | |
| `OPENAI_API_KEY` | — | — | Required by `process_project.py`, the prompt rebuild hooks, and Neocarta. |
| `DIFFBOT_TOKEN` | — | — | Enables `search_news` and `enhance_entity` when set. |
| `LLM_MODEL` | — | `gpt-5.4-mini` | The single model knob for every LLM call: memory extraction and both prompt rebuilds. |
| `MKG_PROMPT_REBUILD_MIN_HOURS` | — | `8` | Minimum hours between prompt rebuilds on Stop (system prompt and memory extraction prompt alike). |
| `MKG_PROMPT_REBUILD_MIN_SUGGESTIONS` | — | `2` | Pending candidate suggestions required before a rebuild runs (both rebuilds). |
| `MKG_PROMPT_MAX_CHARS` | — | `12000` | Length budget for a rebuilt prompt (both rebuilds). |
| `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID` | — | — | Required (with `OPENAI_API_KEY`) to mount the Neocarta catalog tools; `GCP_PROJECT_ID` is also the project queried by `bigquery_execute_query`. |
| `BIGQUERY_MCP_URL` | — | — | Mounts `bigquery_execute_query` when set, e.g. `https://bigquery.googleapis.com/mcp`. For `googleapis.com` URLs a Google ADC bearer token is fetched automatically. |
| `BIGQUERY_MCP_AUTH`, `BIGQUERY_MCP_HEADERS` | — | — | Optional explicit bearer token / JSON dict of extra headers for the BigQuery MCP endpoint. |
| `BIGQUERY_REGION`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` | — | — | Optional; forwarded to the Neocarta subprocess when set. `BIGQUERY_REGION` (default `US`) also sets the dataset location when seeding the sales-agent demo. |
| `GCP_BILLING_PROJECT_ID` | — | falls back to `GCP_PROJECT_ID` | Optional billing project for the sales-agent BigQuery seeder. |
| `GCP_SERVICE_ACCOUNT_JSON` / `GOOGLE_APPLICATION_CREDENTIALS` | — | — | Optional GCP auth for Neocarta: inline service-account JSON (written to a temp file) or a credentials file path. |

## TODO / Roadmap

- [ ] **Improving memory** — richer retrieval (semantic + structural search),
      approval tooling for promoting candidate learnings, and time-decaying
      confidence with re-validation.
- [ ] **GDS intelligence engine** — generate intelligence from the accumulated
      graph: implicit relationship discovery (node similarity), critical
      decision points (betweenness centrality), and behavior clustering
      (community detection).
