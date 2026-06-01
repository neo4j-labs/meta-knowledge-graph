# Meta Knowledge Graph: Self-Learning System for AI Agents

This repository contains the reference implementation for the **Meta Knowledge Graph** (MKG), an **Intelligence Layer for Enterprise AI Agents**.

The MKG is a self-improving, graph-structured metadata layer designed to enhance the reasoning capabilities of AI agents by providing rich enterprise context.

### Key Differentiator: Generating Intelligence

The MKG's core innovation is that it does not just *store* intelligence—it **generates** it through the **Graph Intelligence Engine**.

This engine uses algorithms from the Graph Data Science (GDS) library to:

  * **Discover Implicit Relationships:** Find relationships not explicitly created by humans or agents (using node similarity on bipartite graphs).
  * **Learn Agent Patterns:** Identify critical decision points (using betweenness centrality) and cluster agent behaviors into reusable patterns (using community detection).

Every agent run strengthens patterns, and GDS discovers new ones from the accumulated decision traces, creating a **compounding intelligence effect**.

### Four Metadata Categories Captured

The system harvests and captures decision traces and four categories of metadata from enterprise data platforms and agent interactions:

  * **Technical:** Schemas, column definitions, data types, and lineage.
  * **Operational:** Job execution stats, quality/trust scores, freshness, and pipeline success/failure.
  * **Business:** Glossary terms, KPI definitions, ontologies, and semantic mapping (e.g., table-to-domain).
  * **Agentic:** Decision traces with causal chains (`CAUSED`, `INFLUENCED`, `PRECEDENT_FOR`), policy application, user corrections with rationale, and accumulated patterns.

### Core Architecture

  * **Meta Knowledge Graph (Neo4j):** The central graph store, extending the Context Graph Demo model with nodes like `AgentRun`, `Correction`, `OrgKnowledge`, and `Schema`.
  * **GDS Intelligence Engine:** Runs analytical workflows to discover implicit relationships and learn patterns from accumulated decision traces.
  * **Hybrid Retrieval Agent:** Combines **semantic search** (text embeddings on decision reasoning) with **structural search** (enriched e.g. with graph embeddings) to retrieve precedents that are both semantically and structurally similar.
  * **External Metadata Connector:** Gather catalog metadata (schema, columns, types) from sources like Snowflake or Databricks.
  * **Approval Process:** Knowledge derived from user interactions (chat) requires explicit approval to become permanent rules, and patterns carry time-decaying confidence scores that trigger re-validation.

# Meta Knowledge Graph

A reference implementation of the architecture above for Claude Code agents,
backed by Neo4j. It ships as two halves that form a closed capture-and-recall
loop:

- **MCP server (`meta-knowledge-graph`)** — surfaces project memory, the underlying
  graph, the persisted system prompt, and (optionally) a data catalog and
  warehouse to the agent as tools.
- **Claude Code hooks** — log every session event, inject scoped project
  context on prompt submit, and run an LLM adjudicator at Stop / SessionEnd
  that distills durable `:Learning` / `:Decision` / `:SystemPromptSuggestion`
  candidates from what just happened.

The hooks write to the same graph the MCP tools read from, so each new session
starts with the most relevant prior learnings already injected.

## Architecture

### MCP server

Mounted under the `meta-knowledge-graph` prefix. Tool availability varies by
environment — the data-catalog and warehouse tools are only mounted when the
required env vars are present.

| Tool | Purpose |
|---|---|
| `project_get_context` | Fetch approved + candidate `:Learning` and `:Decision` nodes for the current project, optionally fulltext-ranked by a query. |
| `project_add_learning` | Idempotent direct write of one durable learning. Use for user-asserted constraints the auto-capture would miss; routine work should be left to the adjudicator. |
| `neo4j_get_schema` / `neo4j_read_cypher` | Read access to the graph (proxied from the official neo4j-mcp-server). |
| `import_text_to_kg` | Extract entities and relationships from raw text via an LLM and persist them. |
| `search_news` | Search Diffbot Knowledge Graph Article/news data with a DQL string and a small `max_results` count. Returns concise fields for sales research. Optional; mounted only when `DIFFBOT_TOKEN` or `DIFFBOT_API_TOKEN` is set. |
| `enhance_entity` | Enrich a Diffbot `Organization` or `Person` from sales-friendly identifiers such as name, URL, email, phone, title, employer, or location. Returns concise sales-relevant fields. Optional; mounted only when `DIFFBOT_TOKEN` or `DIFFBOT_API_TOKEN` is set. |
| `bigquery_execute_query` | Read-only SQL against the configured BigQuery project. Optional. |
| `neocarta_*` | Data-catalog navigation plus hybrid vector + fulltext search over schemas, tables, and columns. Optional; requires `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID`, and `OPENAI_API_KEY`. |

Example `search_news` DQL for recent articles:

- Company news from the last 3 days: `type:Article tags.label:"Acme Corp" date<3d language:"en" sortBy:date`
- Topic news from the last 3 days: `type:Article tags.label:"supply chain disruption" date<3d language:"en" sortBy:date`

### Hooks

Wire these into `.claude/settings.json` under the corresponding events. All
hooks swallow their own exceptions so a Neo4j outage never blocks the session.

| Hook event | Script | Behavior |
|---|---|---|
| `SessionStart` | `hooks/inject_system_prompt.py` | Loads `(:SystemPrompt {name: $MKG_PROMPT_NAME})` from Neo4j and injects it. If the node is missing, injects a tool-agnostic bootstrap prompt telling the agent to discover its tools, recall project memory, and persist a refined system prompt back to Neo4j so the next session skips the fallback. |
| `UserPromptSubmit` | `hooks/inject_project_context.py` | Fulltext-ranks `:Learning` and `:Decision` against the new prompt and injects the top hits scoped to the current project. Marks served learnings as used. |
| `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`, `SessionEnd` | `hooks/log_event.py` | Persists each event as an `:Event` node under the current `:Session`, threaded by `:NEXT`. This is the corpus the adjudicator later reads. |
| `Stop`, `SessionEnd` | `hooks/process_project.py` | Pulls the session's events that the current `(project, mode)` hasn't processed yet, builds a tail-preserving corpus, fetches the closest existing learnings/decisions, and asks an LLM to return create/update/ignore actions per category. Writes new `:Learning` / `:Decision` / `:SystemPromptSuggestion` nodes with status `candidate`. |

### Graph model

```
(:Project {id})
   ├─[:HAS_SESSION]→ (:Session)─[:HAS_EVENT]→ (:Event)─[:NEXT]→ ...
   │                            ─[:INJECTED]→ (:Injection)
   ├─[:HAS_LEARNING]→ (:Learning {status: 'candidate'|'approved', confidence})
   ├─[:HAS_DECISION]→ (:Decision)
   ├─[:HAS_SYSTEM_PROMPT_SUGGESTION]→ (:SystemPromptSuggestion)
   └─[:HAS_PROCESSING]→ (:ProjectProcessing)─[:PROCESSED_EVENT]→ (:Event)
                                            ─[:PRODUCED_LEARNING]→ (:Learning)
                                            ─[:UPDATED_LEARNING]→ (:Learning)
                                            ─[:PRODUCED_DECISION]→ (:Decision)
```

Candidate learnings flow through retrieval but are review-gated — they stay
`candidate` until promoted to `approved`. Fulltext indexes
`project_learning_fulltext` and `project_decision_fulltext` back the retrieval
path.

## Quick Start

### Run locally during development

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

Then register the four hook scripts in `.claude/settings.json` under their
corresponding `hooks.SessionStart` / `UserPromptSubmit` / `PreToolUse` /
`PostToolUse` / `Stop` / `SessionEnd` entries.

## Configuration

| Env Variable | CLI Flag | Default | Notes |
|---|---|---|---|
| `NEO4J_URI` | `--db-url` | `bolt://localhost:7687` | |
| `NEO4J_USERNAME` | `--username` | `neo4j` | |
| `NEO4J_PASSWORD` | `--password` | `password` | |
| `NEO4J_DATABASE` | `--database` | `neo4j` | |
| `NEO4J_TRANSPORT` | `--transport` | `stdio` | |
| `OPENAI_API_KEY` | — | — | Required by `import_text_to_kg`, `process_project.py`, and Neocarta. |
| `DIFFBOT_TOKEN` / `DIFFBOT_API_TOKEN` | — | — | Enables `search_news` and `enhance_entity` when set. |
| `LLM_MODEL` | — | `gpt-5.4-mini` | Default model for LLM calls. |
| `MKG_LEARNING_MODEL` | — | falls back to `LLM_MODEL` | Override just the adjudicator model. |
| `MKG_PROMPT_NAME` | — | `default` | Which `(:SystemPrompt {name})` node to load on session start. |
| `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID` | — | — | Required to mount the Neocarta and BigQuery tools. |
