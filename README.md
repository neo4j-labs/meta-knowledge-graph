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
  inject scoped project context on prompt submit, and run an LLM memory extraction
  processor at Stop
  that distills durable `:Learning` candidates (scoped `project` or `user`,
  covering durable facts and decisions alike) from what just happened, plus an
  append-only `:Observation` timeline entry per work window (episodic memory:
  what happened, recalled by recency at session start).

The hooks write to the same graph the MCP tools read from, so each new session
starts with the most relevant prior learnings already injected — both
project-scoped memory and durable facts about the user. The persisted **system
prompt** is frozen at runtime: it is read on start but never rewrites itself. The
**memory extraction prompt** is a fixed code constant — there is no
Neo4j-backed self-improvement loop. The only writers of the system prompt besides
the seed scripts are the deliberate consolidation services — a rate-limited
Stop/SessionEnd hook that folds accumulated user-profile memory into the system
prompt once enough of it has piled up unreviewed, keeping every superseded prompt
as version history.

A complete end-to-end demo — a B2B sales / customer-success assistant for an
enterprise car-rental provider — ships in the repo; see
[Sales agent use case](#sales-agent-use-case) for setup.

## Running MKG

MKG is **harness-agnostic**, and can run in several ways:

- **Claude Code** — install the packaged **plugin** (below). This is the quickest
  path and the one most users want.
- **Codex** — install the packaged **plugin** for lifecycle hooks, the MCP server,
  and Codex skills, or run straight from a **repo checkout** with the committed
  `.codex/` wiring.
- **Other harnesses** — any MCP-capable harness with lifecycle hooks can drive the
  same shared scripts.

Either way MKG is two halves — lifecycle **hooks** (capture + recall) and an
**MCP server** (Neo4j / BigQuery / neocarta tools) — and the only host
prerequisites are [`uv`](https://docs.astral.sh/uv/) and a reachable Neo4j
instance; both halves execute through `uv`.

### Claude Code (plugin)

```
claude plugin marketplace add neo4j-labs/meta-knowledge-graph
claude plugin install meta-knowledge-graph@mkg
```

- The marketplace is named `mkg`; the qualified plugin id is `meta-knowledge-graph@mkg`.
- On the **first MCP start** after install, the plugin launcher falls back to
  `uv run --project` to create the plugin cache virtualenv (one-time; that start
  is slower). Later MCP starts execute the cached `.venv/bin/python -m
  meta_knowledge_graph` directly, with the venv on `PATH` so optional MCP
  subprocesses run from the same environment instead of doing their own `uvx`
  startup install.
- Verify: `claude plugin list` shows `meta-knowledge-graph@mkg` *enabled*; inside
  a session the MKG system prompt is injected and `mcp__meta-knowledge-graph__*`
  tools are available.

```
claude plugin disable meta-knowledge-graph@mkg
claude plugin enable  meta-knowledge-graph@mkg
claude plugin details meta-knowledge-graph@mkg     # component inventory + token cost
```

### Configuring the Claude Code plugin

For an installed Claude Code plugin, the config path is:

```
~/.config/meta-knowledge-graph/.env
```

Credentials live in that one user-global file (mode `600`), read by both the
hooks and the MCP server. It survives plugin updates and is never written into
the ephemeral plugin cache.

> MKG deliberately does **not** use `/plugin configure` — there is no
> `userConfig`/keychain schema, so credentials stay file-based and portable
> across harnesses (Codex, etc.).

Run the wizard in your own terminal (it prompts for secrets):

```
uv run --project ~/.claude/plugins/marketplaces/mkg meta-knowledge-graph setup
```

…or write the file by hand. Example `~/.config/meta-knowledge-graph/.env`:

```
mkdir -p ~/.config/meta-knowledge-graph
cat > ~/.config/meta-knowledge-graph/.env <<'EOF'
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=change-me
NEO4J_DATABASE=neo4j
# LLM for memory extraction (all optional; Claude Code prefers your subscription):
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...   # preferred: `claude setup-token`
OPENAI_API_KEY=sk-...                      # litellm fallback (any provider) + embeddings
ANTHROPIC_API_KEY=sk-ant-api03-...         # litellm fallback (any provider) also works
# GEMINI_API_KEY / OPENROUTER_API_KEY / LLM_MODEL / DIFFBOT_TOKEN also honored
EOF
chmod 600 ~/.config/meta-knowledge-graph/.env
```

Resolution order (first existing wins): `MKG_ENV_FILE` → `<project>/.env`
(repo/demo) → `~/.config/meta-knowledge-graph/.env` (installed/ambient). Override
the dir with `MKG_CONFIG_DIR` or `XDG_CONFIG_HOME`. Start a new session after
changing credentials.

### Codex (plugin)

The installable plugin package lives in [`plugin/`](plugin/), following the
same layout as `claude-mem`: the repo root is the marketplace/catalog checkout,
and `plugin/` is the versioned payload copied into the host plugin cache.

- [`.agents/plugins/marketplace.json`](.agents/plugins/marketplace.json) is the
  Codex marketplace catalog and points at `./plugin`.
- [`plugin/.codex-plugin/plugin.json`](plugin/.codex-plugin/plugin.json)
  packages the plugin metadata, Codex skills, hooks, and MCP server declaration.
- [`plugin/hooks/codex-hooks.json`](plugin/hooks/codex-hooks.json) is the Codex
  hook config. Its commands run the packaged hook scripts with `uv run --script`.
- [`plugin/codex-skills/`](plugin/codex-skills/) exposes Codex equivalents of
  the Claude Code commands:
  `mkg-start`, `mkg-orchestrate`, `sales-agent-demo`, and `mkg-recall`.
- [`plugin/.mcp.json`](plugin/.mcp.json) is shared with the Claude Code plugin
  and starts the same `meta-knowledge-graph` MCP server from the installed
  plugin cache or repo checkout. Warm plugin starts bypass `uv` and run the
  cached virtualenv's Python directly; `uv run --project` remains the fallback
  when the virtualenv has not been created yet.

Install from the remote marketplace repo:

```
codex plugin marketplace add neo4j-labs/meta-knowledge-graph
codex plugin add meta-knowledge-graph@mkg
```

Command and skill ownership is deliberately split by host:

- [`commands/`](commands/) is the Claude Code slash-command surface and the
  canonical procedure source.
- [`plugin/codex-skills/`](plugin/codex-skills/) is the Codex plugin surface.
  Each Codex skill is a thin adapter that reads the matching packaged
  `plugin/commands/*.md` file where one exists, then follows that procedure with
  Codex tools and naming.

Codex also supports a repo-local checkout mode:
[`.codex/hooks.json`](.codex/hooks.json) wires the same lifecycle capture and
recall scripts through `git rev-parse --show-toplevel`, which is convenient when
developing MKG from this repository. Installed plugins use
[`plugin/hooks/codex-hooks.json`](plugin/hooks/codex-hooks.json) instead, so
hook commands resolve the plugin cache rather than the user's active project
checkout. The repo does not use a recursive
`plugins/meta-knowledge-graph -> ..` symlink.

Start a new Codex thread after install so the plugin hooks, skills, and MCP tools
are picked up.

### Codex (repo checkout)

Open this repo in Codex and it works out of the box —
[`.codex/config.toml`](.codex/config.toml) wires the same MCP server (with
approval gates on the query and write tools) and
[`.codex/hooks.json`](.codex/hooks.json) wires the recall, capture, and
Stop-time extraction events:

```toml
[mcp_servers.meta-knowledge-graph]
command = "uv"
args = ["run", "--no-sync", "meta-knowledge-graph"]

[mcp_servers.meta-knowledge-graph.tools.bigquery_execute_query]
approval_mode = "approve"

[mcp_servers.meta-knowledge-graph.tools.project_add_learning]
approval_mode = "approve"
```

Credentials are read from `~/.config/meta-knowledge-graph/.env` exactly as for
Claude Code. Codex doesn't currently document a `SessionEnd` hook, so MKG only
wires Stop-time extraction for Codex.

### Claude Desktop & other harnesses

Plugins are a Claude *Code* feature, so **Claude Desktop** registers the MCP
server manually in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "meta-knowledge-graph": {
      "command": "uv",
      "args": ["--directory", "/path/to/meta-knowledge-graph", "run", "meta-knowledge-graph"]
    }
  }
}
```

Credentials come from `~/.config/meta-knowledge-graph/.env` (or set them inline
under `env`). For **any other harness**, point it at the same two surfaces:
spawn `uv run meta-knowledge-graph` as an MCP server, and call the `hooks/`
scripts from its lifecycle events (JSON payload on stdin, `--client <name>` for
attribution). The graph and memory-extraction loop are identical regardless of
which harness produced the events — and packaged plugins for more harnesses will
follow.

### Developing

Iterate against your working tree without touching the marketplace or cache:

```
claude --plugin-dir /path/to/meta-knowledge-graph     # loads from the repo, this session only
codex plugin marketplace add /path/to/meta-knowledge-graph
codex plugin add meta-knowledge-graph@mkg
```

In a repo checkout the project-local `.env` takes precedence, so demo creds stay
scoped to the repo. Exercise a hook directly with a simulated payload:

```
echo '{"session_id":"dev","hook_event_name":"SessionStart","source":"startup","cwd":"'"$PWD"'"}' \
  | uv run --project . python hooks/inject_system_prompt.py
```

Tests and manifest validation:

```
uv run python -m pytest tests/ -v
claude plugin validate .
```

**Publishing a release** (so `claude plugin update` surfaces it — a version bump
is required):

```
# bump "version" in .claude-plugin/plugin.json, then:
git commit -am "release 0.2.0" && git push
claude plugin tag                                  # creates meta-knowledge-graph--v0.2.0, validates manifest agreement
```

Consumers pull it with:

```
claude plugin marketplace update mkg
claude plugin update meta-knowledge-graph@mkg      # use the qualified id; restart to apply
```

With `autoUpdate: true` on the `mkg` marketplace, the catalog refresh is
automatic — but the version bump is still what makes a new release visible.

**Where code runs from** — two directories back the install:

- `~/.claude/plugins/marketplaces/mkg/` — git clone of the repo (the catalog),
  refreshed by `marketplace update`.
- `~/.claude/plugins/cache/mkg/meta-knowledge-graph/<version>/` — the
  version-pinned copy of `plugin/` that `$CLAUDE_PLUGIN_ROOT` resolves to at
  runtime, with its own `.venv` (re-synced on the first session after each
  update).

## Architecture

### MCP server

Mounted under the `meta-knowledge-graph` prefix, in four groups:

| Group | Tools | Mounted when |
|---|---|---|
| Project memory & graph | `project_get_context`, `project_add_learning`, `project_review_queue`, `project_resolve_learning`, `neo4j_get_schema`, `neo4j_read_cypher` | Always. |
| Diffbot research | `search_news`, `enhance_entity` | `DIFFBOT_TOKEN` is set. |
| BigQuery warehouse | `bigquery_execute_query` | `BIGQUERY_MCP_URL` is set. |
| Neocarta data catalog | `neocarta_*` | `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID`, and the `EMBEDDING_MODEL`'s provider key (OpenAI by default) are set. |

### Hooks

The hook scripts are harness-agnostic: each reads the lifecycle event's JSON
payload from stdin, and `log_event.py` tags events with `--client <name>`
(default `claude_code`), so sessions record which harness produced them. Any
harness that exposes lifecycle hooks — Claude Code, Codex, Cursor, or a custom
loop — can drive the same scripts. The table below shows the Claude Code
wiring from this repo's own `.claude/settings.json`. All hooks swallow their
own exceptions so a Neo4j outage never blocks the session. A few scripts
(`inject_system_prompt.py`, `log_event.py`, `seed_system_prompt.py`) are
exposed under `.claude/hooks/` as symlinks back to the canonical versions in
`hooks/`.

For installed Claude Code plugins, the scripts resolve project memory from the
user's active worktree first (`cwd` / `CLAUDE_PROJECT_DIR`, with a git-root
walk-up) and treat the plugin cache path as the hook implementation root only.
Detached background processors carry that resolved project through
`MKG_PROJECT_ROOT` / `MKG_PROJECT_ID` so Stop-time extraction does not fall back
to the plugin directory.

| Hook event | Script | Behavior |
|---|---|---|
| `SessionStart` (`startup\|resume\|clear`) | `hooks/inject_system_prompt.py` | Loads `(:SystemPrompt {name: 'default'})` from Neo4j and injects it. If the node is missing, injects a tool-agnostic bootstrap prompt telling the agent to discover its tools, recall project memory, and capture user/project facts as scoped learnings. The injection log keeps only a content hash + summary on the `:SystemPromptInjection` node and links it to its source via `[:OF_PROMPT]→(:SystemPrompt)` instead of copying the prompt text. If the same prompt content was already injected into the same session, the hook skips the duplicate (unless the context was wiped by `clear`/`compact`). |
| `SessionStart` (`startup\|resume\|clear`), `UserPromptSubmit` | `hooks/inject_project_context.py` | Fulltext-ranks `:Learning` against the new prompt and injects the top hits: project-scoped learnings for the current project, plus durable user-scoped learnings that follow the user across every project. Every injected item is linked to the session via `[:INJECTED_IN]`; items already injected earlier in the same session — or first produced during it (`[:FROM_SESSION]`) — are excluded from retrieval, so a conversation never receives the same learning twice, nor has its own freshly-extracted memory echoed back. Marks served learnings as used. At `SessionStart` it additionally injects a "Recent project activity" block — the latest `:Observation` timeline entries by recency (never similarity), giving the new session a "previously on this project" recap. |
| `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Notification`, `Stop`, `SubagentStop`, `PreCompact`, `SessionEnd` | `hooks/log_event.py` | Persists each event as a `:SessionEvent` node threaded by `:NEXT`. Main-agent events land under the parent `:Session`; subagent events land under a separate `:Session` keyed by the subagent id and linked back to the parent session, spawn event, start event, and stop event. This is the corpus the memory extraction processor later reads. |
| `Stop` (`--mode turn`) | `hooks/process_project.py` | Runs in the background. Pulls the session's unprocessed events, renders a fixed code-constant memory-extraction prompt, builds a tail-preserving corpus, fetches the closest existing learnings, and asks an LLM to return create/update/ignore actions for `:Learning` nodes (durable facts and decisions alike — a decision's rationale is folded into the learning text). Each learning is classified `project` (a fact about the project/environment) or `user` (a durable fact about the person that holds across projects); user-scoped learnings are keyed on a project-independent namespace so the same fact dedupes everywhere. Writes new nodes with status `candidate` (embedding each so it is searchable), and stores the model used plus `llm_status` (`called`, `skipped`, or `error`) and skip/error reason on `:ProjectProcessing`. `update` actions are ownership-scoped — automation may only rewrite memory this project owns (or a project-independent user fact), never another project's — and a content change to an already-`approved` note demotes it back to `candidate` (clearing its human provenance and re-opening it for the gate), so automation can never silently overwrite the trusted tier. Alongside the selective memory pass, a second, non-evolvable prompt writes the **episodic timeline**: 1–3 `:Observation` records per work window (`type` ∈ change/bugfix/feature/refactor/discovery/decision/problem, `title`, `facts`, `narrative`), append-only, chained per project via `[:NEXT]` + `[:LATEST_OBSERVATION]`, embedded for search, and written in the same transaction as `PROCESSED_EVENT` marking so a processed window can never lose its timeline entry. Observations are never deduplicated, gated, or approved — they record *what happened*, not *what is true*; when the LLM is unavailable the window simply gets no observation. Then runs the **consistency gate** (`hooks/consistency_gate.py`): each new project-scoped candidate is searched against the top-15 `approved` **and fellow `candidate`** learnings in the same project/scope (excluding itself) — semantically via the vector index when `EMBEDDING_MODEL` is set, else via the fulltext index — and an LLM judge decides genuine contradictions, promoting consistent candidates to `approved`, rejecting the items they supersede (`[:SUPERSEDES]`), or vetoing the candidate to `rejected` (`[:CONTRADICTED_BY]`) when an existing item is clearly more reliable. Resolutions apply per candidate, so restatements extracted in one batch collapse onto one canonical item. Finally a bounded **sweep** pushes ungated candidates — `project_add_learning` MCP writes and rows an earlier judge failure skipped — through the same gate, backfilling their embeddings first. |
| `Stop`, `SessionEnd` | `hooks/consolidate_system_prompt.py` | Runs in the background, rate-limited. The system-prompt consolidation service: it only does work when **more than 5** user-profile memories are in need of review — user-scoped `candidate` learnings not yet folded into the prompt (`MKG_PROMPT_CONSOLIDATION_THRESHOLD`) — and not more than once per cooldown window (`MKG_PROMPT_CONSOLIDATION_INTERVAL_HOURS`, default 24h, tracked via `last_consolidated_at` on the node). When both gates pass, it sends the current `(:SystemPrompt {name: 'default'})` plus the pending user facts — fenced as untrusted data — to the LLM, which folds those facts into the persona, then archives the outgoing prompt as a `(:SystemPromptVersion)` history node before overwriting the active one and bumping its version. Folding is **approved-only**: it counts and folds user-scoped learnings with `status = 'approved'`, never raw candidates, so an unreviewed (or poisoned) user fact cannot rewrite the cross-project persona on its own — a human promotes `candidate → approved` through the review queue (`project_review_queue` / `project_resolve_learning`, surfaced by `/mkg-review`). The folded learnings are stamped `consolidated_at` so they drop out of the backlog. |
| `PostToolUse` (matcher on the Diffbot tools `enhance_entity` / `search_news`) | `hooks/ingest_diffbot.py` | Builds Diffbot tool results back into the graph instead of letting them evaporate with the conversation. `enhance_entity` firmographics become `(:Account)-[:HAS_ENRICHMENT]→(:DiffbotOrganization)` or `(:DiffbotPerson)` (matched on domain, name/`allNames`, and employer hints); `search_news` articles become `(:NewsArticle)-[:MENTIONS]→(:Account)` plus `[:MENTIONS]→(:DiffbotOrganization)` when organization tags or account matches identify companies, and `[:TAGGED]→(:NewsTag)`. Only entities carrying a real Diffbot id are stored — references without one are dropped instead of being keyed on synthetic hashes. Diffbot entities and articles link `[:CAPTURED_IN]→(:Session)` for provenance. Handles the harness's response wrappers, including oversized results that arrive as a saved-to-file notice. |
| `PostToolUse` (matcher on the query tools `bigquery_execute_query` / `neo4j_read_cypher`) | `hooks/capture_query_failures.py` | Captures failed or suspicious query outputs as structured `(:QueryExecution)-[:HAS_ISSUE]→(:QueryIssue)` artifacts. The first pass records issues visible in PostToolUse payloads: empty result sets, parser/schema/permission/resource/capability errors, malformed outputs, and Neo4j serialization cases such as temporal values returned as `{}`. Clean successful query results are ignored. |
| `Stop`, `SessionEnd` | `hooks/consolidate_query_errors.py` | Runs in the background, rate-limited **per tool**. The query-error consolidation service: it folds unconsolidated *invalid-query* failures — the consolidatable issue classes (`syntax_error`, `schema_mismatch`, `capability_unavailable`, `serialization_issue`, plus the generic error bucket) — into durable `(:QueryErrorPattern)` nodes carrying the error signature, root cause, and actionable fix. Transient/environmental classes (`timeout`, `resource_limit`, `result_size_limit`, `permission_error`) never consolidate, and the LLM is additionally instructed to discard transient one-offs hiding in the generic bucket. Failures are grouped by the client-independent `tool_key`; a tool consolidates only when **more than** `MKG_QUERY_ERROR_CONSOLIDATION_THRESHOLD` (default 1) of its failures are pending and not more than once per `MKG_QUERY_ERROR_CONSOLIDATION_INTERVAL_HOURS` (default 6h, tracked on the per-tool `(:ToolErrorProfile)`). The LLM merges same-root-cause failures into one pattern and updates existing patterns instead of duplicating them; patterns are embedded for semantic recall, keep `[:DERIVED_FROM]` provenance, and consumed executions are stamped `error_consolidated_at` so they leave the queue. |
| `PostToolUse`, `PostToolUseFailure` (matcher on the query tools `bigquery_execute_query` / `neo4j_read_cypher`) | `hooks/inject_query_error_context.py` | Recall counterpart of the consolidation service. When a query tool fails with a consolidatable invalid-query error (never on successes, timeouts, or other transient failures), it hybrid-searches (vector + fulltext, RRF) the failing query text *and* the error message against the `(:QueryErrorPattern)` library **of that same tool only**, and feeds the top matching fixes back to the model — so guidance is recalled both when the error looks similar and when the query looks similar, and one tool's guidance never leaks into another's failures. Delivery adapts to the event: `PostToolUse` gets `hookSpecificOutput.additionalContext`; `PostToolUseFailure` (no `additionalContext` support) gets `decision: "block"` + `reason`, which simply shows the guidance to the model since the call already failed. Served patterns track `use_count` / `last_used_at` and link `[:INJECTED_IN]` to the session, without per-session dedup — a recurring error should resurface its fix every time. |

### Graph model

```
(:Project {id})
   ├─[:HAS_SESSION]→ (:Session)─[:HAS_EVENT]→ (:SessionEvent)─[:NEXT]→ ...
   │                            ─[:INJECTED]→ (:SystemPromptInjection {content_sha})─[:OF_PROMPT]→ (:SystemPrompt)
   │             └─[:HAS_SUBAGENT]→ (:Session {agent_kind: 'subagent'})
   │                                  ├─[:SUBAGENT_OF]→ (:Session)
   │                                  ├─[:TRIGGERED_BY]→ (:SessionEvent)  # parent spawn_agent result
   │                                  ├─[:STARTED_AT]→ (:SessionEvent)    # SubagentStart
   │                                  └─[:ENDED_AT]→ (:SessionEvent)      # SubagentStop
   ├─[:HAS_LEARNING]→ (:Learning {scope, status, confidence, created_by_model, last_llm_model})
   │                      ─[:INJECTED_IN]→ (:Session)   ─[:FROM_SESSION]→ (:Session)
   ├─[:HAS_OBSERVATION]→ (:Observation {type, title, facts, narrative, started_at, ended_at})─[:NEXT]→ ...
   │                      ─[:FROM_SESSION]→ (:Session)   # episodic timeline: append-only, never gated
   ├─[:LATEST_OBSERVATION]→ (:Observation)               # tail of the per-project timeline
   └─[:HAS_PROCESSING]→ (:ProjectProcessing {llm_model, llm_status, llm_skip_reason, llm_error})
                                            ─[:PROCESSED_EVENT]→ (:SessionEvent)
                                            ─[:PRODUCED_LEARNING]→ (:Learning)
                                            ─[:UPDATED_LEARNING]→ (:Learning)
                                            ─[:PRODUCED_OBSERVATION]→ (:Observation)

# Frozen at runtime (read on start; written only by seed scripts / consolidation):
(:SystemPrompt {name, version, content, last_consolidated_at})
   ─[:HAS_VERSION]→ (:SystemPromptVersion {name, version, content, is_current})  # prompt history
   ─[:CONSOLIDATED]→ (:Learning {scope: 'user'})                                # folded-in user facts

# Produced by hooks/enrich_events.py (on demand):
(:Session)─[:HAS_TURN]→ (:Turn)─[:ISSUED]→ (:ToolCall)─[:USES_TOOL]→ (:Tool)
                                          ─[:RETURNED]→ (:ToolResult)
                                          ─[:HAS_RATIONALE]→ (:ToolRationale)
                                          ─[:TARGETS]→ (:Resource)

# Produced by hooks/ingest_diffbot.py (PostToolUse on the Diffbot tools):
(:Account)─[:HAS_ENRICHMENT]→ (:DiffbotOrganization)─[:CAPTURED_IN]→ (:Session)
(:Account)─[:HAS_ENRICHMENT]→ (:DiffbotPerson)─[:CAPTURED_IN]→ (:Session)
(:DiffbotOrganization)─[:HAS_CEO]→ (:DiffbotPerson)
(:DiffbotPerson)─[:EMPLOYED_BY]→ (:DiffbotOrganization)
(:NewsArticle)─[:MENTIONS]→ (:Account)
(:NewsArticle)─[:MENTIONS]→ (:DiffbotOrganization)
(:NewsArticle)─[:TAGGED]→ (:NewsTag)
(:NewsArticle)─[:CAPTURED_IN]→ (:Session)

# Produced by hooks/capture_query_failures.py (PostToolUse on query tools):
(:Project)─[:HAS_QUERY_EXECUTION]→ (:QueryExecution) ←[:HAS_QUERY_EXECUTION]─(:Session)
(:QueryExecution)─[:HAS_ISSUE]→ (:QueryIssue)

# Produced by hooks/consolidate_query_errors.py (Stop/SessionEnd, background),
# recalled by hooks/inject_query_error_context.py (PostToolUse/PostToolUseFailure):
(:Project)─[:HAS_TOOL_ERROR_PROFILE]→ (:ToolErrorProfile {tool_key, last_consolidated_at})
(:ToolErrorProfile)─[:HAS_ERROR_PATTERN]→ (:QueryErrorPattern {error_signature, resolution, status})
(:QueryErrorPattern)─[:DERIVED_FROM]→ (:QueryExecution)   # provenance
(:QueryErrorPattern)─[:INJECTED_IN]→ (:Session)           # recall observability
```

Candidate learnings flow through retrieval but are review-gated. Project-scoped
candidates are promoted automatically by the **consistency gate** (see the `Stop`
hook), which retrieves the closest `approved` and fellow `candidate`
learnings in the same project/scope (the item itself excluded) and asks
an LLM judge whether the candidate genuinely contradicts or merely restates any
of them — including candidates in retrieval is what lets restatements collapse
(`[:ALREADY_LEARNED_FROM]`) instead of piling up side by side in the review
queue. Retrieval is semantic (a vector index) when the `EMBEDDING_MODEL` is
available and falls back to the fulltext index when it is not. No contradiction,
or a contradiction the candidate wins, promotes it to `approved` (and rejects the
item it supersedes); an existing item that the judge finds clearly more reliable
vetoes the candidate to `rejected`; a genuinely ambiguous conflict leaves it
`candidate` (with `consistency_status = 'ambiguous'`) for a human. That human gate
is real, not a placeholder: `project_review_queue` surfaces exactly the two
populations the gate cannot resolve itself — ambiguous project-scoped conflicts
and user-scoped candidates — oldest first with their contradicting neighbours
attached, and `project_resolve_learning` applies the decision (`approve`,
`reject`, `edit_approve`, or the conflict resolutions `keep_new` / `keep_existing`
/ `keep_both`) using the same status transitions and edges as the automatic gate,
stamping `reviewed_by = 'human'` so a later gate run does not silently overturn
it. The `/mkg-review` command drives this conversationally, and a SessionStart
nudge reports the pending count. Candidates that enter the graph outside the extractor —
the `project_add_learning` MCP tool writes them directly
(embedding at write time when the provider is available), and a judge outage can
leave extracted rows unresolved — are picked up by a bounded per-turn **sweep**
(`MKG_CONSISTENCY_SWEEP_LIMIT`, default 8 per label, 0 disables) that backfills
missing embeddings and runs the exact same gate, so MCP-written memory reaches
`approved` instead of staying a second-class hint forever. Recall everywhere
requires `status IN ['approved', 'candidate']` — there is no tolerance for
status- or scope-less legacy nodes. The gate is additive —
only the LLM judge is required; with the judge unavailable (or
`MKG_CONSISTENCY_GATE=0`) candidates stay `candidate` as before, and user-scoped
candidates always continue through the `consolidate_system_prompt.py` gate
untouched.

Fulltext indexes
`project_learning_fulltext` backs the retrieval
path; the same learning index serves both project-scoped and user-scoped lookups
(the latter filtered to `scope = 'user'` and unbound from any single project).
The consistency gate adds the cosine vector index `project_learning_vector`
over an `embedding` property, with the `SEARCH` clause
pre-filtering `project_id` / `scope` in-index (native filtered vector search,
**Neo4j 2026.02+**). There is no status filter at search time: the gate strips
the `embedding` from items it rejects or folds, so the vector index only ever
contains live memory — dead items keep their text and provenance edges as
tombstones but drop out of retrieval. Run
`scripts/backfill_memory_embeddings.py` once after enabling it so pre-existing
live nodes become retrievable (it skips tombstones). Tunables: `MKG_CONSISTENCY_GATE` (default on),
`MKG_CONSISTENCY_TOPK` (default 15), `MKG_CONSISTENCY_SWEEP_LIMIT` (default 8),
and `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS`.
The query-error pattern library follows the same hybrid retrieval shape with its
own indexes: `query_error_pattern_fulltext` over the pattern's
title/signature/root-cause/resolution/example query, and the cosine vector index
`query_error_pattern_vector` pre-filtering `project_id` / `tool_key` in-index so
recall never crosses tools. Its tunables are
`MKG_QUERY_ERROR_CONSOLIDATION_THRESHOLD` (default 1) and
`MKG_QUERY_ERROR_CONSOLIDATION_INTERVAL_HOURS` (default 6).
The persisted system prompt is frozen at runtime — read on session start but
never self-modified mid-session; the memory-extraction prompt is a fixed code
constant with no self-improvement loop. Improving the system
prompt from the accumulated learning corpus is the job of the rate-limited
`consolidate_system_prompt.py` service (see the hooks table), which folds pending
user-profile memory into the persona and keeps every superseded prompt as a
`:SystemPromptVersion`. It and the seed scripts are the only writers of those
nodes.

## Sales agent use case

The repo ships a complete demo persona in [`import/sales_agent/`](import/sales_agent/):
a sales / customer-success intelligence assistant working a book of ~48
enterprise car-rental customer accounts for **RoadFlex** (a corporate mobility
provider). The minimum setup needs only Neo4j plus an LLM provider key (OpenAI
by default): seed a blank Neo4j
database with the RoadFlex graph, persona, and bootstrap learnings, then start a
session. BigQuery/Neocarta and Diffbot are optional add-ons for warehouse
queries, catalog search, firmographics, and live news.

The companion command for walking through this setup is
[`sales_agent_demo`](commands/sales_agent_demo.md).

### 1. Configure the repo-root `.env`

Create or update **`./.env` at the repo root**. This is the active config for
the sales-agent demo: both the seeders and the hooks load it, and in a repo
checkout it takes precedence over the user-global
`~/.config/meta-knowledge-graph/.env`. Do not edit `.env.example` as the active
config; copy from it if needed:

```bash
cp .env.example .env
chmod 600 .env
```

Then edit `./.env` with these values:

```bash
# Required: Neo4j graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=neo4j

# Required: LLM calls for memory extraction. Calls route through litellm.
# Codex/dev runs default to OpenAI's model (litellm), so set OPENAI_API_KEY.
# Claude Code hook batches default to a Claude model and PREFER the logged-in
# Claude subscription (a Claude Code OAuth token from CLAUDE_CODE_OAUTH_TOKEN or
# the platform credential store). With no subscription token they fall back to
# litellm on whatever provider is configured here — an LLM_MODEL override, or any
# provider key below (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / ...).
OPENAI_API_KEY=<your-openai-api-key>
# Optional: reuse your Claude subscription for Claude Code hook extraction
# (preferred over any API key). Mint one with: claude setup-token
# CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
# Optional: any litellm provider key also serves as the fallback, e.g.:
# ANTHROPIC_API_KEY=sk-ant-api03-...
# Optional explicit model override for every harness (any litellm model):
# LLM_MODEL=anthropic/claude-haiku-4-5

# Optional: Diffbot live news / firmographic enrichment
DIFFBOT_TOKEN=<your-diffbot-token>

# Optional: BigQuery warehouse + Neocarta catalog
GCP_PROJECT_ID=<your-gcp-project>
BIGQUERY_DATASET_ID=acme_corp
BIGQUERY_MCP_URL=https://bigquery.googleapis.com/mcp
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
# or:
# GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account", ...}'

# Optional: Neocarta embedding overrides. With the default model,
# OPENAI_API_KEY above is also the embedding provider key.
# EMBEDDING_MODEL=text-embedding-3-small
# EMBEDDING_DIMENSIONS=1536
# GCP_BILLING_PROJECT_ID=<billing-project>  # defaults to GCP_PROJECT_ID
# BIGQUERY_REGION=US
```

### 2. Seed the minimum graph

You can start from a blank Neo4j database. This path does not require BigQuery
or Diffbot:

```bash
uv run python import/sales_agent/seed_neo4j.py
uv run python import/sales_agent/seed_learnings.py
uv run python import/sales_agent/seed_system_prompt.py
```

Each seeder is independently re-runnable. Neo4j seeders MERGE on natural keys;
seed-owned `USES_PRODUCT` and `HAS_CONTACT` relationships are refreshed so a
re-run reflects the current deterministic dataset.

Optional add-ons:

- **Diffbot:** set `DIFFBOT_TOKEN` in the repo-root `./.env` and restart the MCP
  server. No seeding step is required; `search_news` and `enhance_entity` appear
  when the token is set.
- **BigQuery + Neocarta:** authenticate to GCP, set `GCP_PROJECT_ID`,
  `BIGQUERY_DATASET_ID`, `BIGQUERY_MCP_URL`, and one Google auth method in the
  repo-root `./.env`, then seed the warehouse and catalog. Neocarta also needs
  the provider key for `EMBEDDING_MODEL`; the default OpenAI embedding model uses
  `OPENAI_API_KEY`.

```bash
uv run python import/sales_agent/seed_bigquery.py
uv run python import/sales_agent/run_neocarta.py
```

You can also use the orchestrator. It verifies the mandatory Neo4j connection,
runs the minimum Neo4j-backed seeders, and runs the optional BigQuery/Neocarta
seeders only when GCP env/auth is available:

```bash
uv run python import/sales_agent/seed_all.py
```

`seed_all.py` can run these seeders:

| Seeder | What it loads |
|---|---|
| `seed_bigquery.py` | `accounts`, `products`, `account_product_usage` (monthly time series), `account_contacts`, `account_renewals` under `$GCP_PROJECT_ID.$BIGQUERY_DATASET_ID`. |
| `run_neocarta.py` | Neocarta catalog metadata, then LiteLLM-powered embeddings for the seeded BigQuery dataset. Run after `seed_bigquery.py` so the warehouse tables and descriptions exist. |
| `seed_neo4j.py` | `:Account` / `:Product` / `:Contact` / `:CSM` nodes plus `USES_PRODUCT` (utilization, revenue), `HAS_CONTACT`, and `OWNS` relationships. |
| `seed_learnings.py` | Bootstrap `:Learning` nodes (durable facts and decisions alike) so the first session already has scoped project memory. |
| `seed_system_prompt.py` | Persists `system_prompt.md` (the RoadFlex sales persona) as `(:SystemPrompt {name: 'default'})`. |

To preview the generated dataset without touching any database:
`uv run python import/sales_agent/seed_data.py`. Dataset design notes live in
[`import/sales_agent/README.md`](import/sales_agent/README.md).

### 3. Register the MCP server and hooks

See [Running MKG](#running-mkg) above for your harness — Claude Code via the
plugin, Codex via the plugin's
[`plugin/hooks/codex-hooks.json`](plugin/hooks/codex-hooks.json) or repo-local
[`.codex/config.toml`](.codex/config.toml) /
[`.codex/hooks.json`](.codex/hooks.json), or any custom harness that can spawn
an MCP server and fire lifecycle hooks.
The Neo4j-backed memory and graph tools mount in the minimum setup. With
`DIFFBOT_TOKEN` set, the server mounts `search_news` and `enhance_entity`. With
`BIGQUERY_MCP_URL` set and Google auth available, it mounts
`bigquery_execute_query`; with `GCP_PROJECT_ID` / `BIGQUERY_DATASET_ID` set and
the `EMBEDDING_MODEL`'s provider key available (OpenAI by default), it also
mounts the `neocarta_*` catalog tools.

### 4. Start a session

On SessionStart the hook injects the persisted RoadFlex persona, and prompt
submits inject the most relevant seeded learnings. Try:

- *"Which accounts renew in the next 90 days and which of them are at risk?"*
- *"Where do we have expansion room — accounts running over contracted capacity?"*
- *"Build me a brief on Accenture: footprint, contacts, and recent news."*
- *"Roll up the book of business by CSM."*

From there the loop takes over: every session's events are logged, and memory
extraction distills new learnings (project- and user-scoped, covering durable
facts and decisions alike) on Stop. Each later session starts with the most
relevant of those — plus durable facts about the user — already
injected.

## Configuration

| Env Variable | CLI Flag | Default | Notes |
|---|---|---|---|
| `NEO4J_URI` | `--db-url` | `bolt://localhost:7687` | |
| `NEO4J_USERNAME` | `--username` | `neo4j` | |
| `NEO4J_PASSWORD` | `--password` | `password` | |
| `NEO4J_DATABASE` | `--database` | `neo4j` | |
| `NEO4J_TRANSPORT` | `--transport` | `stdio` | |
| `OPENAI_API_KEY` | — | — | Default provider key for Codex/dev runs: the hooks call `LLM_MODEL` and Neocarta embeds with `EMBEDDING_MODEL`, both via litellm, and both default to OpenAI models outside Claude Code. Claude Code hook batches can default to Claude subscription OAuth when `LLM_MODEL` is unset. For other providers, set the relevant model var and supply that provider's key. |
| `DIFFBOT_TOKEN` | — | — | Enables `search_news` and `enhance_entity` when set. |
| `LLM_MODEL` | — | client-aware | The single model knob for every LLM call (memory extraction), resolved through litellm. Explicit values always win. When unset, Codex/dev runs use `gpt-5.4-mini`; Claude Code hook batches use `anthropic/claude-haiku-4-5`. Any litellm model string works (e.g. `anthropic/claude-haiku-4-5`, `gemini/gemini-2.5-flash`). For Anthropic/Claude models, explicit `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or `ANTHROPIC_BASE_URL` wins; otherwise MKG tries the Claude Code platform credential store, then `CLAUDE_CODE_OAUTH_TOKEN` as a headless fallback. |
| `MKG_DEFAULT_LLM_MODEL` | — | — | Optional fallback override used only when `LLM_MODEL` is unset. |
| `MKG_LLM_BACKEND` | — | `auto` | `auto` uses litellm. Set `claude_cli` only to force the older headless `claude -p` backend. |
| `EMBEDDING_MODEL` | — | `text-embedding-3-small` | litellm embedding model for the optional Neocarta catalog (seed + runtime). Any litellm embedding model works (e.g. `cohere/embed-english-v3.0`, `gemini/text-embedding-004`); supply the matching provider's key. |
| `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID` | — | — | Optional BigQuery/Neocarta settings. Required, with the `EMBEDDING_MODEL` provider key, to mount the Neocarta catalog tools; `GCP_PROJECT_ID` is also the project queried by `bigquery_execute_query`. |
| `BIGQUERY_MCP_URL` | — | — | Optional. Mounts `bigquery_execute_query` when set, e.g. `https://bigquery.googleapis.com/mcp`. For `googleapis.com` URLs a Google ADC bearer token is fetched automatically. |
| `BIGQUERY_MCP_AUTH`, `BIGQUERY_MCP_HEADERS` | — | — | Optional explicit bearer token / JSON dict of extra headers for the BigQuery MCP endpoint. |
| `BIGQUERY_REGION`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` | — | — | Optional; forwarded to the Neocarta subprocess when set. `BIGQUERY_REGION` (default `US`) also sets the dataset location when seeding the sales-agent demo. |
| `GCP_BILLING_PROJECT_ID` | — | falls back to `GCP_PROJECT_ID` | Optional billing project for the sales-agent BigQuery seeder. |
| `GCP_SERVICE_ACCOUNT_JSON` / `GOOGLE_APPLICATION_CREDENTIALS` | — | — | GCP auth for optional BigQuery seeding and Neocarta when application-default credentials are not already available: inline service-account JSON (written to a temp file) or a credentials file path. |

## TODO / Roadmap

- [ ] **Improving memory** — richer retrieval (semantic + structural search),
      approval tooling for promoting candidate learnings, and time-decaying
      confidence with re-validation.
- [ ] **GDS intelligence engine** — generate intelligence from the accumulated
      graph: implicit relationship discovery (node similarity), critical
      decision points (betweenness centrality), and behavior clustering
      (community detection).
