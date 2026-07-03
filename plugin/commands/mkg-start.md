---
description: Onboard a new MKG agent. Check that the meta-knowledge-graph MCP is mounted and which tools are live, inspect the graph state, then either launch the RoadFlex sales demo or build a custom agent persona by capturing user memories.
argument-hint: [optional "sales" or "custom" to skip the menu]
---

# MKG Start

Onboard the user onto the **Meta Knowledge Graph (MKG)**. This command does three
things: confirm the `meta-knowledge-graph` MCP server is actually mounted and
report which tools are live, inspect what is already in the graph, then guide the
user down one of two paths — the **RoadFlex sales demo**, or a **custom agent**
whose persona you build by capturing durable user memories.

Optional argument — pre-pick a path (`sales` or `custom`) to skip the menu:
**$ARGUMENTS**

Be concrete, never echo secret values, and prefer recall over re-asking: if
SessionStart/UserPromptSubmit already injected the user's name, project, or goals,
use them instead of asking again.

## Step 1 — Confirm the MCP and discover live tools

Inspect the tools you can actually call right now. Do **not** assume a tool
exists from prior context — the mounted set varies per session and per `.env`.

1. Look at your available `mcp__meta-knowledge-graph__*` tools (in plugin mode the
   prefix may be `mcp__plugin_meta-knowledge-graph_meta-knowledge-graph__*`).
2. **If none are present**, the MCP server is not mounted. Stop the flow and tell
   the user to set `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` (and an LLM
   credential — a `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` to reuse the
   Claude subscription, or any litellm provider key such as `OPENAI_API_KEY` /
   `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` as the fallback) in
   `~/.config/meta-knowledge-graph/.env`, then restart the session. Do not
   continue until the tools appear.
3. **If they are present**, report which capability groups are live by what you
   see — this is a fact about the environment, not a guess:

   | Group | Tools | Mounted when |
   |---|---|---|
   | Project memory & graph | `project_get_context`, `project_add_learning`, `neo4j_get_schema`, `neo4j_read_cypher` | Always |
   | Diffbot research | `search_news`, `enhance_entity` | `DIFFBOT_TOKEN` set |
   | BigQuery warehouse | `bigquery_execute_query` | `BIGQUERY_MCP_URL` set |
   | Neocarta catalog | `neocarta_*` | `GCP_PROJECT_ID` + `BIGQUERY_DATASET_ID` + embedding key set |

> Matching the toolset to the chosen purpose (provisioning tools dynamically per
> agent) is future work — for now just report what is mounted and proceed.

## Step 2 — Inspect the current graph state

Use `neo4j_read_cypher` (read-only) to see what already exists, so you can route
the user correctly instead of re-seeding over their work. One round-trip:

```cypher
RETURN
  COUNT { MATCH (sp:SystemPrompt {name:'default'}) RETURN sp } AS has_persona,
  COUNT { MATCH (a:Account) RETURN a } AS accounts,
  COUNT {
    MATCH (l:Learning {scope:'user', status:'candidate'})
    WHERE l.consolidated_at IS NULL
       OR toString(coalesce(l.updated_at, l.created_at)) > l.consolidated_at
    RETURN l
  } AS pending_user_memories
```

If `has_persona > 0`, peek at the active persona so you can tell the user what
agent is currently seeded (wrap any temporal in `toString(...)` — Neo4j dates
serialize as `{}` otherwise):

```cypher
MATCH (sp:SystemPrompt {name:'default'})
RETURN sp.version AS version, left(sp.content, 200) AS preview
```

Interpret the state:

- `has_persona = 0` → no custom persona seeded; SessionStart is falling back to
  the generic MKG bootstrap prompt. A clean slate for **either** path.
- `accounts > 0` → the RoadFlex sales graph is already seeded.
- `pending_user_memories` → how many user-scoped memories are queued for the
  persona consolidation service (it fires once **more than 5** are pending; see
  Path B).

## Step 3 — Offer the two paths

If `$ARGUMENTS` already names a path, skip the question. Otherwise ask the user
which they want, in one friendly question, summarizing the state you found:

- **A — RoadFlex sales demo:** the shipped end-to-end persona (a B2B sales /
  customer-success assistant over ~48 enterprise car-rental accounts).
- **B — Custom agent:** you interview the user about their agent's purpose and
  capture it as memories; MKG folds those into a custom persona.

---

## Path A — RoadFlex sales demo

Hand off to the dedicated setup, which configures the env, seeds Neo4j (and
optional Diffbot / BigQuery / Neocarta), and verifies the tools mount:

- Invoke the **`meta-knowledge-graph:sales_agent_demo`** skill, or follow
  [`commands/sales_agent_demo.md`](commands/sales_agent_demo.md).

Before running any sales-demo setup step that modifies env files, seeds Neo4j,
creates/replaces BigQuery tables, or rebuilds the Neocarta catalog, summarize
the intended target(s) and command(s), then wait for explicit user approval.

If Step 2 already showed `accounts > 0` and the sales persona is active, don't
re-seed — confirm it's live and suggest a first query, e.g. *"Which accounts
renew in the next 90 days and which are at risk?"* Restart the session if the
persona was just seeded so SessionStart injects it.

---

## Path B — Custom agent (build the persona from user memories)

You will **learn the purpose of the agent** from the user, store it as durable
**user-scoped** memories, and let MKG's consolidation service fold those into a
custom system prompt. The mechanism: the Stop / SessionEnd
`consolidate_system_prompt` service folds pending **user-scoped** candidate
learnings into `(:SystemPrompt {name:'default'})` once **more than 5** are
pending — so **6 user memories** is the trigger that regenerates the persona.

### B1 — Interview for the purpose

Ask concise questions (skip any already known from injected context). Cover:

1. **Mission** — what should this agent do, for what domain, to drive what outcome?
2. **Who they are** — the user's name and role.
3. **Data & systems** — the assets/sources the agent should reason over.
4. **Working style** — voice, verbosity, how grounded/data-driven answers must be.
5. **Constraints & guardrails** — recurring do's and don'ts.
6. **Success criteria** — what "good" / "done" looks like; standing priorities.

### B2 — Capture exactly six durable user memories

Write each answer as one concise, reusable fact (≤500 chars, no transcripts) via
`project_add_learning` with **`scope: "user"`**. User scope
is required — the consolidation service only counts `scope:'user'` candidates,
and these facts should follow the user across projects. The tool is idempotent on
(scope, text), so make the six **distinct**. Template — fill from the interview:

1. *Mission:* "<Name> wants an agent that <does X> over <domain Y> to achieve <Z>."
2. *Role:* "<Name> is a <role> responsible for <responsibilities>."
3. *Data:* "The agent should ground answers in <systems / assets / sources>."
4. *Voice:* "<Name> prefers <concise / direct / numerate> answers grounded in queried data."
5. *Guardrails:* "Always <constraint>; never <anti-pattern>."
6. *Success:* "Success means <criteria>; standing priorities are <priorities>."

After writing, re-run the `pending_user_memories` count from Step 2 and confirm
it is **> 5** (account for any already pending — capture enough new, distinct
facts to clear the threshold; six fresh ones always do on a clean graph).

### B3 — Trigger and verify the persona

The persona is **frozen at runtime** — it is read at SessionStart and rewritten
only by the seed scripts and the consolidation service. So:

1. When this turn ends, the background `consolidate_system_prompt` Stop hook sees
   `> 5` pending user memories (and, on a fresh setup, no prior consolidation, so
   no cooldown) and folds them into `(:SystemPrompt {name:'default'})`, archiving
   any previous as a `:SystemPromptVersion`.
2. Tell the user to **start a new session** (or `/clear`) so SessionStart injects
   the freshly consolidated persona. They can re-run `/mkg-start` anytime to add
   more memories and re-consolidate.
3. To verify after the next turn, re-read the persona with the `version` /
   `preview` query from Step 2 — the version should have bumped and the content
   should reflect their purpose.

Notes:

- **Clean base:** consolidation *edits the current prompt*. On a graph with no
  persona it starts from the neutral MKG baseline — ideal for a custom build. If
  a different persona (e.g. the sales demo) is already active and they want to
  replace it rather than extend it, re-seed the neutral baseline first with the
  bundled `hooks/seed_system_prompt.py`, then capture the six memories.
- **Demo pacing:** the service is rate-limited to once per
  `MKG_PROMPT_CONSOLIDATION_INTERVAL_HOURS` (default 24h) and gated on
  `MKG_PROMPT_CONSOLIDATION_THRESHOLD` (default 5). For a back-to-back live demo,
  set `MKG_PROMPT_CONSOLIDATION_INTERVAL_HOURS=0` in
  `~/.config/meta-knowledge-graph/.env` to bypass the cooldown.

## Guardrails

- Don't re-seed or overwrite an existing persona or sales graph without telling
  the user what's already there and confirming.
- For the RoadFlex sales demo setup, don't modify env files, seed databases, or
  rebuild warehouse/catalog data without explicit user approval.
- Capture only concise, user-provided facts — no raw transcripts or ephemeral
  state. Routine session signal is handled by the auto-capture pipeline; don't
  double-record it here.
- Never print secret values from the `.env`.
