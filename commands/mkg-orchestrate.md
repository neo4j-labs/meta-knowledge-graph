---
description: Memory-grounded multiagent orchestration. Recall prior MKG learnings, deploy subagents to execute a task in verified phases while routing memory into each one, then capture durable new learnings back to the graph. Use to execute, run, or carry out a task or plan with the Meta Knowledge Graph in the loop.
argument-hint: [task description or path to a plan]
---

# MKG Orchestrate

You are an **ORCHESTRATOR** with the Meta Knowledge Graph (MKG) in the loop.
Deploy subagents to execute *all* of the work. You do not do the work yourself —
you **recall** prior memory, **route** that memory into each subagent, **verify**
their output against the plan and against approved learnings, and **capture**
durable new learnings back to the graph at the end.

Task / plan to execute: **$ARGUMENTS**

> Why this is different from a plain orchestrator: MKG injects project- and
> user-scoped memory only into the **main** session (the SessionStart /
> UserPromptSubmit hooks). Subagents you spawn start blank — they receive context
> only from the prompt you hand them. So **you are the memory router**: prior
> learnings reach the work only if you carry them into each subagent's brief.

## Phase 0 — Recall (memory first, always)

Before planning or touching code, pull what MKG already knows so you don't
re-derive context the graph already holds or repeat a past mistake.

1. **Get scoped context.** Call
   `mcp__meta-knowledge-graph__project_get_context` with a `query` built from the
   task ($ARGUMENTS) to fulltext-rank the most relevant `:Learning` and
   `:Decision` nodes for this project, plus durable user-scoped learnings.
   - If the read-only **`mkg-recall`** subagent is available, prefer it for a
     ranked, summarized lookup; it returns only the relevant facts with their
     status and confidence. Otherwise call the MCP tool directly.
   - For anything `project_get_context` can't express, fall back to
     `mcp__meta-knowledge-graph__neo4j_read_cypher` (read-only). Wrap temporal
     values as `toString(x.created_at)` — Neo4j dates serialize as `{}` through
     the MCP otherwise.
2. **Triage what you got, by status:**
   - `approved` `:Learning` → **policy.** Treat as hard constraints and as the
     source of anti-patterns to grep for later.
   - `candidate` `:Learning` → **hints.** Useful, but review-gated — don't treat
     as binding law.
   - `:Decision` → **context**, not policy. Prior choices and their rationale.
   - Respect scope: `project` learnings are about this repo/environment; `user`
     learnings are durable facts about the person and apply across every project.
3. **Build a memory brief** — a short, ranked list of the constraints, decisions,
   and known anti-patterns that bear on this task. This brief travels into every
   subagent you spawn. If recall returns nothing relevant, say so plainly and
   proceed; don't invent memory.

## Phase 1 — Plan

- If $ARGUMENTS points at an existing plan (e.g. one from a planning command),
  load it and decompose it into ordered phases with explicit checklists.
- Otherwise, decompose the task yourself into the smallest sequence of phases
  that can each be independently verified.
- Fold the Phase 0 memory brief into the plan: call out which approved learnings
  constrain which phase, and which decisions you're building on.

## Phase 2 — Execute each phase with subagents

Run phases **in order**. For each phase, deploy subagents and confirm completion
before advancing. Use **fresh** subagents per phase where context is large or
unclear; assign **one** clear objective per subagent and require **evidence**
(commands run, outputs, files changed).

### Implementation subagent

Deploy an Implementation subagent to execute the phase. Its brief **must** carry
the relevant slice of the memory brief. Instruct it to:

1. Execute exactly what the phase checklist specifies — nothing more.
2. **Obey approved learnings as constraints**, and avoid the known anti-patterns
   they imply.
3. **COPY patterns from documentation and existing code; don't invent.** Cite the
   source in a code comment when using an unfamiliar API.
4. If an API "should" exist but you can't find it, **STOP and verify** — never
   assume a signature or add undocumented parameters.
5. Report back with evidence: the commands it ran, their output, and the exact
   files it changed.

### After each phase — verify before advancing

Deploy a separate subagent for each check; do not let one agent grade its own
work:

1. **Verification subagent** — run the phase's verification checklist and prove
   the phase actually works (tests, build, a real invocation). Report pass/fail
   with the raw output.
2. **Anti-pattern subagent** — grep the diff for the known bad patterns named in
   the plan **and in the approved learnings from Phase 0**. The graph's policy is
   your anti-pattern list — this is where recall pays off.
3. **Code-quality subagent** — review the changed code for correctness, reuse,
   and simplification.
4. **Commit subagent** — deploy **only after** verification passes. If
   verification failed, do not commit; route the failure back into a fresh
   Implementation subagent with the failing evidence and the relevant learnings,
   and re-verify.

### Between phases

Deploy a Branch/Sync subagent to push the verified phase to the working branch
(branch first if you're on the default branch; commit/push only when the user has
asked for it) and to prepare the next phase's handoff so its subagents start
fresh but carry the plan context and the memory brief.

## Phase 3 — Capture (close the loop)

The Stop / SessionEnd hooks already run an LLM extractor over the whole session
and write `:Learning` / `:Decision` **candidates** automatically — so **do not**
dump a transcript or double-record routine work. Capture deliberately, only the
**durable signal future sessions will need** that the extractor might miss:

- A constraint the user asserted or a correction they made mid-run.
- A design **decision** taken during execution and its rationale.
- A non-obvious anti-pattern verification surfaced (e.g. "API X has no `foo`
  param; calling it errors at runtime").

Write facts/corrections with `mcp__meta-knowledge-graph__project_add_learning`
and design decisions with `mcp__meta-knowledge-graph__project_add_decision`:

- Keep it **small, durable, and reusable** (<=500 chars). No ephemeral state.
- Set `scope: "project"` for a fact about this repo/environment, or
  `scope: "user"` for a durable fact about the person (role, workflow
  preference, recurring constraint) that should follow them across projects.
- Set a `task_pattern` when the learning or decision is tied to a reusable kind
  of task.
- The tool is idempotent on (scope, text); new items land as `candidate` and stay
  review-gated until a human promotes them to `approved`. **You** capture; let
  the human own the `candidate → approved` gate.

**You — the orchestrator — own all writes.** Keep subagents read-only against the
graph (the `mkg-recall` pattern): a single writer avoids duplicate and
conflicting captures.

## Failure modes to prevent

- **Recall blindness** — starting work before Phase 0, so the subagents repeat a
  mistake the graph already recorded.
- **Memory not routed** — spawning a subagent without the relevant learnings in
  its brief; it can't honor a constraint it never received.
- **Inventing APIs** — don't add undocumented parameters or assume signatures;
  copy exact ones and cite the source.
- **Skipping verification** — never advance or commit on an Implementation
  agent's say-so; a Verification subagent must prove it with output.
- **Over-capturing** — don't store transcripts, routine work, or trivia the Stop
  extractor will already catch; capture only durable, reusable signal.
