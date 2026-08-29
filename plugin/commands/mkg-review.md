---
description: Review the MKG learning queue. Surface learnings and skill proposals awaiting a human decision — ambiguous contradictions, user-scoped candidate facts, and distilled skills the automatic machinery cannot activate on its own — walk them one at a time, and apply each decision through the resolver tools.
argument-hint: [optional project id; defaults to the current project]
---

# MKG Review

Process the **human-review queue** for the Meta Knowledge Graph. The automatic
consistency gate resolves most freshly-extracted learnings on its own, but it
deliberately leaves two kinds of item for a person:

- **Ambiguous contradictions** — a new project-scoped candidate that genuinely
  conflicts with existing memory, where the judge could not tell which side is
  right.
- **User-scoped candidate facts** — durable facts about the person that would be
  folded into the cross-project persona. These are never auto-approved, because
  an unreviewed fact must not be able to rewrite the agent's identity on its own.
- **Skill proposals** — reusable procedures the background skill-distillation
  service compiled from clusters of approved learnings. A skill changes how the
  agent works, so — exactly like a user fact — it is never activated without a
  human decision. They arrive in the `skill_proposals` list of the same queue.

Nothing reaches the trusted tier (the persona, or live skills) until you
approve it here.

Optional argument — a project id to review instead of the current one:
**$ARGUMENTS**

## Step 1 — Load the queue

Call `project_review_queue` (in plugin mode the prefix may be
`mcp__plugin_meta-knowledge-graph_meta-knowledge-graph__project_review_queue`),
passing `project_id` only if the user named one in the argument.

- If the tool is not mounted, the MCP server is not running the current build.
  Tell the user to restart/reinstall the plugin so the review tools appear, and
  stop — do not hand-edit the graph with raw Cypher to approve memory.
- If `count` is `0`, report that the queue is empty and stop.
- The result carries two lists: `queue` (learnings) and `skill_proposals`
  (distilled skills). Triage learnings first, then skill proposals.

## Step 2 — Triage: show the whole queue first

Present every queued item up front as a compact numbered list, oldest first —
one or two lines per item:

- **Kind, in plain words** — say "a fact about you" for
  `user_scoped_candidate` and "a conflicting project fact" for
  `ambiguous_contradiction`. Never show the raw reason codes.
- **The learning text** (trim to ~120 characters; offer to expand).
- **Age** — a human age derived from `updated_at` ("2 days ago"), not the raw
  timestamp.
- For a conflict, one line per `conflicts` entry — "clashes with: <existing
  text>" — plus the judge's `judge_reason` when present, so the user sees why
  the machine could not decide (e.g. *judge: both describe current retrieval;
  cannot tell which is live*).

Then ask for decisions. Offer the choices in **plain language** and map them to
wire actions yourself — the snake_case action names are API vocabulary and must
not appear in the conversation:

| Item kind | Offer (plain label → wire action) |
|---|---|
| Fact about the user (no conflict) | "keep it" → `approve` · "fix the wording" → `edit_approve` · "discard it" → `reject` |
| Conflicting project fact | "the new one is right" → `keep_new` · "the existing one is right" → `keep_existing` · "both are true" → `keep_both` · "discard the new one" → `reject` |

What each choice does (explain when recommending, or on request):

- **keep it** (`approve`) — promote to trusted memory. A user fact becomes
  eligible for persona consolidation.
- **fix the wording** (`edit_approve`) — the user supplies or confirms a
  rewrite (passed as `edited_text`), then it is promoted.
- **discard it** (`reject`) — drop it from live memory.
- **the new one is right** (`keep_new`) — the candidate wins; the existing
  item it clashes with is superseded and retired.
- **the existing one is right** (`keep_existing`) — existing memory wins; the
  candidate is retired.
- **both are true** (`keep_both`) — the two are compatible; keep the candidate
  and clear the conflict.

Collecting decisions may be batched; deciding for the user may not. The user
can answer several items in one message ("keep 1 and 2, discard 3, show me 4 in
full") — treat that as N explicit decisions. Never resolve an item the user did
not explicitly decide: anything unaddressed stays queued for next time. With
three or fewer items, walking them one at a time is fine too. Expand any item
to full detail (complete text, both sides of a conflict, confidence, judge
reason) whenever the user asks or seems unsure.

Recommend a default when the right call is obvious (e.g. "the new one is
right" when the existing item is clearly stale), but let the user decide.
These are durable writes.

## Step 3 — Apply each decision

For each decided item, call `project_resolve_learning` with its `learning_id`
and the mapped wire `action`. Pass `edited_text` for a rewording. For "the
new/existing one is right" against one specific conflicting item, pass its id
as `conflict_id` (omit to resolve against all of them). Report one plain-words
line per item — e.g. "#2 kept — now trusted memory" — then continue.

## Step 3b — Skill proposals

For each entry in `skill_proposals`, present:

- **What it is, in plain words** — "a new skill: <name>" for a `create`
  proposal, or "an update to the skill <slug>" for an `update` proposal.
- **The description** (this is what retrieval matches on) and the **proposed
  content** — show the content in full; it is the procedure the agent will
  follow. For an update, summarise what changed against `current_content`
  rather than dumping both.
- **Why the proposer thinks it's worth it** — the `rationale`, plus the
  motivating learnings from `derived_from` (one line each), so the user can
  see every step traces back to approved memory.

Offer the choices in plain language and map them to wire actions yourself:

| Offer (plain label → wire action) |
|---|
| "activate it" → `approve` · "fix it first" → `edit_approve` · "discard it" → `reject` |

What each choice does (explain when recommending, or on request):

- **activate it** (`approve`) — the skill goes live: it becomes findable via
  `skill_search`, loadable via `skill_fetch`, and listed in the injected
  catalog line.
- **fix it first** (`edit_approve`) — the user supplies corrections; pass them
  as `edited_content` and/or `edited_description`, then it goes live.
- **discard it** (`reject`) — the proposal is dropped. Rejecting a *new* skill
  frees its learnings for future clustering; rejecting an *update* leaves the
  live skill exactly as it was. Either way no learning is ever touched.

Apply each decision with `project_resolve_skill`, passing the item's
`skill_id` and the mapped `action` (plus `edited_content` /
`edited_description` for a fix-up). The same batching rule applies: the user
may decide several at once, but never resolve one they did not decide.

If the user asks to remove a live skill that is misfiring, that is
`project_resolve_skill` with action `retire` — mention it only when relevant.

## Step 4 — Close out

When the queue is drained (or the user stops), give a short summary: how many
items were approved, rejected, edited, or merged, how many skills were
activated or discarded, and how many remain. If any user-scoped facts were
approved, mention that they will be folded into the persona on a later
background consolidation once enough have accumulated — no action needed now.
If any skills were activated, mention they are now live in `skill_search`.

Never approve, reject, or edit memory the user did not decide on, and never use
raw `write-cypher` to change learning or skill status — always go through
`project_resolve_learning` / `project_resolve_skill` so provenance, version
history, and the contradiction edges stay correct.
