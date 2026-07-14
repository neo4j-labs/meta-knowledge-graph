---
description: Review the MKG learning queue. Surface learnings awaiting a human decision — ambiguous contradictions and user-scoped candidate facts the automatic gate cannot resolve — walk them one at a time, and apply each decision through the resolver tool.
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

Nothing reaches the trusted tier (or the persona) until you approve it here.

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

## Step 2 — Walk the items, oldest first

The queue is ordered oldest-first. For **each** item, show the user:

- the learning `text`, its `scope` (`user` or `project`), and `reason`
  (`user_scoped_candidate` or `ambiguous_contradiction`);
- for a contradiction, every entry in `conflicts` — the existing learning(s) it
  clashes with — so both sides are visible before deciding.

Then ask the user how to resolve it. Present only the choices that fit the item:

| Item kind | Offer |
|---|---|
| User-scoped candidate (no conflict) | **approve**, **edit_approve**, **reject** |
| Ambiguous contradiction | **keep_new**, **keep_existing**, **keep_both**, **reject** |

What each decision means:

- **approve** — promote to `approved` (trusted). For a user fact, this makes it
  eligible for persona consolidation.
- **edit_approve** — fix the wording first (pass `edited_text`), then approve.
- **reject** — drop it from live memory.
- **keep_new** — the new candidate wins; the existing item it contradicts is
  superseded and rejected.
- **keep_existing** — the existing item wins; reject this candidate.
- **keep_both** — the two are actually compatible; approve the candidate and
  clear the contradiction.

Recommend a default when the right call is obvious (e.g. `keep_new` when the
candidate is clearly newer and the old item is stale), but let the user decide.
Do not batch — confirm each item individually. These are durable writes.

## Step 3 — Apply each decision

For each item, call `project_resolve_learning` with its `learning_id` and the
chosen `action`. Pass `edited_text` for `edit_approve`. For `keep_new` /
`keep_existing` against a specific conflicting item, pass its id as `conflict_id`
(omit to resolve against all of them). Report the returned status back to the
user in one line, then move to the next item.

## Step 4 — Close out

When the queue is drained (or the user stops), give a short summary: how many
items were approved, rejected, edited, or merged, and how many remain. If any
user-scoped facts were approved, mention that they will be folded into the
persona on a later background consolidation once enough have accumulated — no
action needed now.

Never approve, reject, or edit memory the user did not decide on, and never use
raw `write-cypher` to change learning status — always go through
`project_resolve_learning` so provenance and the contradiction edges stay correct.
