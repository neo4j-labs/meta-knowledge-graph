# HITL Learning Loop Sample

This sample shows the human-in-the-loop path without writing fake memories into
the durable graph. It models the graph state after the Stop hook extractor and
automatic consistency gate have run.

## Synthetic Input

The extractor writes all new learnings as `status: "candidate"`. The consistency
gate then resolves the easy cases automatically:

```json
[
  {
    "id": "learning:demo:clean",
    "scope": "project",
    "text": "The API smoke test should verify both MCP startup and project context recall.",
    "gate_result": "approved",
    "consistency_status": "clean"
  },
  {
    "id": "learning:demo:duplicate",
    "scope": "project",
    "text": "MCP-written learnings are tagged with source agent-mcp.",
    "gate_result": "already_learned",
    "already_learned_of": "learning:demo:mcp-source"
  },
  {
    "id": "learning:demo:ambiguous-recall",
    "scope": "project",
    "text": "MKG recall is still mostly prompt-text matching and does not yet use graph neighborhoods.",
    "gate_result": "candidate",
    "consistency_status": "ambiguous",
    "conflicts": ["learning:demo:retrieval-hybrid"]
  },
  {
    "id": "learning:user:demo-style",
    "scope": "user",
    "text": "Tomaz wants proactive memory-architecture critique, not feature-only execution.",
    "gate_result": "candidate",
    "consistency_status": "unreviewed"
  }
]
```

The first two do not appear in HITL review. The third is a project-scoped
ambiguous contradiction, and the fourth is a user-scoped candidate. Those two
are the human-owned queue.

## Review Queue Output

`project_review_queue(project_id="demo-memory-agent")` would return:

```json
{
  "project_id": "demo-memory-agent",
  "count": 2,
  "queue": [
    {
      "id": "learning:user:demo-style",
      "scope": "user",
      "text": "Tomaz wants proactive memory-architecture critique, not feature-only execution.",
      "confidence": 0.94,
      "reason": "user_scoped_candidate",
      "consistency_status": "unreviewed",
      "updated_at": "2026-07-15T08:10:00Z",
      "conflicts": []
    },
    {
      "id": "learning:demo:ambiguous-recall",
      "scope": "project",
      "text": "MKG recall is still mostly prompt-text matching and does not yet use graph neighborhoods.",
      "confidence": 0.88,
      "reason": "ambiguous_contradiction",
      "consistency_status": "ambiguous",
      "updated_at": "2026-07-15T08:12:00Z",
      "conflicts": [
        {
          "id": "learning:demo:retrieval-hybrid",
          "status": "approved",
          "confidence": 0.83,
          "text": "MKG recall uses hybrid vector and keyword retrieval over project learnings and recent observations.",
          "judge_reason": "Both describe current retrieval; cannot tell from the texts which reflects the live implementation."
        }
      ]
    }
  ]
}
```

## Walkthrough

`/mkg-review` presents the queue as a numbered triage list first (kind in plain
words, text, age, and for conflicts the clashing item plus the judge's
rationale), and the user may answer several items in one message; each answered
item is still applied individually. The snake_case action names below are wire
vocabulary — the command shows plain labels and maps them.

For the user-scoped candidate, the reviewer sees:

```text
1. A fact about you (3 days ago):
   "Tomaz wants proactive memory-architecture critique, not feature-only execution."
   Choices: keep it (approve) · fix the wording (edit_approve) · discard it (reject)
```

If the reviewer chooses `approve`, `project_resolve_learning` promotes it:

```json
{
  "action": "approve",
  "learning": {
    "id": "learning:user:demo-style",
    "status": "approved",
    "scope": "user",
    "reviewed_by": "human",
    "consistency_status": "human_reviewed"
  }
}
```

That user fact is now trusted memory and becomes eligible for persona
consolidation. It does not rewrite the persona immediately; the consolidation
service only folds approved user facts once more than five are pending.

For the ambiguous project contradiction, the reviewer sees both sides and why
the judge punted:

```text
2. A conflicting project fact (3 days ago):
   New:      "MKG recall is still mostly prompt-text matching and does not yet use graph neighborhoods."
   Clashes:  "MKG recall uses hybrid vector and keyword retrieval over project learnings and recent observations."
   Judge:    Both describe current retrieval; cannot tell from the texts which reflects the live implementation.
   Choices: the new one is right (keep_new) · the existing one is right (keep_existing)
            · both are true (keep_both) · discard the new one (reject)
```

Resolution effects:

```json
[
  {
    "action": "keep_new",
    "effect": "candidate becomes approved; existing learning becomes rejected; candidate -[:SUPERSEDES]-> existing"
  },
  {
    "action": "keep_existing",
    "effect": "candidate becomes rejected; candidate -[:CONTRADICTED_BY]-> existing"
  },
  {
    "action": "keep_both",
    "effect": "candidate becomes approved; unresolved CONTRADICTS edges are deleted"
  },
  {
    "action": "reject",
    "effect": "candidate becomes rejected; embedding is cleared; unresolved CONTRADICTS edges are deleted"
  }
]
```

## Persona Consolidation Loop

After review, the separate Stop / SessionEnd consolidation service checks:

```cypher
MATCH (l:Learning {scope: 'user', status: 'approved'})
WHERE l.consolidated_at IS NULL
   OR coalesce(l.updated_at, l.created_at) > l.consolidated_at
RETURN count(l) AS pending
```

When `pending > MKG_PROMPT_CONSOLIDATION_THRESHOLD` (default `5`) and the
cooldown has elapsed, the service:

1. Fetches approved user facts only.
2. Fences them as untrusted data inside the consolidation prompt.
3. Asks the LLM to revise `(:SystemPrompt {name: "default"})`.
4. Archives the old prompt as `:SystemPromptVersion`.
5. Writes the new current prompt version.
6. Stamps folded learnings with `consolidated_at`.

Raw user-scoped candidates never enter this loop. The human review decision is
the boundary between "captured hint" and "trusted persona input."
