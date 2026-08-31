from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))


def load_hook_module(name: str):
    module_path = ROOT / "hooks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


project_common = load_hook_module("project_common")
consolidate_skills = load_hook_module("consolidate_skills")


NOW = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)


class SkillGateTests(unittest.TestCase):
    def test_below_threshold_skips(self) -> None:
        for count in (0, 3):
            proceed, reason = consolidate_skills.skill_gate(
                pending_count=count,
                threshold=4,
                last_run_at=None,
                interval_hours=24.0,
                now=NOW,
            )
            self.assertFalse(proceed, f"count={count} should not trigger")
            self.assertIn("need at least 4", reason)

    def test_at_threshold_with_no_prior_run_proceeds(self) -> None:
        proceed, reason = consolidate_skills.skill_gate(
            pending_count=4,
            threshold=4,
            last_run_at=None,
            interval_hours=24.0,
            now=NOW,
        )
        self.assertTrue(proceed)
        self.assertIn(">= 4", reason)

    def test_recent_run_is_rate_limited(self) -> None:
        recent = (NOW - timedelta(hours=3)).isoformat()
        proceed, reason = consolidate_skills.skill_gate(
            pending_count=10,
            threshold=4,
            last_run_at=recent,
            interval_hours=24.0,
            now=NOW,
        )
        self.assertFalse(proceed)
        self.assertIn("rate-limited", reason)

    def test_cooldown_elapsed_proceeds(self) -> None:
        stale = (NOW - timedelta(hours=30)).isoformat()
        proceed, _ = consolidate_skills.skill_gate(
            pending_count=5,
            threshold=4,
            last_run_at=stale,
            interval_hours=24.0,
            now=NOW,
        )
        self.assertTrue(proceed)


class GroupingTests(unittest.TestCase):
    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(
            consolidate_skills.cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0
        )
        self.assertAlmostEqual(
            consolidate_skills.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0
        )
        self.assertEqual(consolidate_skills.cosine_similarity([], [1.0]), 0.0)
        self.assertEqual(
            consolidate_skills.cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0
        )

    def test_normalize_task_pattern(self) -> None:
        self.assertEqual(
            project_common.normalize_task_pattern("Plugin failure → verify ~/.claude!"),
            "plugin failure verify claude",
        )
        self.assertEqual(project_common.normalize_task_pattern(None), "")

    def test_match_prefers_exact_normalized_match(self) -> None:
        existing = [
            {"id": "tp-1", "normalized": "verify plugin release", "embedding": [0.0, 1.0]},
            {"id": "tp-2", "normalized": "rebuild cache venv", "embedding": [1.0, 0.0]},
        ]
        matched = consolidate_skills.match_task_pattern(
            "verify plugin release", [1.0, 0.0], existing, floor=0.65
        )
        self.assertEqual(matched, "tp-1")

    def test_match_falls_back_to_embedding_above_floor(self) -> None:
        existing = [
            {"id": "tp-1", "normalized": "verify plugin release", "embedding": [1.0, 0.0]},
        ]
        matched = consolidate_skills.match_task_pattern(
            "check release end to end", [0.95, 0.31], existing, floor=0.65
        )
        self.assertEqual(matched, "tp-1")
        no_match = consolidate_skills.match_task_pattern(
            "debug cypher query", [0.0, 1.0], existing, floor=0.65
        )
        self.assertIsNone(no_match)

    def test_match_without_embedding_is_exact_only(self) -> None:
        existing = [
            {"id": "tp-1", "normalized": "verify plugin release", "embedding": [1.0, 0.0]},
        ]
        self.assertIsNone(
            consolidate_skills.match_task_pattern(
                "verify the plugin release", None, existing, floor=0.65
            )
        )

    def test_group_by_pattern(self) -> None:
        groups = consolidate_skills.group_by_pattern(
            {"l1": "tp-1", "l2": "tp-1", "l3": "tp-2"}
        )
        as_sets = [frozenset(g) for g in groups]
        self.assertIn(frozenset({"l1", "l2"}), as_sets)
        self.assertIn(frozenset({"l3"}), as_sets)

    def test_coactivation_merges_groups_recalled_together(self) -> None:
        groups = [["l1", "l2"], ["l3"], ["l4"]]
        sessions = {
            "l1": {"s1", "s2"},
            "l2": {"s2", "s3"},
            "l3": {"s1", "s2", "s3"},  # jaccard with group 0 = 1.0
            "l4": {"s9"},
        }
        merged = consolidate_skills.merge_groups_by_coactivation(
            groups, sessions, threshold=0.5
        )
        as_sets = [frozenset(g) for g in merged]
        self.assertIn(frozenset({"l1", "l2", "l3"}), as_sets)
        self.assertIn(frozenset({"l4"}), as_sets)

    def test_coactivation_ignores_groups_without_recall_history(self) -> None:
        groups = [["l1"], ["l2"]]
        merged = consolidate_skills.merge_groups_by_coactivation(
            groups, {"l1": {"s1"}}, threshold=0.1
        )
        self.assertEqual(sorted(len(g) for g in merged), [1, 1])

    def test_coactivation_below_threshold_keeps_groups_apart(self) -> None:
        groups = [["l1"], ["l2"]]
        sessions = {"l1": {"s1", "s2", "s3"}, "l2": {"s3", "s4", "s5"}}
        merged = consolidate_skills.merge_groups_by_coactivation(
            groups, sessions, threshold=0.5
        )
        self.assertEqual(sorted(len(g) for g in merged), [1, 1])

    def test_mean_pairwise_similarity(self) -> None:
        embeddings = {
            "a": [1.0, 0.0],
            "b": [1.0, 0.0],
            "c": [0.0, 1.0],
        }
        self.assertAlmostEqual(
            consolidate_skills.mean_pairwise_similarity(["a", "b"], embeddings), 1.0
        )
        self.assertEqual(
            consolidate_skills.mean_pairwise_similarity(["a"], embeddings), 1.0
        )
        mixed = consolidate_skills.mean_pairwise_similarity(["a", "b", "c"], embeddings)
        self.assertAlmostEqual(mixed, 1.0 / 3.0)


class PartitionTests(unittest.TestCase):
    def test_anchor_component_becomes_patch_group_even_at_size_one(self) -> None:
        components = [["anchor-1", "new-1"]]
        patch, create, skipped = consolidate_skills.partition_components(
            components,
            {"anchor-1": ["skill:proj:release"]},
            min_cluster_size=2,
        )
        self.assertEqual(len(patch), 1)
        self.assertEqual(patch[0]["skill_id"], "skill:proj:release")
        self.assertEqual(patch[0]["members"], ["new-1"])
        self.assertEqual(create, [])
        self.assertEqual(skipped, [])

    def test_anchor_free_component_needs_min_cluster_size(self) -> None:
        components = [["new-1", "new-2"], ["lonely"]]
        patch, create, skipped = consolidate_skills.partition_components(
            components, {}, min_cluster_size=2
        )
        self.assertEqual(patch, [])
        self.assertEqual(len(create), 1)
        self.assertEqual(sorted(create[0]["members"]), ["new-1", "new-2"])
        self.assertEqual(skipped, [])

    def test_component_spanning_two_skills_is_surfaced_not_assigned(self) -> None:
        components = [["anchor-1", "anchor-2", "new-1"]]
        patch, create, skipped = consolidate_skills.partition_components(
            components,
            {"anchor-1": ["skill:proj:a"], "anchor-2": ["skill:proj:b"]},
            min_cluster_size=2,
        )
        self.assertEqual(patch, [])
        self.assertEqual(create, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["skill_ids"], ["skill:proj:a", "skill:proj:b"])

    def test_anchor_only_component_is_dropped(self) -> None:
        patch, create, skipped = consolidate_skills.partition_components(
            [["anchor-1"]], {"anchor-1": ["skill:proj:a"]}, min_cluster_size=2
        )
        self.assertEqual((patch, create, skipped), ([], [], []))


class RankingTests(unittest.TestCase):
    def test_patch_outranks_create_and_size_breaks_ties(self) -> None:
        embeddings = {k: [1.0, 0.0] for k in ("a", "b", "c", "d", "e")}
        confidence = {k: 0.5 for k in embeddings}
        groups = [
            {"kind": "create", "skill_id": None, "members": ["a", "b", "c"]},
            {"kind": "patch", "skill_id": "skill:proj:x", "members": ["d"]},
            {"kind": "create", "skill_id": None, "members": ["d", "e"]},
        ]
        ranked = consolidate_skills.rank_groups(groups, embeddings, confidence)
        self.assertEqual(ranked[0]["kind"], "patch")
        self.assertEqual(len(ranked[1]["members"]), 3)
        self.assertEqual(len(ranked[2]["members"]), 2)

    def test_patch_for_stale_skill_outranks_everything(self) -> None:
        embeddings = {k: [1.0, 0.0] for k in ("a", "b", "c", "d", "e")}
        confidence = {k: 0.5 for k in embeddings}
        groups = [
            {"kind": "patch", "skill_id": "skill:proj:fresh", "members": ["a", "b"]},
            {"kind": "patch", "skill_id": "skill:proj:stale", "members": ["c"]},
            {"kind": "create", "skill_id": None, "members": ["d", "e"]},
        ]
        ranked = consolidate_skills.rank_groups(
            groups, embeddings, confidence, {"skill:proj:stale"}
        )
        # The stale skill's patch wins despite being the smaller group.
        self.assertEqual(ranked[0]["skill_id"], "skill:proj:stale")
        self.assertEqual(ranked[1]["skill_id"], "skill:proj:fresh")
        self.assertEqual(ranked[2]["kind"], "create")


class FingerprintTests(unittest.TestCase):
    def test_order_invariant_and_membership_sensitive(self) -> None:
        one = consolidate_skills.group_fingerprint(["b", "a"])
        two = consolidate_skills.group_fingerprint(["a", "b"])
        other = consolidate_skills.group_fingerprint(["a", "b", "c"])
        self.assertEqual(one, two)
        self.assertNotEqual(one, other)
        self.assertEqual(len(one), 16)


class ParseProposalTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        parsed = consolidate_skills.parse_proposal('{"action": "ignore"}')
        self.assertEqual(parsed, {"action": "ignore"})

    def test_fenced_json(self) -> None:
        parsed = consolidate_skills.parse_proposal(
            '```json\n{"action": "create"}\n```'
        )
        self.assertEqual(parsed, {"action": "create"})

    def test_prose_wrapped_json(self) -> None:
        parsed = consolidate_skills.parse_proposal(
            'Here you go: {"action": "update", "content": "x"} hope that helps'
        )
        self.assertEqual(parsed["action"], "update")

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(consolidate_skills.parse_proposal("not json at all"))
        self.assertIsNone(consolidate_skills.parse_proposal('["a", "list"]'))


class ValidateProposalTests(unittest.TestCase):
    CREATE_GROUP = {"kind": "create", "skill_id": None, "members": ["l1", "l2"]}
    PATCH_GROUP = {"kind": "patch", "skill_id": "skill:proj:release", "members": ["l3"]}
    INVENTORY = [
        {
            "id": "skill:proj:release",
            "slug": "release",
            "name": "Release verification",
            "description": "Use when releasing.",
            "status": "approved",
            "content": "## Procedure\n1. ...",
            "version": 1,
            "has_pending": False,
        }
    ]

    FULL_CONTENT = (
        "## When to use\n...\n## Procedure\n1. ...\n"
        "## Pitfalls\n- ...\n## Verification\n- ..."
    )
    ERROR_CONTEXT = [
        {
            "id": "query-error-pattern:proj:neo4j_read_cypher:abc",
            "kind": "pattern",
            "tool_key": "neo4j_read_cypher",
            "title": "Unknown label",
            "error_signature": "no such label",
            "resolution": "check the schema first",
        },
        {
            "id": None,
            "kind": "raw_failure",
            "tool_key": "mcp__github__create_pr",
            "title": "failed tool call",
            "error_signature": "422 validation failed",
            "resolution": None,
        },
    ]

    def _create_proposal(self, **overrides):
        proposal = {
            "action": "create",
            "target_skill_slug": None,
            "name": "Cache venv rebuild",
            "description": "Use when the plugin cache venv loses its symlinks.",
            "content": self.FULL_CONTENT,
            "derived_from": ["l1", "l2"],
            "rationale": "coherent procedure",
        }
        proposal.update(overrides)
        return proposal

    def test_valid_create_normalizes_and_slugs(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(), self.CREATE_GROUP, self.INVENTORY
        )
        self.assertIsNone(error)
        self.assertEqual(normalized["slug"], "cache-venv-rebuild")
        self.assertEqual(normalized["derived_from"], ["l1", "l2"])

    def test_create_slug_collision_is_rejected(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(name="Release"), self.CREATE_GROUP, self.INVENTORY
        )
        self.assertIsNone(normalized)
        self.assertIn("collides", error)

    def test_derived_from_must_be_subset_of_group(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(derived_from=["l1", "outsider"]),
            self.CREATE_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(normalized)
        self.assertIn("unknown entries", error)
        self.assertIn("L1", error)

    def test_derived_from_labels_resolve_to_member_ids(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(derived_from=["L1", "L2"]),
            self.CREATE_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(error)
        self.assertEqual(normalized["derived_from"], ["l1", "l2"])

    def test_create_from_a_single_learning_is_rejected(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(derived_from=["l1"]),
            self.CREATE_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(normalized)
        self.assertIn("at least two learnings", error)

    def test_empty_derived_from_is_rejected(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(derived_from=[]),
            self.CREATE_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(normalized)
        self.assertIn("derived_from", error)

    def test_patch_group_rejects_create_action(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(derived_from=["l3"]),
            self.PATCH_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(normalized)
        self.assertIn("must be update or ignore", error)

    def test_create_group_rejects_update_action(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(action="update", target_skill_slug="release"),
            self.CREATE_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(normalized)
        self.assertIn("must be create or ignore", error)

    def test_valid_update_targets_the_group_skill(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(
                action="update",
                target_skill_slug="release",
                derived_from=["l3"],
                name=None,
                description=None,
            ),
            self.PATCH_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(error)
        self.assertEqual(normalized["slug"], "release")
        # Missing name/description fall back to the live skill's values.
        self.assertEqual(normalized["name"], "Release verification")
        self.assertEqual(normalized["description"], "Use when releasing.")

    def test_update_with_wrong_target_slug_is_rejected(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(
                action="update", target_skill_slug="other", derived_from=["l3"]
            ),
            self.PATCH_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(normalized)
        self.assertIn("target_skill_slug", error)

    def test_ignore_passes_with_rationale_only(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            {"action": "ignore", "rationale": "thematic, not procedural"},
            self.CREATE_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(error)
        self.assertEqual(normalized["action"], "ignore")

    def test_oversized_content_is_rejected(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(content="x" * (project_common.MAX_SKILL_CONTENT + 1)),
            self.CREATE_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(normalized)
        self.assertIn("exceeds", error)

    def test_unknown_action_is_rejected(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            {"action": "deploy"}, self.CREATE_GROUP, self.INVENTORY
        )
        self.assertIsNone(normalized)
        self.assertIn("unknown action", error)

    def test_content_missing_required_sections_is_rejected(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(content="## When to use\n...\n## Procedure\n1. ..."),
            self.CREATE_GROUP,
            self.INVENTORY,
        )
        self.assertIsNone(normalized)
        self.assertIn("missing required sections", error)
        self.assertIn("## Pitfalls", error)
        self.assertIn("## Verification", error)

    def test_informed_by_labels_resolve_to_pattern_ids(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(informed_by=["E1"]),
            self.CREATE_GROUP,
            self.INVENTORY,
            self.ERROR_CONTEXT,
        )
        self.assertIsNone(error)
        self.assertEqual(
            normalized["informed_by"],
            ["query-error-pattern:proj:neo4j_read_cypher:abc"],
        )

    def test_informed_by_raw_digest_label_yields_no_provenance_id(self) -> None:
        # E2 is a raw failure digest with no graph node behind it: valid to
        # cite, but nothing to link.
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(informed_by=["E2"]),
            self.CREATE_GROUP,
            self.INVENTORY,
            self.ERROR_CONTEXT,
        )
        self.assertIsNone(error)
        self.assertEqual(normalized["informed_by"], [])

    def test_informed_by_unknown_label_is_rejected(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(informed_by=["E9"]),
            self.CREATE_GROUP,
            self.INVENTORY,
            self.ERROR_CONTEXT,
        )
        self.assertIsNone(normalized)
        self.assertIn("informed_by", error)
        self.assertIn("E9", error)

    def test_informed_by_defaults_to_empty(self) -> None:
        normalized, error = consolidate_skills.validate_proposal(
            self._create_proposal(), self.CREATE_GROUP, self.INVENTORY
        )
        self.assertIsNone(error)
        self.assertEqual(normalized["informed_by"], [])


class ProposalPromptTests(unittest.TestCase):
    def test_prompt_frames_learnings_as_untrusted_and_lists_inventory(self) -> None:
        group = {"kind": "create", "skill_id": None, "members": ["l1"]}
        learnings = {
            "l1": {
                "id": "l1",
                "text": "uv sync must run in the cache venv",
                "task_pattern": "release verification",
                "confidence": 0.9,
            }
        }
        inventory = [
            {"id": "s1", "slug": "release", "description": "Use when releasing."}
        ]
        prompt = consolidate_skills.build_proposal_prompt(
            "Meta Knowledge Graph", "meta-knowledge-graph", group, learnings, inventory, None
        )
        self.assertIn("<<<LEARNINGS", prompt)
        self.assertIn("LEARNINGS>>>", prompt)
        self.assertIn("UNTRUSTED", prompt)
        self.assertIn("- release — Use when releasing.", prompt)
        self.assertIn("uv sync must run in the cache venv", prompt)
        self.assertIn("candidate for a NEW", prompt)
        self.assertIn(str(project_common.MAX_SKILL_CONTENT), prompt)

    def test_prompt_includes_labelled_tool_failures(self) -> None:
        group = {"kind": "create", "skill_id": None, "members": ["l1"]}
        learnings = {"l1": {"id": "l1", "text": "t", "task_pattern": "p"}}
        error_context = [
            {
                "id": "query-error-pattern:proj:neo4j_read_cypher:abc",
                "tool_key": "neo4j_read_cypher",
                "error_signature": "Unknown label `Persn`",
                "resolution": "check labels with get-schema first",
            },
            {
                "id": None,
                "tool_key": "mcp__github__create_pr",
                "error_signature": "422 validation failed",
                "resolution": None,
            },
        ]
        prompt = consolidate_skills.build_proposal_prompt(
            "P", "p", group, learnings, [], None, error_context
        )
        self.assertIn("<<<FAILURES", prompt)
        self.assertIn("FAILURES>>>", prompt)
        self.assertIn("[E1] tool: neo4j_read_cypher", prompt)
        self.assertIn("Unknown label `Persn`", prompt)
        self.assertIn("known fix: check labels with get-schema first", prompt)
        self.assertIn("[E2] tool: mcp__github__create_pr", prompt)
        self.assertIn("informed_by", prompt)

    def test_prompt_without_error_context_says_none(self) -> None:
        group = {"kind": "create", "skill_id": None, "members": ["l1"]}
        learnings = {"l1": {"id": "l1", "text": "t", "task_pattern": "p"}}
        prompt = consolidate_skills.build_proposal_prompt(
            "P", "p", group, learnings, [], None
        )
        self.assertIn("<<<FAILURES", prompt)
        self.assertIn("(none)", prompt)

    def test_prompt_includes_target_skill_for_patch_groups(self) -> None:
        group = {"kind": "patch", "skill_id": "s1", "members": ["l1"]}
        learnings = {"l1": {"id": "l1", "text": "t", "task_pattern": "p"}}
        target = {
            "id": "s1",
            "slug": "release",
            "name": "Release verification",
            "description": "Use when releasing.",
            "content": "## Procedure\n1. old step",
        }
        prompt = consolidate_skills.build_proposal_prompt(
            "P", "p", group, learnings, [target], target
        )
        self.assertIn("TARGET SKILL", prompt)
        self.assertIn("## Procedure\n1. old step", prompt)


class AskLlmTests(unittest.TestCase):
    GROUP = {"kind": "create", "skill_id": None, "members": ["l1", "l2"]}

    def test_retry_feeds_validation_error_back(self) -> None:
        full_content = (
            "## When to use\\n## Procedure\\n## Pitfalls\\n## Verification"
        )
        responses = [
            '{"action": "create", "name": "", "description": "", '
            '"content": "x", "derived_from": ["l1"], "rationale": "r"}',
            '{"action": "create", "name": "Good name", "description": "Use when x.", '
            f'"content": "{full_content}", '
            '"derived_from": ["l1", "l2"], "rationale": "r"}',
        ]
        calls: list[list[dict]] = []

        def fake_complete(messages, **kwargs):
            calls.append(messages)
            return responses[len(calls) - 1]

        with patch.object(consolidate_skills, "llm_complete", fake_complete):
            proposal, error = consolidate_skills.ask_llm_for_proposal(
                "prompt", self.GROUP, []
            )
        self.assertIsNone(error)
        self.assertEqual(proposal["slug"], "good-name")
        self.assertEqual(len(calls), 2)
        self.assertIn("rejected", calls[1][-1]["content"])

    def test_double_failure_returns_error(self) -> None:
        with patch.object(
            consolidate_skills, "llm_complete", lambda *a, **k: "not json"
        ):
            proposal, error = consolidate_skills.ask_llm_for_proposal(
                "prompt", self.GROUP, []
            )
        self.assertIsNone(proposal)
        self.assertIn("JSON", error)


class ErrorContextTests(unittest.TestCase):
    def test_error_labels_are_ordered(self) -> None:
        rows = [{"id": "a"}, {"id": None}, {"id": "c"}]
        labels = consolidate_skills.error_labels(rows)
        self.assertEqual(list(labels), ["E1", "E2", "E3"])
        self.assertEqual(labels["E1"]["id"], "a")
        self.assertIsNone(labels["E2"]["id"])

    def test_failure_excerpt_unwraps_mcp_error_shape(self) -> None:
        response = (
            '{"content": [{"type": "text", "text": "Neo.ClientError: '
            'Unknown label"}], "isError": true}'
        )
        self.assertEqual(
            consolidate_skills.failure_excerpt(response),
            "Neo.ClientError: Unknown label",
        )

    def test_failure_excerpt_falls_back_to_raw_string(self) -> None:
        self.assertEqual(
            consolidate_skills.failure_excerpt("  plain error text  "),
            "plain error text",
        )
        self.assertEqual(consolidate_skills.failure_excerpt(None), "")

    def test_digest_dedupes_and_skips_covered_tools(self) -> None:
        rows = [
            {"tool_name": "mcp__x__neo4j_read_cypher", "tool_response": "boom"},
            {"tool_name": "Bash", "tool_response": "command not found: uvx"},
            {"tool_name": "Bash", "tool_response": "command not found: uvx"},
            {"tool_name": "WebFetch", "tool_response": ""},
            {"tool_name": "Bash", "tool_response": "permission denied"},
        ]
        digests = consolidate_skills.digest_raw_failures(rows, {"neo4j_read_cypher"})
        self.assertEqual(len(digests), 2)
        self.assertEqual(digests[0]["tool_key"], "Bash")
        self.assertEqual(digests[0]["error_signature"], "command not found: uvx")
        self.assertIsNone(digests[0]["id"])
        self.assertEqual(digests[1]["error_signature"], "permission denied")

    def test_digest_respects_cap(self) -> None:
        rows = [
            {"tool_name": "Bash", "tool_response": f"error {i}"} for i in range(10)
        ]
        digests = consolidate_skills.digest_raw_failures(rows, set(), cap=3)
        self.assertEqual(len(digests), 3)


class SkillConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            for key in (
                "MKG_SKILL_CONSOLIDATION",
                "MKG_SKILL_CONSOLIDATION_THRESHOLD",
                "MKG_SKILL_CONSOLIDATION_INTERVAL_HOURS",
                "MKG_SKILL_MIN_CLUSTER_SIZE",
                "MKG_TASK_PATTERN_SIMILARITY_THRESHOLD",
                "MKG_SKILL_COACTIVATION_THRESHOLD",
                "MKG_SKILL_CATALOG_INJECT",
                "MKG_SKILL_MAX_PROPOSALS_PER_RUN",
            ):
                os.environ.pop(key, None)
            self.assertTrue(project_common.skill_consolidation_enabled())
            self.assertTrue(project_common.skill_catalog_inject_enabled())
            self.assertEqual(project_common.skill_consolidation_threshold(), 4)
            self.assertEqual(project_common.skill_consolidation_interval_hours(), 24.0)
            self.assertEqual(project_common.skill_min_cluster_size(), 2)
            self.assertEqual(project_common.task_pattern_similarity_threshold(), 0.65)
            self.assertEqual(project_common.skill_coactivation_threshold(), 0.5)
            self.assertEqual(project_common.skill_max_proposals_per_run(), 2)

    def test_env_overrides(self) -> None:
        overrides = {
            "MKG_SKILL_CONSOLIDATION": "0",
            "MKG_SKILL_CONSOLIDATION_THRESHOLD": "7",
            "MKG_TASK_PATTERN_SIMILARITY_THRESHOLD": "0.8",
            "MKG_SKILL_COACTIVATION_THRESHOLD": "0.3",
            "MKG_SKILL_CATALOG_INJECT": "off",
            "MKG_SKILL_MAX_PROPOSALS_PER_RUN": "5",
        }
        with patch.dict(os.environ, overrides, clear=False):
            self.assertFalse(project_common.skill_consolidation_enabled())
            self.assertFalse(project_common.skill_catalog_inject_enabled())
            self.assertEqual(project_common.skill_consolidation_threshold(), 7)
            self.assertEqual(project_common.task_pattern_similarity_threshold(), 0.8)
            self.assertEqual(project_common.skill_coactivation_threshold(), 0.3)
            self.assertEqual(project_common.skill_max_proposals_per_run(), 5)

    def test_max_proposals_floor_is_one(self) -> None:
        with patch.dict(
            os.environ, {"MKG_SKILL_MAX_PROPOSALS_PER_RUN": "0"}, clear=False
        ):
            self.assertEqual(project_common.skill_max_proposals_per_run(), 1)

    def test_skill_node_id(self) -> None:
        self.assertEqual(
            project_common.skill_node_id("proj", "release-verification"),
            "skill:proj:release-verification",
        )

    def test_task_pattern_node_id_is_stable_per_normalized_text(self) -> None:
        one = project_common.task_pattern_node_id("proj", "verify plugin release")
        two = project_common.task_pattern_node_id("proj", "verify plugin release")
        other = project_common.task_pattern_node_id("proj", "rebuild cache venv")
        self.assertEqual(one, two)
        self.assertNotEqual(one, other)
        self.assertTrue(one.startswith("taskpattern:proj:"))


class SkillContextFormattingTests(unittest.TestCase):
    PROJECT = project_common.ProjectRef(id="proj", name="Proj")

    def test_catalog_line_lists_slugs(self) -> None:
        context = project_common.format_learning_context(
            self.PROJECT, [], [], skill_slugs=["release-verification", "venv-rebuild"]
        )
        self.assertIn("Learned skills for this project", context)
        self.assertIn("release-verification, venv-rebuild", context)
        self.assertIn("skill_search", context)
        self.assertIn("skill_fetch", context)

    def test_no_skill_data_adds_nothing(self) -> None:
        context = project_common.format_learning_context(self.PROJECT, [], [])
        self.assertEqual(context, "")


class _RecordingDriver:
    """Records execute_query calls; returns queued record lists in order."""

    def __init__(self, results: list[list[dict]] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._results = list(results or [])

    def execute_query(self, query: str, **kwargs):
        kwargs.pop("database_", None)
        self.calls.append((query, kwargs))
        return self._results.pop(0) if self._results else []


SAFETY_PASS = '{"verdict": "pass", "category": "other", "reason": ""}'
SAFETY_BLOCK = (
    '{"verdict": "block", "category": "injection", "reason": "exfil step"}'
)

PENDING = {
    "skill_id": "skill:proj:demo",
    "slug": "demo",
    "skill_status": "candidate",
    "version_id": "skill:proj:demo:v1",
    "action": "create",
    "name": "Demo Skill",
    "description": "Use when demoing.",
    "content": "## When to use\n## Procedure\n## Pitfalls\n## Verification",
    "created_at": "2026-08-29T00:00:00Z",
}


class SkillSafetyScreenTests(unittest.TestCase):
    def test_safety_prompt_fences_the_skill(self) -> None:
        prompt = consolidate_skills.build_skill_safety_prompt(
            "Release check", "Use when releasing.", "## Procedure\n1. run tests"
        )
        self.assertIn("<<<SKILL", prompt)
        self.assertIn("Release check", prompt)
        self.assertIn("SKILL>>>", prompt)
        # A skill is imperative by nature; the screen must say so instead of
        # blocking on imperativeness alone.
        self.assertIn("imperative wording alone is", prompt)

    def test_passing_proposal_activates(self) -> None:
        driver = _RecordingDriver(results=[[dict(PENDING)]])
        with patch.object(
            consolidate_skills, "llm_complete", return_value=SAFETY_PASS
        ), patch.object(
            consolidate_skills, "embed_texts", return_value=[[0.1]]
        ):
            counts = consolidate_skills.activate_pending_proposals(
                driver, "neo4j", "proj", "2026-08-29T12:00:00Z"
            )
        self.assertEqual(counts, {"activated": 1, "blocked": 0, "deferred": 0})
        activation_query = driver.calls[1][0]
        self.assertIn("v.outcome = 'accepted'", activation_query)
        self.assertIn("v.decided_by = 'auto_gate'", activation_query)
        self.assertIn("sk.status = 'approved'", activation_query)
        self.assertIn("DERIVED_FROM", activation_query)
        self.assertIn("INFORMED_BY", activation_query)

    def test_blocked_create_tombstones_the_candidate_skill(self) -> None:
        driver = _RecordingDriver(results=[[dict(PENDING)]])
        with patch.object(
            consolidate_skills, "llm_complete", return_value=SAFETY_BLOCK
        ):
            counts = consolidate_skills.activate_pending_proposals(
                driver, "neo4j", "proj", "2026-08-29T12:00:00Z"
            )
        self.assertEqual(counts, {"activated": 0, "blocked": 1, "deferred": 0})
        block_query, block_params = driver.calls[1]
        self.assertIn("v.outcome = 'blocked'", block_query)
        self.assertIn("sk.status = 'blocked'", block_query)
        self.assertIn("sk.embedding = null", block_query)
        # Only a candidate (create) parent is tombstoned; a live skill whose
        # patch is blocked stays untouched.
        self.assertIn("WHERE sk.status = 'candidate'", block_query)
        self.assertIn("injection", block_params["reason"])

    def test_unusable_verdict_defers_the_proposal(self) -> None:
        driver = _RecordingDriver(results=[[dict(PENDING)]])
        with patch.object(
            consolidate_skills, "llm_complete", return_value="not json"
        ):
            counts = consolidate_skills.activate_pending_proposals(
                driver, "neo4j", "proj", "2026-08-29T12:00:00Z"
            )
        self.assertEqual(counts, {"activated": 0, "blocked": 0, "deferred": 1})
        # Only the fetch ran; no verdict write happened.
        self.assertEqual(len(driver.calls), 1)

    def test_refused_fingerprints_include_blocked(self) -> None:
        driver = _RecordingDriver(results=[[]])
        consolidate_skills.fetch_refused_fingerprints(driver, "neo4j", "proj")
        query = driver.calls[0][0]
        self.assertIn("v.outcome IN ['rejected', 'blocked']", query)


if __name__ == "__main__":
    unittest.main()
