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

    def _create_proposal(self, **overrides):
        proposal = {
            "action": "create",
            "target_skill_slug": None,
            "name": "Cache venv rebuild",
            "description": "Use when the plugin cache venv loses its symlinks.",
            "content": "## When to use\n...\n## Procedure\n1. ...",
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
        responses = [
            '{"action": "create", "name": "", "description": "", '
            '"content": "x", "derived_from": ["l1"], "rationale": "r"}',
            '{"action": "create", "name": "Good name", "description": "Use when x.", '
            '"content": "## When to use", "derived_from": ["l1", "l2"], "rationale": "r"}',
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
            ):
                os.environ.pop(key, None)
            self.assertTrue(project_common.skill_consolidation_enabled())
            self.assertTrue(project_common.skill_catalog_inject_enabled())
            self.assertEqual(project_common.skill_consolidation_threshold(), 4)
            self.assertEqual(project_common.skill_consolidation_interval_hours(), 24.0)
            self.assertEqual(project_common.skill_min_cluster_size(), 2)
            self.assertEqual(project_common.task_pattern_similarity_threshold(), 0.65)
            self.assertEqual(project_common.skill_coactivation_threshold(), 0.5)

    def test_env_overrides(self) -> None:
        overrides = {
            "MKG_SKILL_CONSOLIDATION": "0",
            "MKG_SKILL_CONSOLIDATION_THRESHOLD": "7",
            "MKG_TASK_PATTERN_SIMILARITY_THRESHOLD": "0.8",
            "MKG_SKILL_COACTIVATION_THRESHOLD": "0.3",
            "MKG_SKILL_CATALOG_INJECT": "off",
        }
        with patch.dict(os.environ, overrides, clear=False):
            self.assertFalse(project_common.skill_consolidation_enabled())
            self.assertFalse(project_common.skill_catalog_inject_enabled())
            self.assertEqual(project_common.skill_consolidation_threshold(), 7)
            self.assertEqual(project_common.task_pattern_similarity_threshold(), 0.8)
            self.assertEqual(project_common.skill_coactivation_threshold(), 0.3)

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

    def test_skill_proposal_nudge(self) -> None:
        context = project_common.format_learning_context(
            self.PROJECT, [], [], skill_proposals_pending=2
        )
        self.assertIn("2 learned-skill proposals awaiting your review", context)
        single = project_common.format_learning_context(
            self.PROJECT, [], [], skill_proposals_pending=1
        )
        self.assertIn("1 learned-skill proposal awaiting", single)

    def test_no_skill_data_adds_nothing(self) -> None:
        context = project_common.format_learning_context(self.PROJECT, [], [])
        self.assertEqual(context, "")


if __name__ == "__main__":
    unittest.main()
