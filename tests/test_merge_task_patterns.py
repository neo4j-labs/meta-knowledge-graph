from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch


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
merge_task_patterns = load_hook_module("merge_task_patterns")

NOW = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
PROJECT = project_common.ProjectRef(id="proj", name="Proj")


class _RoutedDriver:
    """Answers ``execute_query`` by the first query fragment that matches."""

    def __init__(self, routes=None, raises=None):
        self.routes = routes or {}
        self.raises = raises or {}
        self.calls: list[tuple[str, dict]] = []

    def execute_query(self, query, **params):
        self.calls.append((query, params))
        for fragment, exc in self.raises.items():
            if fragment in query:
                raise exc
        for fragment, rows in self.routes.items():
            if fragment in query:
                return rows
        return []

    def queries_with(self, fragment: str) -> list[tuple[str, dict]]:
        return [call for call in self.calls if fragment in call[0]]


class _Tx:
    def __init__(self):
        self.runs: list[tuple[str, dict]] = []

    def run(self, query, **params):
        self.runs.append((query, params))


class _SessionDriver:
    def __init__(self):
        self.tx = _Tx()
        self.database = None

    def session(self, database=None):
        self.database = database
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_write(self, fn, *args):
        return fn(self.tx, *args)


def profile(pid, *, reviewed=False, embedded=True, count=1, skills=(), text=None):
    return {
        "id": pid,
        "text": text or f"label {pid}",
        "normalized": text or f"label {pid}",
        "aliases": [],
        "embedded": embedded,
        "reviewed": reviewed,
        "learning_count": count,
        "samples": [f"learning about {pid}"],
        "skill_slugs": list(skills),
    }


def edge(source, target, similarity):
    return {"source": source, "target": target, "similarity": similarity}


class PatternLookupTests(unittest.TestCase):
    def test_live_patterns_answer_for_their_text_and_aliases(self) -> None:
        lookup = merge_task_patterns.build_pattern_lookup(
            [
                {
                    "id": "tp-1",
                    "normalized": "release verification",
                    "aliases": ["verify plugin release"],
                    "status": "active",
                },
                {"id": "tp-2", "normalized": "rebuild venv", "aliases": [], "status": "active"},
            ]
        )
        self.assertEqual(lookup["release verification"], "tp-1")
        self.assertEqual(lookup["verify plugin release"], "tp-1")
        self.assertEqual(lookup["rebuild venv"], "tp-2")

    def test_tombstone_text_follows_the_merge_chain_to_a_live_pattern(self) -> None:
        lookup = merge_task_patterns.build_pattern_lookup(
            [
                {"id": "tp-a", "normalized": "a", "status": "merged", "merged_into": "tp-b"},
                {"id": "tp-b", "normalized": "b", "status": "merged", "merged_into": "tp-c"},
                {"id": "tp-c", "normalized": "c", "status": "active"},
                # A broken chain resolves nowhere rather than to a tombstone.
                {"id": "tp-x", "normalized": "x", "status": "merged", "merged_into": "gone"},
            ]
        )
        self.assertEqual(lookup["a"], "tp-c")
        self.assertEqual(lookup["b"], "tp-c")
        self.assertNotIn("x", lookup)

    def test_match_is_exact_only(self) -> None:
        lookup = {"verify plugin release": "tp-1"}
        self.assertEqual(
            merge_task_patterns.match_task_pattern("verify plugin release", lookup), "tp-1"
        )
        self.assertIsNone(
            merge_task_patterns.match_task_pattern("verify the plugin release", lookup)
        )


class ResolveTaskPatternsTests(unittest.TestCase):
    def test_resolves_exactly_and_embeds_only_minted_patterns(self) -> None:
        driver = _RoutedDriver(
            routes={
                "RETURN l.id AS learning_id, tp.id AS pattern_id": [
                    {"learning_id": "l0", "pattern_id": "tp-old"}
                ],
                "tp.merged_into AS merged_into": [
                    {
                        "id": "tp-1",
                        "normalized": "release verification",
                        "aliases": ["verify plugin release"],
                        "status": "active",
                        "merged_into": None,
                    },
                    {
                        "id": "tp-gone",
                        "normalized": "old release label",
                        "aliases": [],
                        "status": "merged",
                        "merged_into": "tp-1",
                    },
                ],
            }
        )
        rows = [
            {"id": "l0", "task_pattern": "anything"},  # already tagged
            {"id": "l1", "task_pattern": "Verify plugin release"},  # alias
            {"id": "l2", "task_pattern": "Rebuild cache venv!"},  # new
            {"id": "l3", "task_pattern": "rebuild cache venv"},  # same new label
            {"id": "l4", "task_pattern": "old release label"},  # tombstone text
            {"id": "l5", "task_pattern": "   "},  # no pattern
        ]
        embed = MagicMock(return_value=[[0.1, 0.2]])
        with patch.object(merge_task_patterns, "embed_texts", embed):
            tagged = merge_task_patterns.resolve_task_patterns(
                driver, "neo4j", "proj", rows, NOW.isoformat()
            )

        minted = project_common.task_pattern_node_id("proj", "rebuild cache venv")
        self.assertEqual(
            tagged,
            {"l0": "tp-old", "l1": "tp-1", "l2": minted, "l3": minted, "l4": "tp-1"},
        )
        embed.assert_called_once_with(["Rebuild cache venv!"])

        (write_query, write_params), = driver.queries_with("MERGE (l)-[t:TAGGED_WITH]->(tp)")
        self.assertIn("tp.status = 'active'", write_query)
        by_learning = {row["learning_id"]: row for row in write_params["assignments"]}
        self.assertEqual(set(by_learning), {"l1", "l2", "l3", "l4"})
        self.assertEqual(by_learning["l2"]["embedding"], [0.1, 0.2])
        self.assertIsNone(by_learning["l1"]["embedding"])

        (_, hub_params), = driver.queries_with("MERGE (tp)-[o:OBSERVED_IN]->(s)")
        self.assertIn(
            {"learning_id": "l0", "pattern_id": "tp-old"}, hub_params["links"]
        )

    def test_matched_labels_need_no_embedding_call(self) -> None:
        driver = _RoutedDriver(
            routes={
                "tp.merged_into AS merged_into": [
                    {"id": "tp-1", "normalized": "release verification", "status": "active"}
                ]
            }
        )
        embed = MagicMock()
        with patch.object(merge_task_patterns, "embed_texts", embed):
            merge_task_patterns.resolve_task_patterns(
                driver,
                "neo4j",
                "proj",
                [{"id": "l1", "task_pattern": "Release verification"}],
                NOW.isoformat(),
            )
        embed.assert_not_called()


class MergeGateTests(unittest.TestCase):
    def test_no_prior_run_is_due(self) -> None:
        proceed, _ = merge_task_patterns.merge_gate(None, 8.0, NOW)
        self.assertTrue(proceed)

    def test_recent_run_is_rate_limited(self) -> None:
        proceed, reason = merge_task_patterns.merge_gate(
            (NOW - timedelta(hours=7)).isoformat(), 8.0, NOW
        )
        self.assertFalse(proceed)
        self.assertIn("8h cooldown", reason)

    def test_elapsed_cooldown_is_due(self) -> None:
        proceed, _ = merge_task_patterns.merge_gate(
            (NOW - timedelta(hours=8, minutes=1)).isoformat(), 8.0, NOW
        )
        self.assertTrue(proceed)


class SimilarityGraphTests(unittest.TestCase):
    def test_edge_query_converts_the_raw_cosine_floor(self) -> None:
        driver = _RoutedDriver(
            routes={"vector.similarity.cosine": [{"source": "a", "target": "b", "similarity": 0.9}]}
        )
        edges = merge_task_patterns.fetch_similarity_edges(driver, "neo4j", "proj", 0.6)
        self.assertEqual(edges, [edge("a", "b", 0.9)])
        _, params = driver.calls[0]
        self.assertAlmostEqual(params["min_vector_score"], 0.8)

    def test_prune_keeps_an_edge_in_either_endpoints_top_k(self) -> None:
        edges = [
            edge("hub", "a", 0.90),
            edge("hub", "b", 0.80),
            edge("hub", "c", 0.70),
            edge("c", "d", 0.66),
        ]
        kept = merge_task_patterns.prune_edges(edges, top_k=1)
        pairs = {(e["source"], e["target"]) for e in kept}
        # hub's best is a; a, b and c each keep their own best; d keeps c-d.
        self.assertEqual(
            pairs, {("hub", "a"), ("hub", "b"), ("hub", "c"), ("c", "d")}
        )
        kept = merge_task_patterns.prune_edges(
            [edge("x", "y", 0.9), edge("x", "z", 0.8), edge("y", "z", 0.7)], top_k=1
        )
        self.assertNotIn(("y", "z"), {(e["source"], e["target"]) for e in kept})

    def test_connected_components(self) -> None:
        assignment = merge_task_patterns.connected_components(
            [edge("a", "b", 0.9), edge("b", "c", 0.7), edge("x", "y", 0.8)]
        )
        self.assertEqual(assignment["a"], assignment["b"])
        self.assertEqual(assignment["b"], assignment["c"])
        self.assertEqual(assignment["x"], assignment["y"])
        self.assertNotEqual(assignment["a"], assignment["x"])


class CommunityDetectionTests(unittest.TestCase):
    EDGES = [edge("a", "b", 0.9), edge("c", "d", 0.8)]

    def test_leiden_projects_streams_and_drops_the_graph(self) -> None:
        driver = _RoutedDriver(
            routes={
                "gds.version": [{"version": "2.13.0"}],
                "gds.graph.project": [{"node_count": 4, "relationship_count": 2}],
                "gds.leiden.stream": [
                    {"pattern_id": "a", "community": 3},
                    {"pattern_id": "b", "community": 3},
                    {"pattern_id": "c", "community": 5},
                    {"pattern_id": "d", "community": 5},
                ],
            }
        )
        assignment, method = merge_task_patterns.detect_communities(
            driver, "neo4j", "proj", self.EDGES
        )
        self.assertEqual(method, merge_task_patterns.METHOD_LEIDEN)
        self.assertEqual(assignment, {"a": 3, "b": 3, "c": 5, "d": 5})

        project_call, stream_call, drop_call = driver.calls[1:]
        self.assertIn("undirectedRelationshipTypes", project_call[0])
        self.assertEqual(project_call[1]["edges"], self.EDGES)
        self.assertIn("relationshipWeightProperty: 'weight'", stream_call[0])
        self.assertIn("gds.graph.drop", drop_call[0])
        names = {call[1]["graph_name"] for call in (project_call, stream_call, drop_call)}
        self.assertEqual(len(names), 1)

    def test_failed_leiden_still_drops_and_falls_back(self) -> None:
        driver = _RoutedDriver(
            routes={"gds.version": [{"version": "2.13.0"}]},
            raises={"gds.leiden.stream": RuntimeError("bad config")},
        )
        assignment, method = merge_task_patterns.detect_communities(
            driver, "neo4j", "proj", self.EDGES
        )
        self.assertEqual(method, merge_task_patterns.METHOD_COMPONENTS)
        self.assertTrue(driver.queries_with("gds.graph.drop"))
        self.assertEqual(assignment["a"], assignment["b"])

    def test_missing_gds_uses_connected_components(self) -> None:
        driver = _RoutedDriver(raises={"gds.version": RuntimeError("Unknown function")})
        assignment, method = merge_task_patterns.detect_communities(
            driver, "neo4j", "proj", self.EDGES
        )
        self.assertEqual(method, merge_task_patterns.METHOD_COMPONENTS)
        self.assertFalse(driver.queries_with("gds.graph.project"))
        self.assertNotEqual(assignment["a"], assignment["c"])


class ReviewBatchTests(unittest.TestCase):
    def test_selects_unreviewed_multi_member_communities_in_rank_order(self) -> None:
        profiles = {
            "a": profile("a", count=1),
            "b": profile("b", reviewed=True, count=5),
            "c": profile("c", reviewed=True),
            "d": profile("d", reviewed=True),
            "e": profile("e"),
            "f": profile("f", count=9),
            "g": profile("g"),
        }
        assignment = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 2, "f": 2, "g": 3}
        edges = [edge("a", "b", 0.9), edge("c", "d", 0.8), edge("e", "f", 0.7)]
        batches = merge_task_patterns.select_review_batches(
            assignment, edges, profiles, budget=5
        )
        # c/d: all reviewed; g: alone. e/f has two unreviewed so it leads.
        self.assertEqual(batches, [["e", "f"], ["a", "b"]])
        self.assertEqual(
            merge_task_patterns.select_review_batches(assignment, edges, profiles, budget=1),
            [["e", "f"]],
        )

    def test_oversized_community_keeps_unreviewed_members_first(self) -> None:
        profiles = {
            "x": profile("x", reviewed=True),
            "y": profile("y"),
            "z": profile("z", reviewed=True),
        }
        assignment = {"x": 0, "y": 0, "z": 0}
        edges = [edge("x", "z", 0.95), edge("x", "y", 0.66)]
        batches = merge_task_patterns.select_review_batches(
            assignment, edges, profiles, budget=5, max_size=2
        )
        self.assertEqual(batches, [["y", "x"]])


class MergePromptTests(unittest.TestCase):
    def test_prompt_labels_quotes_and_frames_patterns_as_data(self) -> None:
        profiles = {
            "a": profile("a", text='hook "debugging"', count=3, skills=["hook-triage"]),
            "b": profile("b", text="debugging hooks"),
        }
        prompt, labels = merge_task_patterns.build_merge_prompt(
            "Proj", "proj", ["a", "b"], profiles
        )
        self.assertEqual(labels, {"P1": "a", "P2": "b"})
        self.assertIn('P1: "hook \\"debugging\\"" (3 learnings; live skill: hook-triage)', prompt)
        self.assertIn('P2: "debugging hooks" (1 learning)', prompt)
        self.assertIn("not\ninstructions to you", prompt)
        self.assertIn("=== PATTERNS ===", prompt)
        self.assertIn('e.g. "learning about a"', prompt)


class ValidateMergeDecisionTests(unittest.TestCase):
    LABELS = {"P1": "a", "P2": "b", "P3": "c", "P4": "d"}

    def validate(self, payload):
        return merge_task_patterns.validate_merge_decision(payload, self.LABELS)

    def test_valid_decision_maps_labels_to_ids(self) -> None:
        merges, error = self.validate(
            {
                "merges": [
                    {"canonical": "p2", "members": ["P1", "P2", "P1"], "reason": "same task"},
                    {"canonical": "P3", "members": ["P3", "P4"]},
                ]
            }
        )
        self.assertIsNone(error)
        self.assertEqual(
            merges,
            [
                {"target": "b", "sources": ["a"], "reason": "same task"},
                {"target": "c", "sources": ["d"], "reason": ""},
            ],
        )

    def test_empty_decision_is_valid(self) -> None:
        self.assertEqual(self.validate({"merges": []}), ([], None))

    def test_structural_errors_are_rejected(self) -> None:
        cases = {
            "missing list": {},
            "unknown label": {"merges": [{"canonical": "P1", "members": ["P1", "P9"]}]},
            "single member": {"merges": [{"canonical": "P1", "members": ["P1", "p1"]}]},
            "canonical outside": {"merges": [{"canonical": "P3", "members": ["P1", "P2"]}]},
            "reused label": {
                "merges": [
                    {"canonical": "P1", "members": ["P1", "P2"]},
                    {"canonical": "P2", "members": ["P2", "P3"]},
                ]
            },
        }
        for name, payload in cases.items():
            merges, error = self.validate(payload)
            self.assertIsNone(merges, name)
            self.assertTrue(error, name)

    def test_retry_feeds_the_validation_error_back(self) -> None:
        replies = [
            json.dumps({"merges": [{"canonical": "P9", "members": ["P1", "P9"]}]}),
            "```json\n" + json.dumps({"merges": [{"canonical": "P1", "members": ["P1", "P2"]}]}) + "\n```",
        ]
        complete = MagicMock(side_effect=replies)
        with patch.object(merge_task_patterns, "llm_complete", complete):
            merges, error = merge_task_patterns.ask_llm_for_merges("prompt", self.LABELS)
        self.assertIsNone(error)
        self.assertEqual(merges, [{"target": "a", "sources": ["b"], "reason": ""}])
        retry_messages = complete.call_args_list[1].args[0]
        self.assertIn("unknown label", retry_messages[-1]["content"])

    def test_two_failures_return_the_last_error(self) -> None:
        with patch.object(merge_task_patterns, "llm_complete", return_value="nope"):
            merges, error = merge_task_patterns.ask_llm_for_merges("prompt", self.LABELS)
        self.assertIsNone(merges)
        self.assertEqual(error, "response was not a JSON object")


class MergeGuardAndWriteTests(unittest.TestCase):
    def test_merge_spanning_two_live_skills_is_refused(self) -> None:
        profiles = {
            "a": profile("a", skills=["skill-one"]),
            "b": profile("b", skills=["skill-two"]),
            "c": profile("c", skills=["skill-one"]),
            "d": profile("d"),
        }
        kept, refused = merge_task_patterns.guard_skill_boundaries(
            [
                {"target": "a", "sources": ["b"], "reason": ""},
                {"target": "c", "sources": ["d"], "reason": ""},
            ],
            profiles,
        )
        self.assertEqual([m["target"] for m in kept], ["c"])
        self.assertEqual(refused[0]["skills"], ["skill-one", "skill-two"])

    def test_aliases_absorb_source_labels_once(self) -> None:
        aliases = merge_task_patterns.merged_aliases(
            {"normalized": "release verification", "aliases": ["verify release"]},
            [
                {"normalized": "check release", "aliases": ["verify release", "release check"]},
                {"normalized": "release verification", "aliases": []},
            ],
        )
        self.assertEqual(aliases, ["verify release", "check release", "release check"])

    def test_apply_merge_moves_links_and_tombstones_in_one_transaction(self) -> None:
        profiles = {
            "a": {**profile("a"), "aliases": ["old a"]},
            "b": profile("b"),
        }
        driver = _SessionDriver()
        merge_task_patterns.apply_merge(
            driver,
            "neo4j",
            {"target": "a", "sources": ["b"], "reason": "same task"},
            profiles,
            "model-x",
            merge_task_patterns.METHOD_LEIDEN,
            NOW.isoformat(),
        )
        self.assertEqual(driver.database, "neo4j")
        tags, sessions, tombstone, target = driver.tx.runs
        self.assertIn("DELETE t", tags[0])
        self.assertIn("SET l.task_pattern = target.text", tags[0])
        self.assertEqual(tags[1]["source_ids"], ["b"])
        self.assertIn("MERGE (target)-[moved:OBSERVED_IN]->(s)", sessions[0])
        self.assertIn("DELETE o", sessions[0])
        self.assertIn("MERGE (src)-[m:MERGED_INTO]->(target)", tombstone[0])
        self.assertIn("src.status = 'merged'", tombstone[0])
        self.assertEqual(tombstone[1]["reason"], "same task")
        self.assertEqual(tombstone[1]["method"], "gds_leiden")
        self.assertEqual(target[1]["aliases"], ["old a", "label b"])


class RunTaskPatternMergeTests(unittest.TestCase):
    PROFILES = [
        profile("a"),
        profile("b"),
        profile("c"),  # embedded, no neighbour above the floor
        profile("d", embedded=False),
    ]

    def run_pass(self, *, last_run=None, ready=True, dry_run=False, reply=None):
        mocks = {
            "read_last_merge_run": MagicMock(return_value=last_run),
            "llm_ready": MagicMock(return_value=ready),
            "tag_pending_learnings": MagicMock(return_value=0),
            "backfill_pattern_embeddings": MagicMock(return_value=1),
            "fetch_pattern_profiles": MagicMock(return_value=self.PROFILES),
            "fetch_similarity_edges": MagicMock(return_value=[edge("a", "b", 0.9)]),
            "detect_communities": MagicMock(
                return_value=({"a": 0, "b": 0}, merge_task_patterns.METHOD_LEIDEN)
            ),
            "llm_complete": MagicMock(
                return_value=reply
                or json.dumps(
                    {"merges": [{"canonical": "P1", "members": ["P1", "P2"], "reason": "same"}]}
                )
            ),
            "apply_merge": MagicMock(),
            "mark_reviewed": MagicMock(),
            "mark_merge_run": MagicMock(),
            "extraction_model_label": MagicMock(return_value="model-x"),
            "task_pattern_merge_enabled": MagicMock(return_value=True),
            "task_pattern_merge_interval_hours": MagicMock(return_value=8.0),
        }
        with patch.multiple(merge_task_patterns, **mocks):
            summary = merge_task_patterns.run_task_pattern_merge(
                object(), "neo4j", PROJECT, NOW, dry_run=dry_run
            )
        return summary, mocks

    def test_due_pass_merges_and_stamps(self) -> None:
        summary, mocks = self.run_pass()
        self.assertTrue(summary["ran"])
        self.assertEqual(summary["method"], "gds_leiden")
        mocks["apply_merge"].assert_called_once()
        merge = mocks["apply_merge"].call_args.args[2]
        self.assertEqual(merge, {"target": "a", "sources": ["b"], "reason": "same"})
        self.assertEqual(mocks["apply_merge"].call_args.args[5], "gds_leiden")
        # a, b were judged; c had nothing to compare; d is not embedded yet.
        self.assertEqual(mocks["mark_reviewed"].call_args.args[2], ["a", "b", "c"])
        mocks["mark_merge_run"].assert_called_once()

    def test_dry_run_reports_without_writing_merges_or_stamps(self) -> None:
        summary, mocks = self.run_pass(dry_run=True)
        self.assertEqual(len(summary["merges"]), 1)
        mocks["apply_merge"].assert_not_called()
        mocks["mark_reviewed"].assert_not_called()
        mocks["mark_merge_run"].assert_not_called()

    def test_cooldown_skips_before_any_work(self) -> None:
        summary, mocks = self.run_pass(last_run=(NOW - timedelta(hours=2)).isoformat())
        self.assertFalse(summary["ran"])
        mocks["llm_ready"].assert_not_called()
        mocks["tag_pending_learnings"].assert_not_called()
        mocks["mark_merge_run"].assert_not_called()

    def test_missing_llm_leaves_the_run_unstamped(self) -> None:
        summary, mocks = self.run_pass(ready=False)
        self.assertFalse(summary["ran"])
        mocks["tag_pending_learnings"].assert_not_called()
        mocks["mark_merge_run"].assert_not_called()

    def test_failed_review_leaves_its_community_unreviewed(self) -> None:
        _, mocks = self.run_pass(reply="not json")
        mocks["apply_merge"].assert_not_called()
        self.assertEqual(mocks["mark_reviewed"].call_args.args[2], ["c"])
        mocks["mark_merge_run"].assert_called_once()


class SkillServiceWiringTests(unittest.TestCase):
    def test_merge_pass_runs_before_the_skill_gate_and_grouping(self) -> None:
        source = (HOOKS / "consolidate_skills.py").read_text()
        body = source[source.index("def consolidate(") :]
        merge_at = body.index("run_task_pattern_merge(")
        self.assertLess(merge_at, body.index("count_eligible_learnings("))
        self.assertLess(merge_at, body.index("resolve_task_patterns("))
        self.assertIn("from merge_task_patterns import", source)


if __name__ == "__main__":
    unittest.main()
