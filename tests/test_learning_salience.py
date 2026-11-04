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
learning_salience = load_hook_module("learning_salience")

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
PROJECT = project_common.ProjectRef(id="proj", name="Proj")
SALIENCE_ENV = {"MKG_SALIENCE": "1", "MKG_SALIENCE_DECAY": "0.9"}


def at(hours_ago: float) -> datetime:
    return NOW - timedelta(hours=hours_ago)


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
                return rows(params) if callable(rows) else rows
        return []

    def queries_with(self, fragment: str) -> list[tuple[str, dict]]:
        return [call for call in self.calls if fragment in call[0]]


class SalienceConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(project_common.learning_salience_enabled())
            self.assertEqual(project_common.learning_salience_interval_hours(), 4.0)
            self.assertEqual(project_common.learning_salience_weight(), 0.1)
            self.assertEqual(project_common.learning_salience_decay(), 0.9)

    def test_disabling_zeroes_the_recall_weight(self) -> None:
        with patch.dict(os.environ, {"MKG_SALIENCE": "off", "MKG_SALIENCE_WEIGHT": "0.5"}):
            self.assertFalse(project_common.learning_salience_enabled())
            self.assertEqual(project_common.learning_salience_weight(), 0.0)

    def test_weight_and_decay_are_clamped(self) -> None:
        with patch.dict(
            os.environ, {"MKG_SALIENCE_WEIGHT": "7", "MKG_SALIENCE_DECAY": "0"}
        ):
            self.assertEqual(project_common.learning_salience_weight(), 1.0)
            self.assertEqual(project_common.learning_salience_decay(), 0.9)

    def test_rank_limit_widens_only_with_salience(self) -> None:
        self.assertEqual(project_common.salience_rank_limit(5, 0.0), 5)
        self.assertEqual(
            project_common.salience_rank_limit(5, 0.1),
            5 * project_common.SALIENCE_CANDIDATE_POOL,
        )


class SalienceGateTests(unittest.TestCase):
    def test_no_prior_run_is_due(self) -> None:
        proceed, _ = learning_salience.salience_gate(None, 4.0, NOW)
        self.assertTrue(proceed)

    def test_recent_run_is_rate_limited(self) -> None:
        proceed, reason = learning_salience.salience_gate(at(1).isoformat(), 4.0, NOW)
        self.assertFalse(proceed)
        self.assertIn("4h cooldown", reason)

    def test_elapsed_cooldown_is_due(self) -> None:
        proceed, _ = learning_salience.salience_gate(at(5), 4.0, NOW)
        self.assertTrue(proceed)


class SeedTests(unittest.TestCase):
    def test_sessions_since_counts_root_sessions_started_after_the_touch(self) -> None:
        roots = [at(30), at(20), at(10)]
        self.assertEqual(learning_salience.sessions_since(at(5), roots), 0)
        self.assertEqual(learning_salience.sessions_since(at(15), roots), 1)
        self.assertEqual(learning_salience.sessions_since(at(40), roots), 3)

    def test_creation_and_updates_decay_by_project_sessions(self) -> None:
        roots = [at(30), at(20), at(10)]
        rows = [
            {
                # Created two sessions ago, updated in the current session.
                "id": "a",
                "created_at": at(25),
                "edits": [
                    {"kind": "PRODUCED_LEARNING", "at": at(25)},
                    {"kind": "UPDATED_LEARNING", "at": at(5)},
                ],
                "restatements": [],
            },
            {
                # Added through project_add_learning: no extractor edge, so
                # its creation time is the touch.
                "id": "b",
                "created_at": at(5),
                "edits": [],
                "restatements": [at(5)],
            },
            {"id": "c", "created_at": None, "edits": [], "restatements": []},
        ]
        seeds = learning_salience.seed_biases(rows, roots, 0.9)
        self.assertAlmostEqual(seeds["a"], 2.0 * 0.9**2 + 1.0)
        self.assertAlmostEqual(seeds["b"], 2.0 + 1.0)
        self.assertNotIn("c", seeds)

    def test_a_dormant_project_does_not_fade(self) -> None:
        # Six months idle: no new session, so nothing ages.
        roots = [at(24 * 180)]
        rows = [
            {"id": "a", "created_at": at(24 * 179), "edits": [], "restatements": []}
        ]
        seeds = learning_salience.seed_biases(rows, roots, 0.9)
        self.assertAlmostEqual(seeds["a"], 2.0)

    def test_seed_query_excludes_echo_writes_and_restatements(self) -> None:
        driver = _RoutedDriver()
        learning_salience.fetch_learning_rows(driver, "neo4j", "proj")
        query, params = driver.calls[0]
        self.assertEqual(query.count("WHERE NOT coalesce(r.echo, false)"), 2)
        self.assertIn("PRODUCED_LEARNING|UPDATED_LEARNING", query)
        self.assertIn("ALREADY_LEARNED_FROM", query)
        self.assertIn("l.status = 'approved' AND l.scope = 'project'", query)
        self.assertNotIn("INJECTED", query)
        self.assertEqual(params["project_id"], "proj")


class EdgeTests(unittest.TestCase):
    def test_group_edges_weigh_one_over_n_minus_one(self) -> None:
        edges = learning_salience.group_edges([["a", "b", "c"], ["d"]], {"a", "b", "c", "d"})
        self.assertEqual(
            edges, {("a", "b"): 0.5, ("a", "c"): 0.5, ("b", "c"): 0.5}
        )

    def test_group_edges_skip_bulk_groups_and_unknown_members(self) -> None:
        nodes = {f"n{i}" for i in range(60)}
        self.assertEqual(learning_salience.group_edges([sorted(nodes)], nodes), {})
        self.assertEqual(learning_salience.group_edges([["a", "x"]], {"a"}), {})

    def test_similarity_edges_are_undirected_and_unweighted_by_score(self) -> None:
        edges = learning_salience.similarity_edges(
            [
                {"source": "b", "target": "a", "score": 0.95},
                {"source": "a", "target": "b", "score": 0.95},
                {"source": "a", "target": "z", "score": 0.99},
            ],
            {"a", "b"},
        )
        self.assertEqual(edges, {("a", "b"): learning_salience.SIMILAR_EDGE_WEIGHT})

    def test_build_edges_sums_every_kind_of_relatedness(self) -> None:
        rows = [
            {"id": "a", "origin": ["s1"], "kind": "error", "error_signature": "E1"},
            {"id": "b", "origin": ["s1"], "kind": "error", "error_signature": "E1"},
            {"id": "c", "origin": ["s2"], "kind": "fact", "error_signature": None},
            {"id": "d", "origin": [], "kind": "fact", "error_signature": None},
        ]
        edges = learning_salience.build_edges(
            rows,
            [{"skill_id": "sk", "members": ["b", "c"]}],
            [{"source": "c", "target": "d", "score": 0.95}],
        )
        self.assertEqual(
            edges,
            {
                ("a", "b"): 2.0,  # same origin session + same error
                ("b", "c"): 1.0,  # same skill
                ("c", "d"): 0.5,  # similar
            },
        )

    def test_similarity_falls_back_to_exact_cosine(self) -> None:
        driver = _RoutedDriver(
            routes={"vector.similarity.cosine": [{"source": "a", "target": "b", "score": 0.95}]},
            raises={"VECTOR INDEX project_learning_vector": RuntimeError("no index")},
        )
        pairs = learning_salience.fetch_similarity_pairs(driver, "neo4j", "proj")
        self.assertEqual(pairs, [{"source": "a", "target": "b", "score": 0.95}])
        _, params = driver.calls[0]
        self.assertAlmostEqual(params["min_score"], 0.825)  # raw cosine 0.65
        self.assertEqual(params["index_limit"], learning_salience.SIMILAR_TOP_K + 1)


class PageRankTests(unittest.TestCase):
    def test_scores_fall_with_hops_from_the_seed(self) -> None:
        nodes = ["seed", "hop1", "hop2", "island"]
        edges = {("hop1", "seed"): 1.0, ("hop1", "hop2"): 1.0}
        scores = learning_salience.personalized_pagerank(nodes, edges, {"seed": 1.0})
        self.assertGreater(scores["seed"], scores["hop1"])
        self.assertGreater(scores["hop1"], scores["hop2"])
        self.assertGreater(scores["hop2"], 0.0)
        self.assertEqual(scores["island"], 0.0)

    def test_bias_splits_the_restart_mass(self) -> None:
        nodes = ["a", "b"]
        scores = learning_salience.personalized_pagerank(nodes, {}, {"a": 3.0, "b": 1.0})
        self.assertAlmostEqual(scores["a"] / scores["b"], 3.0)

    def test_heavier_edges_carry_more_mass(self) -> None:
        nodes = ["seed", "close", "loose"]
        edges = {("close", "seed"): 2.0, ("loose", "seed"): 0.5}
        scores = learning_salience.personalized_pagerank(nodes, edges, {"seed": 1.0})
        self.assertGreater(scores["close"], scores["loose"])

    def test_no_seeds_scores_nothing(self) -> None:
        scores = learning_salience.personalized_pagerank(["a"], {}, {})
        self.assertEqual(scores, {"a": 0.0})

    def test_normalize_scales_the_top_to_one(self) -> None:
        self.assertEqual(
            learning_salience.normalize({"a": 2.0, "b": 1.0, "c": 0.0}),
            {"a": 1.0, "b": 0.5, "c": 0.0},
        )
        self.assertEqual(learning_salience.normalize({"a": 0.0}), {"a": 0.0})

    def test_skill_salience_sums_sources_then_normalizes(self) -> None:
        scores = learning_salience.skill_salience(
            [
                {"skill_id": "hot", "members": ["a", "b"]},
                {"skill_id": "cold", "members": ["c"]},
            ],
            {"a": 1.0, "b": 0.5, "c": 0.3},
        )
        self.assertEqual(scores, {"hot": 1.0, "cold": 0.2})


class GdsBackendTests(unittest.TestCase):
    NODES = ["a", "b", "lonely"]
    EDGES = {("a", "b"): 1.0}
    SEEDS = {"a": 2.0}

    def test_gds_projects_streams_with_biased_seeds_and_drops(self) -> None:
        driver = _RoutedDriver(
            routes={
                "RETURN gds.version()": [{"version": "2026.10.1"}],
                "gds.pageRank.stream": [
                    {"id": "a", "score": 0.6},
                    {"id": "b", "score": 0.2},
                    {"id": "lonely", "score": 0.0},
                ],
            }
        )
        scores, method = learning_salience.compute_scores(
            driver, "neo4j", "proj", self.NODES, self.EDGES, self.SEEDS
        )
        self.assertEqual(method, learning_salience.METHOD_GDS)
        self.assertEqual(scores, {"a": 0.6, "b": 0.2, "lonely": 0.0})

        (project_query, project_params), = driver.queries_with("gds.graph.project(")
        self.assertIn("undirectedRelationshipTypes: ['*']", project_query)
        self.assertIn("OPTIONAL MATCH (b:Learning {id: row.target})", project_query)
        self.assertIn(
            {"source": "lonely", "target": None, "weight": 0.0}, project_params["rows"]
        )
        (stream_query, stream_params), = driver.queries_with("gds.pageRank.stream")
        self.assertIn("collect([l, seed.bias]) AS source_nodes", stream_query)
        self.assertIn("relationshipWeightProperty: 'weight'", stream_query)
        self.assertEqual(stream_params["seeds"], [{"id": "a", "bias": 2.0}])
        self.assertEqual(stream_params["damping"], 0.5)
        self.assertEqual(len(driver.queries_with("gds.graph.drop")), 1)
        self.assertEqual(
            project_params["graph_name"], stream_params["graph_name"]
        )

    def test_failed_gds_still_drops_and_falls_back_to_python(self) -> None:
        driver = _RoutedDriver(
            routes={"RETURN gds.version()": [{"version": "2026.10.1"}]},
            raises={"gds.pageRank.stream": RuntimeError("boom")},
        )
        scores, method = learning_salience.compute_scores(
            driver, "neo4j", "proj", self.NODES, self.EDGES, self.SEEDS
        )
        self.assertEqual(method, learning_salience.METHOD_PYTHON)
        self.assertEqual(len(driver.queries_with("gds.graph.drop")), 1)
        self.assertGreater(scores["a"], scores["b"])

    def test_missing_gds_computes_in_python(self) -> None:
        driver = _RoutedDriver(raises={"RETURN gds.version()": RuntimeError("no gds")})
        scores, method = learning_salience.compute_scores(
            driver, "neo4j", "proj", self.NODES, self.EDGES, self.SEEDS
        )
        self.assertEqual(method, learning_salience.METHOD_PYTHON)
        self.assertFalse(driver.queries_with("gds.graph.project("))
        self.assertEqual(scores["lonely"], 0.0)


class RunLearningSalienceTests(unittest.TestCase):
    def _driver(self, last_run_at=None) -> _RoutedDriver:
        return _RoutedDriver(
            routes={
                "p.salience_last_run_at AS last_run_at": [{"last_run_at": last_run_at}],
                "RETURN gds.version()": [],
                "COLLECT {": [
                    {
                        "id": "a",
                        "created_at": at(5),
                        "kind": "fact",
                        "error_signature": None,
                        "edits": [{"kind": "PRODUCED_LEARNING", "at": at(5)}],
                        "restatements": [],
                        "origin": ["s1"],
                    },
                    {
                        "id": "b",
                        "created_at": at(50),
                        "kind": "fact",
                        "error_signature": None,
                        "edits": [{"kind": "PRODUCED_LEARNING", "at": at(50)}],
                        "restatements": [],
                        "origin": ["s1"],
                    },
                    {
                        "id": "c",
                        "created_at": at(50),
                        "kind": "fact",
                        "error_signature": None,
                        "edits": [{"kind": "PRODUCED_LEARNING", "at": at(50)}],
                        "restatements": [],
                        "origin": ["s0"],
                    },
                ],
                "NOT ()-[:HAS_SUBAGENT]->(s)": [
                    {"created_at": at(60)},
                    {"created_at": at(10)},
                ],
                "-[:DERIVED_FROM]->(l:Learning)": [
                    {"skill_id": "sk", "members": ["a", "b"]}
                ],
            },
            raises={"RETURN gds.version()": RuntimeError("no gds")},
        )

    def _run(self, driver, **kwargs):
        with patch.dict(os.environ, SALIENCE_ENV):
            return learning_salience.run_learning_salience(
                driver, "neo4j", PROJECT, NOW, **kwargs
            )

    def test_due_pass_writes_normalized_scores_clears_stale_and_stamps(self) -> None:
        driver = self._driver()
        summary = self._run(driver)
        self.assertTrue(summary["ran"])
        self.assertEqual(summary["method"], learning_salience.METHOD_PYTHON)
        self.assertEqual((summary["nodes"], summary["seeds"]), (3, 3))

        (_, learning_params), = driver.queries_with("SET l.salience = row.salience")
        written = {row["id"]: row["salience"] for row in learning_params["rows"]}
        self.assertEqual(max(written.values()), 1.0)
        # The fresh learning tops the project; its co-origin neighbour b
        # outranks c, which only shares b's age.
        self.assertEqual(written["a"], 1.0)
        self.assertGreater(written["b"], written["c"])

        (_, skill_params), = driver.queries_with("SET sk.salience = row.salience")
        self.assertEqual(skill_params["rows"], [{"id": "sk", "salience": 1.0}])
        run_id = learning_params["run_id"]
        for fragment in ("REMOVE l.salience", "REMOVE sk.salience"):
            (_, params), = driver.queries_with(fragment)
            self.assertEqual(params["run_id"], run_id)
        (_, stamp), = driver.queries_with("SET p.salience_last_run_at")
        self.assertEqual(stamp["run_id"], run_id)
        self.assertEqual(stamp["method"], learning_salience.METHOD_PYTHON)

    def test_cooldown_skips_before_any_work(self) -> None:
        driver = self._driver(last_run_at=at(1))
        summary = self._run(driver)
        self.assertFalse(summary["ran"])
        self.assertEqual(len(driver.calls), 1)

    def test_force_ignores_the_cooldown(self) -> None:
        driver = self._driver(last_run_at=at(1))
        self.assertTrue(self._run(driver, force=True)["ran"])
        self.assertFalse(driver.queries_with("p.salience_last_run_at AS last_run_at"))

    def test_dry_run_reports_without_writing_or_stamping(self) -> None:
        driver = self._driver()
        summary = self._run(driver, dry_run=True)
        self.assertEqual(summary["top"][0], ("a", 1.0))
        for fragment in ("SET l.salience", "SET sk.salience", "REMOVE", "SET p.salience"):
            self.assertFalse(driver.queries_with(fragment), fragment)

    def test_disabled_does_nothing(self) -> None:
        driver = self._driver()
        with patch.dict(os.environ, {"MKG_SALIENCE": "0"}):
            summary = learning_salience.run_learning_salience(driver, "neo4j", PROJECT, NOW)
        self.assertFalse(summary["ran"])
        self.assertEqual(driver.calls, [])

    def test_empty_project_clears_old_scores(self) -> None:
        driver = _RoutedDriver(raises={"RETURN gds.version()": RuntimeError("no gds")})
        with patch.dict(os.environ, SALIENCE_ENV):
            summary = learning_salience.run_learning_salience(
                driver, "neo4j", PROJECT, NOW, force=True
            )
        self.assertEqual(summary["nodes"], 0)
        self.assertEqual(len(driver.queries_with("REMOVE l.salience")), 1)
        self.assertFalse(driver.queries_with("gds.graph.project("))


class RecallBoostTests(unittest.TestCase):
    def _capture(self, **kwargs):
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query, **params):
                captured.append((query, params))
                return []

        project_common.fetch_project_learnings(
            FakeDriver(),
            "neo4j",
            project_id="mkg",
            query="How do we deploy the graph schema?",
            query_vector=[0.1, 0.2],
            **kwargs,
        )
        return captured

    def test_gated_recall_boosts_and_widens_the_pool(self) -> None:
        (query, params), = self._capture(min_similarity=0.75, salience_weight=0.1)
        self.assertIn(
            "score * (1.0 + $salience_weight * coalesce(node.salience, 0.0)) AS score", query
        )
        self.assertEqual(query.count("raw_score >= $min_vector_score"), 1)
        self.assertEqual(params["salience_weight"], 0.1)
        self.assertEqual(params["rank_limit"], 5 * project_common.SALIENCE_CANDIDATE_POOL)
        self.assertEqual(params["limit"], 5)

    def test_extractor_dedup_shortlist_ignores_salience(self) -> None:
        query, params = self._capture(include_consolidated=True)[0]
        self.assertEqual(params["salience_weight"], 0.0)
        self.assertEqual(params["rank_limit"], 5)


class SkillServiceWiringTests(unittest.TestCase):
    def test_salience_runs_after_the_merge_and_before_the_skill_gate(self) -> None:
        source = (HOOKS / "consolidate_skills.py").read_text()
        body = source[source.index("def consolidate(") :]
        salience_at = body.index("run_learning_salience(")
        self.assertLess(body.index("run_task_pattern_merge("), salience_at)
        self.assertLess(salience_at, body.index("count_eligible_learnings("))
        self.assertLess(salience_at, body.index("if not llm_ready():"))
        self.assertIn("from learning_salience import run_learning_salience", source)


if __name__ == "__main__":
    unittest.main()
