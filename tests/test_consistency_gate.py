from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))


def load_hook_module(name: str):
    module_path = HOOKS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


consistency_gate = load_hook_module("consistency_gate")


def _neighbours(*ids: str) -> list[dict]:
    return [{"id": i, "text": f"text-{i}"} for i in ids]


class ResolveTests(unittest.TestCase):
    def test_no_contradiction_auto_approves(self):
        res = consistency_gate._resolve("cand", _neighbours("a", "b"), [])
        self.assertEqual(res["outcome"], "approved")
        self.assertEqual(res["consistency"], "clean")
        self.assertEqual(res["superseded_ids"], [])

    def test_new_wins_supersedes_existing(self):
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a"),
            [{"existing_id": "a", "winner": "new", "reason": "newer"}],
        )
        self.assertEqual(res["outcome"], "approved")
        self.assertEqual(res["superseded_ids"], ["a"])
        self.assertEqual(res["contradicted_by_ids"], [])

    def test_existing_veto_rejects_candidate(self):
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a"),
            [{"existing_id": "a", "winner": "existing", "reason": "trusted"}],
        )
        self.assertEqual(res["outcome"], "rejected")
        self.assertEqual(res["contradicted_by_ids"], ["a"])
        self.assertEqual(res["superseded_ids"], [])

    def test_veto_beats_supersede(self):
        # A single existing veto is conservative: candidate loses even if it also
        # wins against another neighbour.
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a", "b"),
            [
                {"existing_id": "a", "winner": "new", "reason": "newer"},
                {"existing_id": "b", "winner": "existing", "reason": "trusted"},
            ],
        )
        self.assertEqual(res["outcome"], "rejected")
        self.assertEqual(res["contradicted_by_ids"], ["b"])
        self.assertEqual(res["superseded_ids"], [])

    def test_unclear_only_leaves_candidate(self):
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a"),
            [{"existing_id": "a", "winner": "unclear", "reason": "??"}],
        )
        self.assertEqual(res["outcome"], "candidate")
        # The judge's reason travels with the unclear conflict so the human
        # review queue can show why the pair was punted.
        self.assertEqual(res["unclear_conflicts"], [{"id": "a", "reason": "??"}])

    def test_conflict_on_unknown_id_is_ignored(self):
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a"),
            [{"existing_id": "ghost", "winner": "existing", "reason": "x"}],
        )
        self.assertEqual(res["outcome"], "approved")

    def test_already_learned_folds_into_canonical(self):
        res = consistency_gate._resolve(
            "cand", _neighbours("a", "b"), [], already_learned_of="a"
        )
        self.assertEqual(res["outcome"], "already_learned")
        self.assertEqual(res["consistency"], "already_learned")
        self.assertEqual(res["already_learned_ids"], ["a"])
        self.assertEqual(res["superseded_ids"], [])

    def test_veto_beats_already_learned(self):
        # A genuine veto is still the most conservative outcome.
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a", "b"),
            [{"existing_id": "b", "winner": "existing", "reason": "trusted"}],
            already_learned_of="a",
        )
        self.assertEqual(res["outcome"], "rejected")
        self.assertEqual(res["contradicted_by_ids"], ["b"])
        self.assertEqual(res["already_learned_ids"], [])

    def test_already_learned_beats_supersede(self):
        # Restatement short-circuits before creating/superseding a new node.
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a", "b"),
            [{"existing_id": "b", "winner": "new", "reason": "newer"}],
            already_learned_of="a",
        )
        self.assertEqual(res["outcome"], "already_learned")
        self.assertEqual(res["already_learned_ids"], ["a"])
        self.assertEqual(res["superseded_ids"], [])

    def test_already_learned_unknown_id_ignored(self):
        res = consistency_gate._resolve(
            "cand", _neighbours("a"), [], already_learned_of="ghost"
        )
        self.assertEqual(res["outcome"], "approved")
        self.assertEqual(res["already_learned_ids"], [])


class ResolveWithoutPromotionTests(unittest.TestCase):
    """promote=False (user scope): the human owns approve/reject, so the only
    automatic transition allowed is the already-learned fold."""

    def test_clean_stays_candidate(self):
        res = consistency_gate._resolve("cand", _neighbours("a"), [], promote=False)
        self.assertEqual(res["outcome"], "candidate")
        self.assertEqual(res["consistency"], "clean")
        self.assertEqual(res["unclear_conflicts"], [])

    def test_restatement_folds(self):
        res = consistency_gate._resolve(
            "cand", _neighbours("a"), [], already_learned_of="a", promote=False
        )
        self.assertEqual(res["outcome"], "already_learned")
        self.assertEqual(res["already_learned_ids"], ["a"])

    def test_veto_queues_conflict_instead_of_rejecting(self):
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a"),
            [{"existing_id": "a", "winner": "existing", "reason": "trusted"}],
            promote=False,
        )
        self.assertEqual(res["outcome"], "candidate")
        self.assertEqual(res["consistency"], "ambiguous")
        self.assertEqual(res["contradicted_by_ids"], [])
        self.assertEqual(len(res["unclear_conflicts"]), 1)
        self.assertEqual(res["unclear_conflicts"][0]["id"], "a")
        self.assertIn("judge preferred the existing item", res["unclear_conflicts"][0]["reason"])
        self.assertIn("trusted", res["unclear_conflicts"][0]["reason"])

    def test_new_win_queues_conflict_instead_of_approving(self):
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a"),
            [{"existing_id": "a", "winner": "new", "reason": "newer"}],
            promote=False,
        )
        self.assertEqual(res["outcome"], "candidate")
        self.assertEqual(res["consistency"], "ambiguous")
        self.assertEqual(res["superseded_ids"], [])
        self.assertEqual(len(res["unclear_conflicts"]), 1)
        self.assertIn("judge preferred the new candidate", res["unclear_conflicts"][0]["reason"])

    def test_conflict_blocks_fold(self):
        # A restatement of one item that contradicts another must stay visible
        # to the human rather than silently merging away.
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a", "b"),
            [{"existing_id": "a", "winner": "unclear", "reason": "cannot tell"}],
            already_learned_of="b",
            promote=False,
        )
        self.assertEqual(res["outcome"], "candidate")
        self.assertEqual(res["consistency"], "ambiguous")
        self.assertEqual(res["already_learned_ids"], [])

    def test_conflict_on_unknown_id_still_folds(self):
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a"),
            [{"existing_id": "ghost", "winner": "existing", "reason": "?"}],
            already_learned_of="a",
            promote=False,
        )
        self.assertEqual(res["outcome"], "already_learned")
        self.assertEqual(res["already_learned_ids"], ["a"])


class ParseJudgeTests(unittest.TestCase):
    def test_plain_json(self):
        out = consistency_gate._parse_judge(
            '{"contradictions": [{"existing_id": "a", "winner": "new", "reason": "r"}]}'
        )
        self.assertEqual(
            out,
            {
                "contradictions": [{"existing_id": "a", "winner": "new", "reason": "r"}],
                "already_learned_of": None,
            },
        )

    def test_fenced_json(self):
        out = consistency_gate._parse_judge(
            '```json\n{"contradictions": []}\n```'
        )
        self.assertEqual(out, {"contradictions": [], "already_learned_of": None})

    def test_garbage_returns_empty(self):
        self.assertEqual(
            consistency_gate._parse_judge("no json here"),
            {"contradictions": [], "already_learned_of": None},
        )

    def test_bad_winner_defaults_unclear(self):
        out = consistency_gate._parse_judge(
            '{"contradictions": [{"existing_id": "a", "winner": "maybe"}]}'
        )
        self.assertEqual(out["contradictions"][0]["winner"], "unclear")

    def test_missing_existing_id_dropped(self):
        out = consistency_gate._parse_judge(
            '{"contradictions": [{"winner": "new"}]}'
        )
        self.assertEqual(out["contradictions"], [])

    def test_parses_already_learned_of(self):
        out = consistency_gate._parse_judge(
            '{"contradictions": [], "already_learned_of": "abc", "already_learned_reason": "same fact"}'
        )
        self.assertEqual(out["already_learned_of"], "abc")
        self.assertEqual(out["contradictions"], [])

    def test_already_learned_null_normalizes_to_none(self):
        out = consistency_gate._parse_judge(
            '{"contradictions": [], "already_learned_of": null}'
        )
        self.assertIsNone(out["already_learned_of"])


class AttachEmbeddingsTests(unittest.TestCase):
    def test_attaches_in_order_and_skips_empty(self):
        rows = [
            {"text": "a", "action": "create"},
            {"text": "", "action": "create"},
            {"text": "c", "action": "create"},
        ]
        with patch.object(
            consistency_gate, "embed_texts", return_value=[[0.1], [0.3]]
        ) as embed:
            produced = consistency_gate.attach_candidate_embeddings(rows)
        self.assertTrue(produced)
        # Only the two rows with text are embedded, in order.
        self.assertEqual(embed.call_args.args[0], ["a", "c"])
        self.assertEqual(rows[0]["embedding"], [0.1])
        self.assertNotIn("embedding", rows[1])
        self.assertEqual(rows[2]["embedding"], [0.3])

    def test_no_text_returns_false(self):
        rows = [{"text": "", "action": "create"}]
        self.assertFalse(consistency_gate.attach_candidate_embeddings(rows))

    def test_failed_embedding_leaves_row_unset(self):
        rows = [{"text": "a", "action": "create"}]
        with patch.object(consistency_gate, "embed_texts", return_value=[None]):
            produced = consistency_gate.attach_candidate_embeddings(rows)
        self.assertFalse(produced)
        self.assertNotIn("embedding", rows[0])


class LuceneQueryTests(unittest.TestCase):
    def test_tokenizes_and_dedupes(self):
        self.assertEqual(
            consistency_gate._lucene_query("Use REST, use REST for the API!"),
            "use rest for the api",
        )

    def test_drops_short_tokens_and_empty(self):
        self.assertEqual(consistency_gate._lucene_query("a to be"), "")

    def test_caps_term_count(self):
        text = " ".join(f"word{i}" for i in range(50))
        self.assertEqual(len(consistency_gate._lucene_query(text).split()), 16)


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_write(self, fn, **kwargs):
        return None


class _FakeDriver:
    def session(self, **kwargs):
        return _FakeSession()


class _FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def consume(self):
        return None


class _RecordingSession:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self._log.append((query, params))
        return _FakeResult([])

    def execute_write(self, fn, **kwargs):
        return None


class _RecordingDriver:
    def __init__(self):
        self.log: list[tuple[str, dict]] = []

    def session(self, **kwargs):
        return _RecordingSession(self.log)


class NeighbourRetrievalTests(unittest.TestCase):
    def test_vector_query_prefilters_project_scope_and_excludes_self(self):
        driver = _RecordingDriver()
        consistency_gate._fetch_neighbours_vector(
            driver,
            "neo4j",
            label="Learning",
            index_name="project_learning_vector",
            project_id="proj",
            scope="project",
            vector=[0.1],
            topk=15,
            candidate_id="learning:proj:self",
        )
        query, params = driver.log[0]
        # Vector retrieval prefilters live memory in-index. Dead items should
        # also have their embeddings stripped, but the query is explicit now.
        self.assertNotIn("node.status =", query)
        self.assertIn("node.status IN ['approved', 'candidate']", query)
        self.assertIn("node.project_id = $project_id", query)
        self.assertIn("node.scope = $scope", query)
        self.assertIn("node.id <> $candidate_id", query)
        self.assertEqual(params["candidate_id"], "learning:proj:self")
        # One extra index slot compensates for the candidate matching itself.
        self.assertEqual(params["limit"], 16)

    def test_hybrid_query_unions_vector_and_keyword_with_rrf(self):
        driver = _RecordingDriver()
        consistency_gate._fetch_neighbours_hybrid_vector_keyword(
            driver,
            "neo4j",
            label="Learning",
            index_name="project_learning_vector",
            fulltext_index="project_learning_fulltext",
            project_id="proj",
            scope="project",
            vector=[0.1],
            text="use rest for the api",
            topk=15,
            candidate_id="learning:proj:self",
        )
        query, params = driver.log[0]
        self.assertIn("VECTOR INDEX project_learning_vector", query)
        self.assertIn("db.index.fulltext.queryNodes", query)
        self.assertIn("UNION ALL", query)
        self.assertIn("sum(1.0 / ($rrf_k + rank)) AS score", query)
        self.assertEqual(params["vector_limit"], 16)
        self.assertEqual(params["keyword_limit"], 15)
        self.assertEqual(params["limit"], 15)
        self.assertEqual(params["rrf_k"], consistency_gate._HYBRID_RRF_K)

    def test_fulltext_query_includes_candidates_and_excludes_self(self):
        driver = _RecordingDriver()
        consistency_gate._fetch_neighbours_fulltext(
            driver,
            "neo4j",
            label="Learning",
            fulltext_index="project_learning_fulltext",
            project_id="proj",
            scope="project",
            text="use rest for the api",
            topk=15,
            candidate_id="learning:proj:self",
        )
        query, params = driver.log[0]
        self.assertIn("node.status IN ['approved', 'candidate']", query)
        self.assertIn("node.id <> $candidate_id", query)
        self.assertEqual(params["candidate_id"], "learning:proj:self")
        self.assertEqual(params["limit"], 15)

    def test_format_neighbour_shows_status(self):
        line = consistency_gate._format_neighbour(
            0, {"id": "a", "text": "t", "status": "candidate"}
        )
        self.assertIn("status=candidate", line)


class TombstoneTests(unittest.TestCase):
    def test_apply_resolutions_strips_dead_embeddings(self):
        captured: list[str] = []

        class _Tx:
            def run(self, query, **params):
                captured.append(query)

        consistency_gate._apply_resolutions(
            _Tx(),
            label="Learning",
            rows=[
                {
                    "id": "cand",
                    "outcome": "approved",
                    "consistency": "superseded_conflicts",
                    "superseded_ids": ["old"],
                    "contradicted_by_ids": [],
                    "unclear_conflicts": [],
                    "already_learned_ids": [],
                }
            ],
            model="m",
            timestamp="2026-07-01T00:00:00Z",
        )
        query = captured[0]
        # Death removes the embedding, which drops the item out of the vector
        # index — that is what lets retrieval skip status filtering entirely.
        self.assertIn("old.embedding = null", query)
        self.assertIn(
            "WHEN row.outcome IN ['rejected', 'already_learned'] THEN null", query
        )

    def test_apply_resolutions_stamps_judge_reason_on_contradicts(self):
        captured: list[str] = []

        class _Tx:
            def run(self, query, **params):
                captured.append(query)

        consistency_gate._apply_resolutions(
            _Tx(),
            label="Learning",
            rows=[
                {
                    "id": "cand",
                    "outcome": "candidate",
                    "consistency": "ambiguous",
                    "superseded_ids": [],
                    "contradicted_by_ids": [],
                    "unclear_conflicts": [{"id": "other", "reason": "both current"}],
                    "already_learned_ids": [],
                }
            ],
            model="m",
            timestamp="2026-07-01T00:00:00Z",
        )
        query = captured[0]
        # The ambiguous edge carries the judge's rationale for the human
        # review queue.
        self.assertIn("UNWIND row.unclear_conflicts AS u", query)
        self.assertIn("r.reason = u.reason", query)


class PerRowApplyTests(unittest.TestCase):
    def test_resolutions_apply_per_row_between_retrievals(self):
        # The write for row 1 must land before row 2's neighbour retrieval, so
        # restatements extracted in one batch collapse instead of mutually
        # folding into each other.
        events: list[str] = []

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute_write(self, fn, **kwargs):
                events.append("write:" + kwargs["rows"][0]["id"])

        class _Driver:
            def session(self, **kwargs):
                return _Session()

        def fake_fetch(*args, **kwargs):
            events.append("fetch:" + kwargs["candidate_id"])
            return [{"id": "other", "text": "t", "status": "candidate"}]

        with patch.object(
            consistency_gate, "_fetch_neighbours_hybrid", side_effect=fake_fetch
        ), patch.object(
            consistency_gate,
            "llm_complete",
            return_value='{"contradictions": [], "already_learned_of": null}',
        ):
            applied = consistency_gate._gate_one_label(
                _Driver(),
                "neo4j",
                project=type("P", (), {"id": "proj"})(),
                label="Learning",
                index_name="project_learning_vector",
                fulltext_index="project_learning_fulltext",
                kind="learning",
                rows=[
                    {"id": "l1", "text": "a", "embedding": [0.1], "scope": "project"},
                    {"id": "l2", "text": "b", "embedding": [0.2], "scope": "project"},
                ],
                model=None,
                timestamp="2026-07-01T00:00:00Z",
            )
        self.assertEqual(applied, 2)
        self.assertEqual(events, ["fetch:l1", "write:l1", "fetch:l2", "write:l2"])


class UserScopePromotionWiringTests(unittest.TestCase):
    def test_gate_one_label_resolves_user_rows_without_promotion(self):
        captured = {}

        def fake_resolve(
            candidate_id, neighbours, contradictions, already_learned_of=None, *, promote=True
        ):
            captured[candidate_id] = promote
            return {
                "id": candidate_id,
                "outcome": "candidate",
                "consistency": "clean",
                "superseded_ids": [],
                "contradicted_by_ids": [],
                "unclear_conflicts": [],
                "already_learned_ids": [],
            }

        with patch.object(
            consistency_gate,
            "_fetch_neighbours_hybrid",
            return_value=[{"id": "n1", "text": "t", "status": "candidate"}],
        ), patch.object(
            consistency_gate,
            "llm_complete",
            return_value='{"contradictions": [], "already_learned_of": null}',
        ), patch.object(consistency_gate, "_resolve", side_effect=fake_resolve):
            consistency_gate._gate_one_label(
                _FakeDriver(),
                "neo4j",
                project=type("P", (), {"id": "proj"})(),
                label="Learning",
                index_name="project_learning_vector",
                fulltext_index="project_learning_fulltext",
                kind="learning",
                rows=[
                    {"id": "u1", "text": "a", "embedding": [0.1], "scope": "user"},
                    {"id": "p1", "text": "b", "embedding": [0.2], "scope": "project"},
                ],
                model=None,
                timestamp="2026-07-01T00:00:00Z",
            )
        self.assertEqual(captured, {"u1": False, "p1": True})


class UngatedFetchTests(unittest.TestCase):
    def test_sweep_fetch_spans_both_scopes(self):
        driver = _RecordingDriver()
        consistency_gate._fetch_ungated_candidates(
            driver,
            "neo4j",
            label="Learning",
            project_id="proj",
            exclude_ids=[],
            limit=5,
        )
        query, params = driver.log[0]
        self.assertIn("n.scope IN ['project', 'user']", query)
        self.assertIn("n.consistency_checked_at IS NULL", query)
        self.assertEqual(params["project_id"], "proj")


class SweepTests(unittest.TestCase):
    def _project(self):
        return type("P", (), {"id": "proj"})()

    def test_disabled_gate_skips(self):
        with patch.dict("os.environ", {"MKG_CONSISTENCY_GATE": "0"}):
            out = consistency_gate.sweep_ungated_candidates(
                None, "neo4j", project=self._project(), model=None, timestamp="t"
            )
        self.assertEqual(out, {"learnings": 0})

    def test_zero_sweep_limit_disables(self):
        with patch.dict(
            "os.environ",
            {"MKG_CONSISTENCY_GATE": "1", "MKG_CONSISTENCY_SWEEP_LIMIT": "0"},
        ):
            out = consistency_gate.sweep_ungated_candidates(
                None, "neo4j", project=self._project(), model=None, timestamp="t"
            )
        self.assertEqual(out, {"learnings": 0})

    def test_judge_unavailable_skips(self):
        with patch.dict("os.environ", {"MKG_CONSISTENCY_GATE": "1"}), patch.object(
            consistency_gate, "llm_readiness_status", return_value=(False, "no creds")
        ):
            out = consistency_gate.sweep_ungated_candidates(
                None, "neo4j", project=self._project(), model=None, timestamp="t"
            )
        self.assertEqual(out, {"learnings": 0})

    def test_sweep_limit_env_override(self):
        with patch.dict("os.environ", {"MKG_CONSISTENCY_SWEEP_LIMIT": "3"}):
            self.assertEqual(consistency_gate.sweep_limit(), 3)
        with patch.dict("os.environ", {"MKG_CONSISTENCY_SWEEP_LIMIT": "bogus"}):
            self.assertEqual(
                consistency_gate.sweep_limit(), consistency_gate.DEFAULT_SWEEP_LIMIT
            )

    def test_sweeps_embeds_and_gates_ungated_rows(self):
        rows = [
            {"id": "l1", "text": "a", "scope": "project", "embedding": None},
            {"id": "l2", "text": "b", "scope": "project", "embedding": [0.2]},
        ]
        fetched: dict[str, dict] = {}
        persisted: dict[str, list[str]] = {}
        gated: dict[str, list[str]] = {}

        def fake_fetch(driver, database, *, label, project_id, exclude_ids, limit):
            fetched[label] = {"exclude_ids": exclude_ids, "limit": limit}
            return [dict(r) for r in rows] if label == "Learning" else []

        def fake_attach(group):
            for row in group:
                row["embedding"] = [0.9]
            return True

        def fake_persist(driver, database, *, label, rows, timestamp):
            persisted[label] = [r["id"] for r in rows]
            return len(rows)

        def fake_gate(driver, database, *, label, rows, **kwargs):
            gated[label] = [r["id"] for r in rows]
            return len(rows)

        with patch.dict("os.environ", {"MKG_CONSISTENCY_GATE": "1"}), patch.object(
            consistency_gate, "llm_readiness_status", return_value=(True, None)
        ), patch.object(
            consistency_gate, "_fetch_ungated_candidates", side_effect=fake_fetch
        ), patch.object(
            consistency_gate, "attach_candidate_embeddings", side_effect=fake_attach
        ), patch.object(
            consistency_gate, "_persist_embeddings", side_effect=fake_persist
        ), patch.object(
            consistency_gate, "_gate_one_label", side_effect=fake_gate
        ):
            out = consistency_gate.sweep_ungated_candidates(
                None,
                "neo4j",
                project=self._project(),
                model=None,
                timestamp="2026-07-01T00:00:00Z",
                exclude_ids=["fresh1"],
            )

        self.assertEqual(out, {"learnings": 2})
        self.assertEqual(fetched["Learning"]["exclude_ids"], ["fresh1"])
        # Only the row without a stored embedding is embedded and persisted.
        self.assertEqual(persisted["Learning"], ["l1"])
        self.assertEqual(gated["Learning"], ["l1", "l2"])
        # Only :Learning is swept now; no other label is touched.
        self.assertEqual(set(persisted), {"Learning"})


class DispatchTests(unittest.TestCase):
    def _run(self, row):
        with patch.object(
            consistency_gate, "_fetch_neighbours_hybrid", return_value=[]
        ) as hybrid:
            consistency_gate._gate_one_label(
                _FakeDriver(),
                "neo4j",
                project=type("P", (), {"id": "proj"})(),
                label="Learning",
                index_name="project_learning_vector",
                fulltext_index="project_learning_fulltext",
                kind="learning",
                rows=[row],
                model=None,
                timestamp="2026-07-01T00:00:00Z",
            )
        return hybrid

    def test_embedding_uses_hybrid_path_with_vector(self):
        hybrid = self._run({"id": "l1", "text": "t", "embedding": [0.1], "scope": "project"})
        self.assertEqual(hybrid.call_count, 1)
        self.assertEqual(hybrid.call_args.kwargs["vector"], [0.1])

    def test_no_embedding_uses_hybrid_path_without_vector(self):
        hybrid = self._run({"id": "l1", "text": "t", "scope": "project"})
        self.assertEqual(hybrid.call_count, 1)
        self.assertIsNone(hybrid.call_args.kwargs["vector"])


class GateGuardTests(unittest.TestCase):
    def test_disabled_via_env(self):
        with patch.dict("os.environ", {"MKG_CONSISTENCY_GATE": "0"}):
            result = consistency_gate.run_consistency_gate(
                driver=None,
                database="neo4j",
                project=type("P", (), {"id": "proj"})(),
                learning_rows=[],
                model=None,
                timestamp="2026-07-01T00:00:00Z",
            )
        self.assertEqual(result, {"learnings": 0})

    def test_creates_in_both_scopes_are_gated(self):
        captured = {}

        def fake_gate(*args, rows, **kwargs):
            captured[kwargs["label"]] = [r["id"] for r in rows]
            return len(rows)

        rows = [
            {"action": "create", "embedding": [0.1], "id": "vec", "text": "t", "scope": "project"},
            {"action": "create", "id": "ft", "text": "t", "scope": "project"},  # no embedding -> fulltext
            {"action": "create", "embedding": [0.1], "id": "usr", "text": "t", "scope": "user"},
            {"action": "update", "id": "upd", "text": "t", "scope": "project"},
            {"action": "create", "id": "notext", "scope": "project"},
        ]
        with patch.dict("os.environ", {"MKG_CONSISTENCY_GATE": "1"}), patch.object(
            consistency_gate, "llm_readiness_status", return_value=(True, None)
        ), patch.object(consistency_gate, "_gate_one_label", side_effect=fake_gate):
            consistency_gate.run_consistency_gate(
                driver=None,
                database="neo4j",
                project=type("P", (), {"id": "proj"})(),
                learning_rows=rows,
                model=None,
                timestamp="2026-07-01T00:00:00Z",
            )
        # Creates with text are gated in both scopes (user rows resolve with
        # promote=False downstream); updates and text-less rows are excluded.
        self.assertEqual(captured["Learning"], ["vec", "ft", "usr"])
        self.assertNotIn("Decision", captured)

    def test_skips_when_judge_unavailable(self):
        with patch.dict("os.environ", {"MKG_CONSISTENCY_GATE": "1"}), patch.object(
            consistency_gate, "llm_readiness_status", return_value=(False, "no creds")
        ):
            result = consistency_gate.run_consistency_gate(
                driver=None,
                database="neo4j",
                project=type("P", (), {"id": "proj"})(),
                learning_rows=[{"action": "create", "embedding": [0.1], "id": "l1"}],
                model=None,
                timestamp="2026-07-01T00:00:00Z",
            )
        self.assertEqual(result, {"learnings": 0})


if __name__ == "__main__":
    unittest.main()
