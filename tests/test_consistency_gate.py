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
        self.assertEqual(res["unclear_ids"], ["a"])

    def test_conflict_on_unknown_id_is_ignored(self):
        res = consistency_gate._resolve(
            "cand",
            _neighbours("a"),
            [{"existing_id": "ghost", "winner": "existing", "reason": "x"}],
        )
        self.assertEqual(res["outcome"], "approved")


class ParseJudgeTests(unittest.TestCase):
    def test_plain_json(self):
        out = consistency_gate._parse_judge(
            '{"contradictions": [{"existing_id": "a", "winner": "new", "reason": "r"}]}'
        )
        self.assertEqual(out, [{"existing_id": "a", "winner": "new", "reason": "r"}])

    def test_fenced_json(self):
        out = consistency_gate._parse_judge(
            '```json\n{"contradictions": []}\n```'
        )
        self.assertEqual(out, [])

    def test_garbage_returns_empty(self):
        self.assertEqual(consistency_gate._parse_judge("no json here"), [])

    def test_bad_winner_defaults_unclear(self):
        out = consistency_gate._parse_judge(
            '{"contradictions": [{"existing_id": "a", "winner": "maybe"}]}'
        )
        self.assertEqual(out[0]["winner"], "unclear")

    def test_missing_existing_id_dropped(self):
        out = consistency_gate._parse_judge(
            '{"contradictions": [{"winner": "new"}]}'
        )
        self.assertEqual(out, [])


class AttachEmbeddingsTests(unittest.TestCase):
    def test_attaches_in_order_and_skips_empty(self):
        rows = [
            {"text": "a", "action": "create"},
            {"text": "", "action": "create"},
            {"text": "c", "rationale": "why", "action": "create"},
        ]
        with patch.object(
            consistency_gate, "embed_texts", return_value=[[0.1], [0.3]]
        ) as embed:
            produced = consistency_gate.attach_candidate_embeddings(rows)
        self.assertTrue(produced)
        # Only the two rows with text are embedded, in order.
        self.assertEqual(embed.call_args.args[0], ["a", "c\nwhy"])
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


class DispatchTests(unittest.TestCase):
    def _run(self, row):
        with patch.object(
            consistency_gate, "_fetch_neighbours_vector", return_value=[]
        ) as vec, patch.object(
            consistency_gate, "_fetch_neighbours_fulltext", return_value=[]
        ) as ft:
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
        return vec, ft

    def test_embedding_uses_vector_path(self):
        vec, ft = self._run({"id": "l1", "text": "t", "embedding": [0.1], "scope": "project"})
        self.assertEqual(vec.call_count, 1)
        self.assertEqual(ft.call_count, 0)

    def test_no_embedding_uses_fulltext_path(self):
        vec, ft = self._run({"id": "l1", "text": "t", "scope": "project"})
        self.assertEqual(vec.call_count, 0)
        self.assertEqual(ft.call_count, 1)


class GateGuardTests(unittest.TestCase):
    def test_disabled_via_env(self):
        with patch.dict("os.environ", {"MKG_CONSISTENCY_GATE": "0"}):
            result = consistency_gate.run_consistency_gate(
                driver=None,
                database="neo4j",
                project=type("P", (), {"id": "proj"})(),
                learning_rows=[],
                decision_rows=[],
                model=None,
                timestamp="2026-07-01T00:00:00Z",
            )
        self.assertEqual(result, {"learnings": 0, "decisions": 0})

    def test_only_project_scoped_creates_are_gated(self):
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
                decision_rows=[],
                model=None,
                timestamp="2026-07-01T00:00:00Z",
            )
        # Project-scoped creates with text are gated (embedding or fulltext);
        # user-scoped, updates, and text-less rows are excluded.
        self.assertEqual(captured["Learning"], ["vec", "ft"])
        self.assertEqual(captured["Decision"], [])

    def test_skips_when_judge_unavailable(self):
        with patch.dict("os.environ", {"MKG_CONSISTENCY_GATE": "1"}), patch.object(
            consistency_gate, "llm_readiness_status", return_value=(False, "no creds")
        ):
            result = consistency_gate.run_consistency_gate(
                driver=None,
                database="neo4j",
                project=type("P", (), {"id": "proj"})(),
                learning_rows=[{"action": "create", "embedding": [0.1], "id": "l1"}],
                decision_rows=[],
                model=None,
                timestamp="2026-07-01T00:00:00Z",
            )
        self.assertEqual(result, {"learnings": 0, "decisions": 0})


if __name__ == "__main__":
    unittest.main()
