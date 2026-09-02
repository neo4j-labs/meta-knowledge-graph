from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import link_learning_session  # noqa: E402


LEARNING_ROW = {
    "id": "learning:meta-knowledge-graph:abcdef0123456789",
    "text": "Plan-v2 uses durability-proof-based promotion.",
    "status": "candidate",
    "scope": "project",
    "confidence": 0.6,
    "action": "created",
}


def _payload(tool_response, *, tool_name: str = "mcp__plugin_mkg_meta-knowledge-graph__project_add_learning", session_id: str = "session-1") -> dict:
    return {
        "hook_event_name": "PostToolUse",
        "session_id": session_id,
        "tool_use_id": "toolu-1",
        "tool_name": tool_name,
        "tool_input": {"text": LEARNING_ROW["text"]},
        "tool_response": tool_response,
    }


class ExtractLearningIdTests(unittest.TestCase):
    def test_extracts_from_mcp_content_block(self) -> None:
        response = {
            "content": [{"type": "text", "text": json.dumps(LEARNING_ROW)}],
            "isError": False,
        }
        self.assertEqual(
            link_learning_session.extract_learning_ids(response), [LEARNING_ROW["id"]]
        )

    def test_extracts_from_plain_json_string(self) -> None:
        self.assertEqual(
            link_learning_session.extract_learning_ids(json.dumps(LEARNING_ROW)),
            [LEARNING_ROW["id"]],
        )

    def test_extracts_from_result_envelope(self) -> None:
        response = {"result": json.dumps(LEARNING_ROW)}
        self.assertEqual(
            link_learning_session.extract_learning_ids(response), [LEARNING_ROW["id"]]
        )

    def test_extracts_from_content_block_list(self) -> None:
        response = [{"type": "text", "text": json.dumps(LEARNING_ROW)}]
        self.assertEqual(
            link_learning_session.extract_learning_ids(response), [LEARNING_ROW["id"]]
        )

    def test_deduplicates_repeated_ids(self) -> None:
        response = {
            "content": [
                {"type": "text", "text": json.dumps(LEARNING_ROW)},
                {"type": "text", "text": json.dumps(LEARNING_ROW)},
            ]
        }
        self.assertEqual(
            link_learning_session.extract_learning_ids(response), [LEARNING_ROW["id"]]
        )

    def test_ignores_non_learning_ids_and_errors(self) -> None:
        self.assertEqual(
            link_learning_session.extract_learning_ids(
                json.dumps({"id": "observation:mkg:s:1", "status": "ok"})
            ),
            [],
        )
        self.assertEqual(
            link_learning_session.extract_learning_ids(
                json.dumps({"status": "error", "error": "text is required"})
            ),
            [],
        )
        self.assertEqual(link_learning_session.extract_learning_ids(None), [])
        self.assertEqual(link_learning_session.extract_learning_ids("not json"), [])


class LinkGuardTests(unittest.TestCase):
    def _link(self, payload: dict) -> int:
        # Any payload that passes the guards would open a driver; the guard
        # tests only exercise paths that return before that import.
        return link_learning_session.link(payload)

    def test_ignores_other_tools(self) -> None:
        payload = _payload(json.dumps(LEARNING_ROW), tool_name="mcp__mkg__episode_fetch")
        self.assertEqual(self._link(payload), 0)

    def test_ignores_missing_or_unknown_session(self) -> None:
        for session_id in ("", "unknown"):
            with self.subTest(session_id=session_id):
                payload = _payload(json.dumps(LEARNING_ROW), session_id=session_id)
                self.assertEqual(self._link(payload), 0)

    def test_ignores_failure_payload(self) -> None:
        payload = _payload(None)
        payload["hook_event_name"] = "PostToolUseFailure"
        payload["error"] = "boom"
        self.assertEqual(self._link(payload), 0)

    def test_ignores_response_without_learning_id(self) -> None:
        payload = _payload(
            {"content": [{"type": "text", "text": '{"status": "error"}'}]}
        )
        self.assertEqual(self._link(payload), 0)

    def test_links_learning_from_success_payload(self) -> None:
        payload = _payload(
            {"content": [{"type": "text", "text": json.dumps(LEARNING_ROW)}]}
        )
        writes = []

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute_write(self, fn, *args):
                writes.append((fn, args))

        class FakeDriver:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def session(self, database=None):
                return FakeSession()

        with mock.patch.dict(
            sys.modules, {"neo4j": mock.Mock(GraphDatabase=mock.Mock(driver=lambda *a, **k: FakeDriver()))}
        ), mock.patch.object(
            link_learning_session,
            "neo4j_config",
            return_value=("bolt://x", "neo4j", "pw", "neo4j"),
        ):
            linked = link_learning_session.link(payload)

        self.assertEqual(linked, 1)
        self.assertEqual(len(writes), 2)
        fn, args = writes[-1]
        self.assertIs(fn, link_learning_session.write_session_links)
        self.assertEqual(args[0], "session-1")
        self.assertEqual(args[1], [LEARNING_ROW["id"]])


class WriteSessionLinksTests(unittest.TestCase):
    def test_write_merges_from_session_edge(self) -> None:
        class FakeTx:
            def __init__(self) -> None:
                self.calls = []

            def run(self, statement: str, **params) -> None:
                self.calls.append((statement, params))

        tx = FakeTx()
        link_learning_session.write_session_links(
            tx, "session-1", [LEARNING_ROW["id"]], "2026-09-01T00:00:00+00:00"
        )

        statement, params = tx.calls[-1]
        self.assertIn("MERGE (l)-[r:FROM_SESSION]->(s)", statement)
        self.assertIn("MERGE (s:Session {session_id: $session_id})", statement)
        self.assertEqual(params["session_id"], "session-1")
        self.assertEqual(params["learning_ids"], [LEARNING_ROW["id"]])


if __name__ == "__main__":
    unittest.main()
