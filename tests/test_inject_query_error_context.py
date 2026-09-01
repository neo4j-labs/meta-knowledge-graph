from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import inject_query_error_context  # noqa: E402


def _payload(
    tool_name: str = "mcp__meta_knowledge_graph__neo4j_read_cypher",
    query: str = "MATCH (n:Account) RETURN n.namee",
    response_text: str = "Neo4jError: Invalid input 'RETRUN'",
    is_error: bool = True,
    **overrides,
) -> dict:
    payload = {
        "session_id": "session-1",
        "tool_use_id": "toolu-1",
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": {"query": query},
        "tool_response": {
            "content": [{"type": "text", "text": response_text}],
            "isError": is_error,
        },
    }
    payload.update(overrides)
    return payload


def _failure_payload(error: str, **overrides) -> dict:
    payload = _payload()
    del payload["tool_response"]
    payload["hook_event_name"] = "PostToolUseFailure"
    payload["error"] = error
    payload.update(overrides)
    return payload


class RecallGateTests(unittest.TestCase):
    def test_syntax_error_triggers_recall(self) -> None:
        request = inject_query_error_context.build_recall_request(
            _payload(response_text="Neo4jError: Syntax error: Invalid input 'RETRUN'")
        )
        self.assertIsNotNone(request)
        self.assertEqual(request["tool_key"], "neo4j_read_cypher")
        self.assertEqual(request["engine"], "neo4j")
        self.assertIn("syntax_error", request["issue_types"])
        self.assertIn("RETRUN", request["search_text"])
        self.assertIn("MATCH (n:Account)", request["search_text"])

    def test_failure_event_is_normalized_and_triggers_recall(self) -> None:
        request = inject_query_error_context.build_recall_request(
            _failure_payload("Unknown function 'apoc.map.fromPair' (not installed?)")
        )
        self.assertIsNotNone(request)
        self.assertEqual(request["source_event"], "PostToolUseFailure")

    def test_clean_success_does_not_trigger(self) -> None:
        request = inject_query_error_context.build_recall_request(
            _payload(
                response_text='[{"n.name": "Acme"}]',
                is_error=False,
            )
        )
        self.assertIsNone(request)

    def test_timeout_only_failure_does_not_trigger(self) -> None:
        request = inject_query_error_context.build_recall_request(
            _payload(
                response_text=(
                    "Neo4jError: The transaction has been terminated. "
                    "The transaction timed out."
                )
            )
        )
        self.assertIsNone(request)

    def test_permission_failure_does_not_trigger(self) -> None:
        request = inject_query_error_context.build_recall_request(
            _payload(
                tool_name="mcp__meta_knowledge_graph__bigquery_execute_query",
                query="SELECT * FROM restricted.table",
                response_text="Access Denied: User does not have permission",
            )
        )
        self.assertIsNone(request)

    def test_empty_result_success_does_not_trigger(self) -> None:
        request = inject_query_error_context.build_recall_request(
            _payload(response_text="[]", is_error=False)
        )
        self.assertIsNone(request)

    def test_interrupt_does_not_trigger(self) -> None:
        request = inject_query_error_context.build_recall_request(
            _failure_payload("Request interrupted by user", is_interrupt=True)
        )
        self.assertIsNone(request)

    def test_non_query_tool_does_not_trigger(self) -> None:
        request = inject_query_error_context.build_recall_request(
            _payload(tool_name="Bash", response_text="Syntax error near 'fi'")
        )
        self.assertIsNone(request)

    def test_missing_query_text_does_not_trigger(self) -> None:
        payload = _payload()
        payload["tool_input"] = {"params": {}}
        self.assertIsNone(inject_query_error_context.build_recall_request(payload))


class FormatContextTests(unittest.TestCase):
    def test_formats_patterns_with_fix_and_examples(self) -> None:
        context = inject_query_error_context.format_pattern_context(
            "neo4j_read_cypher",
            [
                {
                    "id": "query-error-pattern:proj:neo4j_read_cypher:abc",
                    "title": "Temporal values serialize as empty objects",
                    "error_signature": "result contains {}",
                    "root_cause": "Temporal properties need explicit projection",
                    "resolution": "Wrap temporal properties in toString(...)",
                    "example_query": "RETURN n.created_at",
                    "example_fix": "RETURN toString(n.created_at)",
                },
                {
                    "id": "query-error-pattern:proj:neo4j_read_cypher:def",
                    "title": "Unknown property",
                    "error_signature": "Unknown column",
                    "root_cause": None,
                    "resolution": "Check the schema first",
                    "example_query": None,
                    "example_fix": None,
                },
            ],
        )
        self.assertIn("Known neo4j_read_cypher failure patterns", context)
        self.assertIn("1. Temporal values serialize as empty objects", context)
        self.assertIn("Fix: Wrap temporal properties in toString(...)", context)
        self.assertIn("Corrected query: RETURN toString(n.created_at)", context)
        self.assertIn("2. Unknown property", context)
        self.assertNotIn("Root cause: None", context)
        self.assertIn("apply its fix", context)

    def test_empty_patterns_produce_no_context(self) -> None:
        self.assertEqual(
            inject_query_error_context.format_pattern_context("neo4j_read_cypher", []),
            "",
        )


class HookOutputTests(unittest.TestCase):
    def test_post_tool_use_gets_additional_context(self) -> None:
        output = inject_query_error_context.build_hook_output(
            "PostToolUse", "guidance text"
        )
        self.assertEqual(
            output,
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "guidance text",
                }
            },
        )

    def test_post_tool_use_failure_gets_block_reason(self) -> None:
        output = inject_query_error_context.build_hook_output(
            "PostToolUseFailure", "guidance text"
        )
        self.assertEqual(output, {"decision": "block", "reason": "guidance text"})


class FetchPatternsTests(unittest.TestCase):
    class FakeDriver:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        def execute_query(self, query, database_=None, **params):
            self.calls.append((query, params))
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    def test_hybrid_query_scopes_to_project_and_tool(self) -> None:
        driver = self.FakeDriver([[{"id": "p1", "title": "T", "score": 0.5}]])
        rows = inject_query_error_context.fetch_matching_patterns(
            driver,
            "neo4j",
            project_id="proj",
            tool="neo4j_read_cypher",
            search_text="Syntax error Invalid input RETRUN",
            query_vector=[0.1] * 4,
        )
        self.assertEqual(len(rows), 1)
        query, params = driver.calls[0]
        self.assertIn("VECTOR INDEX query_error_pattern_vector", query)
        self.assertIn("query_error_pattern_fulltext", query)
        self.assertEqual(params["project_id"], "proj")
        self.assertEqual(params["tool_key"], "neo4j_read_cypher")

    def test_vector_branch_gated_by_cosine_floor(self) -> None:
        driver = self.FakeDriver([[{"id": "p1", "title": "T", "score": 0.5}]])
        inject_query_error_context.fetch_matching_patterns(
            driver,
            "neo4j",
            project_id="proj",
            tool="neo4j_read_cypher",
            search_text="Syntax error Invalid input RETRUN",
            query_vector=[0.1] * 4,
            min_similarity=0.7,
        )
        query, params = driver.calls[0]
        # Floor gates the vector branch only; the fulltext branch is a lexical
        # match on the error's or query's own terms.
        self.assertEqual(query.count("raw_score >= $min_vector_score"), 1)
        # Raw cosine 0.7 converts to the index's (1 + cos) / 2 scale.
        self.assertAlmostEqual(params["min_vector_score"], 0.85)

    def test_no_floor_defaults_to_ungated_vector_scores(self) -> None:
        driver = self.FakeDriver([[{"id": "p1", "title": "T", "score": 0.5}]])
        inject_query_error_context.fetch_matching_patterns(
            driver,
            "neo4j",
            project_id="proj",
            tool="neo4j_read_cypher",
            search_text="Syntax error Invalid input RETRUN",
            query_vector=[0.1] * 4,
        )
        _, params = driver.calls[0]
        self.assertEqual(params["min_vector_score"], 0.0)

    def test_recall_passes_the_shared_injection_floor(self) -> None:
        from unittest import mock

        captured = {}

        def fake_fetch(*args, **kwargs):
            captured.update(kwargs)
            return []

        with mock.patch.object(
            inject_query_error_context, "fetch_matching_patterns", fake_fetch
        ), mock.patch.object(
            inject_query_error_context, "embed_text", lambda text: [0.1] * 4
        ), mock.patch.object(
            inject_query_error_context, "neo4j_config", lambda: ("bolt://x", "u", "p", "neo4j")
        ), mock.patch.dict(
            sys.modules,
            {"neo4j": mock.Mock(GraphDatabase=mock.Mock(driver=mock.MagicMock()))},
        ):
            inject_query_error_context.recall(_payload())

        self.assertAlmostEqual(captured.get("min_similarity"), 0.7)

    def test_vector_failure_falls_back_to_fulltext(self) -> None:
        driver = self.FakeDriver(
            [RuntimeError("no SEARCH clause"), [{"id": "p1", "title": "T", "score": 0.5}]]
        )
        rows = inject_query_error_context.fetch_matching_patterns(
            driver,
            "neo4j",
            project_id="proj",
            tool="neo4j_read_cypher",
            search_text="Syntax error Invalid input RETRUN",
            query_vector=[0.1] * 4,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(driver.calls), 2)
        fallback_query, _ = driver.calls[1]
        self.assertNotIn("VECTOR INDEX", fallback_query)
        self.assertIn("query_error_pattern_fulltext", fallback_query)

    def test_no_vector_uses_fulltext_only(self) -> None:
        driver = self.FakeDriver([[{"id": "p1", "title": "T", "score": 0.5}]])
        rows = inject_query_error_context.fetch_matching_patterns(
            driver,
            "neo4j",
            project_id="proj",
            tool="neo4j_read_cypher",
            search_text="Syntax error Invalid input RETRUN",
            query_vector=None,
        )
        self.assertEqual(len(rows), 1)
        query, _ = driver.calls[0]
        self.assertNotIn("VECTOR INDEX", query)

    def test_no_signal_returns_empty_without_query(self) -> None:
        driver = self.FakeDriver([])
        rows = inject_query_error_context.fetch_matching_patterns(
            driver,
            "neo4j",
            project_id="proj",
            tool="neo4j_read_cypher",
            search_text="",
            query_vector=None,
        )
        self.assertEqual(rows, [])
        self.assertEqual(driver.calls, [])

    def test_total_retrieval_failure_degrades_to_empty(self) -> None:
        driver = self.FakeDriver(
            [RuntimeError("vector broken"), RuntimeError("fulltext missing")]
        )
        rows = inject_query_error_context.fetch_matching_patterns(
            driver,
            "neo4j",
            project_id="proj",
            tool="neo4j_read_cypher",
            search_text="Syntax error",
            query_vector=[0.1] * 4,
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
