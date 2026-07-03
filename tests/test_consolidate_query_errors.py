from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import capture_query_failures  # noqa: E402
import consolidate_query_errors  # noqa: E402


NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


def _pending_row(
    row_id: str,
    tool_name: str = "mcp__meta_knowledge_graph__neo4j_read_cypher",
    issue_types: list[str] | None = None,
    **overrides,
) -> dict:
    row = {
        "id": row_id,
        "tool_name": tool_name,
        "tool_key": capture_query_failures.tool_key(tool_name),
        "engine": capture_query_failures._engine(tool_name),
        "query_text": "MATCH (n:Account) RETURN n.namee",
        "response_excerpt": "Neo4jError: Unknown column n.namee",
        "issue_types": issue_types or ["schema_mismatch"],
    }
    row.update(overrides)
    return row


class IssueClassificationTests(unittest.TestCase):
    def test_transient_and_consolidatable_classes_are_disjoint(self) -> None:
        overlap = (
            capture_query_failures.CONSOLIDATABLE_ISSUE_TYPES
            & capture_query_failures.TRANSIENT_ISSUE_TYPES
        )
        self.assertEqual(overlap, frozenset())

    def test_invalid_query_classes_are_consolidatable(self) -> None:
        for issue_type in ("syntax_error", "schema_mismatch", "capability_unavailable"):
            self.assertIn(
                issue_type, capture_query_failures.CONSOLIDATABLE_ISSUE_TYPES
            )

    def test_timeouts_and_transients_are_excluded(self) -> None:
        for issue_type in ("timeout", "resource_limit", "result_size_limit", "permission_error"):
            self.assertNotIn(
                issue_type, capture_query_failures.CONSOLIDATABLE_ISSUE_TYPES
            )
            self.assertIn(issue_type, capture_query_failures.TRANSIENT_ISSUE_TYPES)

    def test_tool_key_converges_across_client_mount_names(self) -> None:
        names = (
            "mcp__meta-knowledge-graph__neo4j_read_cypher",
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "mcp__plugin_meta-knowledge-graph_meta-knowledge-graph__neo4j_read_cypher",
        )
        keys = {capture_query_failures.tool_key(name) for name in names}
        self.assertEqual(keys, {"neo4j_read_cypher"})
        self.assertIsNone(capture_query_failures.tool_key("Bash"))

    def test_projection_carries_tool_key(self) -> None:
        projection = capture_query_failures.build_failure_projection(
            {
                "session_id": "session-1",
                "tool_use_id": "toolu-1",
                "tool_name": "mcp__meta_knowledge_graph__bigquery_execute_query",
                "tool_input": {"query": "SELECT FROM x"},
                "tool_response": {
                    "content": [{"type": "text", "text": "Syntax error: SELECT list must not be empty"}],
                    "isError": True,
                },
            }
        )
        self.assertIsNotNone(projection)
        self.assertEqual(projection["query"]["tool_key"], "bigquery_execute_query")


class ConsolidationGateTests(unittest.TestCase):
    def test_below_threshold_skips(self) -> None:
        proceed, reason = consolidate_query_errors.consolidation_gate(
            pending_count=1,
            threshold=1,
            last_consolidated_at=None,
            interval_hours=6.0,
            now=NOW,
        )
        self.assertFalse(proceed)
        self.assertIn("need more than 1", reason)

    def test_above_threshold_without_history_proceeds(self) -> None:
        proceed, _ = consolidate_query_errors.consolidation_gate(
            pending_count=2,
            threshold=1,
            last_consolidated_at=None,
            interval_hours=6.0,
            now=NOW,
        )
        self.assertTrue(proceed)

    def test_cooldown_blocks_recent_run(self) -> None:
        proceed, reason = consolidate_query_errors.consolidation_gate(
            pending_count=5,
            threshold=1,
            last_consolidated_at=(NOW - timedelta(hours=1)).isoformat(),
            interval_hours=6.0,
            now=NOW,
        )
        self.assertFalse(proceed)
        self.assertIn("rate-limited", reason)

    def test_elapsed_cooldown_proceeds(self) -> None:
        proceed, _ = consolidate_query_errors.consolidation_gate(
            pending_count=5,
            threshold=1,
            last_consolidated_at=(NOW - timedelta(hours=7)).isoformat(),
            interval_hours=6.0,
            now=NOW,
        )
        self.assertTrue(proceed)

    def test_unparseable_last_run_does_not_block(self) -> None:
        proceed, _ = consolidate_query_errors.consolidation_gate(
            pending_count=5,
            threshold=1,
            last_consolidated_at="not-a-date",
            interval_hours=6.0,
            now=NOW,
        )
        self.assertTrue(proceed)

    def test_neo4j_datetime_last_run_blocks_recent_run(self) -> None:
        from neo4j.time import DateTime

        proceed, reason = consolidate_query_errors.consolidation_gate(
            pending_count=5,
            threshold=1,
            last_consolidated_at=DateTime.from_native(NOW - timedelta(hours=1)),
            interval_hours=6.0,
            now=NOW,
        )
        self.assertFalse(proceed)
        self.assertIn("rate-limited", reason)


class GroupingTests(unittest.TestCase):
    def test_groups_by_canonical_tool_key(self) -> None:
        rows = [
            _pending_row("q1", "mcp__meta-knowledge-graph__neo4j_read_cypher"),
            _pending_row("q2", "mcp__plugin_x_meta-knowledge-graph__neo4j_read_cypher"),
            _pending_row("q3", "mcp__meta_knowledge_graph__bigquery_execute_query"),
        ]
        grouped = consolidate_query_errors.group_pending_by_tool(rows)
        self.assertEqual(
            {key: len(items) for key, items in grouped.items()},
            {"neo4j_read_cypher": 2, "bigquery_execute_query": 1},
        )

    def test_legacy_rows_without_tool_key_fall_back_to_engine(self) -> None:
        rows = [
            _pending_row("q1", tool_key=None, tool_name="", engine="bigquery"),
        ]
        grouped = consolidate_query_errors.group_pending_by_tool(rows)
        self.assertEqual(list(grouped), ["bigquery_execute_query"])

    def test_unresolvable_rows_are_dropped(self) -> None:
        rows = [_pending_row("q1", tool_key=None, tool_name="Bash", engine="")]
        self.assertEqual(consolidate_query_errors.group_pending_by_tool(rows), {})


class PromptBuildTests(unittest.TestCase):
    def test_prompt_contains_existing_patterns_and_failures(self) -> None:
        prompt = consolidate_query_errors.build_consolidation_prompt(
            "neo4j_read_cypher",
            "neo4j",
            [
                {
                    "id": "query-error-pattern:p:neo4j_read_cypher:abc",
                    "title": "Unknown property",
                    "error_signature": "Unknown column ...",
                    "root_cause": "Typo in property",
                    "resolution": "Check the schema first",
                    "issue_types": ["schema_mismatch"],
                }
            ],
            [_pending_row("query-execution:s:q1")],
        )
        self.assertIn("Tool: neo4j_read_cypher (engine: neo4j)", prompt)
        self.assertIn("query-error-pattern:p:neo4j_read_cypher:abc", prompt)
        self.assertIn("query-execution:s:q1", prompt)
        self.assertIn("MATCH (n:Account) RETURN n.namee", prompt)

    def test_prompt_handles_empty_pattern_library(self) -> None:
        prompt = consolidate_query_errors.build_consolidation_prompt(
            "bigquery_execute_query", "bigquery", [], [_pending_row("q1")]
        )
        self.assertIn("(none yet)", prompt)


class ResponseParsingTests(unittest.TestCase):
    PENDING_IDS = {"query-execution:s:q1", "query-execution:s:q2"}
    EXISTING_IDS = {"query-error-pattern:proj:neo4j_read_cypher:known"}

    def _parse(self, payload, text=None):
        return consolidate_query_errors.parse_consolidation_response(
            text if text is not None else json.dumps(payload),
            project_id="proj",
            tool="neo4j_read_cypher",
            pending_ids=self.PENDING_IDS,
            existing_ids=self.EXISTING_IDS,
        )

    def test_parses_new_pattern_with_deterministic_id(self) -> None:
        result = self._parse(
            {
                "patterns": [
                    {
                        "id": None,
                        "title": "Unknown property name",
                        "error_signature": "Unknown column n.namee",
                        "root_cause": "Property typo",
                        "resolution": "Inspect the schema and use n.name",
                        "example_query": "MATCH (n) RETURN n.namee",
                        "example_fix": "MATCH (n) RETURN n.name",
                        "issue_types": ["schema_mismatch"],
                        "confidence": 0.95,
                        "source_execution_ids": ["query-execution:s:q1", "bogus"],
                    }
                ],
                "skipped_execution_ids": ["query-execution:s:q2", "bogus"],
            }
        )
        self.assertIsNotNone(result)
        patterns, skipped = result
        self.assertEqual(len(patterns), 1)
        self.assertEqual(
            patterns[0]["id"],
            consolidate_query_errors.pattern_id(
                "proj", "neo4j_read_cypher", "Unknown column n.namee"
            ),
        )
        self.assertEqual(patterns[0]["source_execution_ids"], ["query-execution:s:q1"])
        self.assertEqual(skipped, ["query-execution:s:q2"])

    def test_existing_pattern_id_is_kept(self) -> None:
        result = self._parse(
            {
                "patterns": [
                    {
                        "id": "query-error-pattern:proj:neo4j_read_cypher:known",
                        "title": "Known",
                        "error_signature": "sig",
                        "resolution": "fix",
                    }
                ]
            }
        )
        patterns, _ = result
        self.assertEqual(
            patterns[0]["id"], "query-error-pattern:proj:neo4j_read_cypher:known"
        )

    def test_hallucinated_pattern_id_is_replaced(self) -> None:
        result = self._parse(
            {
                "patterns": [
                    {
                        "id": "query-error-pattern:proj:neo4j_read_cypher:madeup",
                        "title": "New",
                        "error_signature": "some signature",
                        "resolution": "fix",
                    }
                ]
            }
        )
        patterns, _ = result
        self.assertEqual(
            patterns[0]["id"],
            consolidate_query_errors.pattern_id(
                "proj", "neo4j_read_cypher", "some signature"
            ),
        )

    def test_fenced_json_is_accepted(self) -> None:
        payload = {
            "patterns": [
                {"id": None, "title": "T", "error_signature": "sig", "resolution": "fix"}
            ],
            "skipped_execution_ids": [],
        }
        result = self._parse(None, text=f"```json\n{json.dumps(payload)}\n```")
        self.assertIsNotNone(result)
        self.assertEqual(len(result[0]), 1)

    def test_incomplete_pattern_rows_are_dropped(self) -> None:
        result = self._parse(
            {
                "patterns": [
                    {"id": None, "title": "T", "error_signature": "", "resolution": "fix"},
                    {"id": None, "title": "T2"},
                    "not-a-dict",
                ]
            }
        )
        patterns, _ = result
        self.assertEqual(patterns, [])

    def test_invalid_confidence_defaults(self) -> None:
        result = self._parse(
            {
                "patterns": [
                    {
                        "id": None,
                        "title": "T",
                        "error_signature": "sig",
                        "resolution": "fix",
                        "confidence": "very high",
                    }
                ]
            }
        )
        patterns, _ = result
        self.assertEqual(patterns[0]["confidence"], 0.7)

    def test_unusable_response_returns_none(self) -> None:
        self.assertIsNone(self._parse(None, text="I could not find any patterns."))
        self.assertIsNone(self._parse(None, text=json.dumps({"patterns": "nope"})))


class EmbeddingAttachTests(unittest.TestCase):
    def test_embeddings_attached_per_pattern(self) -> None:
        calls = {}

        def fake_embed_texts(texts):
            calls["texts"] = texts
            return [[0.1, 0.2], None]

        original = consolidate_query_errors.embed_texts
        consolidate_query_errors.embed_texts = fake_embed_texts
        try:
            patterns = [
                {
                    "title": "T",
                    "error_signature": "sig",
                    "resolution": "fix",
                    "example_query": "MATCH (n) RETURN n",
                },
                {"title": "U", "error_signature": "sig2", "resolution": "fix2"},
            ]
            consolidate_query_errors.attach_pattern_embeddings(patterns)
        finally:
            consolidate_query_errors.embed_texts = original

        self.assertIn("MATCH (n) RETURN n", calls["texts"][0])
        self.assertEqual(patterns[0]["embedding"], [0.1, 0.2])
        self.assertNotIn("embedding", patterns[1])


class WriteConsolidationTests(unittest.TestCase):
    def test_write_runs_profile_pattern_and_stamp_statements(self) -> None:
        executed = []

        class FakeTx:
            def run(self, query, **params):
                executed.append((query, params))

        project = consolidate_query_errors.ProjectRef(id="proj", name="Proj")
        consolidate_query_errors.write_consolidation(
            FakeTx(),
            project,
            "neo4j_read_cypher",
            "neo4j",
            [
                {
                    "id": "query-error-pattern:proj:neo4j_read_cypher:abc",
                    "title": "T",
                    "error_signature": "sig",
                    "root_cause": None,
                    "resolution": "fix",
                    "example_query": None,
                    "example_fix": None,
                    "issue_types": ["schema_mismatch"],
                    "confidence": 0.9,
                    "source_execution_ids": ["query-execution:s:q1"],
                }
            ],
            ["query-execution:s:q1", "query-execution:s:q2"],
            model="test-model",
            timestamp="2026-07-02T12:00:00+00:00",
        )

        self.assertEqual(len(executed), 3)
        profile_query, profile_params = executed[0]
        self.assertIn("ToolErrorProfile", profile_query)
        self.assertEqual(
            profile_params["profile_id"], "tool-error-profile:proj:neo4j_read_cypher"
        )
        pattern_query, pattern_params = executed[1]
        self.assertIn("QueryErrorPattern", pattern_query)
        self.assertIn("DERIVED_FROM", pattern_query)
        self.assertEqual(len(pattern_params["patterns"]), 1)
        stamp_query, stamp_params = executed[2]
        self.assertIn("error_consolidated_at", stamp_query)
        self.assertEqual(len(stamp_params["execution_ids"]), 2)

    def test_write_without_patterns_still_stamps_consumed(self) -> None:
        executed = []

        class FakeTx:
            def run(self, query, **params):
                executed.append((query, params))

        project = consolidate_query_errors.ProjectRef(id="proj", name="Proj")
        consolidate_query_errors.write_consolidation(
            FakeTx(),
            project,
            "bigquery_execute_query",
            "bigquery",
            [],
            ["query-execution:s:q1"],
            model="test-model",
            timestamp="2026-07-02T12:00:00+00:00",
        )
        self.assertEqual(len(executed), 2)
        self.assertIn("ToolErrorProfile", executed[0][0])
        self.assertIn("error_consolidated_at", executed[1][0])


if __name__ == "__main__":
    unittest.main()
