from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import capture_query_failures  # noqa: E402


def _payload(tool_name: str, query: str, response) -> dict:
    return {
        "session_id": "session-1",
        "tool_use_id": "toolu-1",
        "tool_name": tool_name,
        "tool_input": {"query": query},
        "tool_response": response,
    }


def _content(text: str, *, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class QueryFailureCaptureTests(unittest.TestCase):
    def _projection(self, tool_name: str, query: str, response) -> dict:
        projection = capture_query_failures.build_failure_projection(
            _payload(tool_name, query, response)
        )
        self.assertIsNotNone(projection)
        return projection

    def _issue_types(self, projection: dict) -> list[str]:
        return sorted(issue["type"] for issue in projection["issues"])

    def test_ignores_clean_bigquery_result(self) -> None:
        response = _content(
            '{"schema":{"fields":[{"name":"account_id","type":"STRING"}]},'
            '"rows":[{"f":[{"v":"kpmg"}]}],"jobComplete":true,'
            '"queryId":"job_1","totalBytesBilled":"10485760"}'
        )

        projection = capture_query_failures.build_failure_projection(
            _payload(
                "mcp__meta_knowledge_graph__bigquery_execute_query",
                "SELECT * FROM `acme_corp.accounts` LIMIT 1",
                response,
            )
        )

        self.assertIsNone(projection)

    def test_captures_bigquery_empty_result(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__bigquery_execute_query",
            "SELECT * FROM `acme_corp.accounts` WHERE FALSE",
            _content(
                '{"schema":{"fields":[{"name":"account_id","type":"STRING"}]},'
                '"jobComplete":true,"queryId":"job_empty","totalBytesProcessed":"0"}'
            ),
        )

        self.assertEqual(self._issue_types(projection), ["empty_result"])
        self.assertEqual(projection["query"]["row_count"], 0)
        self.assertTrue(projection["query"]["needs_optimization"])
        self.assertEqual(projection["query"]["query_job_id"], "job_empty")

    def test_captures_bigquery_syntax_error_text(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__bigquery_execute_query",
            "SELECT FROM `acme_corp.accounts` LIMIT 1",
            [{"type": "text", "text": "Syntax error: SELECT list must not be empty at [1:8]"}],
        )

        self.assertEqual(self._issue_types(projection), ["syntax_error"])
        self.assertEqual(projection["query"]["status"], "error")

    def test_captures_bigquery_schema_mismatch(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__bigquery_execute_query",
            "SELECT * FROM `acme_corp.__definitely_missing_table__` LIMIT 1",
            _content(
                "Not found: Table llm-experiments-387609:"
                "acme_corp.__definitely_missing_table__ was not found in location US"
            ),
        )

        self.assertEqual(self._issue_types(projection), ["schema_mismatch"])

    def test_captures_neo4j_empty_result(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "MATCH (a:Account {slug: '__missing__'}) RETURN a.slug AS slug",
            _content("[]"),
        )

        self.assertEqual(self._issue_types(projection), ["empty_result"])
        self.assertEqual(projection["query"]["row_count"], 0)

    def test_captures_neo4j_capability_unavailable(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "CALL gds.graph.list() YIELD graphName RETURN graphName LIMIT 1",
            _content(
                "error during GetQueryType: Neo4jError: "
                "Neo.ClientError.Procedure.ProcedureNotFound "
                "(There is no procedure with the name `gds.graph.list` registered "
                "for this database instance.)"
            ),
        )

        self.assertEqual(self._issue_types(projection), ["capability_unavailable"])

    def test_captures_neo4j_temporal_serialization_issue(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "MATCH (a:Account) RETURN a.renewal_date AS renewalDate LIMIT 1",
            _content('[{"renewalDate": {}}]'),
        )

        self.assertEqual(self._issue_types(projection), ["serialization_issue"])
        self.assertTrue(projection["query"]["needs_optimization"])

    def test_captures_missing_tool_response_as_shape_error(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "MATCH (a:Account) RETURN a LIMIT 1",
            None,
        )

        self.assertEqual(self._issue_types(projection), ["tool_output_shape_error"])
        self.assertEqual(projection["query"]["response_shape"], "missing")

    def test_write_projection_does_not_use_driver_reserved_query_parameter(self) -> None:
        class FakeTx:
            def __init__(self) -> None:
                self.calls = []

            def run(self, statement: str, **params) -> None:
                self.calls.append((statement, params))

        projection = self._projection(
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "MATCH (a:Account {id: '__missing__'}) RETURN a.name AS name",
            _content("[]"),
        )
        project = SimpleNamespace(
            id="meta-knowledge-graph",
            name="Meta Knowledge Graph",
            description=None,
            status="active",
            repo_root="/tmp/project",
            source="test",
        )
        tx = FakeTx()

        capture_query_failures.write_failure_projection(
            tx, project, "session-1", projection, "2026-06-12T00:00:00+00:00"
        )

        write_params = tx.calls[-1][1]
        self.assertIn("query_row", write_params)
        self.assertNotIn("query", write_params)


if __name__ == "__main__":
    unittest.main()
