from __future__ import annotations

import json
import sys
import tempfile
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

    def test_namespaced_unknown_function_is_capability(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "RETURN apoc.text.format('%d', [1]) AS x",
            _content(
                "Neo.ClientError.Statement.SyntaxError: "
                "Unknown function 'apoc.text.format' (line 1, column 8)"
            ),
        )

        self.assertEqual(self._issue_types(projection), ["capability_unavailable"])

    def test_bare_unknown_function_typo_is_syntax(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "RETURN datetimee() AS now",
            _content(
                "Neo.ClientError.Statement.SyntaxError: "
                "Unknown function 'datetimee' (line 1, column 8)"
            ),
        )

        self.assertEqual(self._issue_types(projection), ["syntax_error"])

    def test_captures_neo4j_timeout(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "MATCH (a)-[*]->(b) RETURN count(*) AS c",
            _content(
                "The transaction has been terminated. "
                "Retry your operation in a new transaction: the transaction timed out."
            ),
        )

        self.assertEqual(self._issue_types(projection), ["timeout"])
        self.assertTrue(projection["query"]["needs_optimization"])

    def test_captures_bigquery_timeout(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__bigquery_execute_query",
            "SELECT * FROM `acme_corp.account_product_usage`",
            _content("Operation timed out after 600000 ms"),
        )

        self.assertEqual(self._issue_types(projection), ["timeout"])

    def test_captures_bigquery_result_size_limit(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__bigquery_execute_query",
            "SELECT * FROM `acme_corp.account_product_usage`",
            _content(
                "Response too large to return. Consider setting allowLargeResults "
                "to true in your job configuration."
            ),
        )

        self.assertEqual(self._issue_types(projection), ["result_size_limit"])
        self.assertTrue(projection["query"]["needs_optimization"])

    def test_captures_bigquery_resource_limit(self) -> None:
        projection = self._projection(
            "mcp__meta_knowledge_graph__bigquery_execute_query",
            "SELECT * FROM `acme_corp.account_product_usage` ORDER BY mau",
            _content(
                "Resources exceeded during query execution: the query exceeded "
                "memory limit."
            ),
        )

        self.assertEqual(self._issue_types(projection), ["resource_limit"])

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
            tx, project, "session-1", projection, "2026-06-12T00:00:00+00:00", user_id="tomaz@example.com"
        )

        write_params = tx.calls[-1][1]
        self.assertIn("query_row", write_params)
        self.assertNotIn("query", write_params)


class ToolFailureEventTests(unittest.TestCase):
    """Claude Code PostToolUseFailure payloads enter the same unified pipeline."""

    def _failure_payload(self, *, error: str, is_interrupt: bool = False) -> dict:
        return {
            "session_id": "session-1",
            "hook_event_name": "PostToolUseFailure",
            "tool_use_id": "toolu-9",
            "tool_name": "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "tool_input": {"query": "MATCH (n) RETRUN n"},
            "error": error,
            "is_interrupt": is_interrupt,
        }

    def test_failure_payload_normalizes_to_unified_error_response(self) -> None:
        error = "Neo.ClientError.Statement.SyntaxError: Invalid input 'RETRUN'"
        normalized = capture_query_failures.normalize_tool_failure_payload(
            self._failure_payload(error=error)
        )

        self.assertEqual(normalized["hook_event_name"], "PostToolUse")
        self.assertEqual(normalized["source_event"], "PostToolUseFailure")
        self.assertTrue(normalized["tool_error"])
        self.assertIs(normalized["is_interrupt"], False)
        self.assertEqual(
            normalized["tool_response"],
            {"content": [{"type": "text", "text": error}], "isError": True},
        )

    def test_non_failure_payload_passes_through_unchanged(self) -> None:
        payload = _payload(
            "mcp__meta_knowledge_graph__neo4j_read_cypher",
            "MATCH (n) RETURN n LIMIT 1",
            _content("[]"),
        )

        self.assertIs(
            capture_query_failures.normalize_tool_failure_payload(payload), payload
        )

    def test_failure_payload_is_classified_like_any_error_response(self) -> None:
        normalized = capture_query_failures.normalize_tool_failure_payload(
            self._failure_payload(
                error="Neo.ClientError.Statement.SyntaxError: Invalid input 'RETRUN'"
            )
        )
        projection = capture_query_failures.build_failure_projection(normalized)

        self.assertIsNotNone(projection)
        assert projection is not None
        self.assertEqual(projection["query"]["status"], "error")
        self.assertEqual(
            sorted(issue["type"] for issue in projection["issues"]), ["syntax_error"]
        )
        # Same stable id the Stop-time transcript scan derives, so live and
        # fallback capture converge on one QueryExecution node.
        self.assertEqual(
            projection["query"]["id"], "query-execution:session-1:toolu-9"
        )

    def test_interrupted_failure_is_not_captured(self) -> None:
        captured = capture_query_failures.capture(
            self._failure_payload(
                error="[Request interrupted by user]", is_interrupt=True
            )
        )

        self.assertEqual(captured, 0)


class TranscriptExtractionTests(unittest.TestCase):
    """Stop-hook path: failed query tool calls are recovered from the transcript."""

    def _tool_use(self, tool_use_id: str, tool_name: str, query: str) -> dict:
        return {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": {"query": query},
                    }
                ],
            },
        }

    def _tool_result(self, tool_use_id: str, text: str, *, is_error: bool = False) -> dict:
        return {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [{"type": "text", "text": text}],
                        "is_error": is_error,
                    }
                ],
            },
        }

    def _codex_function_call(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict,
        *,
        namespace: str = "mcp__meta_knowledge_graph",
    ) -> dict:
        return {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": tool_name,
                "namespace": namespace,
                "arguments": json.dumps(arguments),
                "call_id": call_id,
            },
        }

    def _codex_tool_end(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict,
        text: str,
        *,
        is_error: bool = False,
        server: str = "meta-knowledge-graph",
    ) -> dict:
        return {
            "type": "event_msg",
            "payload": {
                "type": "mcp_tool_call_end",
                "call_id": call_id,
                "invocation": {
                    "server": server,
                    "tool": tool_name,
                    "arguments": arguments,
                },
                "result": {
                    "Ok": {
                        "content": [{"type": "text", "text": text}],
                        "isError": is_error,
                    }
                },
            },
        }

    def _write_transcript(self, records: list[dict]) -> str:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_extracts_and_classifies_failed_calls_post_tool_use_misses(self) -> None:
        bq = "mcp__meta_knowledge_graph__bigquery_execute_query"
        neo = "mcp__meta_knowledge_graph__neo4j_read_cypher"
        records = [
            # isError calls PostToolUse never delivers:
            self._tool_use("toolu_1", bq, "SELECT FROM acme_corp.accounts"),
            self._tool_result(
                "toolu_1", "Syntax error: SELECT list must not be empty at [1:8]", is_error=True
            ),
            self._tool_use("toolu_2", neo, "CALL gds.graph.list() YIELD graphName RETURN graphName"),
            self._tool_result(
                "toolu_2",
                "Neo.ClientError.Procedure.ProcedureNotFound (There is no procedure "
                "with the name `gds.graph.list` registered for this database instance.)",
                is_error=True,
            ),
            # plain success that must be ignored entirely:
            self._tool_use("toolu_3", neo, "MATCH (a:Account) RETURN a.name AS name LIMIT 1"),
            self._tool_result("toolu_3", '[{"name": "Pfizer"}]'),
            # success-path issue (also seen live by PostToolUse) -> same id, converges:
            self._tool_use("toolu_4", bq, "SELECT account_id FROM acme_corp.accounts WHERE FALSE"),
            self._tool_result(
                "toolu_4",
                '{"schema":{"fields":[{"name":"account_id","type":"STRING"}]},'
                '"jobComplete":true,"queryId":"job_empty"}',
            ),
        ]
        transcript = self._write_transcript(records)

        payloads = capture_query_failures.extract_query_payloads(transcript, "session-x")
        projections = {
            payload["tool_use_id"]: capture_query_failures.build_failure_projection(payload)
            for payload in payloads
        }

        # Four query calls extracted; the clean success yields no projection.
        self.assertEqual(set(projections), {"toolu_1", "toolu_2", "toolu_3", "toolu_4"})
        self.assertIsNone(projections["toolu_3"])
        self.assertEqual(
            sorted(i["type"] for i in projections["toolu_1"]["issues"]), ["syntax_error"]
        )
        self.assertEqual(
            sorted(i["type"] for i in projections["toolu_2"]["issues"]),
            ["capability_unavailable"],
        )
        self.assertEqual(
            sorted(i["type"] for i in projections["toolu_4"]["issues"]), ["empty_result"]
        )
        # Stable id shared with the PostToolUse hook, so re-capture converges.
        self.assertEqual(
            projections["toolu_1"]["query"]["id"], "query-execution:session-x:toolu_1"
        )

    def test_live_failure_event_converges_with_transcript_recovery(self) -> None:
        neo = "mcp__meta_knowledge_graph__neo4j_read_cypher"
        query = "CALL gds.graph.list() YIELD graphName RETURN graphName"
        error = (
            "Neo.ClientError.Procedure.ProcedureNotFound (There is no procedure "
            "with the name `gds.graph.list` registered for this database instance.)"
        )
        live = capture_query_failures.build_failure_projection(
            capture_query_failures.normalize_tool_failure_payload(
                {
                    "session_id": "session-x",
                    "hook_event_name": "PostToolUseFailure",
                    "tool_use_id": "toolu_2",
                    "tool_name": neo,
                    "tool_input": {"query": query},
                    "error": error,
                    "is_interrupt": False,
                }
            )
        )
        transcript = self._write_transcript(
            [
                self._tool_use("toolu_2", neo, query),
                self._tool_result("toolu_2", error, is_error=True),
            ]
        )
        recovered = capture_query_failures.build_failure_projection(
            capture_query_failures.extract_query_payloads(transcript, "session-x")[0]
        )

        self.assertIsNotNone(live)
        self.assertEqual(live, recovered)

    def test_ignores_non_query_tools(self) -> None:
        records = [
            self._tool_use(
                "toolu_n", "mcp__meta_knowledge_graph__search_news", "type:Article tags.label:x"
            ),
            self._tool_result("toolu_n", "Diffbot request failed: boom", is_error=True),
        ]
        transcript = self._write_transcript(records)

        self.assertEqual(
            capture_query_failures.extract_query_payloads(transcript, "session-x"), []
        )

    def test_missing_transcript_file_is_safe(self) -> None:
        self.assertEqual(
            capture_query_failures.extract_query_payloads("/no/such/file.jsonl", "s"), []
        )

    def test_extracts_codex_mcp_tool_call_end_records(self) -> None:
        records = [
            self._codex_function_call(
                "call_bq",
                "bigquery_execute_query",
                {"query": "SELECT FROM `acme_corp.accounts` LIMIT 1"},
            ),
            self._codex_tool_end(
                "call_bq",
                "bigquery_execute_query",
                {"query": "SELECT FROM `acme_corp.accounts` LIMIT 1"},
                "Syntax error: SELECT list must not be empty at [1:8]",
                is_error=True,
            ),
            self._codex_function_call(
                "call_neo",
                "neo4j_read_cypher",
                {
                    "query": "MATCH (q:QueryExecution) RETURN collect(q.id) AS ids "
                    "ORDER BY q.last_seen_at"
                },
            ),
            self._codex_tool_end(
                "call_neo",
                "neo4j_read_cypher",
                {
                    "query": "MATCH (q:QueryExecution) RETURN collect(q.id) AS ids "
                    "ORDER BY q.last_seen_at"
                },
                "Neo.ClientError.Statement.SyntaxError: In a WITH/RETURN with DISTINCT "
                "or an aggregation, it is not possible to access variables declared "
                "before the WITH/RETURN: q",
                is_error=True,
            ),
            self._codex_function_call(
                "call_ok",
                "neo4j_read_cypher",
                {"query": "MATCH (a:Account) RETURN a.name AS name LIMIT 1"},
            ),
            self._codex_tool_end(
                "call_ok",
                "neo4j_read_cypher",
                {"query": "MATCH (a:Account) RETURN a.name AS name LIMIT 1"},
                '[{"name": "Pfizer"}]',
            ),
        ]
        transcript = self._write_transcript(records)

        payloads = capture_query_failures.extract_query_payloads(transcript, "session-codex")
        projections = {
            payload["tool_use_id"]: capture_query_failures.build_failure_projection(payload)
            for payload in payloads
        }

        self.assertEqual(set(projections), {"call_bq", "call_neo", "call_ok"})
        self.assertEqual(
            projections["call_bq"]["query"]["id"],
            "query-execution:session-codex:call_bq",
        )
        self.assertEqual(
            sorted(i["type"] for i in projections["call_bq"]["issues"]),
            ["syntax_error"],
        )
        self.assertEqual(
            sorted(i["type"] for i in projections["call_neo"]["issues"]),
            ["syntax_error"],
        )
        self.assertIsNone(projections["call_ok"])

    def test_extracts_codex_event_msg_without_prior_response_item(self) -> None:
        arguments = {"query": "SELECT account_id FROM `acme_corp.accounts` WHERE FALSE"}
        transcript = self._write_transcript(
            [
                self._codex_tool_end(
                    "call_event_only",
                    "bigquery_execute_query",
                    arguments,
                    '{"schema":{"fields":[{"name":"account_id","type":"STRING"}]},'
                    '"jobComplete":true,"queryId":"job_empty"}',
                )
            ]
        )

        payloads = capture_query_failures.extract_query_payloads(transcript, "session-codex")
        self.assertEqual(len(payloads), 1)
        projection = capture_query_failures.build_failure_projection(payloads[0])

        self.assertIsNotNone(projection)
        self.assertEqual(
            projection["query"]["tool_name"],
            "mcp__meta_knowledge_graph__bigquery_execute_query",
        )
        self.assertEqual(
            sorted(issue["type"] for issue in projection["issues"]), ["empty_result"]
        )


if __name__ == "__main__":
    unittest.main()
