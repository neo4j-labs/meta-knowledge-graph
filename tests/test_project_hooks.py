from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
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
process_project = load_hook_module("process_project")
log_event = load_hook_module("log_event")
inject_project_context = load_hook_module("inject_project_context")


class ProjectHookTests(unittest.TestCase):
    def test_project_id_uses_repo_folder_name(self) -> None:
        project = project_common.resolve_project({}, ROOT)

        self.assertIsNotNone(project)
        assert project is not None
        self.assertEqual(project.id, "meta-knowledge-graph")
        self.assertEqual(project.name, "Meta Knowledge Graph")

    def test_llm_backend_auto_defaults_to_litellm(self) -> None:
        with patch.dict(
            os.environ,
            {"MKG_LLM_BACKEND": "auto", "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"},
            clear=False,
        ):
            self.assertEqual(project_common.llm_backend(), "litellm")

    def test_llm_model_defaults_to_claude_for_claude_client(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_MODEL": "",
                "MKG_DEFAULT_LLM_MODEL": "",
                "CLAUDE_PROJECT_DIR": "",
            },
            clear=False,
        ):
            self.assertEqual(
                project_common.llm_model(client="claude_code"),
                "anthropic/claude-haiku-4-5",
            )

    def test_llm_model_keeps_openai_default_for_codex_client(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_MODEL": "",
                "MKG_DEFAULT_LLM_MODEL": "",
                "CLAUDE_PROJECT_DIR": "",
            },
            clear=False,
        ):
            self.assertEqual(project_common.llm_model(client="codex"), "gpt-5.4-mini")

    def test_litellm_claude_model_uses_oauth_token_when_no_explicit_auth(self) -> None:
        credential = project_common.ClaudeOAuthCredential(
            token="sk-ant-oat01-test",
            source="test",
        )
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": "",
                "ANTHROPIC_BASE_URL": "",
            },
            clear=False,
        ):
            with patch.object(
                project_common,
                "_read_claude_oauth_token",
                return_value=credential,
            ):
                token = project_common._litellm_api_key_for_model(
                    "anthropic/claude-haiku-4-5"
                )

        self.assertEqual(token, "sk-ant-oat01-test")

    def test_litellm_claude_oauth_does_not_override_explicit_auth(self) -> None:
        credential = project_common.ClaudeOAuthCredential(
            token="sk-ant-oat01-test",
            source="test",
        )
        for key, value in (
            ("ANTHROPIC_API_KEY", "sk-ant-api03-explicit"),
            ("ANTHROPIC_AUTH_TOKEN", "gateway-token"),
            ("ANTHROPIC_BASE_URL", "https://gateway.example"),
        ):
            with self.subTest(key=key):
                with patch.dict(
                    os.environ,
                    {
                        "ANTHROPIC_API_KEY": "",
                        "ANTHROPIC_AUTH_TOKEN": "",
                        "ANTHROPIC_BASE_URL": "",
                        key: value,
                    },
                    clear=False,
                ):
                    with patch.object(
                        project_common,
                        "_read_claude_oauth_token",
                        return_value=credential,
                    ):
                        token = project_common._litellm_api_key_for_model(
                            "anthropic/claude-haiku-4-5"
                        )
                self.assertIsNone(token)

    def test_litellm_non_claude_model_does_not_read_oauth(self) -> None:
        with patch.object(
            project_common,
            "_read_claude_oauth_token",
            side_effect=AssertionError("should not read Claude OAuth"),
        ):
            token = project_common._litellm_api_key_for_model("gpt-5.4-mini")

        self.assertIsNone(token)

    def test_parse_claude_oauth_payload_rejects_expired_jwt(self) -> None:
        header = "eyJhbGciOiAibm9uZSJ9"
        payload = "eyJleHAiOiAxfQ"
        expired = f"{header}.{payload}.sig"

        credential = project_common._parse_claude_oauth_payload(
            expired,
            source="test",
        )

        self.assertIsNone(credential)

    def test_memory_action_extraction_returns_model_used(self) -> None:
        completion = '{"learnings": [], "decisions": []}'
        with (
            patch.object(
                process_project,
                "llm_readiness_status",
                return_value=(True, None),
            ) as readiness,
            patch.object(process_project, "llm_complete", return_value=completion) as complete,
        ):
            actions, model, meta = process_project.ask_llm_for_memory_actions_with_model(
                "extract memory",
                model="anthropic/claude-haiku-4-5",
            )

        self.assertEqual(actions, {"learnings": [], "decisions": []})
        self.assertEqual(model, "anthropic/claude-haiku-4-5")
        self.assertEqual(meta, {"status": "called", "skip_reason": None, "error": None})
        readiness.assert_called_once_with("anthropic/claude-haiku-4-5")
        self.assertEqual(complete.call_args.kwargs["model"], "anthropic/claude-haiku-4-5")

    def test_memory_action_extraction_records_model_when_not_ready(self) -> None:
        with (
            patch.object(
                process_project,
                "llm_readiness_status",
                return_value=(False, "Claude OAuth token unavailable"),
            ),
            patch.object(
                process_project,
                "llm_complete",
                side_effect=AssertionError("should not call LLM when unavailable"),
            ),
        ):
            actions, model, meta = process_project.ask_llm_for_memory_actions_with_model(
                "extract memory",
                model="anthropic/claude-haiku-4-5",
            )

        self.assertEqual(actions, {"learnings": [], "decisions": []})
        self.assertEqual(model, "anthropic/claude-haiku-4-5")
        self.assertEqual(
            meta,
            {
                "status": "skipped",
                "skip_reason": "Claude OAuth token unavailable",
                "error": None,
            },
        )

    def test_event_hook_uses_session_event_label(self) -> None:
        captured: list[str] = []

        class FakeTx:
            def run(self, query: str, **params):
                del params
                captured.append(query)

        log_event._ensure_constraints(FakeTx())
        log_event._append_event(
            FakeTx(),
            "session-1",
            "codex",
            {"event_id": "event-1", "timestamp": "2026-06-11T00:00:00+00:00"},
        )

        joined = "\n".join(captured)
        self.assertIn("FOR (e:SessionEvent)", joined)
        self.assertIn("CREATE (e:SessionEvent $event_props)", joined)
        self.assertIn("prev:SessionEvent", joined)
        self.assertIn("FOR (i:SystemPromptInjection)", joined)
        self.assertNotIn("m:Memory", joined)
        self.assertNotIn(":Event", joined)

    def test_memory_prompt_includes_similar_existing_memory(self) -> None:
        events = [
            {
                "event_name": "UserPromptSubmit",
                "prompt": (
                    "Store learnings at Stop and update similar existing ones "
                    "instead of creating duplicates."
                )
            }
        ]
        project = project_common.ProjectRef(id="mkg", name="MKG")

        prompt = process_project.build_memory_extraction_prompt(
            project,
            "turn",
            events,
            [
                {
                    "id": "learning:mkg:existing",
                    "status": "candidate",
                    "task_pattern": "project memory duplicate suppression",
                    "text": "Update similar learnings instead of duplicating them.",
                }
            ],
            [
                {
                    "id": "decision:mkg:existing",
                    "task_pattern": "project memory duplicate suppression",
                    "text": "Ask the LLM whether to update or create memory.",
                    "rationale": "The LLM sees similar existing memory.",
                }
            ],
        )

        self.assertIn("Existing similar learnings", prompt)
        self.assertIn("learning:mkg:existing", prompt)
        self.assertIn("Existing similar decisions", prompt)
        self.assertIn("decision:mkg:existing", prompt)
        self.assertIn('"action": "create|update|ignore"', prompt)
        self.assertIn("routing precedence", prompt)
        self.assertIn("Do not also create a learning", prompt)
        self.assertIn("we decided", prompt)
        self.assertIn('"scope": "project|user"', prompt)
        self.assertIn("durable fact about the *person*", prompt)
        self.assertIn("Every decision also has a scope", prompt)
        self.assertIn("cross-project working preferences", prompt)
        self.assertIn("sensitive personal data", prompt)
        # The self-rewriting prompt-suggestion buckets are gone.
        self.assertNotIn("system_prompt_updates", prompt)
        self.assertNotIn("memory_extraction_prompt_updates", prompt)

    def test_memory_prompt_falls_back_when_template_lacks_required_tokens(self) -> None:
        prompt = process_project.build_memory_extraction_prompt(
            project_common.ProjectRef(id="mkg", name="MKG"),
            "turn",
            [{"event_name": "UserPromptSubmit", "prompt": "remember duplicate handling"}],
            [],
            [],
            template="Missing the dynamic placeholders.",
        )

        self.assertIn("Project: MKG (mkg)", prompt)
        self.assertIn("remember duplicate handling", prompt)
        self.assertIn("Existing similar learnings", prompt)

    def test_llm_action_rows_skip_ignored_memory(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        actions = {
            "learnings": [
                {
                    "action": "update",
                    "existing_id": "learning:mkg:existing",
                    "text": "Refined learning text.",
                    "task_pattern": "project memory duplicate suppression",
                    "confidence": 0.8,
                },
                {"action": "ignore", "reason": "Routine work."},
            ],
            "decisions": [
                {
                    "action": "create",
                    "text": "Ask the LLM to extract memory writes.",
                    "rationale": "It can compare against similar existing memory.",
                    "task_pattern": "project memory llm extraction",
                    "confidence": 0.9,
                }
            ],
        }

        learning_rows, decision_rows = process_project._memory_rows_from_actions(
            project, "turn", actions
        )

        self.assertEqual(len(learning_rows), 1)
        self.assertEqual(learning_rows[0]["id"], "learning:mkg:existing")
        self.assertEqual(learning_rows[0]["action"], "update")
        self.assertEqual(learning_rows[0]["scope"], "project")
        self.assertEqual(len(decision_rows), 1)
        self.assertEqual(decision_rows[0]["action"], "create")
        self.assertEqual(decision_rows[0]["scope"], "project")

    def test_llm_action_rows_include_model_provenance(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        actions = {
            "learnings": [
                {
                    "action": "create",
                    "text": "MKG stores model provenance.",
                    "confidence": 0.8,
                }
            ],
            "decisions": [
                {
                    "action": "create",
                    "text": "Store extraction model on produced memory.",
                    "confidence": 0.9,
                }
            ],
        }

        learning_rows, decision_rows = process_project._memory_rows_from_actions(
            project,
            "turn",
            actions,
            llm_model="anthropic/claude-haiku-4-5",
        )

        self.assertEqual(learning_rows[0]["llm_model"], "anthropic/claude-haiku-4-5")
        self.assertEqual(decision_rows[0]["llm_model"], "anthropic/claude-haiku-4-5")

    def test_processing_events_default_to_claude_model_for_claude_client(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_MODEL": "",
                "MKG_DEFAULT_LLM_MODEL": "",
                "CLAUDE_PROJECT_DIR": "",
            },
            clear=False,
        ):
            model = process_project._llm_model_for_processing_events(
                [
                    {"event_name": "UserPromptSubmit", "client": "claude_code"},
                    {"event_name": "Stop", "client": "claude_code"},
                ]
            )

        self.assertEqual(model, "anthropic/claude-haiku-4-5")

    def test_write_processing_stores_llm_model_provenance(self) -> None:
        queries: list[str] = []
        params: list[dict] = []

        class FakeTx:
            def run(self, query: str, **kwargs):
                queries.append(query)
                params.append(kwargs)

        project = project_common.ProjectRef(id="mkg", name="MKG")
        model = "anthropic/claude-haiku-4-5"
        learning_rows = [
            {
                "id": "learning:mkg:a",
                "action": "create",
                "text": "MKG stores model provenance.",
                "task_pattern": "model provenance",
                "confidence": 0.8,
                "status": "candidate",
                "scope": "project",
                "source": "project_turn_llm",
                "summary": "MKG stores model provenance.",
                "reason": "Reusable implementation fact.",
                "llm_model": model,
            }
        ]
        decision_rows = [
            {
                "id": "decision:mkg:a",
                "action": "create",
                "text": "Store extraction model on produced memory.",
                "rationale": "It makes processing provenance queryable.",
                "task_pattern": "model provenance",
                "confidence": 0.9,
                "scope": "project",
                "source": "project_turn_llm",
                "summary": "Store extraction model on produced memory.",
                "related_learning_id": None,
                "reason": "Implementation decision.",
                "llm_model": model,
            }
        ]

        process_project._write_processing(
            FakeTx(),
            project,
            "session-1",
            "turn",
            [{"event_id": "event-1"}],
            learning_rows,
            decision_rows,
            "default",
            3,
            model,
            "called",
            None,
            None,
            "2026-06-17T12:00:00+00:00",
        )

        joined = "\n".join(queries)
        self.assertIn("pp.llm_model = $llm_model", queries[0])
        self.assertIn("pp.llm_status = $llm_status", queries[0])
        self.assertEqual(params[0]["llm_model"], model)
        self.assertEqual(params[0]["llm_status"], "called")
        self.assertIn("USED_MEMORY_EXTRACTION_PROMPT", queries[1])
        self.assertIn("mep.last_used_model", queries[1])
        self.assertIn("r.llm_status = $llm_status", queries[1])
        self.assertIn("l.created_by_model = row.llm_model", joined)
        self.assertIn("l.last_llm_model", joined)
        self.assertIn("d.created_by_model = row.llm_model", joined)
        self.assertIn("d.last_llm_model", joined)
        self.assertIn("produced.llm_model = row.llm_model", joined)

    def test_write_processing_stores_llm_skip_reason(self) -> None:
        queries: list[str] = []
        params: list[dict] = []

        class FakeTx:
            def run(self, query: str, **kwargs):
                queries.append(query)
                params.append(kwargs)

        process_project._write_processing(
            FakeTx(),
            project_common.ProjectRef(id="mkg", name="MKG"),
            "session-1",
            "turn",
            [{"event_id": "event-1"}],
            [],
            [],
            "default",
            3,
            "anthropic/claude-haiku-4-5",
            "skipped",
            "Claude OAuth token unavailable",
            None,
            "2026-06-17T12:00:00+00:00",
        )

        self.assertEqual(params[0]["llm_status"], "skipped")
        self.assertEqual(params[0]["llm_skip_reason"], "Claude OAuth token unavailable")
        self.assertEqual(params[1]["llm_status"], "skipped")
        self.assertEqual(params[1]["llm_skip_reason"], "Claude OAuth token unavailable")

    def test_user_scoped_learning_is_namespaced_above_the_project(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        actions = {
            "learnings": [
                {
                    "action": "create",
                    "scope": "user",
                    "text": "Prefers terse, data-grounded answers.",
                    "confidence": 0.9,
                },
                {
                    "action": "create",
                    "scope": "bogus-scope",
                    "text": "A plain project fact.",
                    "confidence": 0.7,
                },
            ],
            "decisions": [],
        }

        learning_rows, _ = process_project._memory_rows_from_actions(
            project, "turn", actions
        )

        by_scope = {row["scope"]: row for row in learning_rows}
        self.assertEqual(set(by_scope), {"user", "project"})
        # User facts collapse onto a project-independent namespace so the same
        # fact dedupes across every project the user touches.
        self.assertTrue(by_scope["user"]["id"].startswith("learning:user:"))
        self.assertTrue(by_scope["project"]["id"].startswith("learning:mkg:"))

    def test_user_scoped_decision_is_namespaced_above_the_project(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        actions = {
            "learnings": [],
            "decisions": [
                {
                    "action": "create",
                    "scope": "user",
                    "text": "Use terse status updates across projects.",
                    "rationale": "The user made this a durable working agreement.",
                    "confidence": 0.9,
                },
                {
                    "action": "create",
                    "scope": "bogus-scope",
                    "text": "Keep hook injection project-specific.",
                    "rationale": "This is a project implementation policy.",
                    "confidence": 0.8,
                },
            ],
        }

        _, decision_rows = process_project._memory_rows_from_actions(
            project, "turn", actions
        )

        by_scope = {row["scope"]: row for row in decision_rows}
        self.assertEqual(set(by_scope), {"user", "project"})
        self.assertTrue(by_scope["user"]["id"].startswith("decision:user:"))
        self.assertTrue(by_scope["project"]["id"].startswith("decision:mkg:"))

    def test_learning_context_marks_candidates_as_hints(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        context = project_common.format_learning_context(
            project,
            [
                {
                    "text": "Keep project learning simple.",
                    "status": "candidate",
                    "confidence": 0.7,
                }
            ],
        )

        self.assertIn("Project context for MKG", context)
        self.assertIn("Keep project learning simple", context)
        self.assertIn("candidate learnings as hints", context)

    def test_learning_context_includes_decisions(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        context = project_common.format_learning_context(
            project,
            [],
            [
                {
                    "text": "Use repo folder name for project id.",
                    "confidence": 0.9,
                }
            ],
        )

        self.assertIn("Relevant project decisions", context)
        self.assertIn("Use repo folder name", context)

    def test_learning_context_includes_user_facts(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        context = project_common.format_learning_context(
            project,
            [],
            None,
            [
                {
                    "text": "Prefers terse, data-grounded answers.",
                    "status": "approved",
                    "confidence": 0.9,
                }
            ],
        )

        self.assertIn("What we know about the user", context)
        self.assertIn("Prefers terse", context)
        self.assertIn("user facts and user-scoped decisions", context)

    def test_learning_context_includes_user_decisions(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        context = project_common.format_learning_context(
            project,
            [],
            None,
            None,
            [
                {
                    "text": "Use user-scoped memory only at SessionStart.",
                    "confidence": 0.8,
                }
            ],
        )

        self.assertIn("User-scoped decisions", context)
        self.assertIn("SessionStart", context)

    def test_fetch_learnings_excludes_injected_and_in_session_memory(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                return []

        project_common.fetch_project_learnings(
            FakeDriver(),
            "neo4j",
            project_id="mkg",
            query=None,
            exclude_session_id="session-1",
        )
        project_common.fetch_project_decisions(
            FakeDriver(),
            "neo4j",
            project_id="mkg",
            query=None,
            exclude_session_id="session-1",
        )

        for query, params in captured:
            # Skip memory already shown in this session AND memory first produced
            # in it: re-surfacing it would only echo the live conversation.
            self.assertIn("INJECTED_IN", query)
            self.assertIn("FROM_SESSION", query)
            self.assertEqual(params["session_id"], "session-1")

    def test_fetch_project_learnings_restricts_to_project_scope(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                return []

        project_common.fetch_project_learnings(
            FakeDriver(), "neo4j", project_id="mkg", query=None
        )

        self.assertTrue(captured)
        self.assertIn("coalesce(l.scope, 'project') = 'project'", captured[0][0])

    def test_fetch_project_decisions_restricts_to_project_scope(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                return []

        project_common.fetch_project_decisions(
            FakeDriver(), "neo4j", project_id="mkg", query=None
        )

        self.assertTrue(captured)
        self.assertIn("coalesce(d.scope, 'project') = 'project'", captured[0][0])

    def test_fetch_user_learnings_spans_projects_and_filters_scope(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                return []

        project_common.fetch_user_learnings(
            FakeDriver(), "neo4j", query=None, exclude_session_id="session-1"
        )

        self.assertTrue(captured)
        query, params = captured[0]
        self.assertIn("(l:Learning {scope: 'user'})", query)
        self.assertNotIn("HAS_LEARNING", query)
        self.assertIn("FROM_SESSION", query)
        self.assertEqual(params["session_id"], "session-1")

    def test_fetch_user_learnings_excludes_consolidated_prompt_facts(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                if "fulltext.queryNodes" in query:
                    return [
                        {
                            "id": "learning:user:a",
                            "text": "The user's name is Tomaz.",
                            "status": "candidate",
                            "confidence": 1.0,
                            "task_pattern": "user profile",
                            "score": 1.0,
                        }
                    ]
                return []

        project_common.fetch_user_learnings(FakeDriver(), "neo4j", query="Tomaz")
        project_common.fetch_user_learnings(FakeDriver(), "neo4j", query=None)

        self.assertEqual(len(captured), 2)
        for query, _ in captured:
            self.assertIn("consolidated_at IS NULL", query)
            self.assertIn("coalesce(", query)
            self.assertIn("> ", query)
            self.assertIn("consolidated_at", query)

    def test_fetch_user_decisions_spans_projects_and_filters_scope(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                return []

        project_common.fetch_user_decisions(
            FakeDriver(), "neo4j", query=None, exclude_session_id="session-1"
        )

        self.assertTrue(captured)
        query, params = captured[0]
        self.assertIn("(d:Decision {scope: 'user'})", query)
        self.assertNotIn("HAS_DECISION", query)
        self.assertIn("FROM_SESSION", query)
        self.assertEqual(params["session_id"], "session-1")

    def test_context_injection_scope_follows_hook_event(self) -> None:
        self.assertEqual(
            inject_project_context.context_scope_for_hook("SessionStart"), "user"
        )
        self.assertEqual(
            inject_project_context.context_scope_for_hook("UserPromptSubmit"),
            "project",
        )

    def test_mark_injected_in_session_links_memory_to_session(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                return []

        project_common.mark_injected_in_session(
            FakeDriver(),
            "neo4j",
            "session-1",
            ["learning:mkg:a"],
            ["decision:mkg:b"],
            "UserPromptSubmit",
            prompt="show renewal risk",
        )

        self.assertEqual(len(captured), 4)
        learning_query, learning_params = captured[0]
        decision_query, decision_params = captured[2]
        self.assertIn("MATCH (m:Learning {id: memory_id})", learning_query)
        self.assertIn("MERGE (m)-[r:INJECTED_IN]->(s)", learning_query)
        self.assertEqual(learning_params["ids"], ["learning:mkg:a"])
        self.assertIn("MATCH (m:Decision {id: memory_id})", decision_query)
        self.assertEqual(decision_params["ids"], ["decision:mkg:b"])

    def test_mark_injected_in_session_links_memory_to_hook_event(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                return []

        project_common.mark_injected_in_session(
            FakeDriver(),
            "neo4j",
            "session-1",
            ["learning:mkg:a"],
            [],
            "SessionStart",
            source="startup",
        )

        self.assertEqual(len(captured), 2)
        event_query, event_params = captured[1]
        self.assertIn("(e:SessionEvent {event_name: $hook_event})", event_query)
        self.assertIn("MERGE (m)-[r:INJECTED_AT]->(e)", event_query)
        self.assertIn("$prompt IS NULL OR e.prompt = $prompt", event_query)
        self.assertIn("$source IS NULL OR e.source = $source", event_query)
        self.assertEqual(event_params["hook_event"], "SessionStart")
        self.assertEqual(event_params["source"], "startup")
        self.assertIsNone(event_params["prompt"])
        self.assertIn("since", event_params)

    def test_log_event_backfills_injected_at_for_context_events(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeTx:
            def run(self, query: str, **params):
                captured.append((query, params))

        log_event._link_injected_memory(
            FakeTx(), "session-1", "event-1", "UserPromptSubmit"
        )

        self.assertEqual(
            log_event.INJECTION_CONTEXT_EVENTS, {"SessionStart", "UserPromptSubmit"}
        )
        query, params = captured[0]
        self.assertIn("(m:Learning OR m:Decision)", query)
        self.assertIn("inj.hook_event = $event_name", query)
        self.assertIn("MERGE (m)-[r:INJECTED_AT]->(e)", query)
        self.assertEqual(params["event_id"], "event-1")
        self.assertEqual(params["event_name"], "UserPromptSubmit")

    def test_mark_injected_in_session_skips_unknown_session(self) -> None:
        class ExplodingDriver:
            def execute_query(self, query: str, **params):
                raise AssertionError("should not write for unknown session")

        project_common.mark_injected_in_session(
            ExplodingDriver(), "neo4j", "unknown", ["learning:mkg:a"], [], "SessionStart"
        )
        project_common.mark_injected_in_session(
            ExplodingDriver(), "neo4j", None, ["learning:mkg:a"], [], "SessionStart"
        )
        project_common.mark_injected_in_session(
            ExplodingDriver(), "neo4j", "session-1", [], [], "SessionStart"
        )

    def test_mark_learnings_used_casts_iso_timestamp(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                return []

        project_common.mark_learnings_used(FakeDriver(), "neo4j", ["learning:mkg:a"])

        self.assertEqual(len(captured), 1)
        query, params = captured[0]
        self.assertIn("l.last_used_at = datetime($timestamp)", query)
        self.assertNotIn("last_used_at = datetime()", query)
        self.assertIn("T", params["timestamp"])
        self.assertEqual(params["learning_ids"], ["learning:mkg:a"])

    def test_background_processor_is_fire_and_forget(self) -> None:
        with patch.object(process_project.subprocess, "Popen") as popen:
            process_project._spawn_background("turn", 200, "session-1")

        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertIn("--session-id", command)
        self.assertIn("session-1", command)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_codex_stop_hook_logs_then_processes_project(self) -> None:
        config = json.loads((ROOT / ".codex" / "hooks.json").read_text())
        stop_hooks = config["hooks"]["Stop"][0]["hooks"]

        # The self-rewriting prompt-rebuild Stop hooks are gone; logging, memory
        # extraction, and the rate-limited prompt-consolidation service remain.
        self.assertEqual(len(stop_hooks), 3)
        self.assertIn("hooks/log_event.py", stop_hooks[0]["command"])
        self.assertIn("--client codex", stop_hooks[0]["command"])
        self.assertIn("hooks/process_project.py", stop_hooks[1]["command"])
        self.assertIn("--mode turn --background", stop_hooks[1]["command"])
        self.assertIn("hooks/consolidate_system_prompt.py", stop_hooks[2]["command"])
        self.assertIn("--background", stop_hooks[2]["command"])
        joined = "\n".join(hook["command"] for hook in stop_hooks)
        self.assertNotIn("apply_system_prompt.py", joined)
        self.assertNotIn("apply_memory_extraction_prompt.py", joined)

    def test_codex_hooks_inject_project_context_for_supported_context_events(self) -> None:
        config = json.loads((ROOT / ".codex" / "hooks.json").read_text())

        session_start_hooks = config["hooks"]["SessionStart"][0]["hooks"]
        self.assertEqual(config["hooks"]["SessionStart"][0]["matcher"], "startup|resume|clear|compact")
        self.assertTrue(
            any("hooks/inject_system_prompt.py" in hook["command"] for hook in session_start_hooks)
        )
        self.assertTrue(
            any("hooks/inject_project_context.py" in hook["command"] for hook in session_start_hooks)
        )

        prompt_hooks = config["hooks"]["UserPromptSubmit"][0]["hooks"]
        self.assertIn("hooks/inject_project_context.py", prompt_hooks[0]["command"])
        self.assertIn("hooks/log_event.py", prompt_hooks[1]["command"])
        self.assertIn("--client codex", prompt_hooks[1]["command"])

    def test_codex_post_tool_use_captures_query_failures(self) -> None:
        config = json.loads((ROOT / ".codex" / "hooks.json").read_text())
        post_tool_groups = config["hooks"]["PostToolUse"]
        query_groups = [
            group
            for group in post_tool_groups
            if "bigquery_execute_query" in group.get("matcher", "")
            and "neo4j_read_cypher" in group.get("matcher", "")
        ]

        self.assertEqual(len(query_groups), 1)
        commands = [hook["command"] for hook in query_groups[0]["hooks"]]
        self.assertTrue(
            any("hooks/capture_query_failures.py" in command for command in commands)
        )

    def test_claude_post_tool_use_captures_query_failures(self) -> None:
        config = json.loads((ROOT / ".claude" / "settings.json").read_text())
        post_tool_groups = config["hooks"]["PostToolUse"]
        query_groups = [
            group
            for group in post_tool_groups
            if "bigquery_execute_query" in group.get("matcher", "")
            and "neo4j_read_cypher" in group.get("matcher", "")
        ]

        self.assertEqual(len(query_groups), 1)
        commands = [hook["command"] for hook in query_groups[0]["hooks"]]
        self.assertTrue(
            any("hooks/capture_query_failures.py" in command for command in commands)
        )

    def test_codex_logs_documented_lifecycle_events_without_session_end(self) -> None:
        config = json.loads((ROOT / ".codex" / "hooks.json").read_text())
        expected_logged_events = {
            "SessionStart",
            "SubagentStart",
            "PreToolUse",
            "PermissionRequest",
            "PostToolUse",
            "PreCompact",
            "PostCompact",
            "UserPromptSubmit",
            "SubagentStop",
            "Stop",
        }

        self.assertTrue(expected_logged_events.issubset(config["hooks"].keys()))
        self.assertNotIn("SessionEnd", config["hooks"])

        for event in expected_logged_events:
            commands = [
                hook["command"]
                for group in config["hooks"][event]
                for hook in group["hooks"]
            ]
            self.assertTrue(
                any("hooks/log_event.py" in command and "--client codex" in command for command in commands),
                event,
            )


if __name__ == "__main__":
    unittest.main()
