from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
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
enrich_events = load_hook_module("enrich_events")


class ProjectHookTests(unittest.TestCase):
    def test_project_id_uses_repo_folder_name(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MKG_PROJECT_ROOT": "",
                "MKG_PROJECT_DIR": "",
                "CLAUDE_PROJECT_DIR": "",
                "CODEX_WORKSPACE_ROOT": "",
                "CLAUDE_PLUGIN_ROOT": "",
            },
            clear=False,
        ):
            project = project_common.resolve_project({}, ROOT)

        self.assertIsNotNone(project)
        assert project is not None
        self.assertEqual(project.id, "meta-knowledge-graph")
        self.assertEqual(project.name, "Meta Knowledge Graph")

    def test_project_id_prefers_payload_cwd_over_hook_root(self) -> None:
        hook_root = Path("/Users/test/.claude/plugins/cache/mkg/meta-knowledge-graph/0.1.7")
        project = project_common.resolve_project(
            {"cwd": "/Users/test/work/customer-portal"},
            hook_root,
        )

        self.assertIsNotNone(project)
        assert project is not None
        self.assertEqual(project.id, "customer-portal")
        self.assertEqual(project.name, "Customer Portal")
        self.assertEqual(project.repo_root, "/Users/test/work/customer-portal")
        self.assertEqual(project.source, "payload.cwd")

    def test_project_id_uses_nearest_git_root_for_payload_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "billing-api"
            package_dir = repo / "src" / "billing"
            package_dir.mkdir(parents=True)
            (repo / ".git").mkdir()

            project = project_common.resolve_project({"cwd": str(package_dir)}, ROOT)

        self.assertIsNotNone(project)
        assert project is not None
        self.assertEqual(project.id, "billing-api")
        self.assertEqual(project.name, "Billing Api")
        self.assertEqual(project.repo_root, str(repo))
        self.assertEqual(project.source, "payload.cwd")

    def test_project_id_uses_claude_project_dir_for_background_plugin_hook(self) -> None:
        hook_root = Path("/Users/test/.claude/plugins/cache/mkg/meta-knowledge-graph/0.1.7")
        with patch.dict(
            os.environ,
            {
                "CLAUDE_PLUGIN_ROOT": str(hook_root),
                "CLAUDE_PROJECT_DIR": "/Users/test/work/claims-service",
                "MKG_PROJECT_ROOT": "",
                "MKG_PROJECT_DIR": "",
            },
            clear=False,
        ):
            project = project_common.resolve_project({"session_id": "session-1"}, hook_root)

        self.assertIsNotNone(project)
        assert project is not None
        self.assertEqual(project.id, "claims-service")
        self.assertEqual(project.repo_root, "/Users/test/work/claims-service")
        self.assertEqual(project.source, "env.CLAUDE_PROJECT_DIR")

    def test_project_id_skips_installed_plugin_root_candidate(self) -> None:
        hook_root = Path("/Users/test/.claude/plugins/cache/mkg/meta-knowledge-graph/0.1.7")
        with patch.dict(
            os.environ,
            {
                "CLAUDE_PLUGIN_ROOT": str(hook_root),
                "CLAUDE_PROJECT_DIR": "/Users/test/work/claims-service",
                "MKG_PROJECT_ROOT": "",
                "MKG_PROJECT_DIR": "",
            },
            clear=False,
        ):
            project = project_common.resolve_project({"cwd": str(hook_root)}, hook_root)

        self.assertIsNotNone(project)
        assert project is not None
        self.assertEqual(project.id, "claims-service")
        self.assertEqual(project.repo_root, "/Users/test/work/claims-service")
        self.assertEqual(project.source, "env.CLAUDE_PROJECT_DIR")

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

    def test_litellm_claude_oauth_still_used_with_base_url_only(self) -> None:
        credential = project_common.ClaudeOAuthCredential(
            token="sk-ant-oat01-test",
            source="test",
        )
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": "",
                "ANTHROPIC_BASE_URL": "https://gateway.example",
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

    def test_agent_context_marks_main_agent_from_parent_transcript(self) -> None:
        session_id = "019ed509-16e5-7321-8d67-461f857b944a"
        props = project_common.agent_context_props(
            {
                "transcript_path": (
                    "/Users/test/.codex/sessions/rollout-2026-06-17T12-03-23-"
                    f"{session_id}.jsonl"
                )
            },
            session_id,
            "PreToolUse",
        )

        self.assertEqual(props["agent_kind"], "main")
        self.assertFalse(props["is_subagent"])
        self.assertEqual(props["agent_id"], session_id)
        self.assertEqual(props["agent_transcript_id"], session_id)
        self.assertNotIn("parent_session_id", props)

    def test_agent_context_marks_subagent_from_child_transcript(self) -> None:
        parent_session_id = "019ed509-16e5-7321-8d67-461f857b944a"
        subagent_id = "019ed50c-cd1b-7480-a143-6a5ee9401ed5"
        props = project_common.agent_context_props(
            {
                "transcript_path": (
                    "/Users/test/.codex/sessions/rollout-2026-06-17T12-07-26-"
                    f"{subagent_id}.jsonl"
                )
            },
            parent_session_id,
            "PostToolUse",
        )

        self.assertEqual(props["agent_kind"], "subagent")
        self.assertTrue(props["is_subagent"])
        self.assertEqual(props["agent_id"], subagent_id)
        self.assertEqual(props["agent_transcript_id"], subagent_id)
        self.assertEqual(props["parent_session_id"], parent_session_id)

    def test_agent_context_marks_subagent_stop_from_event_name(self) -> None:
        parent_session_id = "019ed509-16e5-7321-8d67-461f857b944a"
        subagent_id = "019ed50c-cd1b-7480-a143-6a5ee9401ed5"
        props = project_common.agent_context_props(
            {"agent_path": subagent_id},
            parent_session_id,
            "SubagentStop",
        )

        self.assertEqual(props["agent_kind"], "subagent")
        self.assertTrue(props["is_subagent"])
        self.assertEqual(props["agent_id"], subagent_id)
        self.assertEqual(props["parent_session_id"], parent_session_id)

    def test_agent_context_marks_subagent_from_explicit_agent_id(self) -> None:
        # Claude Code internal subagent tool hook: parent session id + parent
        # transcript, subagent identified only by an explicit agent id field.
        parent_session_id = "3344fd58-f279-4e4c-9b82-de7616d7c3aa"
        subagent_id = "a72a9fb53032168f1"
        props = project_common.agent_context_props(
            {
                "agent_id": subagent_id,
                "transcript_path": (
                    "/Users/test/.claude/projects/mkg/"
                    f"{parent_session_id}.jsonl"
                ),
            },
            parent_session_id,
            "PreToolUse",
        )

        self.assertEqual(props["agent_kind"], "subagent")
        self.assertTrue(props["is_subagent"])
        self.assertEqual(props["agent_id"], subagent_id)
        self.assertEqual(props["parent_session_id"], parent_session_id)

    def test_agent_context_keeps_main_agent_when_agent_id_matches_session(self) -> None:
        # A main-agent tool hook may echo agent_id == session_id; it must stay
        # classified as the main agent.
        session_id = "3344fd58-f279-4e4c-9b82-de7616d7c3aa"
        props = project_common.agent_context_props(
            {
                "agent_id": session_id,
                "transcript_path": (
                    f"/Users/test/.claude/projects/mkg/{session_id}.jsonl"
                ),
            },
            session_id,
            "PreToolUse",
        )

        self.assertEqual(props["agent_kind"], "main")
        self.assertFalse(props["is_subagent"])
        self.assertEqual(props["agent_id"], session_id)

    def test_log_event_routes_subagent_lifecycle_to_parent_timeline(self) -> None:
        parent_session_id = "019ed509-16e5-7321-8d67-461f857b944a"
        subagent_id = "019ed50c-cd1b-7480-a143-6a5ee9401ed5"
        captured: list[tuple[str, dict]] = []

        class FakeTx:
            def run(self, query: str, **params):
                captured.append((query, params))

        log_event._append_event(
            FakeTx(),
            parent_session_id,
            "codex",
            {
                "event_id": "event-start",
                "event_name": "SubagentStart",
                "timestamp": "2026-06-17T10:07:34+00:00",
                "agent_kind": "subagent",
                "is_subagent": True,
                "agent_id": subagent_id,
                "agent_transcript_id": subagent_id,
                "agent_type": "general-purpose",
                "parent_session_id": parent_session_id,
            },
        )

        append_query, append_params = captured[0]
        linked_query, linked_params = captured[1]
        # SubagentStart is a marker on the parent timeline (trigger point), so it
        # is owned by the parent session, not the subagent.
        self.assertIn("MERGE (s:Session {session_id: $event_session_id})", append_query)
        self.assertEqual(append_params["event_session_id"], parent_session_id)
        # The parent Session node must not be mislabeled with the subagent's
        # actor identity.
        self.assertEqual(append_params["session_props"], {"client": "codex"})
        # ...but the subagent session it describes is still linked + marked.
        self.assertIn("MERGE (child:Session {session_id: $subagent_session_id})", linked_query)
        self.assertIn("MERGE (parent)-[has:HAS_SUBAGENT]->(child)", linked_query)
        self.assertIn("MERGE (child)-[of:SUBAGENT_OF]->(parent)", linked_query)
        self.assertIn("MERGE (child)-[:STARTED_AT]->(e)", linked_query)
        self.assertEqual(linked_params["parent_session_id"], parent_session_id)
        self.assertEqual(linked_params["subagent_session_id"], subagent_id)
        self.assertEqual(linked_params["agent_type"], "general-purpose")

    def test_log_event_routes_subagent_internal_event_to_subagent_session(self) -> None:
        # Claude Code delivers a subagent's internal tool hooks under the parent
        # session id, flagged only by an explicit agent id (is_subagent below is
        # what agent_context_props derives from that). Those events must be owned
        # by the subagent session, with HAS_SUBAGENT wired to the parent.
        parent_session_id = "019ed509-16e5-7321-8d67-461f857b944a"
        subagent_id = "a72a9fb53032168f1"
        captured: list[tuple[str, dict]] = []

        class FakeTx:
            def run(self, query: str, **params):
                captured.append((query, params))

        log_event._append_event(
            FakeTx(),
            parent_session_id,
            "claude_code",
            {
                "event_id": "event-pre",
                "event_name": "PreToolUse",
                "timestamp": "2026-06-17T10:07:40+00:00",
                "agent_kind": "subagent",
                "is_subagent": True,
                "agent_id": subagent_id,
                "tool_name": "Bash",
            },
        )

        append_query, append_params = captured[0]
        linked_query, linked_params = captured[1]
        self.assertEqual(append_params["event_session_id"], subagent_id)
        self.assertEqual(append_params["session_props"]["agent_kind"], "subagent")
        self.assertIn("MERGE (parent)-[has:HAS_SUBAGENT]->(child)", linked_query)
        # An internal event is not a lifecycle marker, so no STARTED_AT/ENDED_AT.
        self.assertEqual(linked_params["parent_session_id"], parent_session_id)
        self.assertEqual(linked_params["subagent_session_id"], subagent_id)

    def test_log_event_does_not_store_parent_transcript_on_subagent_session(self) -> None:
        parent_session_id = "019ed509-16e5-7321-8d67-461f857b944a"
        subagent_id = "019ed50c-cd1b-7480-a143-6a5ee9401ed5"

        session_props = log_event._session_props(
            subagent_id,
            parent_session_id,
            "codex",
            {
                "agent_kind": "subagent",
                "is_subagent": True,
                "agent_id": subagent_id,
                "agent_transcript_id": parent_session_id,
                "parent_session_id": parent_session_id,
            },
        )

        self.assertEqual(session_props["agent_id"], subagent_id)
        self.assertEqual(session_props["parent_session_id"], parent_session_id)
        self.assertNotIn("agent_transcript_id", session_props)

    def test_log_event_links_spawn_response_to_subagent_session(self) -> None:
        parent_session_id = "019ed509-16e5-7321-8d67-461f857b944a"
        subagent_id = "019ed50c-cd1b-7480-a143-6a5ee9401ed5"
        captured: list[tuple[str, dict]] = []

        class FakeTx:
            def run(self, query: str, **params):
                captured.append((query, params))

        self.assertEqual(
            log_event._spawned_subagent_id(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "spawn_agent",
                    "tool_response": {"agent_id": subagent_id, "nickname": "Hubble"},
                },
                "PostToolUse",
            ),
            subagent_id,
        )

        log_event._append_event(
            FakeTx(),
            parent_session_id,
            "codex",
            {
                "event_id": "event-spawn",
                "event_name": "PostToolUse",
                "timestamp": "2026-06-17T10:07:30+00:00",
                "agent_kind": "main",
                "is_subagent": False,
                "agent_id": parent_session_id,
                "spawned_subagent_id": subagent_id,
            },
        )

        trigger_query, trigger_params = captured[1]
        self.assertIn("MERGE (child:Session {session_id: $subagent_session_id})", trigger_query)
        self.assertIn("MERGE (child)-[:TRIGGERED_BY]->(e)", trigger_query)
        self.assertEqual(trigger_params["parent_session_id"], parent_session_id)
        self.assertEqual(trigger_params["subagent_session_id"], subagent_id)

    def test_event_enrichment_preserves_subagent_actor_fields(self) -> None:
        parent_session_id = "019ed509-16e5-7321-8d67-461f857b944a"
        subagent_id = "019ed50c-cd1b-7480-a143-6a5ee9401ed5"
        transcript_path = (
            "/Users/test/.codex/sessions/rollout-2026-06-17T12-07-26-"
            f"{subagent_id}.jsonl"
        )
        events = [
            {
                "event_id": "event-prompt",
                "event_name": "UserPromptSubmit",
                "timestamp": "2026-06-17T10:07:34Z",
                "turn_id": "turn-subagent",
                "prompt": "Run a diagnostic command.",
                "transcript_path": transcript_path,
            },
            {
                "event_id": "event-pre",
                "event_name": "PreToolUse",
                "timestamp": "2026-06-17T10:07:47Z",
                "turn_id": "turn-subagent",
                "tool_name": "Bash",
                "tool_use_id": "call-1",
                "tool_input": json.dumps({"command": "pwd"}),
                "transcript_path": transcript_path,
            },
            {
                "event_id": "event-post",
                "event_name": "PostToolUse",
                "timestamp": "2026-06-17T10:07:48Z",
                "turn_id": "turn-subagent",
                "tool_name": "Bash",
                "tool_use_id": "call-1",
                "tool_input": json.dumps({"command": "pwd"}),
                "tool_response": "/Users/test/project\n",
                "transcript_path": transcript_path,
            },
        ]

        projection = enrich_events.build_event_enrichment_projection(
            project_common.ProjectRef(id="mkg", name="MKG"),
            parent_session_id,
            "turn",
            events,
        )

        self.assertEqual(len(projection["turns"]), 1)
        self.assertEqual(len(projection["tool_calls"]), 1)
        turn = projection["turns"][0]
        call = projection["tool_calls"][0]
        for row in (turn, call):
            self.assertEqual(row["agent_kind"], "subagent")
            self.assertTrue(row["is_subagent"])
            self.assertEqual(row["agent_id"], subagent_id)
            self.assertEqual(row["agent_transcript_id"], subagent_id)
            self.assertEqual(row["parent_session_id"], parent_session_id)

    def test_parent_session_processors_include_subagent_sessions(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeResult:
            records: list[dict] = []

        class FakeDriver:
            def execute_query(self, query: str, **params):
                captured.append((query, params))
                return FakeResult()

        process_project._fetch_unprocessed_events(
            FakeDriver(), "neo4j", "mkg", "parent-session", "turn", 50
        )
        enrich_events._fetch_unprocessed_events(
            FakeDriver(), "neo4j", "mkg", "parent-session", "turn", 50
        )

        for query, params in captured:
            self.assertIn("OPTIONAL MATCH (s)-[:HAS_SUBAGENT*1..]->(sub:Session)", query)
            self.assertIn("MATCH (scoped_session)-[:HAS_EVENT]->(e:SessionEvent)", query)
            self.assertEqual(params["session_id"], "parent-session")

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

    def test_llm_action_rows_use_unified_hook_source(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        actions = {
            "learnings": [{"action": "create", "text": "A durable fact.", "confidence": 0.8}],
            "decisions": [{"action": "create", "text": "A durable decision.", "confidence": 0.9}],
        }

        # The verbose source tag is uniform regardless of processing mode so
        # hook-written memory is always identifiable as `hooks-stop`.
        for mode in ("turn", "session"):
            learning_rows, decision_rows = process_project._memory_rows_from_actions(
                project, mode, actions
            )
            self.assertEqual(learning_rows[0]["source"], "hooks-stop")
            self.assertEqual(decision_rows[0]["source"], "hooks-stop")

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
                "source": "hooks-stop",
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
                "source": "hooks-stop",
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

    def test_background_processor_pins_resolved_project_env(self) -> None:
        project = project_common.ProjectRef(
            id="claims-service",
            name="Claims Service",
            repo_root="/Users/test/work/claims-service",
        )

        with patch.object(process_project.subprocess, "Popen") as popen:
            process_project._spawn_background("turn", 200, "session-1", project)

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["MKG_PROJECT_ID"], "claims-service")
        self.assertEqual(env["MKG_PROJECT_NAME"], "Claims Service")
        self.assertEqual(env["MKG_PROJECT_ROOT"], "/Users/test/work/claims-service")

    def test_project_merge_preserves_first_source_and_records_last_source(self) -> None:
        captured: list[tuple[str, dict]] = []

        class FakeTx:
            def run(self, query: str, **params):
                captured.append((query, params))

        project = project_common.ProjectRef(
            id="claims-service",
            name="Claims Service",
            repo_root="/Users/test/work/claims-service",
            source="env.MKG_PROJECT_ROOT",
        )

        project_common.merge_project_and_session(
            FakeTx(),
            project,
            "session-1",
            "2026-06-18T08:52:24+00:00",
        )

        query, params = captured[0]
        self.assertIn("p.source = coalesce(p.source, $project_source)", query)
        self.assertIn("p.last_source = $project_source", query)
        self.assertIn("SET p += $project_update_props", query)
        self.assertEqual(params["project_source"], "env.MKG_PROJECT_ROOT")
        self.assertIn("source", params["project_props"])
        self.assertNotIn("source", params["project_update_props"])

    def test_start_end_only_events_are_not_project_work(self) -> None:
        self.assertFalse(
            project_common.has_project_work_events(
                [
                    {"event_name": "SessionStart"},
                    {"event_name": "SessionEnd"},
                ]
            )
        )
        self.assertTrue(
            project_common.has_project_work_events(
                [
                    {"event_name": "SessionStart"},
                    {"event_name": "UserPromptSubmit"},
                ]
            )
        )

    def test_log_event_keeps_lifecycle_events_project_neutral(self) -> None:
        project = project_common.ProjectRef(id="claims-service", name="Claims Service")

        self.assertIsNone(log_event._event_project_id(project, "SessionStart"))
        self.assertIsNone(log_event._event_project_id(project, "SessionEnd"))
        self.assertEqual(
            log_event._event_project_id(project, "UserPromptSubmit"),
            "claims-service",
        )
        self.assertIsNone(log_event._event_project_id(None, "UserPromptSubmit"))

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

    def test_codex_plugin_package_points_at_child_payload(self) -> None:
        manifest = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text())
        package_manifest = json.loads((ROOT / "plugin" / ".codex-plugin" / "plugin.json").read_text())
        plugin_config = json.loads((ROOT / "plugin" / "hooks" / "codex-hooks.json").read_text())
        checkout_config = json.loads((ROOT / ".codex" / "hooks.json").read_text())

        self.assertEqual(manifest["skills"], "./plugin/codex-skills/")
        self.assertEqual(manifest["mcpServers"], "./plugin/.mcp.json")
        self.assertEqual(manifest["hooks"], "./plugin/hooks/codex-hooks.json")
        self.assertEqual(package_manifest["skills"], "./codex-skills/")
        self.assertEqual(package_manifest["mcpServers"], "./.mcp.json")
        self.assertEqual(package_manifest["hooks"], "./hooks/codex-hooks.json")
        self.assertEqual(set(plugin_config["hooks"].keys()), set(checkout_config["hooks"].keys()))
        self.assertIn("SessionStart", plugin_config["hooks"])
        self.assertIn("Stop", plugin_config["hooks"])
        self.assertNotIn("SessionEnd", plugin_config["hooks"])

    def test_codex_plugin_hooks_resolve_from_installed_plugin_root(self) -> None:
        config = json.loads((ROOT / "plugin" / "hooks" / "codex-hooks.json").read_text())
        commands = [
            hook["command"]
            for groups in config["hooks"].values()
            for group in groups
            for hook in group["hooks"]
        ]

        self.assertTrue(commands)
        for command in commands:
            self.assertIn("CODEX_PLUGIN_ROOT", command)
            self.assertIn("plugins/cache", command)
            self.assertIn("MKG_HOOK_ROOT", command)
            self.assertIn("uv run --script", command)
            self.assertIn("installed Codex plugin cache", command)
            self.assertNotIn("git rev-parse", command)
            self.assertNotIn("CODEX_PLUGIN_ROOT:-$PWD", command)
            self.assertNotIn("MKG_PLUGIN_ROOT", command)
            self.assertNotIn("CLAUDE_PLUGIN_ROOT", command)
            self.assertNotIn("$PWD/hooks", command)
            self.assertNotIn("--project", command)

        stop_commands = [
            hook["command"] for hook in config["hooks"]["Stop"][0]["hooks"]
        ]
        self.assertIn("log_event.py", stop_commands[0])
        self.assertIn("--client codex", stop_commands[0])
        self.assertIn("process_project.py", stop_commands[1])
        self.assertIn("--mode turn --background", stop_commands[1])
        self.assertIn("consolidate_system_prompt.py", stop_commands[2])
        self.assertIn("--background", stop_commands[2])

    def test_codex_checkout_hooks_resolve_from_repo_root(self) -> None:
        config = json.loads((ROOT / ".codex" / "hooks.json").read_text())
        commands = [
            hook["command"]
            for groups in config["hooks"].values()
            for group in groups
            for hook in group["hooks"]
        ]

        self.assertTrue(commands)
        for command in commands:
            self.assertIn("git rev-parse --show-toplevel", command)
            self.assertIn("MKG_HOOK_ROOT", command)
            self.assertIn("uv run --script", command)
            self.assertNotIn("CODEX_PLUGIN_ROOT", command)
            self.assertNotIn("--project", command)
            self.assertNotIn("--no-sync", command)

    def test_claude_session_end_does_not_run_memory_extraction(self) -> None:
        for hooks_path in (
            ROOT / ".claude" / "settings.json",
            ROOT / "hooks" / "hooks.json",
            ROOT / "plugin" / "hooks" / "hooks.json",
        ):
            with self.subTest(hooks_path=hooks_path):
                config = json.loads(hooks_path.read_text())
                commands = [
                    hook["command"]
                    for group in config["hooks"]["SessionEnd"]
                    for hook in group["hooks"]
                ]

                self.assertTrue(any("hooks/log_event.py" in command for command in commands))
                self.assertTrue(
                    any("hooks/consolidate_system_prompt.py" in command for command in commands)
                )
                self.assertFalse(
                    any("hooks/process_project.py" in command for command in commands)
                )


if __name__ == "__main__":
    unittest.main()
