from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "hooks" / "inject_system_prompt.py"
SPEC = importlib.util.spec_from_file_location("inject_system_prompt", MODULE_PATH)
inject_system_prompt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inject_system_prompt)


class InjectSystemPromptTests(unittest.TestCase):
    def test_fallback_prompt_bootstraps_missing_neo4j_context(self) -> None:
        prompt = inject_system_prompt.FALLBACK_BOOTSTRAP_PROMPT

        self.assertIn("Session bootstrap", prompt)
        self.assertIn("Inspect the available ``meta-knowledge-graph`` MCP tools", prompt)
        self.assertIn("ask", prompt)
        self.assertIn("already know what", prompt)
        self.assertIn("help getting started", prompt)
        self.assertIn("the user's name", prompt)
        self.assertIn("what project they are working on", prompt)
        self.assertIn("what goals or", prompt)
        self.assertIn("success criteria", prompt)
        self.assertIn("user-scoped learnings", prompt)
        # The prompt no longer rewrites itself at runtime.
        self.assertNotIn("persist a refined ``SystemPrompt`` back", prompt)

    def test_fallback_injection_log_is_concise(self) -> None:
        summary = inject_system_prompt.summarize_injection_content(
            "__missing__",
            inject_system_prompt.FALLBACK_BOOTSTRAP_PROMPT,
            "default",
        )

        self.assertLess(len(summary), 300)
        self.assertIn("default MKG SystemPrompt", summary)
        self.assertIn("help starting", summary)
        self.assertIn("project", summary)
        self.assertIn("goals", summary)
        self.assertNotIn("You are the Intelligence Agent", summary)

    def _record_injection(self, *, created: bool, captured: dict) -> bool:
        class FakeDriver:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def execute_query(self, query: str, **params):
                captured["query"] = query
                captured["params"] = params
                return [{"created": created}]

        class FakeGraphDatabase:
            @staticmethod
            def driver(uri: str, *, auth: tuple[str, str]):
                captured["uri"] = uri
                captured["auth"] = auth
                return FakeDriver()

        fake_neo4j = types.ModuleType("neo4j")
        fake_neo4j.GraphDatabase = FakeGraphDatabase
        content = "You are the Intelligence Agent for the Meta Knowledge Graph."

        with patch.dict(sys.modules, {"neo4j": fake_neo4j}):
            with patch.object(
                inject_system_prompt,
                "neo4j_config",
                return_value=("bolt://example", "neo4j", "password", "neo4j"),
            ):
                return inject_system_prompt.record_injection(
                    session_id="session-1",
                    hook_event="SessionStart",
                    target="additionalContext",
                    prompt_name="default",
                    content=content,
                    source="neo4j",
                    user_id="tomaz@example.com",
                )

    def test_injection_links_prompt_instead_of_copying_content(self) -> None:
        captured: dict[str, object] = {}
        should_inject = self._record_injection(created=True, captured=captured)

        self.assertTrue(should_inject)
        params = captured["params"]
        query = captured["query"]
        content = "You are the Intelligence Agent for the Meta Knowledge Graph."
        self.assertNotIn("content", params)
        self.assertEqual(
            params["content_sha"], inject_system_prompt.content_sha(content)
        )
        self.assertEqual(params["database_"], "neo4j")
        self.assertNotEqual(params["content_summary"], content)
        self.assertEqual(params["char_count"], len(content))
        self.assertEqual(params["summary_char_count"], len(params["content_summary"]))
        self.assertNotIn("content: $content", query)
        self.assertIn("content_sha: $content_sha", query)
        self.assertIn("SystemPromptInjection", query)
        self.assertNotIn(":Injection", query)
        self.assertIn("OF_PROMPT", query)
        self.assertIn("SystemPrompt {name: $prompt_name}", query)

    def test_duplicate_injection_in_same_session_is_skipped(self) -> None:
        captured: dict[str, object] = {}
        should_inject = self._record_injection(created=False, captured=captured)

        self.assertFalse(should_inject)


class ComposePromptTests(unittest.TestCase):
    def test_profile_section_is_appended_to_frozen_base(self) -> None:
        base = "You are the agent.\n"
        composed = inject_system_prompt.compose_prompt(
            base, "- Prefers terse answers.\n- Works in Python."
        )

        # The base persona survives verbatim; the section is appended after it.
        self.assertTrue(composed.startswith("You are the agent."))
        self.assertIn(inject_system_prompt.USER_PROFILE_HEADER, composed)
        self.assertIn("- Prefers terse answers.", composed)
        self.assertLess(
            composed.index("You are the agent."),
            composed.index(inject_system_prompt.USER_PROFILE_HEADER),
        )
        self.assertIn("human-approved memory", composed)

    def test_missing_or_empty_profile_returns_base_unchanged(self) -> None:
        base = "You are the agent."
        self.assertEqual(inject_system_prompt.compose_prompt(base, None), base)
        self.assertEqual(inject_system_prompt.compose_prompt(base, "   "), base)

    def test_stale_profile_carries_caution_note(self) -> None:
        composed = inject_system_prompt.compose_prompt(
            "Base.", "- A bullet.", needs_revision=True
        )
        self.assertIn("retracted after consolidation", composed)
        fresh = inject_system_prompt.compose_prompt("Base.", "- A bullet.")
        self.assertNotIn("retracted after consolidation", fresh)


if __name__ == "__main__":
    unittest.main()
