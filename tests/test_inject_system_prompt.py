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
        self.assertIn("the user's name", prompt)
        self.assertIn("what project they are working on", prompt)
        self.assertIn("what goals or", prompt)
        self.assertIn("success criteria", prompt)
        self.assertIn("persist a refined ``SystemPrompt`` back", prompt)

    def test_fallback_injection_log_is_concise(self) -> None:
        summary = inject_system_prompt.summarize_injection_content(
            "__missing__",
            inject_system_prompt.FALLBACK_BOOTSTRAP_PROMPT,
            "default",
        )

        self.assertLess(len(summary), 300)
        self.assertIn("default MKG SystemPrompt", summary)
        self.assertIn("project", summary)
        self.assertIn("goals", summary)
        self.assertNotIn("You are the Intelligence Agent", summary)

    def test_injection_log_stores_raw_content_and_separate_summary(self) -> None:
        captured: dict[str, object] = {}

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def run(self, query: str, **params):
                captured["query"] = query
                captured["params"] = params

        class FakeDriver:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def session(self, *, database: str):
                captured["database"] = database
                return FakeSession()

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
                "_neo4j_config",
                return_value=("bolt://example", "neo4j", "password", "neo4j"),
            ):
                inject_system_prompt.log_injection(
                    session_id="session-1",
                    hook_event="SessionStart",
                    target="additionalContext",
                    prompt_name="default",
                    content=content,
                    source="neo4j",
                )

        params = captured["params"]
        self.assertEqual(params["content"], content)
        self.assertNotEqual(params["content_summary"], content)
        self.assertEqual(params["char_count"], len(content))
        self.assertEqual(params["original_char_count"], len(content))
        self.assertEqual(params["stored_char_count"], len(content))
        self.assertEqual(params["summary_char_count"], len(params["content_summary"]))
        self.assertIn("content: $content", captured["query"])
        self.assertIn("content_summary: $content_summary", captured["query"])


if __name__ == "__main__":
    unittest.main()
