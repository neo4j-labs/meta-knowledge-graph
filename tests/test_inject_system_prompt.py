from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "hooks" / "inject_system_prompt.py"
SPEC = importlib.util.spec_from_file_location("inject_system_prompt", MODULE_PATH)
inject_system_prompt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inject_system_prompt)


class InjectSystemPromptTests(unittest.TestCase):
    def test_fallback_prompt_bootstraps_missing_neo4j_context(self) -> None:
        prompt = inject_system_prompt.FALLBACK_BOOTSTRAP_PROMPT

        self.assertIn("Neo4j did not return", prompt)
        self.assertIn("Inspect the available ``metagraph-mcp`` MCP tools", prompt)
        self.assertIn("ask", prompt)
        self.assertIn("what project they are working on", prompt)
        self.assertIn("what goals or", prompt)
        self.assertIn("success criteria", prompt)
        self.assertIn("name and interests", prompt)

    def test_fallback_injection_log_is_concise(self) -> None:
        summary = inject_system_prompt.summarize_injection_content(
            "__missing__",
            inject_system_prompt.FALLBACK_BOOTSTRAP_PROMPT,
            "default",
        )

        self.assertLess(len(summary), 300)
        self.assertIn("fallback MKG bootstrap prompt", summary)
        self.assertIn("project", summary)
        self.assertIn("goals", summary)
        self.assertNotIn("You are the Intelligence Agent", summary)


if __name__ == "__main__":
    unittest.main()
