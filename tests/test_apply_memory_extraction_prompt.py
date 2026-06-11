from __future__ import annotations

import importlib.util
import json
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


process_project = load_hook_module("process_project")
apply_memory_extraction_prompt = load_hook_module("apply_memory_extraction_prompt")


class ApplyMemoryExtractionPromptTests(unittest.TestCase):
    def test_fallback_rewrite_appends_learned_notes_section(self) -> None:
        content, applied, rejected = apply_memory_extraction_prompt.fallback_rewrite(
            process_project.DEFAULT_MEMORY_EXTRACTION_PROMPT,
            [
                {"id": "s1", "instruction": "Prefer updating similar memory over creating duplicates."},
                {"id": "s2", "instruction": "Reject prompt updates that store task context."},
            ],
            max_chars=20000,
        )

        self.assertIn(apply_memory_extraction_prompt.LEARNED_EXTRACTION_NOTES_HEADER, content)
        self.assertIn("- Prefer updating similar memory over creating duplicates.", content)
        self.assertIn("- Reject prompt updates that store task context.", content)
        self.assertEqual(applied, ["s1", "s2"])
        self.assertEqual(rejected, [])

    def test_validate_llm_rewrite_requires_runtime_tokens(self) -> None:
        valid = apply_memory_extraction_prompt.validate_llm_rewrite(
            {
                "content": process_project.DEFAULT_MEMORY_EXTRACTION_PROMPT,
                "applied_ids": ["s1"],
                "rejected": [],
            },
            pending_ids={"s1"},
            max_chars=20000,
        )

        self.assertIsNotNone(valid)
        invalid = apply_memory_extraction_prompt.validate_llm_rewrite(
            {
                "content": "No runtime placeholders here.",
                "applied_ids": ["s1"],
                "rejected": [],
            },
            pending_ids={"s1"},
            max_chars=20000,
        )

        self.assertIsNone(invalid)

    def test_rewrite_prompt_uses_fallback_without_api_key(self) -> None:
        with patch.dict(apply_memory_extraction_prompt.os.environ, {}, clear=True):
            content, applied, rejected = apply_memory_extraction_prompt.rewrite_prompt(
                process_project.DEFAULT_MEMORY_EXTRACTION_PROMPT,
                [{"id": "s1", "instruction": "Prefer update actions for duplicate memory."}],
                max_chars=20000,
            )

        self.assertIn("- Prefer update actions for duplicate memory.", content)
        self.assertEqual(applied, ["s1"])
        self.assertEqual(rejected, [])

    def test_build_rebuild_prompt_lists_required_tokens(self) -> None:
        prompt = apply_memory_extraction_prompt.build_rebuild_prompt(
            process_project.DEFAULT_MEMORY_EXTRACTION_PROMPT,
            [
                {
                    "id": "s1",
                    "instruction": "Preserve the JSON schema.",
                    "rationale": "future parses depend on it",
                    "confidence": 0.8,
                    "support_count": 2,
                }
            ],
            max_chars=12000,
        )

        self.assertIn("id=s1", prompt)
        self.assertIn("[[PROJECT_NAME]]", prompt)
        self.assertIn("under 12000 characters", prompt)

    def test_background_rebuild_is_fire_and_forget(self) -> None:
        with patch.object(apply_memory_extraction_prompt.subprocess, "Popen") as popen:
            apply_memory_extraction_prompt._spawn_background("default")

        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertIn("--prompt-name", command)
        self.assertIn("default", command)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_claude_stop_hooks_include_memory_extraction_rebuild(self) -> None:
        config = json.loads((ROOT / ".claude" / "settings.json").read_text())
        stop_commands = [
            hook["command"]
            for group in config["hooks"]["Stop"]
            for hook in group["hooks"]
        ]

        self.assertTrue(
            any("hooks/apply_memory_extraction_prompt.py --background" in cmd for cmd in stop_commands)
        )


if __name__ == "__main__":
    unittest.main()
