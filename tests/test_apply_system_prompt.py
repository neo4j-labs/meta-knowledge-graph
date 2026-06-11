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


apply_system_prompt = load_hook_module("apply_system_prompt")


class ApplySystemPromptTests(unittest.TestCase):
    def test_fallback_rewrite_appends_learned_notes_section(self) -> None:
        content, applied, rejected = apply_system_prompt.fallback_rewrite(
            "You are the MKG agent.",
            [
                {"id": "s1", "instruction": "Wrap temporal values in toString()."},
                {"id": "s2", "instruction": "Check gds.* procedures exist before calling them."},
            ],
            max_chars=2000,
        )

        self.assertIn(apply_system_prompt.LEARNED_NOTES_HEADER, content)
        self.assertIn("- Wrap temporal values in toString().", content)
        self.assertIn("- Check gds.* procedures exist before calling them.", content)
        self.assertEqual(applied, ["s1", "s2"])
        self.assertEqual(rejected, [])

    def test_fallback_rewrite_marks_verbatim_duplicates_applied_without_growth(self) -> None:
        current = (
            "You are the MKG agent.\n\n"
            f"{apply_system_prompt.LEARNED_NOTES_HEADER}\n"
            "- Wrap temporal values in toString().\n"
        )

        content, applied, rejected = apply_system_prompt.fallback_rewrite(
            current,
            [{"id": "s1", "instruction": "Wrap temporal values in toString()."}],
            max_chars=2000,
        )

        self.assertEqual(applied, ["s1"])
        self.assertEqual(rejected, [])
        self.assertEqual(content.count("Wrap temporal values"), 1)

    def test_fallback_rewrite_leaves_over_budget_suggestions_as_candidates(self) -> None:
        current = "You are the MKG agent."

        content, applied, _ = apply_system_prompt.fallback_rewrite(
            current,
            [{"id": "s1", "instruction": "x" * 500}],
            max_chars=len(current) + 50,
        )

        self.assertEqual(applied, [])
        self.assertNotIn("xxx", content)

    def test_validate_llm_rewrite_filters_unknown_ids(self) -> None:
        validated = apply_system_prompt.validate_llm_rewrite(
            {
                "content": "rewritten prompt",
                "applied_ids": ["s1", "unknown"],
                "rejected": [
                    {"id": "s2", "reason": "duplicate"},
                    {"id": "s1", "reason": "already applied"},
                    {"id": "ghost", "reason": "not pending"},
                ],
            },
            pending_ids={"s1", "s2"},
            max_chars=2000,
        )

        self.assertIsNotNone(validated)
        assert validated is not None
        content, applied, rejected = validated
        self.assertEqual(content, "rewritten prompt")
        self.assertEqual(applied, ["s1"])
        self.assertEqual(rejected, [{"id": "s2", "reason": "duplicate"}])

    def test_validate_llm_rewrite_rejects_oversize_or_empty_content(self) -> None:
        self.assertIsNone(
            apply_system_prompt.validate_llm_rewrite(
                {"content": "x" * 100, "applied_ids": ["s1"]},
                pending_ids={"s1"},
                max_chars=50,
            )
        )
        self.assertIsNone(
            apply_system_prompt.validate_llm_rewrite(
                {"content": "   ", "applied_ids": ["s1"]},
                pending_ids={"s1"},
                max_chars=50,
            )
        )

    def test_rewrite_prompt_uses_fallback_without_api_key(self) -> None:
        with patch.dict(apply_system_prompt.os.environ, {}, clear=True):
            content, applied, rejected = apply_system_prompt.rewrite_prompt(
                "You are the MKG agent.",
                [{"id": "s1", "instruction": "Recall before asking."}],
                max_chars=2000,
            )

        self.assertIn("- Recall before asking.", content)
        self.assertEqual(applied, ["s1"])
        self.assertEqual(rejected, [])

    def test_rewrite_prompt_falls_back_when_llm_fails(self) -> None:
        with patch.dict(apply_system_prompt.os.environ, {"OPENAI_API_KEY": "test"}):
            with patch.object(
                apply_system_prompt,
                "ask_llm_for_rewrite",
                side_effect=RuntimeError("boom"),
            ):
                content, applied, _ = apply_system_prompt.rewrite_prompt(
                    "You are the MKG agent.",
                    [{"id": "s1", "instruction": "Recall before asking."}],
                    max_chars=2000,
                )

        self.assertIn("- Recall before asking.", content)
        self.assertEqual(applied, ["s1"])

    def test_build_rebuild_prompt_lists_suggestions_and_budget(self) -> None:
        prompt = apply_system_prompt.build_rebuild_prompt(
            "current prompt",
            [
                {
                    "id": "s1",
                    "instruction": "Recall before asking.",
                    "rationale": "saves a round trip",
                    "confidence": 0.8,
                    "support_count": 3,
                }
            ],
            max_chars=12000,
        )

        self.assertIn("id=s1", prompt)
        self.assertIn("Recall before asking.", prompt)
        self.assertIn("under 12000 characters", prompt)

    def test_background_rebuild_is_fire_and_forget(self) -> None:
        with patch.object(apply_system_prompt.subprocess, "Popen") as popen:
            apply_system_prompt._spawn_background("default")

        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertIn("--prompt-name", command)
        self.assertIn("default", command)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_claude_stop_hooks_include_prompt_rebuild(self) -> None:
        config = json.loads((ROOT / ".claude" / "settings.json").read_text())
        stop_commands = [
            hook["command"]
            for group in config["hooks"]["Stop"]
            for hook in group["hooks"]
        ]

        self.assertTrue(
            any("hooks/apply_system_prompt.py --background" in cmd for cmd in stop_commands)
        )


if __name__ == "__main__":
    unittest.main()
