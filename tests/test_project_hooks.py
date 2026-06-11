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


project_common = load_hook_module("project_common")
process_project = load_hook_module("process_project")


class ProjectHookTests(unittest.TestCase):
    def test_project_id_uses_repo_folder_name(self) -> None:
        project = project_common.resolve_project({}, ROOT)

        self.assertIsNotNone(project)
        assert project is not None
        self.assertEqual(project.id, "meta-knowledge-graph")
        self.assertEqual(project.name, "Meta Knowledge Graph")

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
        self.assertIn("system_prompt_updates", prompt)
        self.assertIn("memory_extraction_prompt_updates", prompt)
        self.assertIn("rate-limited rebuild", prompt)
        self.assertIn("MemoryExtractionPrompt", prompt)
        self.assertIn("high-level stable information about the user", prompt)
        self.assertIn("broad interests", prompt)
        self.assertIn("communication/workflow preferences", prompt)
        self.assertIn("sensitive personal data", prompt)

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
            "system_prompt_updates": [
                {
                    "action": "suggest",
                    "prompt_name": "default",
                    "instruction": (
                        "When project context is missing, ask for project goals "
                        "and success criteria."
                    ),
                    "rationale": "This should apply across future MKG sessions.",
                    "confidence": 0.85,
                },
                {"action": "ignore", "reason": "Too specific."},
            ],
            "memory_extraction_prompt_updates": [
                {
                    "action": "suggest",
                    "prompt_name": "default",
                    "instruction": "Prefer update over create when similar memory already exists.",
                    "rationale": "This improves future duplicate handling.",
                    "confidence": 0.8,
                },
                {"action": "ignore", "reason": "Already covered."},
            ],
        }

        (
            learning_rows,
            decision_rows,
            system_prompt_rows,
            extraction_prompt_rows,
        ) = process_project._memory_rows_from_actions(project, "turn", actions)

        self.assertEqual(len(learning_rows), 1)
        self.assertEqual(learning_rows[0]["id"], "learning:mkg:existing")
        self.assertEqual(learning_rows[0]["action"], "update")
        self.assertEqual(len(decision_rows), 1)
        self.assertEqual(decision_rows[0]["action"], "create")
        self.assertEqual(len(system_prompt_rows), 1)
        self.assertEqual(system_prompt_rows[0]["prompt_name"], "default")
        self.assertIn("project goals", system_prompt_rows[0]["instruction"])
        self.assertEqual(len(extraction_prompt_rows), 1)
        self.assertEqual(extraction_prompt_rows[0]["prompt_name"], "default")
        self.assertIn("Prefer update", extraction_prompt_rows[0]["instruction"])

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

        self.assertEqual(len(stop_hooks), 4)
        self.assertIn("hooks/log_event.py --client codex", stop_hooks[0]["command"])
        self.assertIn("hooks/process_project.py --mode turn --background", stop_hooks[1]["command"])
        self.assertIn("hooks/apply_system_prompt.py --background", stop_hooks[2]["command"])
        self.assertIn(
            "hooks/apply_memory_extraction_prompt.py --background",
            stop_hooks[3]["command"],
        )


if __name__ == "__main__":
    unittest.main()
