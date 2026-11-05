"""Background agents, restarts, and the newer Claude Code hook events.

Subagents run in the background by default, so a subagent's report reaches
the session as a prompt the harness submits; a main-agent Stop can be a pause
for agents still running; a restarted background session repeats its resume
SessionStart word for word. These tests pin how capture records each of those,
how extraction reads them, and which newer events are wired.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HOOKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


project_common = load_module("project_common")
log_event = load_module("log_event")
process_project = load_module("process_project")

USER = project_common.UserRef("tomaz@example.com", "env")
NEWER_EVENTS = (
    "InstructionsLoaded",
    "UserPromptExpansion",
    "PermissionDenied",
    "StopFailure",
    "TaskCreated",
    "TaskCompleted",
    "PostCompact",
    "PostModelSwitch",
)


def capture(payload: dict, client: str = "claude_code", env: dict | None = None):
    """Run log_event on ``payload`` against a fake driver and return the
    ``(event_props, job_id)`` it handed to ``_append_event``."""
    writes: list = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute_write(self, fn, *args):
            writes.append((fn, args))

    class FakeDriver:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def session(self, database=None):
            return FakeSession()

    fake_neo4j = types.ModuleType("neo4j")
    fake_neo4j.GraphDatabase = types.SimpleNamespace(driver=lambda uri, auth: FakeDriver())
    clean_env = {k: v for k, v in os.environ.items() if k not in ("CLAUDE_PID", "CLAUDE_JOB_DIR")}
    clean_env.update(env or {})
    with patch.dict(sys.modules, {"neo4j": fake_neo4j}), patch.dict(
        os.environ, clean_env, clear=True
    ), patch.object(log_event, "resolve_user", return_value=USER), patch.object(
        log_event, "resolve_project", return_value=None
    ), patch.object(log_event, "_read_transcript", return_value="SNAPSHOT"):
        log_event.log_event(payload, client=client)
    args = next(args for fn, args in writes if fn is log_event._append_event)
    return args[2], args[3]


class _RecordingTx:
    def __init__(self) -> None:
        self.calls: list = []

    def run(self, query, **params):
        self.calls.append((query, params))


class PromptOriginTests(unittest.TestCase):
    def test_harness_wrappers_are_told_from_typed_text(self) -> None:
        cases = {
            '<agent-message from="reviewer">done</agent-message>': "agent-message",
            "  \n<task-notification>\n<task-id>x</task-id>": "task-notification",
            '<teammate-message teammate_id="a">hi': "teammate-message",
            "<cross-session-message>ping": "cross-session-message",
            '<channel source="telegram">hello': "channel",
            "fix the bug in <agent-message>": "user",
            "<div>not a harness tag</div>": "user",
            "/loop 5m check the deploy": "user",
            "": "user",
            None: "user",
        }
        for prompt, origin in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(project_common.prompt_origin(prompt), origin)


class CaptureTests(unittest.TestCase):
    def test_submitted_prompt_records_its_origin(self) -> None:
        props, _ = capture(
            {
                "session_id": "s1",
                "hook_event_name": "UserPromptSubmit",
                "prompt": '<agent-message from="tests">all green</agent-message>',
                "permission_mode": "auto",
                "effort": {"level": "high"},
            }
        )
        self.assertEqual(props["prompt_origin"], "agent-message")
        self.assertEqual(props["permission_mode"], "auto")
        self.assertEqual(props["effort_level"], "high")
        typed, _ = capture(
            {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "hi"}
        )
        self.assertEqual(typed["prompt_origin"], "user")

    def test_main_stop_records_background_work_and_subagent_stop_does_not(self) -> None:
        tasks = [
            {
                "type": "local_agent",
                "status": "running",
                "agent_type": "Explore",
                "description": "map the\nhooks",
            },
            {"type": "local_bash", "status": "running", "command": "pytest -x"},
        ] + [{"type": "workflow", "status": "running", "name": f"w{i}"} for i in range(25)]
        crons = [{"schedule": "*/5 * * * *", "recurring": True, "prompt": "check CI"}]
        props, _ = capture(
            {
                "session_id": "s1",
                "hook_event_name": "Stop",
                "last_assistant_message": "Waiting on the agents.",
                "background_tasks": tasks,
                "session_crons": crons,
            }
        )
        self.assertEqual(
            props["background_tasks"][0], "local_agent (running): Explore · map the hooks"
        )
        self.assertEqual(props["background_tasks"][1], "local_bash (running): pytest -x")
        self.assertEqual(len(props["background_tasks"]), log_event.MAX_LISTED + 1)
        self.assertEqual(props["background_tasks"][-1], "... and 7 more")
        self.assertEqual(props["session_crons"], ["*/5 * * * * (recurring): check CI"])

        sub, _ = capture(
            {
                "session_id": "s1",
                "hook_event_name": "SubagentStop",
                "agent_id": "a1",
                "agent_type": "Explore",
                "agent_transcript_path": "/tmp/a1.jsonl",
                "background_tasks": tasks,
            }
        )
        self.assertNotIn("background_tasks", sub)
        self.assertEqual(sub["agent_transcript_path"], "/tmp/a1.jsonl")

    def test_stop_failure_keeps_the_api_error_apart_from_tool_errors(self) -> None:
        props, _ = capture(
            {
                "session_id": "s1",
                "hook_event_name": "StopFailure",
                "error": "rate_limit",
                "error_details": "x" * 5000,
            }
        )
        self.assertEqual(props["error"], "rate_limit")
        self.assertNotIn("tool_error", props)
        self.assertLess(len(props["error_details"]), 1100)
        # The transcript snapshot stays: StopFailure is read by extraction.
        self.assertEqual(props["transcript"], "SNAPSHOT")

        failed, _ = capture(
            {
                "session_id": "s1",
                "hook_event_name": "PostToolUseFailure",
                "tool_name": "Bash",
                "tool_input": {"command": "false"},
                "error": "exit 1",
            }
        )
        self.assertIs(failed["tool_error"], True)
        self.assertNotIn("error", failed)

    def test_newer_events_keep_their_fields_without_a_transcript_snapshot(self) -> None:
        cases = {
            "InstructionsLoaded": (
                {"file_path": "/r/CLAUDE.md", "memory_type": "Project", "load_reason": "session_start"},
                {"file_path": "/r/CLAUDE.md", "memory_type": "Project"},
            ),
            "UserPromptExpansion": (
                {"prompt": "/mkg-start", "expansion_type": "slash_command", "command_name": "mkg-start"},
                {"command_name": "mkg-start", "expansion_type": "slash_command"},
            ),
            "PermissionDenied": (
                {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}, "reason": "destructive"},
                {"tool_name": "Bash", "reason": "destructive"},
            ),
            "TaskCreated": (
                {"task_id": "t1", "task_subject": "Write tests", "teammate_name": "qa"},
                {"task_id": "t1", "task_subject": "Write tests", "teammate_name": "qa"},
            ),
            "PostCompact": (
                {"trigger": "auto", "compact_summary": "s" * 9000},
                {"trigger": "auto"},
            ),
            "PostModelSwitch": (
                {"from_model": "claude-opus-5-5", "to_model": "claude-sonnet-5-5"},
                {"from_model": "claude-opus-5-5", "to_model": "claude-sonnet-5-5"},
            ),
        }
        for event_name, (fields, expected) in cases.items():
            with self.subTest(event=event_name):
                props, _ = capture(
                    {"session_id": "s1", "hook_event_name": event_name, "transcript_path": "/t", **fields}
                )
                for key, value in expected.items():
                    self.assertEqual(props[key], value)
                self.assertNotIn("transcript", props)
        compact, _ = capture(
            {"session_id": "s1", "hook_event_name": "PostCompact", "compact_summary": "s" * 9000}
        )
        self.assertLess(len(compact["compact_summary"]), log_event.MAX_FIELD_CHARS + 100)

    def test_harness_pid_separates_restarts_but_not_parallel_hook_configs(self) -> None:
        resume = {"session_id": "s1", "hook_event_name": "SessionStart", "source": "resume"}
        first, _ = capture(resume, env={"CLAUDE_PID": "100"})
        same_process, _ = capture(resume, env={"CLAUDE_PID": "100"})
        restarted, _ = capture(resume, env={"CLAUDE_PID": "200"})
        self.assertEqual(first["event_id"], same_process["event_id"])
        self.assertNotEqual(first["event_id"], restarted["event_id"])
        self.assertEqual(first["harness_pid"], "100")
        # A Codex hook never reads Claude Code's variable.
        codex_a, _ = capture(resume, client="codex", env={"CLAUDE_PID": "100"})
        codex_b, _ = capture(resume, client="codex", env={"CLAUDE_PID": "200"})
        self.assertEqual(codex_a["event_id"], codex_b["event_id"])
        self.assertNotIn("harness_pid", codex_a)

    def test_background_session_records_its_job_on_its_own_session(self) -> None:
        payload = {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "go"}
        _, job_id = capture(payload, env={"CLAUDE_JOB_DIR": "/home/u/.claude/jobs/job-42/"})
        self.assertEqual(job_id, "job-42")
        _, none = capture(payload)
        self.assertIsNone(none)

        tx = _RecordingTx()
        log_event._append_event(
            tx, "s1", "claude_code", {"event_id": "e1", "timestamp": "2026-11-04T00:00:00+00:00"}, "job-42"
        )
        self.assertEqual(tx.calls[0][1]["session_props"]["job_id"], "job-42")
        # A subagent inside the job inherits CLAUDE_JOB_DIR but is not the job.
        tx = _RecordingTx()
        log_event._append_event(
            tx,
            "s1",
            "claude_code",
            {
                "event_id": "e2",
                "event_name": "PreToolUse",
                "timestamp": "2026-11-04T00:00:00+00:00",
                "agent_id": "a1",
                "is_subagent": True,
            },
            "job-42",
        )
        self.assertEqual(tx.calls[0][1]["event_session_id"], "a1")
        self.assertNotIn("job_id", tx.calls[0][1]["session_props"])

    def test_main_ignores_sigterm(self) -> None:
        with patch.object(log_event.signal, "signal") as set_handler, patch.object(
            log_event, "log_event"
        ), patch.object(log_event, "load_mkg_env"), patch.object(
            sys, "argv", ["log_event.py"]
        ), patch.object(sys, "stdin", io.StringIO("{}")), patch.dict(
            os.environ, {"MKG_IN_EXTRACTION": ""}
        ):
            self.assertEqual(log_event.main(), 0)
        set_handler.assert_any_call(log_event.signal.SIGTERM, log_event.signal.SIG_IGN)


class CorpusTests(unittest.TestCase):
    def test_harness_prompts_are_labeled_by_sender(self) -> None:
        events = [
            {
                "event_name": "UserPromptSubmit",
                "timestamp": "2026-11-04T10:00:00+00:00",
                "prompt": "run the reviewers in the background",
                "prompt_origin": "user",
            },
            {
                "event_name": "UserPromptSubmit",
                "timestamp": "2026-11-04T10:05:00+00:00",
                "prompt": '<agent-message from="reviewer">I prefer tabs</agent-message>',
                "prompt_origin": "agent-message",
            },
            # Captured before prompt_origin existed: the tag is read instead.
            {
                "event_name": "UserPromptSubmit",
                "timestamp": "2026-11-04T10:06:00+00:00",
                "prompt": "<task-notification>workflow finished</task-notification>",
            },
        ]
        corpus = process_project._event_corpus(events)
        self.assertIn("prompt: run the reviewers in the background", corpus)
        self.assertIn('subagent_report: <agent-message from="reviewer">I prefer tabs', corpus)
        self.assertIn("background_task_notification: <task-notification>workflow", corpus)
        self.assertNotIn("prompt: <agent-message", corpus)
        self.assertNotIn("prompt: <task-notification", corpus)

    def test_stop_lists_pending_background_work_and_stop_failure_says_why(self) -> None:
        events = [
            {
                "event_name": "Stop",
                "timestamp": "2026-11-04T10:00:00+00:00",
                "last_assistant_message": "Started three reviewers.",
                "background_tasks": ["local_agent (running): reviewer · check the diff"],
                "session_crons": ["*/5 * * * * (recurring): check CI"],
            },
            {
                "event_name": "StopFailure",
                "timestamp": "2026-11-04T10:01:00+00:00",
                "error": "overloaded",
                "error_details": "529 from the API",
            },
        ]
        corpus = process_project._event_corpus(events)
        self.assertIn(
            "background_work_at_turn_end: still running: local_agent (running): reviewer"
            " · check the diff | scheduled: */5 * * * * (recurring): check CI",
            corpus,
        )
        self.assertIn("turn_ended_by_api_error: overloaded: 529 from the API", corpus)

    def test_background_work_survives_the_budget_with_what_was_said(self) -> None:
        tools = [
            {
                "event_name": "PostToolUse",
                "timestamp": f"2026-11-04T10:{i:02d}:00+00:00",
                "tool_name": "Bash",
                "tool_input": "x" * 500,
            }
            for i in range(40)
        ]
        stop = {
            "event_name": "Stop",
            "timestamp": "2026-11-04T09:00:00+00:00",
            "background_tasks": ["workflow (running): review-changes"],
        }
        corpus = process_project._event_corpus([stop, *tools], limit=3000)
        self.assertIn("events elided", corpus)
        self.assertIn("background_work_at_turn_end: still running: workflow", corpus)

    def test_capture_only_events_stay_out_of_the_corpus(self) -> None:
        events = [
            {"event_name": "InstructionsLoaded", "file_path": "/r/CLAUDE.md"},
            {"event_name": "UserPromptExpansion", "prompt": "/mkg-start", "command_name": "mkg-start"},
            {"event_name": "UserPromptSubmit", "prompt": "/mkg-start"},
            {"event_name": "PermissionDenied", "tool_name": "Bash", "tool_input": "DENIED-INPUT"},
            {"event_name": "PostCompact", "compact_summary": "COMPACT-SUMMARY"},
            {"event_name": "PostModelSwitch", "to_model": "claude-sonnet-5-5"},
            {"event_name": "TaskCreated", "task_subject": "Write tests"},
        ]
        corpus = process_project._event_corpus(events)
        self.assertEqual(corpus.count("/mkg-start"), 1)
        for absent in ("InstructionsLoaded", "UserPromptExpansion", "DENIED-INPUT",
                       "PostCompact", "PostModelSwitch", "TaskCreated"):
            self.assertNotIn(absent, corpus)

    def test_both_prompts_explain_who_sent_what(self) -> None:
        memory = process_project.DEFAULT_MEMORY_EXTRACTION_PROMPT
        observation = process_project.DEFAULT_OBSERVATION_EXTRACTION_PROMPT
        for key in [*project_common.HARNESS_PROMPT_TAGS.values(),
                    "background_work_at_turn_end", "turn_ended_by_api_error"]:
            with self.subTest(key=key):
                self.assertIn(key, " ".join(memory.split()))
                self.assertIn(key, " ".join(observation.split()))
        self.assertIn("never take it as evidence of a fact about the user", " ".join(memory.split()))


class WiringTests(unittest.TestCase):
    def test_newer_events_are_wired_to_capture_in_every_claude_config(self) -> None:
        for path in (
            ROOT / "hooks" / "hooks.json",
            ROOT / "plugin" / "hooks" / "hooks.json",
            ROOT / ".claude" / "settings.json",
        ):
            hooks = json.loads(path.read_text())["hooks"]
            for event in NEWER_EVENTS:
                with self.subTest(config=str(path.relative_to(ROOT)), event=event):
                    commands = [h["command"] for g in hooks.get(event, []) for h in g["hooks"]]
                    self.assertTrue(any("log_event.py" in c for c in commands))
            # A WorktreeCreate hook replaces the harness's own worktree handling.
            self.assertNotIn("WorktreeCreate", hooks)
            self.assertNotIn("WorktreeRemove", hooks)

    def test_capture_only_events_are_the_newer_events_but_stop_failure(self) -> None:
        self.assertEqual(
            project_common.CAPTURE_ONLY_EVENTS, frozenset(NEWER_EVENTS) - {"StopFailure"}
        )


if __name__ == "__main__":
    unittest.main()
