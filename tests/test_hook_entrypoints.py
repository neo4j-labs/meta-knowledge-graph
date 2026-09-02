"""Entrypoint smoke tests: a hook's ``main()`` run end to end against a stub
driver, the way the harness runs it.

Unit tests cover the helpers; these cover the plumbing between them, which
is where a hook dies silently. inject_project_context once ran for weeks
with a shadowed variable that raised on the first attribute access inside
``main()``: every helper was tested, the source text was asserted on, and no
test had executed the function the harness calls. A smoke test asserts on
what the harness sees (exit code, stdout JSON, stderr) and on the real
content of the injected context, because a failing hook now emits a
degraded-context line of its own and "non-empty" alone would not tell the
two apart.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))


def load_hook_module(name: str):
    module_path = HOOKS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


project_common = load_hook_module("project_common")
inject_project_context = load_hook_module("inject_project_context")


PROJECT = project_common.ProjectRef(
    id="mkg", name="Meta Knowledge Graph", repo_root="/tmp/mkg-checkout", source="test"
)
USER = project_common.UserRef(id="tester@example.com", source="test")
ENV = {
    "NEO4J_URI": "bolt://fake:7687",
    "NEO4J_USERNAME": "fake-user",
    "NEO4J_PASSWORD": "fake-pw",
    "NEO4J_DATABASE": "neo4j",
    "MKG_SKILL_ACTIVATION": "auto",
    "MKG_SKILL_CATALOG_INJECT": "1",
}

OBSERVATION = {
    "id": "obs-1",
    "type": "discovery",
    "title": "Found a dead project-context hook despite passing tests",
    "facts": None,
    "narrative": None,
    "ended_epoch": time.time() - 3600,
}


def _learning(scope: str, text: str) -> dict:
    return {
        "id": f"learning:{scope}:1",
        "text": text,
        "status": "approved",
        "confidence": 0.9,
        "task_pattern": None,
        "scope": scope,
        "kind": "fact",
        "tool_key": None,
        "error_signature": None,
        "resolved": None,
        "score": 0.9,
        "sources": ["vector"],
    }


USER_LEARNING = _learning("user", "Prefers small focused commits")
PROJECT_LEARNING = _learning(
    "project", "Hook sources are mirrored into plugin/ by scripts/sync_plugin_payload.py"
)


class _FakeTx:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def run(self, query: str, **params):
        self.statements.append(query)
        return []


class _FakeSession:
    def __init__(self, driver: "FakeDriver") -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_write(self, fn, *args, **kwargs):
        tx = _FakeTx()
        result = fn(tx, *args, **kwargs)
        self.driver.writes.extend(tx.statements)
        return result


class FakeDriver:
    """Answers ``execute_query`` from ``(predicate, rows)`` rules, records
    everything, and gives ``session().execute_write`` a transaction stub."""

    def __init__(self, answers=()) -> None:
        self.answers = list(answers)
        self.queries: list[tuple[str, dict]] = []
        self.writes: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def session(self, database=None):
        return _FakeSession(self)

    def execute_query(self, query: str, **params):
        self.queries.append((query, params))
        for matches, rows in self.answers:
            if matches(query, params):
                return rows
        return []


def memory_driver() -> FakeDriver:
    return FakeDriver(
        [
            (lambda q, p: "HAS_OBSERVATION" in q and "count(o)" not in q, [OBSERVATION]),
            (lambda q, p: "count(o) AS total" in q, [{"total": 4}]),
            # Session start carries no prompt, so user facts come from the
            # recency fallback rather than the hybrid retrieval path.
            (
                lambda q, p: "Learning {scope: 'user'}" in q
                or ("query_vector" in p and p.get("scope") == "user"),
                [USER_LEARNING],
            ),
            (lambda q, p: "query_vector" in p and p.get("scope") == "project", [PROJECT_LEARNING]),
        ]
    )


def fake_neo4j(driver_factory) -> types.ModuleType:
    """A stand-in ``neo4j`` package: the hook imports ``GraphDatabase`` inside
    ``main()``, so swapping ``sys.modules['neo4j']`` is enough."""
    module = types.ModuleType("neo4j")

    class GraphDatabase:
        opened: list[tuple[str, tuple]] = []

        @staticmethod
        def driver(uri, auth=None, **kwargs):
            GraphDatabase.opened.append((uri, auth))
            return driver_factory(uri, auth)

    module.GraphDatabase = GraphDatabase  # type: ignore[attr-defined]
    return module


def run_hook(module, payload: dict, driver_factory) -> tuple[int, str, str]:
    """Run ``module.main()`` as the harness would: payload on stdin, stdout
    and stderr captured, identity and embeddings stubbed, Neo4j faked."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {"neo4j": fake_neo4j(driver_factory)}))
        stack.enter_context(patch.dict(os.environ, ENV))
        stack.enter_context(patch.object(module, "load_mkg_env", lambda root: root / ".env"))
        stack.enter_context(patch.object(module, "resolve_project", lambda payload, root: PROJECT))
        stack.enter_context(patch.object(module, "resolve_user", lambda root=None, client=None: USER))
        stack.enter_context(patch.object(module, "embed_text", lambda text, **kw: [0.1, 0.2, 0.3]))
        stack.enter_context(patch.object(sys, "stdin", io.StringIO(json.dumps(payload))))
        stack.enter_context(contextlib.redirect_stdout(stdout))
        stack.enter_context(contextlib.redirect_stderr(stderr))
        code = module.main()
    return code, stdout.getvalue(), stderr.getvalue()


def hook_output(stdout: str) -> dict:
    payload = json.loads(stdout)
    return payload["hookSpecificOutput"]


class InjectProjectContextEntrypointTests(unittest.TestCase):
    def test_session_start_injects_user_facts_and_recent_episodes(self) -> None:
        driver = memory_driver()
        payload = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-1",
            "source": "startup",
            "cwd": PROJECT.repo_root,
        }

        code, stdout, stderr = run_hook(inject_project_context, payload, lambda uri, auth: driver)

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "", "a healthy hook writes nothing to stderr")
        output = hook_output(stdout)
        self.assertEqual(output["hookEventName"], "SessionStart")
        context = output["additionalContext"]
        self.assertIn("Project context for Meta Knowledge Graph (mkg):", context)
        self.assertIn("What we know about the user (tester@example.com):", context)
        self.assertIn("Prefers small focused commits", context)
        self.assertIn("Recent project activity", context)
        self.assertIn("Found a dead project-context hook", context)
        self.assertIn("3 earlier episodes are on record", context)
        # Project learnings are prompt-scoped; session start does not query them.
        self.assertNotIn("Relevant project learnings", context)

        # The driver was opened with the configured database credentials, the
        # schema was ensured, and the served learning was linked to the session.
        self.assertTrue(any("CREATE CONSTRAINT" in w for w in driver.writes))
        injected = [params for query, params in driver.queries if "MERGE (m)-[r:INJECTED_IN]" in query]
        self.assertTrue(injected, "served memory must be linked to the session")
        self.assertEqual(injected[0]["ids"], [USER_LEARNING["id"]])
        self.assertEqual(injected[0]["session_id"], "sess-1")

    def test_user_prompt_injects_relevant_project_learnings(self) -> None:
        driver = memory_driver()
        payload = {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess-1",
            "prompt": "how do the hooks get mirrored into the plugin payload?",
            "cwd": PROJECT.repo_root,
        }

        code, stdout, stderr = run_hook(inject_project_context, payload, lambda uri, auth: driver)

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        output = hook_output(stdout)
        self.assertEqual(output["hookEventName"], "UserPromptSubmit")
        context = output["additionalContext"]
        self.assertIn("Relevant project learnings:", context)
        self.assertIn("mirrored into plugin/", context)
        self.assertNotIn("What we know about the user", context)
        injected = [params for query, params in driver.queries if "MERGE (m)-[r:INJECTED_IN]" in query]
        self.assertEqual(injected[0]["ids"], [PROJECT_LEARNING["id"]])

    def test_driver_credentials_come_from_neo4j_config(self) -> None:
        opened: list[tuple[str, tuple]] = []

        def factory(uri, auth):
            opened.append((uri, auth))
            return memory_driver()

        payload = {"hook_event_name": "SessionStart", "session_id": "sess-1", "source": "startup"}
        code, _, stderr = run_hook(inject_project_context, payload, factory)

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(opened, [("bolt://fake:7687", ("fake-user", "fake-pw"))])

    def test_human_review_mode_counts_pending_skill_proposals(self) -> None:
        # The count is a session-start nudge for the human publisher; in auto
        # mode the sweep drains the queue itself, so no query runs.
        driver = memory_driver()
        driver.answers.append(
            (lambda q, p: "SkillVersion {outcome: 'pending'}" in q, [{"pending": 2}])
        )
        payload = {"hook_event_name": "SessionStart", "session_id": "sess-1", "source": "startup"}

        with patch.dict(os.environ, {"MKG_SKILL_ACTIVATION": "human"}):
            with patch.dict(ENV, {"MKG_SKILL_ACTIVATION": "human"}):
                code, stdout, stderr = run_hook(
                    inject_project_context, payload, lambda uri, auth: driver
                )
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn(
            "2 distilled skill proposals are waiting for a human",
            hook_output(stdout)["additionalContext"],
        )

        auto = memory_driver()
        code, stdout, stderr = run_hook(inject_project_context, payload, lambda uri, auth: auto)
        self.assertEqual((code, stderr), (0, ""))
        self.assertFalse(any("SkillVersion {outcome: 'pending'}" in q for q, _ in auto.queries))
        self.assertNotIn("waiting for a human", hook_output(stdout)["additionalContext"])

    def test_memory_failure_is_visible_in_the_injected_context(self) -> None:
        def factory(uri, auth):
            raise RuntimeError(f"connection refused to {uri}")

        payload = {"hook_event_name": "SessionStart", "session_id": "sess-1", "source": "startup"}
        code, stdout, stderr = run_hook(inject_project_context, payload, factory)

        # Never crash the session, never fail silently either.
        self.assertEqual(code, 0)
        self.assertIn("[inject_project_context] error: connection refused", stderr)
        output = hook_output(stdout)
        self.assertEqual(output["hookEventName"], "SessionStart")
        context = output["additionalContext"]
        self.assertIn("Persistent memory could not be loaded for this session", context)
        self.assertIn("RuntimeError: connection refused to bolt://fake:7687", context)
        self.assertIn("Memory recall is degraded", context)

    def test_degraded_context_is_one_bounded_line(self) -> None:
        exc = ValueError("x" * 1000 + "\n\n  multi   line")
        line = inject_project_context.format_degraded_context("UserPromptSubmit", exc)
        self.assertNotIn("\n", line)
        self.assertIn("for this prompt", line)
        self.assertLess(len(line), 400)


if __name__ == "__main__":
    unittest.main()
