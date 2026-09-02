"""User identity: who wrote and who may recall what.

Every session, learning, observation, and profile carries the id (an email
where one is discoverable) of the person driving the harness. These tests pin
the resolution chain, the per-user namespacing of user facts, the user tag on
every write, the per-user filter on every user-scoped read, and the parity
between the hooks and the MCP server, which resolve the identity separately.
"""

from __future__ import annotations

import base64
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
SRC = ROOT / "src"
for path in (HOOKS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


project_common = load_module(HOOKS / "project_common.py", "project_common")
log_event = load_module(HOOKS / "log_event.py", "log_event")
process_project = load_module(HOOKS / "process_project.py", "process_project")
consistency_gate = load_module(HOOKS / "consistency_gate.py", "consistency_gate")
link_learning_session = load_module(
    HOOKS / "link_learning_session.py", "link_learning_session"
)
inject_system_prompt = load_module(
    HOOKS / "inject_system_prompt.py", "inject_system_prompt"
)
from meta_knowledge_graph import server  # noqa: E402

USER = project_common.UserRef("tomaz@example.com", "env")


def _jwt(claims: dict) -> str:
    def segment(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()

    return f"{segment({'alg': 'none'})}.{segment(claims)}.signature"


class _RecordingTx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        return _RecordingResult()


class _RecordingResult:
    def single(self):
        return {"old_version": 0, "new_version": 1, "total": 0}

    def consume(self):
        return None

    def __iter__(self):
        return iter([])


class _RecordingDriver:
    def __init__(self, rows=None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._rows = rows or []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_query(self, query: str, **params):
        self.calls.append((query, params))
        return list(self._rows)

    def session(self, **kwargs):
        return _RecordingSession(self.calls)


class _RecordingSession:
    def __init__(self, calls) -> None:
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params):
        self._calls.append((query, params))
        return _RecordingResult()

    def execute_write(self, fn, **kwargs):
        return None


class ResolveUserTests(unittest.TestCase):
    """The resolution chain: explicit env, harness account, git, OS."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _env(self, **extra: str) -> dict[str, str]:
        env = {
            "HOME": str(self.tmp),
            "CLAUDE_CONFIG_DIR": str(self.tmp / "claude"),
            "CODEX_HOME": str(self.tmp / "codex"),
            "PATH": os.environ.get("PATH", ""),
        }
        env.update(extra)
        return env

    def _write_claude_account(self, email: str) -> None:
        (self.tmp / "claude").mkdir(exist_ok=True)
        (self.tmp / "claude" / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"emailAddress": email}})
        )

    def _write_codex_account(self, email: str) -> None:
        (self.tmp / "codex").mkdir(exist_ok=True)
        (self.tmp / "codex" / "auth.json").write_text(
            json.dumps({"tokens": {"id_token": _jwt({"email": email, "sub": "x"})}})
        )

    def test_explicit_env_wins_and_is_normalized(self) -> None:
        self._write_claude_account("claude@example.com")
        with patch.dict(os.environ, self._env(MKG_USER_ID="  Tomaz@Example.COM "), clear=True):
            user = project_common.resolve_user()
        self.assertEqual(user, project_common.UserRef("tomaz@example.com", "env"))

    def test_pinned_source_rides_along_with_a_pinned_id(self) -> None:
        # A background worker respawned with MKG_USER_ID pinned reports the
        # original source, not "env".
        env = self._env(MKG_USER_ID="tomaz@example.com", MKG_USER_ID_SOURCE="claude_code")
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(project_common.resolve_user().source, "claude_code")

    def test_claude_account_email_is_read_from_claude_json(self) -> None:
        self._write_claude_account("Claude@Example.com")
        with patch.dict(os.environ, self._env(), clear=True):
            user = project_common.resolve_user()
        self.assertEqual(user, project_common.UserRef("claude@example.com", "claude_code"))

    def test_codex_account_email_is_read_from_the_id_token(self) -> None:
        self._write_codex_account("codex@example.com")
        with patch.dict(os.environ, self._env(), clear=True):
            user = project_common.resolve_user()
        self.assertEqual(user, project_common.UserRef("codex@example.com", "codex"))

    def test_active_harness_account_is_preferred(self) -> None:
        self._write_claude_account("claude@example.com")
        self._write_codex_account("codex@example.com")
        with patch.dict(os.environ, self._env(), clear=True):
            # No signal: Claude Code first.
            self.assertEqual(project_common.resolve_user().id, "claude@example.com")
            # An explicit client hint flips it.
            self.assertEqual(
                project_common.resolve_user(client="codex").id, "codex@example.com"
            )
        with patch.dict(os.environ, self._env(MKG_CLIENT="codex"), clear=True):
            self.assertEqual(project_common.resolve_user().id, "codex@example.com")
        with patch.dict(os.environ, self._env(CODEX_PLUGIN_ROOT="/x"), clear=True):
            self.assertEqual(project_common.resolve_user().id, "codex@example.com")
        with patch.dict(
            os.environ, self._env(CODEX_PLUGIN_ROOT="/x", MKG_CLIENT="claude_code"), clear=True
        ):
            self.assertEqual(project_common.resolve_user().id, "claude@example.com")

    def test_git_email_is_the_fallback_after_harness_accounts(self) -> None:
        with patch.dict(os.environ, self._env(), clear=True), patch.object(
            project_common, "_git_user_email", return_value="git@example.com"
        ) as git:
            user = project_common.resolve_user(self.tmp)
        self.assertEqual(user, project_common.UserRef("git@example.com", "git"))
        git.assert_called_once_with(self.tmp)

    def test_os_user_is_the_last_resort_and_never_empty(self) -> None:
        with patch.dict(os.environ, self._env(), clear=True), patch.object(
            project_common, "_git_user_email", return_value=None
        ), patch.object(project_common.getpass, "getuser", return_value="Tomaz"):
            user = project_common.resolve_user()
        self.assertEqual(user, project_common.UserRef("tomaz", "os"))

    def test_malformed_account_files_are_ignored(self) -> None:
        (self.tmp / "claude").mkdir()
        (self.tmp / "claude" / ".claude.json").write_text("{not json")
        (self.tmp / "codex").mkdir()
        (self.tmp / "codex" / "auth.json").write_text(json.dumps({"tokens": {"id_token": "x"}}))
        with patch.dict(os.environ, self._env(), clear=True), patch.object(
            project_common, "_git_user_email", return_value="git@example.com"
        ):
            self.assertEqual(project_common.resolve_user().source, "git")

    def test_normalize_user_id(self) -> None:
        self.assertEqual(project_common.normalize_user_id(" A@B.c "), "a@b.c")
        self.assertIsNone(project_common.normalize_user_id(""))
        self.assertIsNone(project_common.normalize_user_id(None))
        self.assertIsNone(project_common.normalize_user_id("two words"))
        self.assertEqual(
            len(project_common.normalize_user_id("x" * 500)),
            project_common.USER_ID_MAX_CHARS,
        )

    def test_project_env_pins_the_user_for_background_workers(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG", repo_root="/tmp/mkg")
        env = project_common.project_env(project, project_common.UserRef("a@b.c", "git"))
        self.assertEqual(env["MKG_USER_ID"], "a@b.c")
        self.assertEqual(env["MKG_USER_ID_SOURCE"], "git")
        self.assertEqual(env["MKG_PROJECT_ID"], "mkg")
        # No user: nothing pinned beyond what the shell already had.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MKG_USER_ID", None)
            os.environ.pop("MKG_USER_ID_SOURCE", None)
            unpinned = project_common.project_env(project)
        self.assertNotIn("MKG_USER_ID", unpinned)
        self.assertNotIn("MKG_USER_ID_SOURCE", unpinned)

    def test_background_extractor_pins_the_user(self) -> None:
        with patch.object(process_project.subprocess, "Popen") as popen:
            process_project._spawn_background("turn", 200, "session-1", None, USER)
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["MKG_USER_ID"], "tomaz@example.com")
        self.assertEqual(env["MKG_USER_ID_SOURCE"], "env")


class UserNamespaceTests(unittest.TestCase):
    """User facts are keyed per person; project facts stay per project."""

    def test_learning_namespace_is_per_user(self) -> None:
        self.assertEqual(
            project_common.learning_namespace("mkg", "user", "a@b.c"), "user:a@b.c"
        )
        self.assertEqual(project_common.learning_namespace("mkg", "project", "a@b.c"), "mkg")

    def test_extractor_rows_carry_the_user_and_per_user_ids(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        actions = {
            "learnings": [
                {"action": "create", "scope": "user", "text": "Prefers uv scripts."},
                {"action": "create", "scope": "project", "text": "Tests run via pytest."},
            ]
        }
        rows = process_project._memory_rows_from_actions(
            project, "turn", actions, user_id="a@b.c"
        )
        by_scope = {row["scope"]: row for row in rows}
        self.assertTrue(by_scope["user"]["id"].startswith("learning:user:a@b.c:"))
        self.assertTrue(by_scope["project"]["id"].startswith("learning:mkg:"))
        self.assertEqual({row["user_id"] for row in rows}, {"a@b.c"})
        # The same sentence about a different person is a different node.
        other = process_project._memory_rows_from_actions(
            project, "turn", actions, user_id="x@y.z"
        )
        self.assertNotEqual(other[0]["id"], rows[0]["id"])
        self.assertEqual(other[1]["id"], rows[1]["id"])

    def test_per_user_id_still_reads_as_user_scope_on_update(self) -> None:
        self.assertEqual(
            process_project._scope_from_action({}, "learning:user:a@b.c:deadbeef"), "user"
        )

    def test_observation_rows_carry_the_user(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        rows = process_project._observation_rows_from_items(
            project,
            "session-1",
            [{"event_id": "e1", "timestamp": "2026-09-02T09:00:00+00:00"}],
            [{"type": "change", "title": "Did a thing"}],
            user_id="a@b.c",
        )
        self.assertEqual(rows[0]["user_id"], "a@b.c")


class GraphWriteTests(unittest.TestCase):
    """Every write is tagged and hung off the (:User)."""

    def test_schema_declares_the_user_node_and_per_user_profile(self) -> None:
        tx = _RecordingTx()
        project_common.ensure_project_schema(tx)
        joined = "\n".join(query for query, _ in tx.calls)
        self.assertIn("FOR (u:User) REQUIRE u.id IS UNIQUE", joined)
        self.assertIn("FOR (up:UserProfile) REQUIRE up.user_id IS UNIQUE", joined)
        self.assertIn("FOR (l:Learning) ON (l.user_id)", joined)
        self.assertIn("FOR (s:Session) ON (s.user_id)", joined)

    def test_project_session_merge_owns_the_session(self) -> None:
        tx = _RecordingTx()
        project = project_common.ProjectRef(id="mkg", name="MKG")
        project_common.merge_project_and_session(tx, project, "s1", "2026-09-02T09:00:00+00:00", "a@b.c")
        query, params = tx.calls[0]
        self.assertIn("s.user_id = coalesce(s.user_id, $user_id)", query)
        self.assertIn("MERGE (u:User {id: $user_id})", query)
        self.assertIn("MERGE (u)-[:HAS_SESSION]->(s)", query)
        self.assertEqual(params["user_id"], "a@b.c")

    def test_merge_user_sessions_dedupes_and_skips_unknown(self) -> None:
        tx = _RecordingTx()
        project_common.merge_user_sessions(
            tx, USER, ["s1", "s1", "unknown", "", "s2"], "2026-09-02T09:00:00+00:00"
        )
        query, params = tx.calls[0]
        self.assertEqual(params["session_ids"], ["s1", "s2"])
        self.assertEqual(params["user_id"], "tomaz@example.com")
        self.assertEqual(params["source"], "env")
        self.assertIn("MERGE (u)-[r:HAS_SESSION]->(s)", query)
        # The first owner sticks.
        self.assertIn("s.user_id = coalesce(s.user_id, $user_id)", query)
        tx2 = _RecordingTx()
        project_common.merge_user_sessions(tx2, USER, ["unknown"], "2026-09-02T09:00:00+00:00")
        self.assertEqual(tx2.calls, [])

    def test_log_event_stamps_the_event_and_owns_the_session(self) -> None:
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
        with patch.dict(sys.modules, {"neo4j": fake_neo4j}), patch.object(
            log_event, "resolve_user", return_value=USER
        ) as resolve, patch.object(log_event, "resolve_project", return_value=None):
            log_event.log_event(
                {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "hi"},
                client="codex",
            )
        # The client hint reaches the resolver so the right harness account wins.
        self.assertEqual(resolve.call_args.args[1], "codex")
        by_fn = {fn: args for fn, args in writes}
        event_props = by_fn[log_event._append_event][2]
        self.assertEqual(event_props["user_id"], "tomaz@example.com")
        user, session_ids, _ = by_fn[project_common.merge_user_sessions]
        self.assertEqual(user, USER)
        self.assertEqual(session_ids, ["s1", "s1"])

    def test_session_props_carry_the_user_without_leaking_nones(self) -> None:
        tx = _RecordingTx()
        log_event._append_event(
            tx,
            "s1",
            "claude_code",
            {"event_id": "e1", "timestamp": "2026-09-02T09:00:00+00:00", "user_id": "a@b.c"},
        )
        self.assertEqual(tx.calls[0][1]["session_props"]["user_id"], "a@b.c")
        tx = _RecordingTx()
        log_event._append_event(
            tx,
            "parent",
            "codex",
            {
                "event_id": "e2",
                "event_name": "SubagentStart",
                "timestamp": "2026-09-02T09:00:00+00:00",
                "agent_id": "child",
                "is_subagent": True,
                "user_id": "a@b.c",
            },
        )
        # Lifecycle markers stamp the user on the parent session but never
        # the subagent's actor identity.
        self.assertEqual(tx.calls[0][1]["session_props"], {"client": "codex", "user_id": "a@b.c"})

    def test_extraction_write_tags_learnings_observations_and_processing(self) -> None:
        tx = _RecordingTx()
        project = project_common.ProjectRef(id="mkg", name="MKG")
        events = [{"event_id": "e1", "timestamp": "2026-09-02T09:00:00+00:00"}]
        learning_rows = [
            {
                "id": "learning:user:a@b.c:1",
                "action": "create",
                "text": "Prefers uv.",
                "scope": "user",
                "status": "candidate",
                "confidence": 0.8,
                "user_id": "a@b.c",
            },
            {"id": "learning:mkg:2", "action": "update", "text": "x", "scope": "project", "confidence": 0.5, "user_id": "a@b.c"},
        ]
        observation_rows = [
            {"id": "observation:mkg:s1:d:0", "type": "change", "title": "t", "facts": [], "user_id": "a@b.c"}
        ]
        process_project._write_processing(
            tx, project, "s1", "turn", events, learning_rows, "m", "called", None, None,
            "2026-09-02T09:00:00+00:00", observation_rows=observation_rows, user_id="a@b.c",
        )
        head, creates, updates, observations = [q for q, _ in tx.calls[:4]]
        self.assertIn("pp.user_id = $user_id", head)
        self.assertIn("s.user_id = coalesce(s.user_id, $user_id)", head)
        self.assertIn("MERGE (u)-[:HAS_SESSION]->(s)", head)
        self.assertEqual(tx.calls[0][1]["user_id"], "a@b.c")
        self.assertIn("l.user_id = row.user_id", creates)
        self.assertIn("l.last_user_id = coalesce(row.user_id, l.last_user_id)", creates)
        self.assertIn("CASE WHEN row.scope = 'user' THEN [1] ELSE [] END", creates)
        self.assertIn("MERGE (u)-[:HAS_LEARNING]->(l)", creates)
        self.assertIn("l.last_user_id = coalesce(row.user_id, l.last_user_id)", updates)
        self.assertIn("o.user_id = coalesce(row.user_id, o.user_id)", observations)

    def test_mcp_learning_link_hook_owns_session_and_tags_learning(self) -> None:
        tx = _RecordingTx()
        link_learning_session.write_session_links(
            tx, "s1", ["learning:mkg:1"], "2026-09-02T09:00:00+00:00", "a@b.c"
        )
        query, params = tx.calls[-1]
        self.assertIn("MERGE (l)-[r:FROM_SESSION]->(s)", query)
        self.assertIn("MERGE (u)-[:HAS_SESSION]->(s)", query)
        self.assertIn("l.user_id = coalesce(l.user_id, $user_id)", query)
        self.assertEqual(params["user_id"], "a@b.c")

    def test_prompt_injection_record_is_tagged(self) -> None:
        captured: dict = {}

        class FakeDriver:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute_query(self, query, **params):
                captured["query"] = query
                captured["params"] = params
                return [{"created": True}]

        fake_neo4j = types.ModuleType("neo4j")
        fake_neo4j.GraphDatabase = types.SimpleNamespace(driver=lambda uri, auth: FakeDriver())
        with patch.dict(sys.modules, {"neo4j": fake_neo4j}), patch.object(
            inject_system_prompt,
            "neo4j_config",
            return_value=("bolt://x", "neo4j", "pw", "neo4j"),
        ):
            inject_system_prompt.record_injection(
                session_id="s1",
                hook_event="SessionStart",
                target="additionalContext",
                prompt_name="default",
                content="Base.",
                source="neo4j",
                user_id="a@b.c",
            )
        self.assertEqual(captured["params"]["user_id"], "a@b.c")
        self.assertIn("i.user_id = $user_id", captured["query"])
        self.assertIn("s.user_id = coalesce(s.user_id, $user_id)", captured["query"])
        self.assertIn("MERGE (u)-[:HAS_SESSION]->(s)", captured["query"])


class RecallScopingTests(unittest.TestCase):
    """User-scoped reads are one person's; project reads stay shared."""

    def test_user_learning_fallbacks_filter_by_user(self) -> None:
        driver = _RecordingDriver()
        project_common.fetch_user_learnings(driver, "neo4j", query=None, user_id="a@b.c")
        query, params = driver.calls[0]
        self.assertIn("l.user_id = $user_id", query)
        self.assertEqual(params["user_id"], "a@b.c")

    def test_user_learning_search_filters_by_user_on_every_path(self) -> None:
        driver = _RecordingDriver()
        project_common.fetch_user_learnings(driver, "neo4j", query="uv scripts", user_id="a@b.c")
        # Hybrid (keyword-only without a vector), then the fulltext fallback,
        # then the recency tail: every path is scoped to the user.
        self.assertGreaterEqual(len(driver.calls), 2)
        for query, params in driver.calls:
            self.assertIn("user_id = $user_id", query)
            self.assertEqual(params["user_id"], "a@b.c")

    def test_project_learning_search_is_not_user_filtered(self) -> None:
        driver = _RecordingDriver()
        project_common._fetch_memory_hybrid(
            driver,
            "neo4j",
            label="Learning",
            query="uv scripts",
            query_vector=None,
            scope="project",
            statuses=["approved"],
            limit=5,
            project_id="mkg",
            user_id="a@b.c",
        )
        query, _ = driver.calls[0]
        self.assertNotIn("node.user_id = $user_id", query)

    def test_gate_blocked_count_scopes_user_blocks_to_the_user(self) -> None:
        driver = _RecordingDriver(rows=[{"blocked": 1}])
        project_common.count_gate_blocked(driver, "neo4j", "mkg", user_id="a@b.c")
        query, params = driver.calls[0]
        self.assertIn("l.user_id = $user_id", query)
        self.assertIn("l.scope = 'project' AND l.project_id = $project_id", query)
        self.assertEqual(params["user_id"], "a@b.c")

    def test_profile_backlog_and_stale_facts_are_per_user(self) -> None:
        for fn in (
            project_common.count_user_profile_memories_pending,
            project_common.fetch_user_profile_memories_pending,
            project_common.fetch_user_profile_stale_facts,
        ):
            with self.subTest(fn=fn.__name__):
                driver = _RecordingDriver(rows=[{"pending": 0}])
                fn(driver, "neo4j", user_id="a@b.c")
                query, params = driver.calls[0]
                self.assertIn("l.user_id = $user_id", query)
                self.assertEqual(params["user_id"], "a@b.c")

    def test_profile_is_keyed_by_user(self) -> None:
        driver = _RecordingDriver()
        state = project_common.read_user_profile_state(driver, "neo4j", "a@b.c")
        query, params = driver.calls[0]
        self.assertIn("UserProfile {user_id: $user_id}", query)
        self.assertEqual(params["user_id"], "a@b.c")
        self.assertEqual(state["version"], 0)

        tx = _RecordingTx()
        project_common.snapshot_and_update_user_profile(
            tx,
            user_id="a@b.c",
            new_content="- Prefers uv.",
            folded_learning_ids=["learning:user:a@b.c:1"],
            unfolded_learning_ids=[],
            model="m",
            session_id="s1",
            now="2026-09-02T09:00:00+00:00",
        )
        archive, fold = tx.calls[0][0], tx.calls[1][0]
        self.assertIn("MERGE (up:UserProfile {user_id: $user_id})", archive)
        self.assertIn("up.name = $user_id", archive)
        self.assertIn("$user_id + ':v' + toString(old_version + 1)", archive)
        self.assertIn("MERGE (u)-[:HAS_PROFILE]->(up)", archive)
        self.assertIn("MATCH (up:UserProfile {user_id: $user_id})", fold)
        self.assertEqual(tx.calls[0][1]["user_id"], "a@b.c")

    def test_prompt_bundle_reads_the_users_profile(self) -> None:
        driver = _RecordingDriver()

        def execute_query(query, **params):
            driver.calls.append((query, params))
            if "SystemPrompt" in query:
                return [{"content": "Base."}]
            return [{"content": "- A bullet.", "needs_revision": False}]

        driver.execute_query = execute_query
        fake_neo4j = types.ModuleType("neo4j")
        fake_neo4j.GraphDatabase = types.SimpleNamespace(driver=lambda uri, auth: driver)
        with patch.dict(sys.modules, {"neo4j": fake_neo4j}), patch.object(
            inject_system_prompt,
            "neo4j_config",
            return_value=("bolt://x", "neo4j", "pw", "neo4j"),
        ):
            base, profile, stale = inject_system_prompt.fetch_prompt_bundle_from_neo4j(
                "default", "a@b.c"
            )
        self.assertEqual((base, profile, stale), ("Base.", "- A bullet.", False))
        profile_query, profile_params = driver.calls[1]
        self.assertIn("UserProfile {user_id: $user_id}", profile_query)
        self.assertEqual(profile_params["user_id"], "a@b.c")

    def test_gate_neighbours_for_a_user_fact_are_that_users_only(self) -> None:
        driver = _RecordingDriver()
        consistency_gate._fetch_neighbours_vector(
            driver,
            "neo4j",
            label="Learning",
            index_name="project_learning_vector",
            project_id="mkg",
            scope="user",
            vector=[0.1],
            topk=5,
            candidate_id="c",
            user_id="a@b.c",
        )
        consistency_gate._fetch_neighbours_fulltext(
            driver,
            "neo4j",
            label="Learning",
            fulltext_index="project_learning_fulltext",
            project_id="mkg",
            scope="user",
            text="prefers uv",
            topk=5,
            candidate_id="c",
            user_id="a@b.c",
        )
        for query, params in driver.calls:
            self.assertIn("($user_id IS NULL OR node.user_id = $user_id)", query)
            self.assertEqual(params["user_id"], "a@b.c")

    def test_sweep_returns_the_row_owner(self) -> None:
        driver = _RecordingDriver()
        consistency_gate._fetch_ungated_candidates(
            driver, "neo4j", label="Learning", project_id="mkg", exclude_ids=[], limit=5
        )
        self.assertIn("n.user_id AS user_id", driver.calls[0][0])

    def test_injected_context_names_the_user(self) -> None:
        project = project_common.ProjectRef(id="mkg", name="MKG")
        context = project_common.format_learning_context(
            project, [], [{"text": "Prefers uv.", "status": "approved"}], user_id="a@b.c"
        )
        self.assertIn("What we know about the user (a@b.c):", context)


class ServerParityTests(unittest.TestCase):
    """The MCP server resolves and applies the same identity as the hooks."""

    IDENTITY_FUNCTIONS = (
        "normalize_user_id",
        "_claude_account_email",
        "_jwt_claims",
        "_codex_account_email",
        "_git_user_email",
        "_active_harness",
        "_harness_account_lookups",
        "resolve_user",
    )
    IDENTITY_CONSTANTS = (
        "USER_ID_ENV_VAR",
        "USER_ID_SOURCE_ENV_VAR",
        "CLIENT_HINT_ENV_VAR",
        "USER_ID_SOURCE_ENV",
        "USER_ID_SOURCE_CLAUDE",
        "USER_ID_SOURCE_CODEX",
        "USER_ID_SOURCE_GIT",
        "USER_ID_SOURCE_OS",
        "USER_ID_MAX_CHARS",
        "IDENTITY_LOOKUP_TIMEOUT_SECONDS",
        "CLAUDE_HARNESS_ENV_VARS",
        "CODEX_HARNESS_ENV_VARS",
    )

    def test_identity_resolution_is_textually_identical(self) -> None:
        for name in self.IDENTITY_FUNCTIONS:
            with self.subTest(function=name):
                self.assertEqual(
                    inspect.getsource(getattr(server, name)),
                    inspect.getsource(getattr(project_common, name)),
                )
        for name in self.IDENTITY_CONSTANTS:
            with self.subTest(constant=name):
                self.assertEqual(getattr(server, name), getattr(project_common, name))

    def test_server_resolves_from_env(self) -> None:
        with patch.dict(os.environ, {"MKG_USER_ID": " Tomaz@Example.com "}):
            self.assertEqual(server._resolve_user_id(), "tomaz@example.com")

    def test_server_namespaces_user_facts_per_user(self) -> None:
        self.assertEqual(server._learning_namespace("mkg", "user", "a@b.c"), "user:a@b.c")
        self.assertEqual(
            server._learning_namespace("mkg", "user", "a@b.c"),
            project_common.learning_namespace("mkg", "user", "a@b.c"),
        )
        self.assertEqual(server._learning_namespace("mkg", "project", "a@b.c"), "mkg")

    def test_server_tools_tag_writes_and_scope_reads(self) -> None:
        source = Path(server.__file__).read_text()
        # project_add_learning tags the learning and owns user facts.
        self.assertIn("l.user_id = coalesce(l.user_id, $user_id)", source)
        self.assertIn("l.last_user_id = $user_id", source)
        self.assertIn("MERGE (u:User {id: $user_id})", source)
        self.assertIn("FOREACH (_ IN CASE WHEN $scope = 'user' THEN [1] ELSE [] END", source)
        # project_get_context recalls only the caller's user facts.
        self.assertIn("MATCH (l:Learning {scope: 'user', user_id: $user_id})", source)
        self.assertIn('"user_id": user_id,', source)
        # project_gate_audit scopes the user-scoped record to the caller.
        self.assertIn("(l.scope = 'user' AND l.user_id = $user_id)", source)
        # project_resolve_learning attributes the human override and flags
        # the owner's profile.
        self.assertIn("l.reviewed_by_user_id = $user_id", source)
        self.assertIn("up.user_id = coalesce(l.user_id, $user_id)", source)

    def test_setup_wizard_offers_the_user_id(self) -> None:
        from meta_knowledge_graph import SETUP_FIELDS

        self.assertIn("MKG_USER_ID", [key for key, _, _ in SETUP_FIELDS])


class HarnessWiringTests(unittest.TestCase):
    def test_codex_hooks_and_mcp_carry_the_client_hint(self) -> None:
        for path in (ROOT / ".codex" / "hooks.json", ROOT / "plugin" / "hooks" / "codex-hooks.json"):
            config = json.loads(path.read_text())
            commands = [
                hook["command"]
                for groups in config["hooks"].values()
                for group in groups
                for hook in group["hooks"]
            ]
            self.assertTrue(commands, str(path))
            for command in commands:
                self.assertIn("MKG_CLIENT=codex", command, str(path))
        self.assertIn('MKG_CLIENT = "codex"', (ROOT / ".codex" / "config.toml").read_text())


if __name__ == "__main__":
    unittest.main()
