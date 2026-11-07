"""Text I/O must be UTF-8 whatever the platform's code page is.

On Windows before Python 3.15, text Python reads or writes without an explicit
encoding uses the ANSI code page (cp1252 on a western install). The harness
pipes UTF-8, so a hook payload with ``č`` or ``”`` failed to decode and the
event was lost, ``ž`` was stored as ``Å¾``, and a user folder like
``C:\\Users\\Tomaž`` garbled the ``cwd`` and ``transcript_path`` of every event.

macOS and Linux default to UTF-8, so the Windows behaviour is reproduced with
``PYTHONIOENCODING=cp1252``, which makes the child's stdio decode the way a
Windows pipe does. Each cp1252 test has a control that shows the plain
``sys.stdin.read()`` / ``print`` path really does break under that setting, so
the test can't pass just because the environment failed to reproduce Windows.
The guard at the bottom keeps new code from reintroducing an implicit encoding.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))


def load_hook_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HOOKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


project_common = load_hook_module("project_common")

# ž and — decode under cp1252 but garble; č and ” contain bytes cp1252 leaves
# undefined (0x8D, 0x9D), so they raise; → can't be encoded to cp1252 at all.
SAMPLE = "Tomaž — “Bratanič” → done"


def windows_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONUTF8", None)
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONPATH"] = str(HOOKS)
    return env


def run_child(code: str, stdin: bytes = b"", env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        input=stdin,
        capture_output=True,
        env=env or windows_env(),
        cwd=tempfile.gettempdir(),
        timeout=60,
    )


def utf8_payload(**fields) -> bytes:
    # The harness serialises with JSON.stringify: raw UTF-8, no \\u escapes.
    return json.dumps(fields, ensure_ascii=False).encode("utf-8")


class HookStdinTests(unittest.TestCase):
    def test_control_plain_stdin_read_breaks_under_cp1252(self) -> None:
        proc = run_child(
            "import sys; sys.stdin.read()",
            stdin=utf8_payload(prompt=SAMPLE),
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"UnicodeDecodeError", proc.stderr)

    def test_read_hook_payload_decodes_utf8_under_cp1252(self) -> None:
        folder = "C:\\Users\\Tomaž\\code\\repo"
        proc = run_child(
            """
            import json, sys
            import project_common
            sys.stdout.write(json.dumps(project_common.read_hook_payload()))
            """,
            stdin=utf8_payload(prompt=SAMPLE, cwd=folder, session_id="s1"),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["prompt"], SAMPLE)
        self.assertEqual(payload["cwd"], folder)

    def test_read_hook_payload_keeps_the_event_when_a_byte_is_not_utf8(self) -> None:
        raw = b'{"session_id": "s1", "prompt": "bad \xff byte"}'
        proc = run_child(
            """
            import json, sys
            import project_common
            sys.stdout.write(json.dumps(project_common.read_hook_payload()))
            """,
            stdin=raw,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["session_id"], "s1")
        self.assertEqual(payload["prompt"], "bad \ufffd byte")

    def test_read_hook_payload_is_empty_for_missing_or_non_object_input(self) -> None:
        for raw in (b"", b"   ", b"not json", b"[1, 2]"):
            with self.subTest(raw=raw):
                proc = run_child(
                    "import json, sys, project_common; "
                    "sys.stdout.write(json.dumps(project_common.read_hook_payload()))",
                    stdin=raw,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(json.loads(proc.stdout), {})

    def test_read_stdin_text_reads_an_in_memory_stream(self) -> None:
        import io

        with patch.object(sys, "stdin", io.StringIO(SAMPLE)):
            self.assertEqual(project_common.read_stdin_text(), SAMPLE)
        with patch.object(sys, "stdin", None):
            self.assertEqual(project_common.read_stdin_text(), "")

    def test_log_event_captures_prompt_and_transcript_under_cp1252(self) -> None:
        """End to end through log_event.main(): the prompt arrives intact, and
        the transcript under a non-ASCII user folder is found and decoded."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Tomaž"
            folder.mkdir()
            transcript = folder / "session.jsonl"
            record = {"type": "user", "message": {"role": "user", "content": SAMPLE}}
            transcript.write_bytes(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
            out = Path(tmp) / "captured.json"
            proc = run_child(
                f"""
                import json, sys
                from pathlib import Path
                sys.argv = ["log_event.py", "--client", "claude_code"]
                import log_event
                log_event.load_mkg_env = lambda root: None
                log_event.in_extraction_subprocess = lambda: False
                def capture(data, client):
                    Path({str(out)!r}).write_text(json.dumps({{
                        "data": data,
                        "transcript": log_event._read_transcript(data.get("transcript_path")),
                    }}), encoding="utf-8")
                log_event.log_event = capture
                sys.exit(log_event.main())
                """,
                stdin=utf8_payload(
                    hook_event_name="UserPromptSubmit",
                    session_id="s1",
                    prompt=SAMPLE,
                    cwd=str(folder),
                    transcript_path=str(transcript),
                ),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, b"", "log_event reported an error")
            captured = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(captured["data"]["prompt"], SAMPLE)
        self.assertEqual(captured["data"]["cwd"], str(folder))
        self.assertIsNotNone(captured["transcript"], "transcript was not read")
        self.assertIn(SAMPLE, json.loads(captured["transcript"])["message"]["content"])


class HookOutputTests(unittest.TestCase):
    def test_control_plain_print_breaks_under_cp1252(self) -> None:
        proc = run_child("print('→')")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"UnicodeEncodeError", proc.stderr)

    def test_use_utf8_output_writes_utf8_under_cp1252(self) -> None:
        proc = run_child(
            """
            import sys, project_common
            project_common.use_utf8_output()
            print(%r)
            print(%r, file=sys.stderr)
            """ % (SAMPLE, SAMPLE)
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.decode("utf-8").strip(), SAMPLE)
        self.assertEqual(proc.stderr.decode("utf-8").strip(), SAMPLE)

    def test_use_utf8_output_leaves_unreconfigurable_streams_alone(self) -> None:
        import io

        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            project_common.use_utf8_output()
            self.assertIs(sys.stdout, out)
            self.assertIs(sys.stderr, err)


class EnvFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.env = self.tmp / ".env"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def readers(self):
        import meta_knowledge_graph

        sys.path.insert(0, str(ROOT / "import" / "sales_agent"))
        try:
            import _env as seed_env
        finally:
            sys.path.remove(str(ROOT / "import" / "sales_agent"))
        return {
            "hooks": project_common.read_env_text,
            "server": meta_knowledge_graph._read_env_text,
            "seeders": seed_env.read_env_text,
        }

    def test_utf8_values_round_trip(self) -> None:
        self.env.write_bytes("NEO4J_PASSWORD=Bratanič”\n".encode("utf-8"))
        for name, read in self.readers().items():
            with self.subTest(reader=name):
                self.assertEqual(read(self.env), "NEO4J_PASSWORD=Bratanič”\n")

    def test_byte_order_mark_is_dropped(self) -> None:
        self.env.write_bytes("\ufeffNEO4J_URI=bolt://example:7687\n".encode("utf-8"))
        for name, read in self.readers().items():
            with self.subTest(reader=name):
                self.assertEqual(read(self.env), "NEO4J_URI=bolt://example:7687\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEO4J_URI", None)
            project_common.load_dotenv(self.env)
            self.assertEqual(os.environ.get("NEO4J_URI"), "bolt://example:7687")

    def test_legacy_code_page_file_falls_back_to_the_locale_encoding(self) -> None:
        # What an older `meta-knowledge-graph setup` wrote on a cp1252 Windows box.
        self.env.write_bytes("NEO4J_PASSWORD=Tomaž\n".encode("cp1252"))
        with patch("locale.getpreferredencoding", return_value="cp1252"):
            for name, read in self.readers().items():
                with self.subTest(reader=name):
                    self.assertEqual(read(self.env), "NEO4J_PASSWORD=Tomaž\n")

    def test_hooks_load_non_ascii_credentials(self) -> None:
        self.env.write_bytes('NEO4J_PASSWORD="Tomaž č"\n'.encode("utf-8"))
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEO4J_PASSWORD", None)
            project_common.load_dotenv(self.env)
            self.assertEqual(os.environ["NEO4J_PASSWORD"], "Tomaž č")
            os.environ["NEO4J_PASSWORD"] = "ambient"
            project_common._override_credentials_from(self.env)
            self.assertEqual(os.environ["NEO4J_PASSWORD"], "Tomaž č")


class IdentityFileTests(unittest.TestCase):
    def test_claude_account_email_survives_non_ascii_elsewhere_in_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = {
                "oauthAccount": {"emailAddress": "User@Example.com"},
                "projects": {"C:\\Users\\Bratanič\\repo": {"history": ["“quoted”"]}},
            }
            Path(tmp, ".claude.json").write_bytes(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
            real_read_text = Path.read_text

            def windows_read_text(self, encoding=None, errors=None, *args, **kwargs):
                return real_read_text(self, encoding or "cp1252", errors, *args, **kwargs)

            with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": tmp}), patch.object(
                Path, "read_text", windows_read_text
            ):
                self.assertEqual(project_common._claude_account_email(), "user@example.com")


class ClaudeCliEncodingTests(unittest.TestCase):
    def test_claude_cli_pipes_utf8_both_ways(self) -> None:
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": SAMPLE}), stderr="")

        with patch.object(project_common.subprocess, "run", fake_run):
            result = project_common._complete_claude_cli_once([{"role": "user", "content": SAMPLE}])
        self.assertEqual(result, SAMPLE)
        self.assertEqual(seen["input"], SAMPLE)
        self.assertEqual(seen["encoding"], "utf-8")
        self.assertNotIn("text", seen)


# --------------------------------------------------------------------------- #
# Guard: no implicit encodings in shipped code
# --------------------------------------------------------------------------- #
SHIPPED_SOURCES = sorted(
    [
        *HOOKS.glob("*.py"),
        *(ROOT / "src" / "meta_knowledge_graph").glob("*.py"),
        *(ROOT / "import" / "sales_agent").glob("*.py"),
        *(ROOT / "scripts").glob("*.py"),
    ]
)
SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call", "call"}


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        owner = func.value
        prefix = owner.id + "." if isinstance(owner, ast.Name) else "."
        return prefix + func.attr
    return ""


def _kw(node: ast.Call, name: str):
    return next((k.value for k in node.keywords if k.arg == name), None)


def _is_true(value) -> bool:
    return isinstance(value, ast.Constant) and value.value is True


def implicit_encoding_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"read", "readline", "readlines"}:
            owner = node.value
            if (
                isinstance(owner, ast.Attribute)
                and owner.attr == "stdin"
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "sys"
            ):
                found.append(f"{node.lineno}: sys.stdin.{node.attr} (use read_stdin_text)")
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        has_encoding = _kw(node, "encoding") is not None
        if name.endswith(".read_text") and not (has_encoding or node.args):
            found.append(f"{node.lineno}: read_text() without encoding")
        elif name.endswith(".write_text") and not (has_encoding or len(node.args) >= 2):
            found.append(f"{node.lineno}: write_text() without encoding")
        elif name in {"open", "io.open", "os.fdopen", ".open"}:
            mode = node.args[1] if len(node.args) > 1 else _kw(node, "mode")
            text_mode = not (isinstance(mode, ast.Constant) and "b" in str(mode.value))
            if text_mode and not (has_encoding or len(node.args) > 3):
                found.append(f"{node.lineno}: {name}() in text mode without encoding")
        elif name.split(".")[-1] in SUBPROCESS_CALLS and name.startswith(("subprocess.", ".")):
            if (_is_true(_kw(node, "text")) or _is_true(_kw(node, "universal_newlines"))) and not has_encoding:
                found.append(f"{node.lineno}: {name}(text=True) without encoding")
    return found


class NoImplicitEncodingTests(unittest.TestCase):
    def test_shipped_code_names_utf8_for_every_text_read_and_write(self) -> None:
        self.assertTrue(SHIPPED_SOURCES)
        for path in SHIPPED_SOURCES:
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertEqual(implicit_encoding_violations(path), [])

    def test_guard_catches_each_pattern(self) -> None:
        sample = textwrap.dedent(
            """
            import subprocess, sys
            from pathlib import Path
            sys.stdin.read()
            Path("x").read_text()
            Path("x").write_text("y")
            open("x")
            open("x", "w")
            subprocess.run(["x"], text=True)
            open("x", "rb")
            Path("x").read_text(encoding="utf-8")
            subprocess.run(["x"], capture_output=True, encoding="utf-8")
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.py"
            path.write_text(sample, encoding="utf-8")
            violations = implicit_encoding_violations(path)
        self.assertEqual(len(violations), 6, violations)

    def test_every_hook_entrypoint_switches_output_to_utf8_first(self) -> None:
        for path in sorted(HOOKS.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            main = next(
                (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"),
                None,
            )
            if main is None:
                continue
            with self.subTest(hook=path.name):
                first = main.body[0]
                self.assertTrue(
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Call)
                    and _call_name(first.value) == "use_utf8_output",
                    f"{path.name}: main() must call use_utf8_output() first",
                )


if __name__ == "__main__":
    unittest.main()
