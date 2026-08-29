from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meta_knowledge_graph import server  # noqa: E402


def _server_function_def(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(Path(server.__file__).read_text())
    defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(defs) == 1
    return defs[0]


def test_neocarta_transport_uses_isolated_uvx_environment() -> None:
    transport = server._neocarta_transport({"NEO4J_URI": "bolt://example"})

    assert transport.command == "uvx"
    assert transport.args == ["--from", "neocarta[mcp]>=0.8.0", "neocarta-mcp"]
    assert transport.env == {"NEO4J_URI": "bolt://example"}


def test_plugin_mcp_server_prefers_cached_venv_with_uv_fallback() -> None:
    mcp_config = json.loads((ROOT / "plugin" / ".mcp.json").read_text())
    mkg_server = mcp_config["mcpServers"]["meta-knowledge-graph"]
    launcher = (ROOT / "plugin" / "scripts" / "mcp-launcher.sh").read_text()
    command = " ".join(mkg_server["args"])

    assert mkg_server["command"] == "bash"
    assert "scripts/mcp-launcher.sh" in command
    assert "plugin/scripts/mcp-launcher.sh" in command
    assert "ls -dt" in command
    assert "sort -V" not in command
    assert 'exec "$ROOT/.venv/bin/python" -m meta_knowledge_graph' in launcher
    assert 'export PATH="$ROOT/.venv/bin:${PATH:-}"' in launcher
    assert 'exec uv run --project "$ROOT" meta-knowledge-graph' in launcher
    assert "uvx" not in command + launcher


def _session_start_commands(hooks_path: Path) -> list[str]:
    hooks = json.loads(hooks_path.read_text())
    return [
        hook["command"]
        for group in hooks["hooks"]["SessionStart"]
        for hook in group["hooks"]
        if hook["type"] == "command"
    ]


def test_plugin_session_start_hooks_can_bootstrap_uv_environment() -> None:
    for hooks_path in (
        ROOT / "plugin" / "hooks" / "codex-hooks.json",
        ROOT / ".codex" / "hooks.json",
        ROOT / "hooks" / "hooks.json",
        ROOT / "plugin" / "hooks" / "hooks.json",
        ROOT / ".claude" / "settings.json",
    ):
        commands = _session_start_commands(hooks_path)

        assert commands
        assert all("--no-sync" not in command for command in commands)


def test_mcp_project_resolver_prefers_explicit_project_env(monkeypatch) -> None:
    monkeypatch.setenv("MKG_PROJECT_ID", "Claims Service")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/Users/test/work/other-project")

    assert server._resolve_project_id(None) == "claims-service"


def test_mcp_project_resolver_uses_claude_project_dir(monkeypatch) -> None:
    monkeypatch.delenv("MKG_PROJECT_ID", raising=False)
    monkeypatch.delenv("MKG_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("MKG_PROJECT_DIR", raising=False)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/Users/test/work/claims-service")

    assert server._resolve_project_id(None) == "claims-service"


def test_mcp_learning_source_is_stable_verbose_tag() -> None:
    # MCP-written memory is always tagged with this stable verbose source so it
    # is uniform across Codex and Claude, paralleling the hook's `hooks-stop`.
    assert server.MCP_LEARNING_SOURCE == "agent-mcp"


def test_project_add_learning_has_no_source_parameter() -> None:
    # The writer provenance is fixed by the tool, not caller-supplied. The tool
    # is a closure inside create_mcp_server, so inspect its definition via AST.
    fn = _server_function_def("project_add_learning")
    arg_names = {arg.arg for arg in fn.args.args + fn.args.kwonlyargs}
    assert "source" not in arg_names


def test_embed_learning_text_returns_none_without_credentials(monkeypatch) -> None:
    # The write path must degrade to a plain (un-embedded) learning when the
    # embedding provider has no credentials; the sweep embeds it later.
    monkeypatch.setattr(server, "_llm_credentials_present", lambda model: False)
    assert server._embed_learning_text_sync("some text") is None


def test_project_add_learning_writes_embedding_property() -> None:
    # MCP-written learnings are embedded at write time so the consistency
    # gate's vector retrieval sees them immediately; coalesce keeps the stored
    # vector when embedding is unavailable for a given call.
    source = Path(server.__file__).read_text()
    assert "l.embedding = coalesce($embedding, l.embedding)" in source


def test_project_get_context_uses_hybrid_retrieval() -> None:
    source = Path(server.__file__).read_text()
    assert "description=\"Optional free-text query for hybrid ranking" in source
    assert "VECTOR INDEX {vector_index}" in source
    assert "UNION ALL'.join(branches)" in source
    assert "sum(1.0 / ($rrf_k + rank)) AS score" in source


def test_project_get_context_has_no_decision_surface() -> None:
    source = Path(server.__file__).read_text()
    assert "Decision" not in source
    assert "project_add_decision" not in source
    assert '"user_learnings": user_learnings' in source
