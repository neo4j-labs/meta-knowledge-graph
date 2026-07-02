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


def test_neocarta_transport_uses_project_environment_entrypoint() -> None:
    transport = server._neocarta_transport({"NEO4J_URI": "bolt://example"})

    assert transport.command == "neocarta-mcp"
    assert transport.args == []
    assert transport.env == {"NEO4J_URI": "bolt://example"}


def test_plugin_mcp_server_uses_project_uv_environment() -> None:
    mcp_config = json.loads((ROOT / "plugin" / ".mcp.json").read_text())
    mkg_server = mcp_config["mcpServers"]["meta-knowledge-graph"]

    assert mkg_server["command"] == "bash"
    assert "uv run --project" in " ".join(mkg_server["args"])
    assert "uvx" not in " ".join(mkg_server["args"])


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
    tree = ast.parse(Path(server.__file__).read_text())
    defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "project_add_learning"
    ]
    assert len(defs) == 1
    arg_names = {arg.arg for arg in defs[0].args.args + defs[0].args.kwonlyargs}
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
