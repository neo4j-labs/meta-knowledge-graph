from __future__ import annotations

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
    mcp_config = json.loads((ROOT / ".mcp.json").read_text())
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
        ROOT / "hooks.json",
        ROOT / ".codex" / "hooks.json",
        ROOT / "hooks" / "hooks.json",
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
