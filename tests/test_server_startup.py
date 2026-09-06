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


def test_gate_audit_replaces_the_review_queue() -> None:
    # The gate is autonomous: no learning ever waits on a person, and the
    # retired project_review_queue stays gone. (Skill publishing is the one
    # opt-in exception — MKG_SKILL_ACTIVATION=human — covered below.)
    # project_gate_audit is the accountability record — blocked learnings and
    # skill proposals with reasons, kept-both conflicts, stale skills — and the
    # resolver tools remain as the human override surface.
    source = Path(server.__file__).read_text()
    assert 'name="project_gate_audit"' in source
    assert "project_review_queue" not in source
    assert "blocked_learnings" in source
    assert "kept_conflicts" in source
    assert "ambiguous_kept_both" in source


def test_skill_review_queue_is_the_opt_in_publishing_gate() -> None:
    # MKG_SKILL_ACTIVATION=human is the one place a person becomes a
    # dependency: screened skill proposals wait in skill_review_queue until
    # project_resolve_skill publishes them. The queue carries the reviewer's
    # evidence — sources with their current status, the live content to diff
    # a patch against, the safety verdict — and the audit reports the length.
    source = Path(server.__file__).read_text()
    assert 'name="skill_review_queue"' in source
    assert 'SKILL_ACTIVATION_ENV_VAR = "MKG_SKILL_ACTIVATION"' in source
    assert "coalesce(v.safety_status, 'unscreened') AS safety_status" in source
    assert "sk.content AS current_content" in source
    assert "RETURN l.id AS id, l.text AS text, l.status AS status, " in source
    assert '"pending_skills": pending_skills' in source
    assert '"activation_mode": mode' in source


def test_skill_activation_mode_mirrors_hooks(monkeypatch) -> None:
    monkeypatch.delenv("MKG_SKILL_ACTIVATION", raising=False)
    assert server._skill_activation_mode() == "auto"
    for value in ("human", "REVIEW", " hitl ", "manual"):
        monkeypatch.setenv("MKG_SKILL_ACTIVATION", value)
        assert server._skill_activation_mode() == "human"
    monkeypatch.setenv("MKG_SKILL_ACTIVATION", "yes")
    assert server._skill_activation_mode() == "auto"


def test_project_get_context_counts_agent_retrieval_separately() -> None:
    # On-demand pulls by an agent bump retrieval_count; the hooks keep
    # inject_count; use_count is the combined tally both paths share.
    source = Path(server.__file__).read_text()
    assert "l.retrieval_count = coalesce(l.retrieval_count, 0) + 1" in source
    assert "l.last_retrieved_at = datetime($timestamp)" in source
    assert "l.use_count = coalesce(l.use_count, 0) + 1" in source
    assert "inject_count = coalesce" not in source
    # MCP-written learnings start every counter at zero.
    assert "l.inject_count = 0" in source
    assert "l.retrieval_count = 0" in source


def test_skill_approval_takes_source_learnings_out_of_recall() -> None:
    # Human publishing (project_resolve_skill approve) is the twin of the
    # auto-gate activation: the DERIVED_FROM learnings are flagged
    # consolidated (recall pre-filters it in-index) and carry compiled_at,
    # their embedding stays for dedup, and retire clears the flag on sources
    # no other live skill serves.
    fn = _server_function_def("project_resolve_skill")
    source = ast.get_source_segment(Path(server.__file__).read_text(), fn)
    assert source is not None
    approve = source.split("MERGE (sk)-[d:DERIVED_FROM]->(l)")[-1]
    assert "l.consolidated = true" in approve
    assert "l.embedding" not in approve
    assert "l.compiled_at = coalesce(l.compiled_at, datetime($ts))" in approve
    assert "l.compiled_skill_id = sk.id" in approve
    retire = source.split('if action == "retire":')[1].split("else:")[0]
    assert "(other:Skill {status: 'approved'})-[:DERIVED_FROM]->(l)" in retire
    assert "l.consolidated = false" in retire
    assert "l.compiled_at = null" in retire
    assert "l.embedding" not in retire
    # Rejecting a proposal still never touches a learning.
    reject = source.split('if action == "reject":')[1].split("else:  # approve")[0]
    assert "l.consolidated" not in reject


def test_project_get_context_excludes_consolidated_memory_in_index() -> None:
    # The MCP context tool mirrors the hooks: memory the profile or a live
    # skill already serves is pre-filtered inside the vector index for both
    # scopes, post-filtered on the fulltext branch, and left out of the
    # recency fallbacks — all on the consolidated flag, never on embeddings.
    text = Path(server.__file__).read_text()
    hybrid = ast.get_source_segment(text, _server_function_def("_fetch_context_memory_hybrid"))
    assert hybrid is not None
    search_clause = hybrid.split("SCORE AS raw_score")[0]
    assert "{consolidated_filter}" in search_clause
    assert '"AND node.consolidated = false" if exclude_consolidated' in hybrid
    assert "coalesce(node.consolidated, false) = false" in hybrid
    assert "consolidated_at" not in hybrid
    context = ast.get_source_segment(text, _server_function_def("project_get_context"))
    assert context is not None
    assert context.count("exclude_consolidated=True") == 2
    assert context.count("coalesce(l.consolidated, false) = false") == 2
    assert "consolidated_at" not in context


def test_learning_writes_maintain_the_recall_flag_in_mcp() -> None:
    # Born false on project_add_learning; every human or agent write that
    # bumps updated_at re-derives it, so a folded user fact re-enters recall
    # while a compiled learning stays served by its skill.
    source = Path(server.__file__).read_text()
    assert "l.consolidated = false," in source
    assert source.count("l.consolidated = (l.compiled_at IS NOT NULL)") == 3
    assert "c.consolidated = (c.compiled_at IS NOT NULL)" in source


def test_resolve_learning_restores_embedding_on_reinstate() -> None:
    # Reinstating a blocked/rejected tombstone must re-embed it, or it stays
    # invisible to vector retrieval forever.
    source = Path(server.__file__).read_text()
    assert "l.embedding = coalesce(l.embedding, $embedding)" in source
    assert "c.embedding = coalesce(c.embedding, $embedding)" in source


def test_mcp_task_pattern_drops_paragraph_values() -> None:
    # The MCP tool is the second writer of the same grouping key. A procedure
    # pasted into it clusters with nothing, exactly as through the extractor.
    assert (
        server._task_pattern(
            "To bootstrap UserProfile: (1) extract or MCP-add six user-scoped "
            "Learning candidates; (2) ensure the window has non-lifecycle events."
        )
        is None
    )
    assert server._task_pattern("when you debug the hook and it fails again") is None
    assert server._task_pattern(None) is None
    assert server._task_pattern("   ") is None


def test_mcp_task_pattern_keeps_short_labels() -> None:
    assert server._task_pattern("  hook pipeline\n  debugging ") == "hook pipeline debugging"
    assert server._task_pattern("cypher schema migration") == "cypher schema migration"


def test_mcp_task_pattern_caps_match_the_extractor() -> None:
    # Two entry points, one grouping key: capping them differently would let a
    # pattern the extractor rejects enter the graph through the tool.
    hooks_source = (ROOT / "hooks" / "process_project.py").read_text()
    for constant in ("MAX_TASK_PATTERN_CHARS", "MAX_TASK_PATTERN_WORDS"):
        value = getattr(server, constant)
        assert f"{constant} = {value}\n" in hooks_source
