"""Exercise the Codex launch commands without contacting Neo4j or an LLM."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = json.loads((ROOT / 'plugin/hooks/codex-hooks.json').read_text())['hooks']
MCP = json.loads((ROOT / 'plugin/.codex-mcp.json').read_text())['mcpServers']['meta-knowledge-graph']


def clean_env(tmp_path):
    return {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'CODEX_HOME': str(tmp_path / 'codex')}


@pytest.mark.parametrize('root_var', ['PLUGIN_ROOT', 'CODEX_PLUGIN_ROOT', 'CLAUDE_PLUGIN_ROOT'])
def test_every_hook_runs_from_selected_install_with_payload(tmp_path, root_var):
    install = tmp_path / 'installed plugin'
    (install / 'hooks').mkdir(parents=True)
    for script in (ROOT / 'hooks').glob('*.py'):
        (install / 'hooks' / script.name).touch()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    uv = bin_dir / 'uv'
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$MKG_CLIENT" "$MKG_HOOK_ROOT" "$@"\ncat\n')
    uv.chmod(0o755)
    env = clean_env(tmp_path)
    env.update({root_var: str(install), 'PATH': str(bin_dir) + ':' + env['PATH']})
    if root_var == 'PLUGIN_ROOT':
        env['CODEX_PLUGIN_ROOT'] = str(tmp_path / 'stale install')
    for event, groups in HOOKS.items():
        for group in groups:
            for hook in group['hooks']:
                payload = json.dumps({'hook_event_name': event, 'session_id': 'test-session'})
                result = subprocess.run(['bash', '-c', hook['command']], input=payload,
                                        text=True, capture_output=True, env=env, cwd=tmp_path)
                assert result.returncode == 0, result.stderr
                assert result.stdout.startswith(f'codex\n{install}\nrun\n--script\n{install}/hooks/')
                assert result.stdout.endswith(payload)


@pytest.mark.parametrize('root_var', ['PLUGIN_ROOT', 'CODEX_PLUGIN_ROOT', 'CLAUDE_PLUGIN_ROOT'])
@pytest.mark.parametrize('layout', ['package', 'repository'])
def test_mcp_launch_uses_same_runtime_and_codex_identity(tmp_path, root_var, layout):
    install = tmp_path / 'installed plugin'
    scripts = install / ('scripts' if layout == 'package' else 'plugin/scripts')
    scripts.mkdir(parents=True)
    (install / 'pyproject.toml').touch()
    (scripts / 'mcp-launcher.sh').write_text('printf "%s\\n" "$MKG_CLIENT" "$1"\n')
    env = clean_env(tmp_path)
    env.update(MCP['env'])
    env[root_var] = str(install)
    result = subprocess.run([MCP['command'], *MCP['args']], env=env, cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['codex', str(install)]


def test_invalid_explicit_root_does_not_run_another_cached_version(tmp_path):
    cache = tmp_path / 'codex/plugins/cache/mkg/meta-knowledge-graph/999'
    (cache / 'hooks').mkdir(parents=True)
    (cache / 'hooks/log_event.py').touch()
    (cache / 'pyproject.toml').touch()
    env = clean_env(tmp_path)
    env['PLUGIN_ROOT'] = str(tmp_path / 'missing')
    hook = HOOKS['Stop'][0]['hooks'][0]['command']
    result = subprocess.run(['bash', '-c', hook], env=env, text=True, capture_output=True)
    assert result.returncode == 0
    assert 'cannot locate' in result.stderr
    result = subprocess.run([MCP['command'], *MCP['args']], env=env, text=True, capture_output=True)
    assert result.returncode == 1
    assert 'Codex plugin root not found' in result.stderr


def test_session_end_matches_claude_pipeline_and_has_bounded_timeout():
    claude = json.loads((ROOT / 'hooks/hooks.json').read_text())['hooks']
    expected = ['log_event.py', 'consolidate_system_prompt.py', 'consolidate_skills.py']
    for path in (ROOT / '.codex/hooks.json', ROOT / 'plugin/hooks/codex-hooks.json'):
        hooks = json.loads(path.read_text())['hooks']
        end = hooks['SessionEnd'][0]['hooks']
        assert len(end) == len(claude['SessionEnd'][0]['hooks']) == len(expected)
        for hook, script in zip(end, expected):
            assert script in hook['command']
            assert hook['timeout'] == 3
        assert 'fork' in hooks['SessionStart'][0]['matcher'].split('|')
        assert all('process_project.py' not in hook['command'] for hook in end)
