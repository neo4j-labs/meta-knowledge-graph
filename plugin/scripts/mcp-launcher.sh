#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-}"

if [ -z "$ROOT" ] || [ ! -f "$ROOT/pyproject.toml" ]; then
  echo "[mkg mcp] plugin root not found" >&2
  exit 1
fi

if [ -x "$ROOT/.venv/bin/python" ]; then
  export MKG_PLUGIN_ROOT="$ROOT"
  export VIRTUAL_ENV="$ROOT/.venv"
  export PATH="$ROOT/.venv/bin:${PATH:-}"
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "$ROOT/.venv/bin/python" -m meta_knowledge_graph
fi

exec uv run --project "$ROOT" meta-knowledge-graph
