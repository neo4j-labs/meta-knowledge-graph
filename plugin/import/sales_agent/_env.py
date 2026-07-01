"""Shared ``.env`` resolution for the sales-agent seeders.

Mirrors :mod:`meta_knowledge_graph.__init__`: load the current project / cwd
``.env`` first (repo-checkout / demo mode), then the user-global MKG ``.env``
authoritatively with ``override=True``. The global file is what an installed
Claude Code plugin configures (``~/.config/meta-knowledge-graph/.env``), so
seeding works from a plugin cache where there is no repo ``.env``.

Kept dependency-free (no ``hooks`` import) so every seeder can use it regardless
of how it lays out ``sys.path``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def mkg_config_dir() -> Path:
    """User-global MKG config dir. Mirrors ``hooks/project_common.mkg_config_dir``
    and ``meta_knowledge_graph.__init__._mkg_config_dir`` so the seeders read
    credentials from the same place the MCP server and hooks do. Precedence:
    ``MKG_CONFIG_DIR`` -> ``XDG_CONFIG_HOME`` -> ``~/.config``, under
    ``meta-knowledge-graph``."""
    override = os.environ.get("MKG_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "meta-knowledge-graph"


def load_seed_env() -> None:
    """Load cwd ``.env`` (if any), then the user-global MKG ``.env`` over it."""
    load_dotenv()
    load_dotenv(mkg_config_dir() / ".env", override=True)
