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

import io
import locale
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


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


def read_env_text(path: Path) -> str:
    """Text of a ``.env`` file. Mirrors ``hooks/project_common.read_env_text``:
    UTF-8 (byte-order mark dropped), or the locale encoding for a file an older
    ``meta-knowledge-graph setup`` wrote in the Windows code page."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False), errors="replace")


def _load_env_file(path: str | Path, *, override: bool = False) -> None:
    if path and Path(path).is_file():
        load_dotenv(stream=io.StringIO(read_env_text(Path(path))), override=override)


def load_seed_env() -> None:
    """Load cwd ``.env`` (if any), then the user-global MKG ``.env`` over it."""
    _load_env_file(find_dotenv())
    _load_env_file(mkg_config_dir() / ".env", override=True)
