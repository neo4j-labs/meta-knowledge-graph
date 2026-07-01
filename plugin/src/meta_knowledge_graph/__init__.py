import argparse
import asyncio
import getpass
import os
from pathlib import Path

from dotenv import load_dotenv

from meta_knowledge_graph import server

# Credentials/config the setup wizard prompts for. Mirrors the keys the hooks
# consume (hooks/project_common.CREDENTIAL_ENV_VARS) so both read the same .env.
# Each entry is (env var, is_secret, default).
SETUP_FIELDS = (
    ("NEO4J_URI", False, "bolt://localhost:7687"),
    ("NEO4J_USERNAME", False, "neo4j"),
    ("NEO4J_PASSWORD", True, None),
    ("NEO4J_DATABASE", False, "neo4j"),
    ("OPENAI_API_KEY", True, None),
    ("ANTHROPIC_API_KEY", True, None),
    ("GEMINI_API_KEY", True, None),
    ("OPENROUTER_API_KEY", True, None),
    ("DIFFBOT_TOKEN", True, None),
)


def _mkg_config_dir() -> Path:
    """User-global config dir; mirrors hooks/project_common.mkg_config_dir so the
    server and hooks read credentials from the same place. Precedence:
    MKG_CONFIG_DIR -> XDG_CONFIG_HOME -> ~/.config, under meta-knowledge-graph.
    Secrets live here, never in the plugin cache (wiped on update) or the repo."""
    override = os.environ.get("MKG_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "meta-knowledge-graph"


def _run_setup() -> int:
    """Interactively write credentials to <config-dir>/.env with mode 600."""
    target = _mkg_config_dir() / ".env"
    existing: dict[str, str] = {}
    if target.exists():
        for raw in target.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                existing[key.strip()] = value.strip().strip('"').strip("'")

    print(f"MKG setup — writing to {target}")
    print("Press Enter to keep the current value; leave blank to skip a field.\n")
    values: dict[str, str] = {}
    for key, secret, default in SETUP_FIELDS:
        current = existing.get(key, "")
        hint = ("set" if current else (default or "")) if secret else (current or default or "")
        suffix = f" [{hint}]" if hint else ""
        entered = (getpass.getpass(f"{key}{suffix}: ") if secret else input(f"{key}{suffix}: ")).strip()
        chosen = entered or current or (default or "")
        if chosen:
            values[key] = chosen

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    target.chmod(0o600)
    print(f"\nWrote {len(values)} value(s) to {target} (mode 600).")
    return 0


# Project/cwd .env first (repo-checkout / demo mode), then the user-global MKG
# .env authoritatively (installed plugin/CLI mode — survives plugin updates).
load_dotenv()
load_dotenv(_mkg_config_dir() / ".env", override=True)


def main():
    parser = argparse.ArgumentParser(description="Meta Knowledge Graph MCP Server")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "setup",
        help="Write Neo4j/LLM credentials to the user-global MKG .env "
        "(~/.config/meta-knowledge-graph/.env)",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URL",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD", "password"),
        help="Neo4j password",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default=os.environ.get("NEO4J_TRANSPORT", "stdio"),
        help="MCP transport type",
    )
    args = parser.parse_args()
    if args.command == "setup":
        raise SystemExit(_run_setup())
    asyncio.run(
        server.main(
            db_url=args.db_url,
            username=args.username,
            password=args.password,
            database=args.database,
            transport=args.transport,
        )
    )
