#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Mirror the checkout's hook/server sources into the plugin/ payload.

The repo root is the dev checkout and plugin/ is what the marketplace installs
(marketplace.json points its source at "./plugin"). The two hold verbatim copies
of the same trees, so the copies drift silently unless something re-syncs them.
The suite imports the checkout, which means drift ships code no test has run.

``--check`` reports drift without writing, mirroring the assertion in
tests/test_project_hooks.py::PluginPayloadParityTests.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Kept in step with tests/test_project_hooks.py.
MIRRORED_TREES = ("hooks", "src", "commands", "import")
PACKAGE_ONLY_FILES = {"hooks": frozenset({"codex-hooks.json"})}
IGNORED_DIRS = frozenset({"__pycache__", ".venv", ".pytest_cache", ".ruff_cache"})


def payload_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root)
        if IGNORED_DIRS.intersection(relative.parts):
            continue
        files[relative.as_posix()] = path
    return files


def sync_tree(tree: str, *, check: bool) -> list[str]:
    source_root = ROOT / tree
    target_root = ROOT / "plugin" / tree
    exempt = PACKAGE_ONLY_FILES.get(tree, frozenset())

    source = payload_files(source_root)
    target = payload_files(target_root)
    drifted: list[str] = []

    for relative, source_path in sorted(source.items()):
        target_path = target_root / relative
        if target_path.exists() and filecmp.cmp(source_path, target_path, shallow=False):
            continue
        drifted.append(f"{tree}/{relative}")
        if not check:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

    # Files the payload has but the checkout dropped, minus the package-only set.
    for relative in sorted(set(target) - set(source) - set(exempt)):
        drifted.append(f"plugin/{tree}/{relative} (not in {tree}/)")
        if not check:
            (target_root / relative).unlink()

    return drifted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift and exit non-zero instead of writing.",
    )
    args = parser.parse_args()

    drifted: list[str] = []
    for tree in MIRRORED_TREES:
        drifted.extend(sync_tree(tree, check=args.check))

    if not drifted:
        print("plugin/ payload is in sync with the checkout.")
        return 0

    verb = "drifted" if args.check else "synced"
    print(f"{len(drifted)} file(s) {verb}:")
    for entry in drifted:
        print(f"  {entry}")
    if args.check:
        print("\nRun `uv run scripts/sync_plugin_payload.py` to sync.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
