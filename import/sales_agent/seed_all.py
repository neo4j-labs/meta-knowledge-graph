"""Seed the whole sales-agent demo: BigQuery, then Neo4j, then the system prompt.

Each underlying script stays independently runnable for partial refreshes.

Run:
    uv run python import/sales_agent/seed_all.py            # data + persona (name 'sales_agent')
    uv run python import/sales_agent/seed_all.py --default  # also make the persona the active prompt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    prompt_args = ["--default"] if "--default" in argv else []
    steps: list[tuple[str, list[str]]] = [
        ("seed_bigquery.py", []),
        ("seed_neo4j.py", []),
        ("seed_system_prompt.py", prompt_args),
    ]
    for name, extra in steps:
        print(f"\n--- {name} {' '.join(extra)} ---")
        result = subprocess.run([sys.executable, str(here / name), *extra], check=False)
        if result.returncode != 0:
            print(f"{name} exited with code {result.returncode}", file=sys.stderr)
            return result.returncode
    print("\nall seeders completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
