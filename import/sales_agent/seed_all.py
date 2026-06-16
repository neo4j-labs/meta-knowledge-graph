"""Seed the sales-agent demo using only configured, reachable services.

Neo4j and OpenAI are the mandatory baseline. BigQuery/Neocarta are optional:
when GCP project/dataset settings and usable Google auth are present, the
warehouse and catalog seeders run; otherwise they are skipped with an explicit
message. Diffbot has no seed step; setting ``DIFFBOT_TOKEN`` enables the runtime
tools when the MCP server starts.

Run:
    uv run python import/sales_agent/seed_all.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REQUIRED_ENV = (
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
    "OPENAI_API_KEY",
)
BIGQUERY_ENV = ("GCP_PROJECT_ID", "BIGQUERY_DATASET_ID")


class SeedAllError(RuntimeError):
    """Baseline configuration or connectivity problem that should stop seeding."""


def _missing_env(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not os.environ.get(name)]


def _require_baseline_env() -> None:
    missing = _missing_env(REQUIRED_ENV)
    if missing:
        raise SeedAllError(
            "Missing required environment variables for the minimum sales-agent "
            f"seed: {', '.join(missing)}"
        )


def _verify_neo4j() -> None:
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ["NEO4J_DATABASE"]

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            driver.execute_query("RETURN 1", database_=database)
    except Exception as exc:  # noqa: BLE001 - driver-specific failures vary.
        raise SeedAllError(
            "Neo4j credentials are configured but not usable. Check "
            "NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, and NEO4J_DATABASE."
        ) from exc


def _prepare_inline_gcp_credentials() -> tuple[Path | None, str]:
    """Allow GCP_SERVICE_ACCOUNT_JSON to drive Google client libraries."""
    service_account_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not service_account_json or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return None, ""

    try:
        json.loads(service_account_json)
    except json.JSONDecodeError as exc:
        return None, f"GCP_SERVICE_ACCOUNT_JSON is not valid JSON: {exc}"

    fd, path = tempfile.mkstemp(prefix="sales-agent-sa-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(service_account_json)
        os.chmod(path, 0o600)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    return Path(path), ""


def _bigquery_access() -> tuple[bool, str]:
    missing = _missing_env(BIGQUERY_ENV)
    if missing:
        return False, f"missing optional BigQuery env vars: {', '.join(missing)}"

    try:
        from google.cloud import bigquery

        project = os.environ["GCP_PROJECT_ID"]
        billing_project = os.environ.get("GCP_BILLING_PROJECT_ID") or project
        client = bigquery.Client(project=billing_project)
        client.query("SELECT 1").result(timeout=30)
    except Exception as exc:  # noqa: BLE001 - optional preflight should skip plainly.
        return False, f"BigQuery auth/access check failed: {exc}"

    return True, ""


def _steps(include_bigquery: bool) -> list[str]:
    steps = [
        "seed_neo4j.py",
        "seed_learnings.py",
        "seed_system_prompt.py",
    ]
    if include_bigquery:
        steps.extend(["seed_bigquery.py", "run_neocarta.py"])
    return steps


def _run_step(name: str) -> int:
    print(f"\n--- {name} ---")
    result = subprocess.run([sys.executable, str(HERE / name)], check=False)
    if result.returncode != 0:
        print(f"{name} exited with code {result.returncode}", file=sys.stderr)
    return result.returncode


def main(_argv: list[str]) -> int:
    load_dotenv(REPO_ROOT / ".env")
    temp_credentials = None
    try:
        _require_baseline_env()
        _verify_neo4j()

        missing_bigquery_env = _missing_env(BIGQUERY_ENV)
        if missing_bigquery_env:
            include_bigquery = False
            reason = (
                "missing optional BigQuery env vars: "
                f"{', '.join(missing_bigquery_env)}"
            )
        else:
            temp_credentials, reason = _prepare_inline_gcp_credentials()
            if reason:
                include_bigquery = False
            else:
                include_bigquery, reason = _bigquery_access()

        if include_bigquery:
            print("BigQuery access verified; warehouse and Neocarta seeders will run.")
        else:
            print(f"Skipping optional BigQuery/Neocarta seeders: {reason}")

        if os.environ.get("DIFFBOT_TOKEN"):
            print("Diffbot token detected; no seed step is needed for Diffbot.")
        else:
            print("Diffbot token not set; runtime Diffbot tools will remain disabled.")

        for name in _steps(include_bigquery):
            returncode = _run_step(name)
            if returncode != 0:
                return returncode

        print("\nseeders completed.")
        return 0
    except SeedAllError as exc:
        print(f"seed_all.py: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_credentials:
            temp_credentials.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
