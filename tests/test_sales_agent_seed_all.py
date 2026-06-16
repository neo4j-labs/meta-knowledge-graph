from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "import" / "sales_agent" / "seed_all.py"
SPEC = importlib.util.spec_from_file_location("sales_agent_seed_all", MODULE_PATH)
seed_all = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["sales_agent_seed_all"] = seed_all
SPEC.loader.exec_module(seed_all)


BASELINE_ENV = {
    "NEO4J_URI": "bolt://example:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "secret",
    "NEO4J_DATABASE": "neo4j",
    "OPENAI_API_KEY": "sk-test",
}


class SalesAgentSeedAllTests(unittest.TestCase):
    def test_sales_prompt_uses_mkg_memory_contract(self) -> None:
        prompt = (ROOT / "import" / "sales_agent" / "system_prompt.md").read_text()
        normalized_prompt = " ".join(prompt.split())

        self.assertIn("Meta Knowledge Graph memory", prompt)
        self.assertIn("always mean the Meta Knowledge Graph (MKG) memory", prompt)
        self.assertIn("system: Neo4j-backed", prompt)
        self.assertIn("Neo4j-backed `:Learning` / `:Decision` nodes", prompt)
        self.assertIn("`project_get_context`, `project_add_learning`", prompt)
        self.assertIn(
            "Do not refer to, rely on, or imply any separate memory provider",
            normalized_prompt,
        )

    def test_baseline_env_requires_neo4j_and_openai(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(seed_all.SeedAllError) as ctx:
                seed_all._require_baseline_env()

        message = str(ctx.exception)
        self.assertIn("NEO4J_URI", message)
        self.assertIn("NEO4J_PASSWORD", message)
        self.assertIn("OPENAI_API_KEY", message)

    def test_steps_include_bigquery_only_when_enabled(self) -> None:
        self.assertEqual(
            seed_all._steps(False),
            ["seed_neo4j.py", "seed_learnings.py", "seed_system_prompt.py"],
        )
        self.assertEqual(
            seed_all._steps(True),
            [
                "seed_neo4j.py",
                "seed_learnings.py",
                "seed_system_prompt.py",
                "seed_bigquery.py",
                "run_neocarta.py",
            ],
        )

    def test_main_skips_bigquery_when_optional_env_is_missing(self) -> None:
        calls: list[str] = []

        with patch.dict(os.environ, BASELINE_ENV, clear=True):
            with patch.object(seed_all, "load_dotenv", return_value=True):
                with patch.object(seed_all, "_verify_neo4j", return_value=None):
                    with patch.object(
                        seed_all,
                        "_run_step",
                        side_effect=lambda name: calls.append(name) or 0,
                    ):
                        self.assertEqual(seed_all.main(["seed_all.py"]), 0)

        self.assertEqual(
            calls,
            ["seed_neo4j.py", "seed_learnings.py", "seed_system_prompt.py"],
        )

    def test_main_runs_bigquery_steps_when_access_check_passes(self) -> None:
        calls: list[str] = []
        env = {
            **BASELINE_ENV,
            "GCP_PROJECT_ID": "project",
            "BIGQUERY_DATASET_ID": "acme_corp",
        }

        with patch.dict(os.environ, env, clear=True):
            with patch.object(seed_all, "load_dotenv", return_value=True):
                with patch.object(seed_all, "_verify_neo4j", return_value=None):
                    with patch.object(
                        seed_all,
                        "_prepare_inline_gcp_credentials",
                        return_value=(None, ""),
                    ):
                        with patch.object(
                            seed_all,
                            "_bigquery_access",
                            return_value=(True, ""),
                        ):
                            with patch.object(
                                seed_all,
                                "_run_step",
                                side_effect=lambda name: calls.append(name) or 0,
                            ):
                                self.assertEqual(seed_all.main(["seed_all.py"]), 0)

        self.assertEqual(
            calls,
            [
                "seed_neo4j.py",
                "seed_learnings.py",
                "seed_system_prompt.py",
                "seed_bigquery.py",
                "run_neocarta.py",
            ],
        )

    def test_inline_gcp_credentials_become_adc_file(self) -> None:
        env = {"GCP_SERVICE_ACCOUNT_JSON": '{"type": "service_account"}'}
        with patch.dict(os.environ, env, clear=True):
            path, reason = seed_all._prepare_inline_gcp_credentials()
            assert path is not None
            self.assertEqual(reason, "")
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(), '{"type": "service_account"}')
            self.assertEqual(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], str(path))
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
