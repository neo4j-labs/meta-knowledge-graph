from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import ingest_diffbot  # noqa: E402


class DiffbotIngestTests(unittest.TestCase):
    def test_enhance_rows_label_organizations(self) -> None:
        rows = ingest_diffbot._enhance_rows(
            {
                "data": [
                    {
                        "entity": {
                            "id": "org-1",
                            "type": "Organization",
                            "name": "Acme Corp",
                            "homepageUri": "https://www.acme.com",
                            "allNames": [{"name": "Acme"}],
                            "ceo": {
                                "name": "Jane Doe",
                                "diffbotUri": "http://diffbot.com/entity/EJane",
                            },
                        }
                    }
                ]
            },
            {"entity_type": "Organization", "url": "https://acme.com"},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entity_kind"], "organization")
        self.assertEqual(rows[0]["domain"], "acme.com")
        self.assertEqual(rows[0]["props"]["entity_type"], "Organization")
        self.assertEqual(rows[0]["props"]["ceo"], "Jane Doe")
        self.assertEqual(
            rows[0]["ceo"],
            {"id": "http://diffbot.com/entity/EJane", "name": "Jane Doe"},
        )
        self.assertEqual(rows[0]["employer_refs"], [])
        self.assertIn("acme", rows[0]["name_keys"])

    def test_enhance_rows_label_people_and_keep_employers(self) -> None:
        rows = ingest_diffbot._enhance_rows(
            {
                "data": [
                    {
                        "entity": {
                            "id": "person-1",
                            "type": "Person",
                            "name": "Jane Buyer",
                            "title": "VP Procurement",
                            "employments": [
                                {
                                    "employer": {
                                        "name": "Acme Corp",
                                        "diffbotUri": "http://diffbot.com/entity/EAcme",
                                    }
                                },
                            ],
                        }
                    }
                ]
            },
            {"entity_type": "Person", "employer": "Acme Corp"},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entity_kind"], "person")
        self.assertEqual(rows[0]["props"]["entity_type"], "Person")
        self.assertEqual(rows[0]["props"]["employers"], ["Acme Corp"])
        self.assertIsNone(rows[0]["ceo"])
        self.assertEqual(
            rows[0]["employer_refs"],
            [{"id": "http://diffbot.com/entity/EAcme", "name": "Acme Corp"}],
        )
        self.assertIn("acme corp", rows[0]["name_keys"])

    def test_enhance_rows_id_employer_hint_without_duplicating_named_refs(self) -> None:
        rows = ingest_diffbot._enhance_rows(
            {
                "data": [
                    {
                        "entity": {
                            "id": "person-2",
                            "type": "Person",
                            "name": "Sam Seller",
                            "employments": [
                                {"employer": {"name": "Acme Corp"}},
                            ],
                        }
                    }
                ]
            },
            {"entity_type": "Person", "employer": "Globex"},
        )

        refs = rows[0]["employer_refs"]
        self.assertEqual([ref["name"] for ref in refs], ["Acme Corp", "Globex"])
        for ref in refs:
            self.assertTrue(ref["id"].startswith("diffbot-ref:"))
        self.assertNotEqual(refs[0]["id"], refs[1]["id"])

    def test_news_rows_extract_organization_tags_as_companies(self) -> None:
        rows = ingest_diffbot._news_rows(
            {
                "data": [
                    {
                        "entity": {
                            "pageUrl": "https://news.example/acme",
                            "title": "Acme expands field operations",
                            "tags": [
                                {
                                    "label": "Acme Corp",
                                    "uri": "diffbot://org/acme",
                                    "types": [{"name": "Organization"}],
                                },
                                {
                                    "label": "Jane Buyer",
                                    "types": [{"name": "Person"}],
                                },
                            ],
                        }
                    }
                ]
            },
            {"dql": 'type:Article tags.label:"Acme Corp"'},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["account_keys"], ["acme corp", "jane buyer"])
        self.assertEqual(
            rows[0]["companies"],
            [
                {
                    "id": "diffbot://org/acme",
                    "props": {
                        "name": "Acme Corp",
                        "entity_type": "Organization",
                        "diffbot_uri": "diffbot://org/acme",
                    },
                }
            ],
        )

    def test_news_rows_do_not_treat_untyped_tags_as_companies(self) -> None:
        rows = ingest_diffbot._news_rows(
            {
                "data": [
                    {
                        "entity": {
                            "pageUrl": "https://news.example/topic",
                            "tags": [
                                {"label": "supply chain disruption"},
                                {"label": "Acme Company"},
                            ],
                        }
                    }
                ]
            },
            {"dql": 'type:Article tags.label:"supply chain disruption"'},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["companies"], [])


if __name__ == "__main__":
    unittest.main()
