from __future__ import annotations

import sys
import unittest
from hashlib import sha256
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
        self.assertNotIn("entity_type", rows[0]["props"])
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
        self.assertEqual(rows[0]["props"]["employers"], ["Acme Corp"])
        self.assertIsNone(rows[0]["ceo"])
        self.assertEqual(
            rows[0]["employer_refs"],
            [{"id": "http://diffbot.com/entity/EAcme", "name": "Acme Corp"}],
        )
        self.assertIn("acme corp", rows[0]["name_keys"])

    def test_enhance_rows_drop_refs_without_diffbot_id(self) -> None:
        rows = ingest_diffbot._enhance_rows(
            {
                "data": [
                    {
                        "entity": {
                            "id": "person-2",
                            "type": "Person",
                            "name": "Sam Seller",
                            "employments": [
                                {"employer": {"name": "and CEO of Acme"}},
                                {
                                    "employer": {
                                        "name": "Acme Corp",
                                        "diffbotUri": "https://diffbot.com/entity/EAcme",
                                    }
                                },
                                {
                                    "employer": {
                                        "name": "Acme",
                                        "diffbotUri": "http://diffbot.com/entity/EAcme",
                                    }
                                },
                            ],
                        }
                    }
                ]
            },
            {"entity_type": "Person", "employer": "Globex"},
        )

        # The malformed name-only employer produces no node ref, and the
        # https/http URI variants converge on one canonical id.
        self.assertEqual(
            rows[0]["employer_refs"],
            [{"id": "http://diffbot.com/entity/EAcme", "name": "Acme Corp"}],
        )
        self.assertEqual(rows[0]["props"]["employers"], ["Acme Corp"])
        # The employer hint still participates in account matching without
        # fabricating an organization node.
        self.assertIn("globex", rows[0]["name_keys"])

    def test_enhance_rows_skip_entities_without_id_or_kind(self) -> None:
        rows = ingest_diffbot._enhance_rows(
            {
                "data": [
                    {"entity": {"type": "Person", "name": "No Id Person"}},
                    {"entity": {"id": "x-1", "name": "No Kind Entity"}},
                ]
            },
            {},
        )

        self.assertEqual(rows, [])

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
                                {
                                    "label": "Untagged Org",
                                    "types": [{"name": "Organization"}],
                                },
                            ],
                        }
                    }
                ]
            },
            {"dql": 'type:Article tags.label:"Acme Corp"'},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["account_keys"], ["acme corp", "jane buyer", "untagged org"]
        )
        # Organization tags without any Diffbot id are dropped from companies.
        self.assertEqual(
            rows[0]["companies"],
            [
                {
                    "id": "diffbot://org/acme",
                    "props": {
                        "name": "Acme Corp",
                        "diffbot_uri": "diffbot://org/acme",
                    },
                }
            ],
        )

    def test_news_rows_include_article_text_metadata(self) -> None:
        article_text = "Acme opened a new field-service hub for regional teams."
        rows = ingest_diffbot._news_rows(
            {
                "data": [
                    {
                        "entity": {
                            "pageUrl": "https://news.example/acme",
                            "title": "Acme expands field operations",
                            "summary": "Acme expands its field support footprint.",
                            "text": article_text,
                            "tags": [{"label": "Acme Corp"}],
                        }
                    }
                ]
            },
            {"dql": 'type:Article tags.label:"Acme Corp"'},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["props"],
            {
                "title": "Acme expands field operations",
                "summary": "Acme expands its field support footprint.",
                "text": article_text,
                "text_chars": len(article_text),
                "text_sha256": sha256(article_text.encode("utf-8")).hexdigest(),
            },
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
