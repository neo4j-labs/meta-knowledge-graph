from __future__ import annotations

import asyncio
import json
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meta_knowledge_graph import server  # noqa: E402


class FakeDiffbotResponse:
    reason_phrase = "Unprocessable Entity"

    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    def json(self) -> dict[str, object]:
        return self._payload


class FakeAsyncClient:
    def __init__(self, responses: list[FakeDiffbotResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(
        self,
        _url: str,
        *,
        params: dict[str, object],
        headers: dict[str, str],
    ) -> FakeDiffbotResponse:
        self.calls.append(params.copy())
        return self.responses.pop(0)


class DiffbotToolHelperTests(unittest.TestCase):
    def test_diffbot_token_is_read_and_stripped(self) -> None:
        with patch.dict(server.os.environ, {"DIFFBOT_TOKEN": " primary "}, clear=True):
            self.assertEqual(server._diffbot_token(), "primary")

    def test_diffbot_tools_are_enabled_only_when_token_is_present(self) -> None:
        with patch.dict(server.os.environ, {}, clear=True):
            self.assertFalse(server._has_diffbot_token())

        with patch.dict(server.os.environ, {"DIFFBOT_TOKEN": "   "}, clear=True):
            self.assertFalse(server._has_diffbot_token())

        with patch.dict(server.os.environ, {"DIFFBOT_TOKEN": "token"}, clear=True):
            self.assertTrue(server._has_diffbot_token())

    def test_diffbot_response_filters_use_basic_field_mode(self) -> None:
        filters = [
            server.DIFFBOT_NEWS_FILTER,
            server.DIFFBOT_ORGANIZATION_ENHANCE_FILTER,
            server.DIFFBOT_PERSON_ENHANCE_FILTER,
        ]

        for response_filter in filters:
            self.assertNotIn(";", response_filter)
            self.assertNotIn("  ", response_filter)

        self.assertIn("tags.label", server.DIFFBOT_NEWS_FILTER.split())
        self.assertIn("summary", server.DIFFBOT_NEWS_FILTER.split())
        self.assertIn("text", server.DIFFBOT_NEWS_FILTER.split())

    def test_diffbot_request_retries_without_filter_when_filter_parse_fails(
        self,
    ) -> None:
        client = FakeAsyncClient(
            [
                FakeDiffbotResponse(
                    422,
                    {"error": True, "message": "Parse filter failed"},
                ),
                FakeDiffbotResponse(200, {"data": [{"title": "ok"}]}),
            ]
        )

        with (
            patch.dict(server.os.environ, {"DIFFBOT_TOKEN": "token"}, clear=True),
            patch.object(server.httpx, "AsyncClient", return_value=client),
        ):
            result = asyncio.run(
                server._diffbot_get_json(
                    "dql",
                    {
                        "query": 'type:Article tags.label:"Neo4j"',
                        "filter": server.DIFFBOT_NEWS_FILTER,
                    },
                )
            )

        self.assertEqual(json.loads(result), {"data": [{"title": "ok"}]})
        self.assertEqual(len(client.calls), 2)
        self.assertIn("filter", client.calls[0])
        self.assertNotIn("filter", client.calls[1])
        self.assertEqual(client.calls[1]["query"], 'type:Article tags.label:"Neo4j"')

    def test_diffbot_response_payload_limits_locations(self) -> None:
        payload = server._diffbot_response_payload(
            FakeDiffbotResponse(
                200,
                {
                    "data": [
                        {
                            "entity": {
                                "name": "Acme Corp",
                                "locations": [
                                    {"address": "one"},
                                    {"address": "two"},
                                    {"address": "three"},
                                    {"address": "four"},
                                ],
                            }
                        }
                    ]
                },
            )
        )

        locations = payload["data"][0]["entity"]["locations"]
        self.assertEqual(len(locations), server.MAX_DIFFBOT_LOCATIONS)
        self.assertEqual([loc["address"] for loc in locations], ["one", "two", "three"])

    def test_diffbot_error_payload_limits_locations(self) -> None:
        payload = server._diffbot_response_payload(
            FakeDiffbotResponse(
                500,
                {
                    "message": "server error",
                    "locations": [
                        {"address": "one"},
                        {"address": "two"},
                        {"address": "three"},
                        {"address": "four"},
                    ],
                },
            )
        )

        self.assertEqual(payload["status"], "error")
        self.assertEqual(
            len(payload["response"]["locations"]),
            server.MAX_DIFFBOT_LOCATIONS,
        )

    def test_diffbot_payload_compacts_entity_references(self) -> None:
        payload = server._diffbot_response_payload(
            FakeDiffbotResponse(
                200,
                {
                    "data": [
                        {
                            "entity": {
                                "name": "Acme Corp",
                                "ceo": {
                                    "name": "Jane Doe",
                                    "diffbotUri": "http://diffbot.com/entity/E1",
                                    "types": ["Person"],
                                },
                                "categories": [
                                    {"name": f"Category {i}", "diffbotUri": "uri"}
                                    for i in range(10)
                                ],
                                "allNames": [f"name-{i}" for i in range(10)],
                                "locations": [
                                    {
                                        "address": "branch",
                                        "city": {"name": "Berlin", "diffbotUri": "uri"},
                                        "latitude": 52.5,
                                        "metroArea": {"name": "Berlin metro"},
                                    },
                                    {
                                        "address": "hq",
                                        "city": {
                                            "name": "New York City",
                                            "diffbotUri": "uri",
                                        },
                                        "isPrimary": True,
                                    },
                                ],
                            }
                        }
                    ]
                },
            )
        )

        entity = payload["data"][0]["entity"]
        self.assertEqual(
            entity["ceo"],
            {"name": "Jane Doe", "diffbotUri": "http://diffbot.com/entity/E1"},
        )
        self.assertEqual(
            entity["categories"],
            [f"Category {i}" for i in range(server.MAX_DIFFBOT_LIST_ITEMS)],
        )
        self.assertEqual(len(entity["allNames"]), server.MAX_DIFFBOT_LIST_ITEMS)
        self.assertEqual(
            entity["locations"][0],
            {"address": "hq", "city": "New York City", "isPrimary": True},
        )
        self.assertEqual(
            entity["locations"][1],
            {"address": "branch", "city": "Berlin"},
        )

    def test_bounded_diffbot_json_drops_heavy_fields_and_notes_truncation(self) -> None:
        payload = {
            "data": [
                {
                    "entity": {
                        "name": "Acme Corp",
                        "description": "x" * server.MAX_DIFFBOT_RESPONSE_CHARS,
                        "revenue": {"value": 1},
                    }
                }
            ]
        }

        text = server._bounded_diffbot_json(payload)

        self.assertLessEqual(len(text), server.MAX_DIFFBOT_RESPONSE_CHARS)
        result = json.loads(text)
        entity = result["data"][0]["entity"]
        self.assertNotIn("description", entity)
        self.assertEqual(entity["revenue"], {"value": 1})
        self.assertIn("description", result["truncated"])

    def test_bounded_diffbot_json_keeps_first_match_as_last_resort(self) -> None:
        payload = {
            "data": [
                {"entity": {"name": f"match-{i}", "blob": "x" * 20000}}
                for i in range(5)
            ]
        }

        text = server._bounded_diffbot_json(payload)

        self.assertLessEqual(len(text), server.MAX_DIFFBOT_RESPONSE_CHARS)
        result = json.loads(text)
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["data"][0]["entity"]["name"], "match-0")
        self.assertIn("first match", result["truncated"])

    def test_bounded_diffbot_json_leaves_small_payloads_untouched(self) -> None:
        payload = {"data": [{"entity": {"name": "Acme Corp"}}]}

        result = json.loads(server._bounded_diffbot_json(payload))

        self.assertEqual(result, payload)
        self.assertNotIn("truncated", result)

    def test_compact_clips_article_text_and_keeps_all_articles(self) -> None:
        payload = {
            "data": [
                {"entity": {"title": f"Article {i}", "text": "y" * 50000}}
                for i in range(server.MAX_DIFFBOT_ARTICLES)
            ]
        }

        result = json.loads(
            server._bounded_diffbot_json(server._compact_diffbot_payload(payload))
        )

        self.assertEqual(len(result["data"]), server.MAX_DIFFBOT_ARTICLES)
        self.assertNotIn("truncated", result)
        for item in result["data"]:
            text = item["entity"]["text"]
            self.assertTrue(text.endswith("…[clipped]"))
            self.assertLessEqual(
                len(text), server.MAX_DIFFBOT_ARTICLE_TEXT_CHARS + len("…[clipped]")
            )

    def test_enhance_params_require_valid_entity_type(self) -> None:
        params, error = server._build_diffbot_enhance_params(
            "Company",
            {"name": "Diffbot"},
        )

        self.assertIsNone(params)
        self.assertIn("Organization", error or "")
        self.assertIn("Person", error or "")

    def test_enhance_params_require_an_identifier(self) -> None:
        params, error = server._build_diffbot_enhance_params(
            "Organization",
            {"customId": "row-1"},
        )

        self.assertIsNone(params)
        self.assertIn("At least one identifier", error or "")

    def test_enhance_params_reject_person_only_fields_for_organization(self) -> None:
        params, error = server._build_diffbot_enhance_params(
            "Organization",
            {"name": "Diffbot", "email": "person@example.com"},
        )

        self.assertIsNone(params)
        self.assertIn("only valid for Person", error or "")

    def test_enhance_params_keep_valid_organization_identifiers(self) -> None:
        params, error = server._build_diffbot_enhance_params(
            "Organization",
            {
                "name": "Diffbot",
                "url": "https://www.diffbot.com",
                "phone": "+1 555 0100",
            },
        )

        self.assertIsNone(error)
        self.assertEqual(
            params,
            {
                "type": "Organization",
                "name": "Diffbot",
                "url": "https://www.diffbot.com",
                "phone": "+1 555 0100",
            },
        )

    def test_enhance_filter_is_sales_focused_by_entity_type(self) -> None:
        self.assertIn("homepageUri", server._diffbot_enhance_filter("Organization"))
        self.assertIn("employments", server._diffbot_enhance_filter("Person"))
        self.assertIn("diffbotUri", server._diffbot_enhance_filter("Organization"))
        self.assertIn("diffbotUri", server._diffbot_enhance_filter("Person"))


if __name__ == "__main__":
    unittest.main()
