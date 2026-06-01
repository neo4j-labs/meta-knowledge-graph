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
    def test_diffbot_token_prefers_primary_env_var(self) -> None:
        with patch.dict(
            server.os.environ,
            {"DIFFBOT_TOKEN": "primary", "DIFFBOT_API_TOKEN": "secondary"},
            clear=True,
        ):
            self.assertEqual(server._diffbot_token(), "primary")

    def test_diffbot_tools_are_enabled_only_when_token_is_present(self) -> None:
        with patch.dict(server.os.environ, {}, clear=True):
            self.assertFalse(server._has_diffbot_token())

        with patch.dict(server.os.environ, {"DIFFBOT_TOKEN": "   "}, clear=True):
            self.assertFalse(server._has_diffbot_token())

        with patch.dict(server.os.environ, {"DIFFBOT_API_TOKEN": "secondary"}, clear=True):
            self.assertTrue(server._has_diffbot_token())

        with patch.dict(server.os.environ, {"DIFFBOT_API_KEY": "fallback"}, clear=True):
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


if __name__ == "__main__":
    unittest.main()
