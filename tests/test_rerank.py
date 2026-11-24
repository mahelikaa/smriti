import json

import httpx
import pytest

from smriti.embed import VOYAGE_BASE_URL
from smriti.rerank import RERANK_MODEL, RerankError, rerank


def client_with(handler):
    return httpx.AsyncClient(
        base_url=VOYAGE_BASE_URL, transport=httpx.MockTransport(handler)
    )


async def test_returns_indexes_and_scores_in_api_order():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        data = [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4},
        ]
        return httpx.Response(200, json={"data": data})

    async with client_with(handler) as c:
        order = await rerank(c, "shaadi kab hai", ["a", "b", "c"], top_k=2)

    assert order == [(2, 0.9), (0, 0.4)]
    assert calls[0] == {
        "query": "shaadi kab hai",
        "documents": ["a", "b", "c"],
        "model": RERANK_MODEL,
        "top_k": 2,
    }


async def test_no_documents_makes_no_call():
    def handler(request):
        raise AssertionError("should not call the API")

    async with client_with(handler) as c:
        assert await rerank(c, "q", [], top_k=5) == []


@pytest.mark.parametrize("status, retryable", [(429, True), (502, True), (400, False)])
async def test_http_errors(status, retryable):
    async with client_with(lambda request: httpx.Response(status)) as c:
        with pytest.raises(RerankError) as e:
            await rerank(c, "q", ["a"], top_k=1)

    assert e.value.retryable is retryable


async def test_index_out_of_range_is_rejected():
    def handler(request):
        return httpx.Response(
            200, json={"data": [{"index": 5, "relevance_score": 0.9}]}
        )

    async with client_with(handler) as c:
        with pytest.raises(RerankError, match="bad response shape"):
            await rerank(c, "q", ["a", "b"], top_k=1)
