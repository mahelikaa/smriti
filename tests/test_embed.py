import json

import httpx
import pytest

from smriti.embed import EMBED_DIMS, EMBED_MODEL, EmbeddingError, embed, make_client


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")


def vec(marker: float, dims: int = EMBED_DIMS) -> list[float]:
    return [marker] + [0.0] * (dims - 1)


def client_with(handler):
    return make_client(transport=httpx.MockTransport(handler))


def ok(vectors_by_index: dict[int, list[float]]):
    def handler(request):
        data = [{"index": i, "embedding": v} for i, v in vectors_by_index.items()]
        return httpx.Response(200, json={"data": data})

    return handler


async def test_empty_list_makes_no_call():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    assert await embed(client_with(handler), [], "document") == []
    assert calls == []


async def test_sends_model_input_type_and_key():
    seen = []

    def handler(request):
        seen.append(request)
        return ok({0: vec(0.1)})(request)

    await embed(client_with(handler), ["hello"], "query")

    req = seen[0]
    assert req.url.path == "/v1/embeddings"
    assert req.headers["Authorization"] == "Bearer test-key"
    assert json.loads(req.content) == {
        "input": ["hello"],
        "model": EMBED_MODEL,
        "input_type": "query",
    }


async def test_vectors_are_sorted_by_index():
    # API answers out of order: index 1 first, then index 0
    handler = ok({1: vec(0.2), 0: vec(0.1)})

    vectors = await embed(client_with(handler), ["wedding", "mornings"], "document")

    assert vectors[0][0] == 0.1  # wedding gets its own vector
    assert vectors[1][0] == 0.2


@pytest.mark.parametrize(
    "status, retryable",
    [(429, True), (500, True), (503, True), (400, False), (401, False)],
)
async def test_http_errors(status, retryable):
    def handler(request):
        return httpx.Response(status)

    with pytest.raises(EmbeddingError) as exc:
        await embed(client_with(handler), ["hi"], "document")
    assert exc.value.retryable is retryable


async def test_timeout_is_retryable():
    def handler(request):
        raise httpx.ConnectTimeout("too slow")

    with pytest.raises(EmbeddingError) as exc:
        await embed(client_with(handler), ["hi"], "document")
    assert exc.value.retryable is True


@pytest.mark.parametrize(
    "vectors_by_index",
    [
        {0: vec(0.1)},  # 1 vector for 2 texts
        {0: vec(0.1, dims=512), 1: vec(0.2, dims=512)},  # wrong size
    ],
)
async def test_bad_shape_is_not_retryable(vectors_by_index):
    with pytest.raises(EmbeddingError) as exc:
        await embed(client_with(ok(vectors_by_index)), ["a", "b"], "document")
    assert exc.value.retryable is False


async def test_malformed_body_is_not_retryable():
    def handler(request):
        return httpx.Response(200, json={"oops": 1})

    with pytest.raises(EmbeddingError) as exc:
        await embed(client_with(handler), ["hi"], "document")
    assert exc.value.retryable is False
