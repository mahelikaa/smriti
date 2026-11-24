import json

import httpx
import pytest
from fakes import fake_embedder

from smriti.embed import EMBED_DIMS, EMBED_MODEL, VOYAGE_BASE_URL, to_pgvector
from smriti.main import app

TOPICS = {"wedding": 0, "married": 0, "shaadi": 0, "morning": 1, "pune": 2}


def topic_vector(text: str) -> list[float]:
    v = [0.0] * EMBED_DIMS
    words = text.lower()
    dim = next((d for word, d in TOPICS.items() if word in words), 3)
    v[dim] = 1.0
    return v


async def add_memory(user_id, content, model=EMBED_MODEL):
    vector = topic_vector(content) if model else None
    async with app.state.pool.acquire() as conn:
        turn_id = await conn.fetchval(
            """INSERT INTO turns (user_id, external_id, role, text, sent_at)
               VALUES ($1, $2, 'user', 'hi', now()) RETURNING id""",
            user_id,
            content,
        )
        await conn.execute(
            """INSERT INTO memories (user_id, turn_id, kind, content, importance,
                                     embedding, embedding_model)
               VALUES ($1, $2, 'fact', $3, 3, $4::vector, $5)""",
            user_id,
            turn_id,
            content,
            to_pgvector(vector),
            model,
        )


@pytest.fixture
def search_calls(monkeypatch):
    calls = []
    embedder = fake_embedder(calls=calls, vector_for=topic_vector)
    monkeypatch.setattr(app.state, "embed_client", embedder)
    return calls


async def search(client, q, user_id="alice", **params):
    return await client.get(
        f"/v1/users/{user_id}/memories/search", params={"q": q, **params}
    )


async def test_finds_memory_by_meaning(client, search_calls):
    await add_memory("alice", "Hates mornings")
    await add_memory("alice", "Lives in Pune")
    await add_memory("alice", "Sister Riya's wedding is in December")

    response = await search(client, "behen ki shaadi kab hai?")

    items = response.json()["items"]
    assert response.status_code == 200
    assert items[0]["content"] == "Sister Riya's wedding is in December"
    assert items[0]["score"] == pytest.approx(1.0)
    assert items[1]["score"] == pytest.approx(0.0)
    assert search_calls[0]["input_type"] == "query"


async def test_never_returns_other_users_memories(client, search_calls):
    await add_memory("bob", "Bob's wedding is in May")
    await add_memory("alice", "Hates mornings")

    response = await search(client, "who is getting married?")

    contents = [item["content"] for item in response.json()["items"]]
    assert contents == ["Hates mornings"]


async def test_k_limits_results(client, search_calls):
    for i in range(4):
        await add_memory("alice", f"memory {i}")

    response = await search(client, "anything", k=2)

    assert len(response.json()["items"]) == 2


async def test_skips_unembedded_and_other_model_memories(client, search_calls):
    await add_memory("alice", "Wedding with no vector", model=None)
    await add_memory("alice", "Wedding from old model", model="old-model")
    await add_memory("alice", "Wedding in December")

    response = await search(client, "shaadi")

    contents = [item["content"] for item in response.json()["items"]]
    assert contents == ["Wedding in December"]


@pytest.mark.parametrize(
    "status, expected, retry_after",
    [(429, 503, "1"), (503, 503, "1"), (401, 502, None)],
)
async def test_embedding_errors(client, monkeypatch, status, expected, retry_after):
    monkeypatch.setattr(app.state, "embed_client", fake_embedder(status=status))

    response = await search(client, "shaadi")

    assert response.status_code == expected
    assert response.headers.get("Retry-After") == retry_after


@pytest.mark.parametrize("params", [{"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 21}])
async def test_bad_params_are_422(client, params):
    response = await client.get("/v1/users/alice/memories/search", params=params)

    assert response.status_code == 422


async def test_skips_superseded_memories(client, search_calls):
    await add_memory("alice", "Wedding is in March")
    await add_memory("alice", "Wedding moved to December")
    async with app.state.pool.acquire() as conn:
        await conn.execute(
            """UPDATE memories SET superseded_by = (
                   SELECT id FROM memories WHERE content = 'Wedding moved to December')
               WHERE content = 'Wedding is in March'"""
        )

    response = await search(client, "shaadi")

    contents = [item["content"] for item in response.json()["items"]]
    assert contents == ["Wedding moved to December"]


def fake_voyage(rerank_order=None, rerank_status=200, rerank_calls=None):
    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            data = [
                {"index": i, "embedding": topic_vector(t)}
                for i, t in enumerate(body["input"])
            ]
            return httpx.Response(200, json={"data": data})
        if rerank_calls is not None:
            rerank_calls.append(body)
        if rerank_status != 200:
            return httpx.Response(rerank_status)
        data = [{"index": i, "relevance_score": s} for i, s in rerank_order]
        return httpx.Response(200, json={"data": data})

    return httpx.AsyncClient(
        base_url=VOYAGE_BASE_URL, transport=httpx.MockTransport(handler)
    )


async def test_rerank_reorders_results(client, monkeypatch):
    for content in ["Wedding in December", "Lives in Pune", "Hates mornings"]:
        await add_memory("alice", content)
    calls = []
    monkeypatch.setattr(
        app.state,
        "embed_client",
        fake_voyage(rerank_order=[(2, 0.8), (0, 0.3)], rerank_calls=calls),
    )

    response = await search(client, "shaadi", k=2, rerank="true")

    body = response.json()
    documents = calls[0]["documents"]
    assert body["reranked"] is True
    assert [i["content"] for i in body["items"]] == [documents[2], documents[0]]
    assert [i["score"] for i in body["items"]] == [0.8, 0.3]
    assert len(documents) == 3
    assert calls[0]["top_k"] == 2


async def test_rerank_failure_falls_back_to_vector_order(client, monkeypatch):
    await add_memory("alice", "Wedding in December")
    await add_memory("alice", "Lives in Pune")
    monkeypatch.setattr(app.state, "embed_client", fake_voyage(rerank_status=503))

    response = await search(client, "shaadi", rerank="true")

    body = response.json()
    assert response.status_code == 200
    assert body["reranked"] is False
    assert body["items"][0]["content"] == "Wedding in December"


async def test_rerank_is_off_by_default(client, monkeypatch):
    await add_memory("alice", "Wedding in December")
    calls = []
    monkeypatch.setattr(app.state, "embed_client", fake_voyage(rerank_calls=calls))

    response = await search(client, "shaadi")

    assert response.json()["reranked"] is False
    assert calls == []
