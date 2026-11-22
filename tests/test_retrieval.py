import json

import pytest

from smriti.embed import EmbeddingError
from smriti.retrieval import (
    Query,
    embed_with_retry,
    hit_at,
    load_set,
    summarize,
)


def test_retrieval_set_is_valid():
    data = load_set()
    assert len(data.memories) >= 40
    assert len(data.queries) >= 40


def test_query_pointing_at_superseded_memory_is_rejected(tmp_path):
    path = tmp_path / "set.json"
    path.write_text(
        json.dumps(
            {
                "memories": [
                    {"id": "a", "kind": "fact", "content": "Lives in Pune"},
                    {
                        "id": "b",
                        "kind": "fact",
                        "content": "Lives in Delhi",
                        "superseded_by": "a",
                    },
                ],
                "queries": [{"text": "kahan rehti hoon", "relevant": ["b"]}],
            }
        )
    )

    with pytest.raises(ValueError):
        load_set(path)


@pytest.mark.parametrize(
    ("k", "expected"),
    [(1, False), (2, True), (5, True)],
)
def test_hit_at_only_looks_at_top_k(k, expected):
    assert hit_at(["x", "a", "y"], ["a"], k) is expected


def test_any_relevant_memory_counts_as_a_hit():
    assert hit_at(["b", "x"], ["a", "b"], 1) is True


def test_summarize_counts_queries_not_memories():
    queries = [Query(text="q1", relevant=["a"]), Query(text="q2", relevant=["b"])]
    rankings = [["a", "x"], ["x", "y", "z", "w", "v", "b"]]

    summary = summarize(rankings, queries)

    assert summary == {"queries": 2, "recall@1": 0.5, "recall@5": 0.5}


class FlakyEmbed:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self, client, texts, input_type):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return [[0.0] for _ in texts]


async def test_retries_once_on_rate_limit(monkeypatch):
    fake = FlakyEmbed([EmbeddingError("http 429", retryable=True)])
    monkeypatch.setattr("smriti.embed.embed", fake)

    vectors = await embed_with_retry(None, ["a", "b"], "document", wait=0)

    assert fake.calls == 2
    assert len(vectors) == 2


async def test_gives_up_after_second_rate_limit(monkeypatch):
    fake = FlakyEmbed([EmbeddingError("http 429", retryable=True)] * 2)
    monkeypatch.setattr("smriti.embed.embed", fake)

    with pytest.raises(EmbeddingError):
        await embed_with_retry(None, ["a"], "document", wait=0)
    assert fake.calls == 2


async def test_does_not_retry_permanent_errors(monkeypatch):
    fake = FlakyEmbed([EmbeddingError("http 401", retryable=False)])
    monkeypatch.setattr("smriti.embed.embed", fake)

    with pytest.raises(EmbeddingError):
        await embed_with_retry(None, ["a"], "document", wait=0)
    assert fake.calls == 1
