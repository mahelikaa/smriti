import os
from typing import Literal

import httpx

EMBED_MODEL = "voyage-4"
EMBED_DIMS = 1024
VOYAGE_BASE_URL = "https://ai.mongodb.com/v1"
TIMEOUT_SECONDS = 10.0

InputType = Literal["query", "document"]


class EmbeddingError(Exception):
    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def to_pgvector(v: list[float] | None) -> str | None:
    if v is None:
        return None
    return "[" + ",".join(map(str, v)) + "]"


def make_client(transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=VOYAGE_BASE_URL,
        headers={"Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}"},
        timeout=TIMEOUT_SECONDS,
        transport=transport,
    )


async def embed(
    client: httpx.AsyncClient, texts: list[str], input_type: InputType
) -> list[list[float]]:
    if not texts:
        return []

    try:
        resp = await client.post(
            "/embeddings",
            json={"input": texts, "model": EMBED_MODEL, "input_type": input_type},
        )
    except httpx.TransportError as e:
        raise EmbeddingError(f"transport: {type(e).__name__}", retryable=True) from e

    if resp.status_code == 429 or resp.status_code >= 500:
        raise EmbeddingError(f"http {resp.status_code}", retryable=True)
    if resp.status_code != 200:
        raise EmbeddingError(f"http {resp.status_code}", retryable=False)

    try:
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        vectors = [d["embedding"] for d in data]
    except (ValueError, KeyError, TypeError) as e:
        raise EmbeddingError("malformed response body", retryable=False) from e

    if len(vectors) != len(texts) or any(len(v) != EMBED_DIMS for v in vectors):
        raise EmbeddingError("bad response shape", retryable=False)
    return vectors
