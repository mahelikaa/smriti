import httpx

from smriti.embed import EmbeddingError

RERANK_MODEL = "rerank-3"
CANDIDATES = 20


class RerankError(EmbeddingError):
    pass


async def rerank(
    client: httpx.AsyncClient,
    query: str,
    documents: list[str],
    top_k: int,
    model: str = RERANK_MODEL,
) -> list[tuple[int, float]]:
    if not documents:
        return []

    try:
        resp = await client.post(
            "/rerank",
            json={
                "query": query,
                "documents": documents,
                "model": model,
                "top_k": top_k,
            },
        )
    except httpx.TransportError as e:
        raise RerankError(f"transport: {type(e).__name__}", retryable=True) from e

    if resp.status_code == 429 or resp.status_code >= 500:
        raise RerankError(f"http {resp.status_code}", retryable=True)
    if resp.status_code != 200:
        raise RerankError(f"http {resp.status_code}", retryable=False)

    try:
        order = [(d["index"], d["relevance_score"]) for d in resp.json()["data"]]
    except (ValueError, KeyError, TypeError) as e:
        raise RerankError("malformed response body", retryable=False) from e

    if len(order) > top_k or any(not 0 <= i < len(documents) for i, _ in order):
        raise RerankError("bad response shape", retryable=False)
    return order
