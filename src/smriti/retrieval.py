import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
from pydantic import BaseModel

from smriti import embed as embedding
from smriti.embed import EMBED_MODEL, EmbeddingError, to_pgvector
from smriti.main import DATABASE_URL, SEARCH_MEMORIES

ROOT = Path(__file__).resolve().parents[2]
SET_PATH = ROOT / "eval" / "retrieval.json"
RESULTS_PATH = ROOT / "eval" / "retrieval_results.json"
EVAL_USER = "eval-retrieval"
K = 5
RETRY_WAIT_SECONDS = 20.0

INSERT_TURN = """
INSERT INTO turns (user_id, external_id, role, text, sent_at)
VALUES ($1, 'seed', 'user', 'seed', now())
RETURNING id
"""
INSERT_MEMORY = """
INSERT INTO memories (user_id, turn_id, kind, content, importance,
                      embedding, embedding_model)
VALUES ($1, $2, $3, $4, 3, $5::vector, $6)
RETURNING id
"""
MARK_SUPERSEDED = "UPDATE memories SET superseded_by = $1 WHERE id = $2"


class SeedMemory(BaseModel):
    id: str
    kind: str
    content: str
    superseded_by: str | None = None


class Query(BaseModel):
    text: str
    relevant: list[str]


class RetrievalSet(BaseModel):
    memories: list[SeedMemory]
    queries: list[Query]


def load_set(path: Path = SET_PATH) -> RetrievalSet:
    data = RetrievalSet.model_validate_json(path.read_text())
    ids = [m.id for m in data.memories]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate memory id")
    live = {m.id for m in data.memories if m.superseded_by is None}
    for m in data.memories:
        if m.superseded_by is not None and m.superseded_by not in live:
            raise ValueError(f"{m.id}: superseded_by points at {m.superseded_by}")
    for q in data.queries:
        if not q.relevant or not set(q.relevant) <= live:
            raise ValueError(f"bad relevant ids for query {q.text!r}")
    return data


def hit_at(ranked: list[str], relevant: list[str], k: int) -> bool:
    return bool(set(ranked[:k]) & set(relevant))


def summarize(rankings: list[list[str]], queries: list[Query]) -> dict:
    n = len(queries)
    return {
        "queries": n,
        "recall@1": round(
            sum(hit_at(r, q.relevant, 1) for r, q in zip(rankings, queries)) / n, 3
        ),
        f"recall@{K}": round(
            sum(hit_at(r, q.relevant, K) for r, q in zip(rankings, queries)) / n, 3
        ),
    }


async def embed_with_retry(
    client, texts: list[str], input_type: str, wait: float = RETRY_WAIT_SECONDS
) -> list[list[float]]:
    try:
        return await embedding.embed(client, texts, input_type)
    except EmbeddingError as e:
        if not e.retryable:
            raise
        print(f"voyage: {e}, retrying in {wait:.0f}s")
        await asyncio.sleep(wait)
        return await embedding.embed(client, texts, input_type)


async def seed(conn, data: RetrievalSet, vectors: list[list[float]]) -> dict[int, str]:
    turn_id = await conn.fetchval(INSERT_TURN, EVAL_USER)
    db_ids = {}
    for m, v in zip(data.memories, vectors):
        db_ids[m.id] = await conn.fetchval(
            INSERT_MEMORY,
            EVAL_USER,
            turn_id,
            m.kind,
            m.content,
            to_pgvector(v),
            EMBED_MODEL,
        )
    for m in data.memories:
        if m.superseded_by is not None:
            await conn.execute(MARK_SUPERSEDED, db_ids[m.superseded_by], db_ids[m.id])
    return {db_id: key for key, db_id in db_ids.items()}


async def run(data: RetrievalSet) -> list[list[str]]:
    async with embedding.make_client() as client:
        doc_vectors = await embed_with_retry(
            client, [m.content for m in data.memories], "document"
        )
        query_vectors = await embed_with_retry(
            client, [q.text for q in data.queries], "query"
        )

    conn = await asyncpg.connect(DATABASE_URL)
    tx = conn.transaction()
    await tx.start()
    try:
        keys = await seed(conn, data, doc_vectors)
        rankings = []
        for v in query_vectors:
            rows = await conn.fetch(
                SEARCH_MEMORIES, EVAL_USER, to_pgvector(v), EMBED_MODEL, K
            )
            rankings.append([keys[r["id"]] for r in rows])
        return rankings
    finally:
        await tx.rollback()
        await conn.close()


def main() -> None:
    data = load_set()
    rankings = asyncio.run(run(data))
    summary = summarize(rankings, data.queries)
    content = {m.id: m.content for m in data.memories}

    print(
        f"recall@1 {summary['recall@1']:.0%}   recall@{K} {summary[f'recall@{K}']:.0%}"
    )
    print("\nmisses:")
    for q, r in zip(data.queries, rankings):
        if not hit_at(r, q.relevant, K):
            print(f"  {q.text!r}: wanted {q.relevant} | got {r}")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "summary": summary,
                "queries": [
                    {
                        "text": q.text,
                        "relevant": q.relevant,
                        "top": [f"{k}: {content[k]}" for k in r],
                        "rank": next(
                            (i + 1 for i, k in enumerate(r) if k in q.relevant), None
                        ),
                    }
                    for q, r in zip(data.queries, rankings)
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
