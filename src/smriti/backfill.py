import asyncio

from smriti import embed as embedding
from smriti.db import create_pool_with_retry
from smriti.embed import EMBED_MODEL, to_pgvector
from smriti.main import DATABASE_URL

BATCH_SIZE = 64

SELECT_STALE = """
SELECT id, content FROM memories
WHERE id > $1 AND (embedding IS NULL OR embedding_model <> $2)
ORDER BY id
LIMIT $3
"""

SET_EMBEDDING = """
UPDATE memories
SET embedding = $2::vector, embedding_model = $3
WHERE id = $1 AND (embedding IS NULL OR embedding_model <> $3)
"""


async def backfill(pool, embed_client, batch_size: int = BATCH_SIZE) -> int:
    done = 0
    last_id = 0
    while True:
        async with pool.acquire() as conn:
            rows = await conn.fetch(SELECT_STALE, last_id, EMBED_MODEL, batch_size)
        if not rows:
            return done

        contents = [r["content"] for r in rows]
        vectors = await embedding.embed(embed_client, contents, "document")

        async with pool.acquire() as conn:
            await conn.executemany(
                SET_EMBEDDING,
                [
                    (r["id"], to_pgvector(v), EMBED_MODEL)
                    for r, v in zip(rows, vectors, strict=True)
                ],
            )
        done += len(rows)
        last_id = rows[-1]["id"]


async def main() -> None:
    pool = await create_pool_with_retry(DATABASE_URL, min_size=1, max_size=2)
    embed_client = embedding.make_client()
    try:
        done = await backfill(pool, embed_client)
        print(f"embedded {done} memories with {EMBED_MODEL}")
    finally:
        await embed_client.aclose()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
