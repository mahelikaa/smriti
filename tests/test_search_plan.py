from smriti.embed import EMBED_DIMS, EMBED_MODEL, to_pgvector
from smriti.main import SEARCH_MEMORIES, app


async def test_search_can_use_user_index(client):
    query_vector = to_pgvector([0.1] * EMBED_DIMS)
    async with app.state.pool.acquire() as conn, conn.transaction():
        await conn.execute("SET LOCAL enable_seqscan = off")
        rows = await conn.fetch(
            "EXPLAIN " + SEARCH_MEMORIES, "alice", query_vector, EMBED_MODEL, 5
        )

    plan = "\n".join(row["QUERY PLAN"] for row in rows)
    assert "memories_user_model_idx" in plan
    assert "Seq Scan" not in plan
