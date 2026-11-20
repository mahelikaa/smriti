import pytest
from fakes import fake_embedder

from smriti.backfill import backfill
from smriti.embed import EMBED_DIMS, EMBED_MODEL, EmbeddingError, to_pgvector
from smriti.main import app


async def add_memories(contents, embedding=None, model=None):
    async with app.state.pool.acquire() as conn:
        turn_id = await conn.fetchval(
            """INSERT INTO turns (user_id, external_id, role, text, sent_at)
               VALUES ('alice', $1, 'user', 'hi', now()) RETURNING id""",
            f"m{len(contents)}-{model}",
        )
        await conn.executemany(
            """INSERT INTO memories (user_id, turn_id, kind, content, importance,
                                     embedding, embedding_model)
               VALUES ('alice', $1, 'fact', $2, 3, $3::vector, $4)""",
            [(turn_id, c, to_pgvector(embedding), model) for c in contents],
        )


async def embedded_rows():
    async with app.state.pool.acquire() as conn:
        return await conn.fetch(
            """SELECT content, embedding_model, vector_dims(embedding) AS dims
               FROM memories ORDER BY id"""
        )


async def test_fills_missing_vectors(client):
    await add_memories(["Lives in Pune", "Hates mornings", "Sister is Riya"])

    done = await backfill(app.state.pool, fake_embedder())

    rows = await embedded_rows()
    assert done == 3
    assert all(r["embedding_model"] == EMBED_MODEL for r in rows)
    assert all(r["dims"] == EMBED_DIMS for r in rows)


async def test_second_run_does_nothing(client):
    await add_memories(["Lives in Pune"])
    await backfill(app.state.pool, fake_embedder())
    calls = []

    done = await backfill(app.state.pool, fake_embedder(calls=calls))

    assert done == 0
    assert calls == []


async def test_works_in_batches(client):
    await add_memories([f"memory {i}" for i in range(5)])
    calls = []

    done = await backfill(app.state.pool, fake_embedder(calls=calls), batch_size=2)

    assert done == 5
    assert [len(c["input"]) for c in calls] == [2, 2, 1]
    assert all(c["input_type"] == "document" for c in calls)


async def test_reembeds_old_model_and_skips_current(client):
    await add_memories(["old one"], embedding=[0.2] * EMBED_DIMS, model="old-model")
    await add_memories(["current"], embedding=[0.3] * EMBED_DIMS, model=EMBED_MODEL)
    calls = []

    done = await backfill(app.state.pool, fake_embedder(calls=calls))

    rows = await embedded_rows()
    assert done == 1
    assert calls[0]["input"] == ["old one"]
    assert [r["embedding_model"] for r in rows] == [EMBED_MODEL, EMBED_MODEL]


async def test_embed_failure_stops_and_leaves_rows_unchanged(client):
    await add_memories(["Lives in Pune"])

    with pytest.raises(EmbeddingError):
        await backfill(app.state.pool, fake_embedder(status=503))

    rows = await embedded_rows()
    assert rows[0]["embedding_model"] is None
