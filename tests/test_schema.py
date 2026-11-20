import asyncpg
import pytest

from smriti.main import app

VEC_1024 = "[" + ",".join(["0.1"] * 1024) + "]"
VEC_512 = "[" + ",".join(["0.1"] * 512) + "]"

INSERT = """
INSERT INTO memories (user_id, turn_id, kind, content, importance,
                      embedding, embedding_model)
VALUES ('alice', $1, 'fact', 'Lives in Pune', 3, $2::vector, $3)
"""


async def make_turn(conn) -> int:
    return await conn.fetchval(
        """INSERT INTO turns (user_id, external_id, role, text, sent_at)
           VALUES ('alice', 'm1', 'user', 'hi', now()) RETURNING id"""
    )


async def test_memory_with_vector_and_model_is_saved(client):
    async with app.state.pool.acquire() as conn:
        turn_id = await make_turn(conn)
        await conn.execute(INSERT, turn_id, VEC_1024, "voyage-4")

        dims = await conn.fetchval("SELECT vector_dims(embedding) FROM memories")

    assert dims == 1024


async def test_memory_without_vector_is_allowed(client):
    async with app.state.pool.acquire() as conn:
        turn_id = await make_turn(conn)
        await conn.execute(INSERT, turn_id, None, None)


@pytest.mark.parametrize("vec, model", [(VEC_1024, None), (None, "voyage-4")])
async def test_vector_and_model_must_come_together(client, vec, model):
    async with app.state.pool.acquire() as conn:
        turn_id = await make_turn(conn)
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(INSERT, turn_id, vec, model)


async def test_wrong_size_vector_is_rejected(client):
    async with app.state.pool.acquire() as conn:
        turn_id = await make_turn(conn)
        with pytest.raises(asyncpg.DataError):
            await conn.execute(INSERT, turn_id, VEC_512, "voyage-4")
