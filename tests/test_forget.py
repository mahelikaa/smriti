from fakes import fake_embedder

from smriti.main import app


async def add_memory(user_id, content, superseded_by=None):
    async with app.state.pool.acquire() as conn:
        turn_id = await conn.fetchval(
            """INSERT INTO turns (user_id, external_id, role, text, sent_at)
               VALUES ($1, $2, 'user', 'hi', now()) RETURNING id""",
            user_id,
            content,
        )
        await conn.execute(
            "INSERT INTO extraction_jobs (turn_id, status) VALUES ($1, 'done')",
            turn_id,
        )
        return await conn.fetchval(
            """INSERT INTO memories (user_id, turn_id, kind, content, importance,
                                     superseded_by)
               VALUES ($1, $2, 'fact', $3, 3, $4) RETURNING id""",
            user_id,
            turn_id,
            content,
            superseded_by,
        )


async def count(sql, *args):
    async with app.state.pool.acquire() as conn:
        return await conn.fetchval(sql, *args)


async def test_get_memory(client):
    memory_id = await add_memory("alice", "Lives in Pune")

    response = await client.get(f"/v1/users/alice/memories/{memory_id}")

    assert response.status_code == 200
    assert response.json()["content"] == "Lives in Pune"


async def test_forget_deletes_row_and_vector(client):
    memory_id = await add_memory("alice", "Lives in Pune")

    response = await client.delete(f"/v1/users/alice/memories/{memory_id}")

    assert response.status_code == 204
    assert await count("SELECT count(*) FROM memories WHERE id = $1", memory_id) == 0


async def test_forgotten_memory_is_410_not_404(client):
    memory_id = await add_memory("alice", "Lives in Pune")
    await client.delete(f"/v1/users/alice/memories/{memory_id}")

    get = await client.get(f"/v1/users/alice/memories/{memory_id}")
    again = await client.delete(f"/v1/users/alice/memories/{memory_id}")
    never = await client.get("/v1/users/alice/memories/999")

    assert get.status_code == 410
    assert again.status_code == 410
    assert never.status_code == 404


async def test_cannot_forget_another_users_memory(client):
    bob_memory = await add_memory("bob", "Lives in Delhi")

    response = await client.delete(f"/v1/users/alice/memories/{bob_memory}")

    assert response.status_code == 404
    assert await count("SELECT count(*) FROM memories WHERE id = $1", bob_memory) == 1


async def test_forgetting_a_fact_also_forgets_its_old_versions(client):
    new_id = await add_memory("alice", "Lives in Bangalore")
    old_id = await add_memory("alice", "Lives in Gurgaon", superseded_by=new_id)
    oldest_id = await add_memory("alice", "Lives in Pune", superseded_by=old_id)

    response = await client.delete(f"/v1/users/alice/memories/{new_id}")

    assert response.status_code == 204
    assert await count("SELECT count(*) FROM memories WHERE user_id = 'alice'") == 0
    for memory_id in (old_id, oldest_id):
        gone = await client.get(f"/v1/users/alice/memories/{memory_id}")
        assert gone.status_code == 410


async def test_forget_user_removes_everything_for_that_user_only(client):
    alice_memory = await add_memory("alice", "Lives in Pune")
    await add_memory("bob", "Lives in Delhi")

    response = await client.delete("/v1/users/alice")

    assert response.status_code == 204
    for table in ("turns", "memories"):
        sql = f"SELECT count(*) FROM {table} WHERE user_id = $1"
        assert await count(sql, "alice") == 0
        assert await count(sql, "bob") == 1
    assert await count("SELECT count(*) FROM extraction_jobs") == 1
    gone = await client.get(f"/v1/users/alice/memories/{alice_memory}")
    assert gone.status_code == 410


async def test_forgotten_memory_never_shows_in_search(client, monkeypatch):
    monkeypatch.setattr(app.state, "embed_client", fake_embedder())
    memory_id = await add_memory("alice", "Lives in Pune")
    await client.delete(f"/v1/users/alice/memories/{memory_id}")

    response = await client.get(
        "/v1/users/alice/memories/search", params={"q": "where do I live"}
    )

    assert response.json()["items"] == []
