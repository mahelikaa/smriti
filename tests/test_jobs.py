import asyncpg
import pytest

from smriti.jobs import backoff_delay, claim_job, complete_job, fail_job
from smriti.main import app


@pytest.mark.parametrize(
    "attempts, expected",
    [
        (1, 2),
        (2, 4),
        (3, 8),
        (4, 16),
        (5, 32),
    ],
)
def test_backoff_numbers(attempts, expected):
    assert backoff_delay(attempts, jitter=1.0) == expected


def test_backoff_cap():
    assert backoff_delay(attempts=20, jitter=1.0) == 300


def test_backoff_jitter():
    assert backoff_delay(attempts=3, jitter=0.5) == 4.0


async def test_claim_empty_queue(client):
    async with app.state.pool.acquire() as conn:
        job = await claim_job(conn)

    assert job is None


async def test_claim_pending_job(client):
    response = await client.post(
        "/v1/users/alice/turns",
        json={
            "external_id": "msg1",
            "role": "user",
            "text": "Hello",
            "sent_at": "2026-11-14T10:00:00Z",
        },
    )

    assert response.status_code == 201

    async with app.state.pool.acquire() as conn:
        job = await claim_job(conn)
        assert job is not None

        db_job = await conn.fetchrow(
            "SELECT * FROM extraction_jobs WHERE id = $1",
            job["id"],
        )

        lease_is_future = await conn.fetchval(
            "SELECT locked_until > now() FROM extraction_jobs WHERE id = $1",
            job["id"],
        )

    assert job["turn_id"] == response.json()["id"]
    assert job["attempts"] == 1

    assert db_job["status"] == "running"
    assert db_job["attempts"] == 1
    assert lease_is_future is True


async def test_claim_respects_run_after(client):
    response = await client.post(
        "/v1/users/alice/turns",
        json={
            "external_id": "msg1",
            "role": "user",
            "text": "Hello",
            "sent_at": "2026-11-14T10:00:00Z",
        },
    )

    assert response.status_code == 201

    async with app.state.pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE extraction_jobs
            SET run_after = now() + interval '1 hour'
            WHERE turn_id = $1
            """,
            response.json()["id"],
        )

        job = await claim_job(conn)

    assert job is None


async def test_claim_ignores_live_lease(client):
    response = await client.post(
        "/v1/users/alice/turns",
        json={
            "external_id": "msg1",
            "role": "user",
            "text": "Hello",
            "sent_at": "2026-11-14T10:00:00Z",
        },
    )

    assert response.status_code == 201

    async with app.state.pool.acquire() as conn:
        first_claim = await claim_job(conn)
        second_claim = await claim_job(conn)

    assert first_claim is not None
    assert first_claim["attempts"] == 1
    assert second_claim is None


async def test_claim_reclaims_expired_lease(client):
    response = await client.post(
        "/v1/users/alice/turns",
        json={
            "external_id": "msg1",
            "role": "user",
            "text": "Hello",
            "sent_at": "2026-11-14T10:00:00Z",
        },
    )

    assert response.status_code == 201

    async with app.state.pool.acquire() as conn:
        first_claim = await claim_job(conn)

        assert first_claim is not None
        assert first_claim["attempts"] == 1

        # simulate the worker's lease expiring.
        await conn.execute(
            """
            UPDATE extraction_jobs
            SET locked_until = now() - interval '1 second'
            WHERE id = $1
            """,
            first_claim["id"],
        )

        second_claim = await claim_job(conn)

    assert second_claim is not None
    assert second_claim["id"] == first_claim["id"]
    assert second_claim["attempts"] == 2


async def test_two_workers_claim_different_jobs(client):
    for external_id in ["msg1", "msg2"]:
        response = await client.post(
            "/v1/users/alice/turns",
            json={
                "external_id": external_id,
                "role": "user",
                "text": "Hello",
                "sent_at": "2026-11-14T10:00:00Z",
            },
        )

        assert response.status_code == 201

    pool = app.state.pool
    async with (
        pool.acquire() as conn_a,
        pool.acquire() as conn_b,
        conn_a.transaction(),
    ):
        job_a = await claim_job(conn_a)
        await conn_b.execute("SET lock_timeout = '500ms'")
        job_b = await claim_job(conn_b)

    assert job_a is not None
    assert job_b is not None
    assert job_a["id"] != job_b["id"]
    assert job_a["attempts"] == 1
    assert job_b["attempts"] == 1


MEMORIES = [
    {"kind": "fact", "content": "lives in Pune", "importance": 3},
    {"kind": "preference", "content": "hates mornings", "importance": 2},
]


async def create_and_claim(client, external_id="msg1"):
    response = await client.post(
        "/v1/users/alice/turns",
        json={
            "external_id": external_id,
            "role": "user",
            "text": "main Pune mein rehti hoon, subah se nafrat hai",
            "sent_at": "2026-11-14T10:00:00Z",
        },
    )
    assert response.status_code == 201

    async with app.state.pool.acquire() as conn:
        job = await claim_job(conn)

    assert job is not None
    return job


async def test_complete_saves_memories(client):
    job = await create_and_claim(client)

    async with app.state.pool.acquire() as conn:
        ok = await complete_job(conn, job["id"], job["attempts"], MEMORIES)
        db_job = await conn.fetchrow(
            "SELECT * FROM extraction_jobs WHERE id = $1", job["id"]
        )
        rows = await conn.fetch("SELECT * FROM memories ORDER BY id")

    assert ok is True
    assert db_job["status"] == "done"
    assert db_job["locked_until"] is None
    assert len(rows) == 2
    assert all(r["user_id"] == "alice" for r in rows)
    assert all(r["turn_id"] == job["turn_id"] for r in rows)
    assert [r["kind"] for r in rows] == ["fact", "preference"]


async def test_complete_replaces_old_memories(client):
    job = await create_and_claim(client)

    async with app.state.pool.acquire() as conn:
        await complete_job(conn, job["id"], job["attempts"], MEMORIES)

        # simulate a rerun: put the job back to running by hand
        await conn.execute(
            "UPDATE extraction_jobs SET status = 'running' WHERE id = $1",
            job["id"],
        )
        ok = await complete_job(conn, job["id"], job["attempts"], MEMORIES[:1])
        rows = await conn.fetch("SELECT * FROM memories")

    assert ok is True
    assert len(rows) == 1
    assert rows[0]["content"] == "lives in Pune"


async def test_complete_with_stale_token(client):
    job = await create_and_claim(client)

    async with app.state.pool.acquire() as conn:
        # worker 1's lease expires, worker 2 takes the job back (attempts → 2)
        await conn.execute(
            "UPDATE extraction_jobs SET locked_until = now() - interval '1 second'"
            " WHERE id = $1",
            job["id"],
        )
        reclaimed = await claim_job(conn)
        assert reclaimed["attempts"] == 2

        # worker 1 comes back with its old token (attempts=1)
        ok = await complete_job(conn, job["id"], 1, MEMORIES)
        status = await conn.fetchval(
            "SELECT status FROM extraction_jobs WHERE id = $1", job["id"]
        )
        count = await conn.fetchval("SELECT count(*) FROM memories")

    assert ok is False
    assert count == 0
    assert status == "running"


async def test_complete_rolls_back_on_bad_memory(client):
    job = await create_and_claim(client)
    bad = [{"kind": "bogus", "content": "x", "importance": 3}]

    async with app.state.pool.acquire() as conn:
        # the job is marked done first, then the INSERT hits the CHECK on kind
        with pytest.raises(asyncpg.CheckViolationError):
            await complete_job(conn, job["id"], job["attempts"], bad)

        status = await conn.fetchval(
            "SELECT status FROM extraction_jobs WHERE id = $1", job["id"]
        )
        count = await conn.fetchval("SELECT count(*) FROM memories")

    assert status == "running"
    assert count == 0


async def test_fail_retryable(client):
    job = await create_and_claim(client)

    async with app.state.pool.acquire() as conn:
        ok = await fail_job(
            conn,
            job["id"],
            job["attempts"],
            "429 rate limited",
            retryable=True,
            jitter=1.0,
        )
        db_job = await conn.fetchrow(
            """
            SELECT *,
                   run_after > now() AS waiting,
                   run_after <= now() + interval '3 seconds' AS not_too_far
            FROM extraction_jobs WHERE id = $1
            """,
            job["id"],
        )

    assert ok is True
    assert db_job["status"] == "pending"
    assert db_job["waiting"] is True
    assert db_job["not_too_far"] is True  # attempt 1, jitter 1.0 → 2 s
    assert db_job["last_error"] == "429 rate limited"
    assert db_job["locked_until"] is None


async def test_fail_at_max_attempts(client):
    response = await client.post(
        "/v1/users/alice/turns",
        json={
            "external_id": "msg1",
            "role": "user",
            "text": "Hello",
            "sent_at": "2026-11-14T10:00:00Z",
        },
    )
    assert response.status_code == 201

    async with app.state.pool.acquire() as conn:
        # pretend 7 attempts already happened; this claim makes it 8
        await conn.execute(
            "UPDATE extraction_jobs SET attempts = 7 WHERE turn_id = $1",
            response.json()["id"],
        )
        job = await claim_job(conn)
        assert job["attempts"] == 8

        ok = await fail_job(
            conn,
            job["id"],
            job["attempts"],
            "529 overloaded",
            retryable=True,
            jitter=1.0,
        )
        status = await conn.fetchval(
            "SELECT status FROM extraction_jobs WHERE id = $1", job["id"]
        )

    assert ok is True
    assert status == "dead"


async def test_fail_not_retryable(client):
    job = await create_and_claim(client)

    async with app.state.pool.acquire() as conn:
        ok = await fail_job(
            conn,
            job["id"],
            job["attempts"],
            "400 invalid request",
            retryable=False,
            jitter=1.0,
        )
        status = await conn.fetchval(
            "SELECT status FROM extraction_jobs WHERE id = $1", job["id"]
        )

    assert ok is True
    assert job["attempts"] == 1
    assert status == "dead"


async def test_fail_with_stale_token(client):
    job = await create_and_claim(client)

    async with app.state.pool.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET locked_until = now() - interval '1 second'"
            " WHERE id = $1",
            job["id"],
        )
        reclaimed = await claim_job(conn)
        assert reclaimed["attempts"] == 2

        ok = await fail_job(conn, job["id"], 1, "timeout", retryable=True, jitter=1.0)
        db_job = await conn.fetchrow(
            "SELECT status, last_error FROM extraction_jobs WHERE id = $1",
            job["id"],
        )

    assert ok is False
    assert db_job["status"] == "running"
    assert db_job["last_error"] is None


async def test_exhausted_expired_job_goes_dead(client):
    job = await create_and_claim(client)

    async with app.state.pool.acquire() as conn:
        # the worker process died on its 8th attempt: lease expired, no fail_job
        await conn.execute(
            """
            UPDATE extraction_jobs
            SET attempts = 8, locked_until = now() - interval '1 second'
            WHERE id = $1
            """,
            job["id"],
        )
        reclaimed = await claim_job(conn)
        db_job = await conn.fetchrow(
            "SELECT status, last_error, locked_until FROM extraction_jobs"
            " WHERE id = $1",
            job["id"],
        )

    assert reclaimed is None
    assert db_job["status"] == "dead"
    assert db_job["last_error"] == "lease expired after max attempts"
    assert db_job["locked_until"] is None
