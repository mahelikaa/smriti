import pytest

from smriti.jobs import backoff_delay, claim_job
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
