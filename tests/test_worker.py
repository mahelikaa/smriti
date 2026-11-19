import asyncio
import logging
from types import SimpleNamespace

import anthropic
import httpx

from smriti import worker
from smriti.extract import Extraction, Memory
from smriti.main import app
from smriti.worker import process_one, run_worker


class FakeMessages:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0.01)  # pretend Claude takes time
        if self.error:
            raise self.error
        text = kwargs["messages"][0]["content"]
        return SimpleNamespace(
            stop_reason="end_turn",
            parsed_output=Extraction(
                memories=[Memory(kind="fact", content=text, importance=3)]
            ),
            usage=SimpleNamespace(input_tokens=120, output_tokens=40),
        )


class FakeClient:
    def __init__(self, error=None):
        self.messages = FakeMessages(error=error)


async def post_turn(client, external_id="msg1", text="main Pune mein rehti hoon"):
    response = await client.post(
        "/v1/users/alice/turns",
        json={
            "external_id": external_id,
            "role": "user",
            "text": text,
            "sent_at": "2026-11-14T10:00:00Z",
        },
    )
    assert response.status_code == 201


async def get_job(conn):
    return await conn.fetchrow("SELECT * FROM extraction_jobs ORDER BY id LIMIT 1")


async def test_empty_queue_returns_false(client):
    assert await process_one(app.state.pool, FakeClient()) is False


async def test_success_saves_memories(client):
    await post_turn(client)

    did_work = await process_one(app.state.pool, FakeClient())

    async with app.state.pool.acquire() as conn:
        job = await get_job(conn)
        rows = await conn.fetch("SELECT * FROM memories")

    assert did_work is True
    assert job["status"] == "done"
    assert len(rows) == 1
    assert rows[0]["content"] == "main Pune mein rehti hoon"


async def test_retryable_error_schedules_retry(client):
    await post_turn(client)
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.RateLimitError(
        "slow down", response=httpx.Response(429, request=req), body=None
    )

    await process_one(app.state.pool, FakeClient(error=error))

    async with app.state.pool.acquire() as conn:
        job = await get_job(conn)

    assert job["status"] == "pending"
    assert "RateLimitError" in job["last_error"]


async def test_non_retryable_error_goes_dead(client):
    await post_turn(client)
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.BadRequestError(
        "bad request", response=httpx.Response(400, request=req), body=None
    )

    await process_one(app.state.pool, FakeClient(error=error))

    async with app.state.pool.acquire() as conn:
        job = await get_job(conn)

    assert job["status"] == "dead"
    assert job["attempts"] == 1
    assert "BadRequestError" in job["last_error"]


async def test_unexpected_crash_goes_dead(client):
    await post_turn(client)

    await process_one(app.state.pool, FakeClient(error=RuntimeError("oops")))

    async with app.state.pool.acquire() as conn:
        job = await get_job(conn)

    assert job["status"] == "dead"
    assert job["last_error"].startswith("bug: RuntimeError")


async def test_two_workers_drain_queue(client):
    for i in range(5):
        await post_turn(client, external_id=f"msg{i}", text=f"turn {i}")

    fake = FakeClient()
    stop = asyncio.Event()
    workers = [
        asyncio.create_task(run_worker(app.state.pool, fake, stop)) for _ in range(2)
    ]

    async with app.state.pool.acquire() as conn:
        for _ in range(100):  # wait upto 5s
            left = await conn.fetchval(
                "SELECT count(*) FROM extraction_jobs WHERE status <> 'done'"
            )
            if left == 0:
                break
            await asyncio.sleep(0.05)

        stop.set()
        await asyncio.wait_for(asyncio.gather(*workers), timeout=2)

        attempts = await conn.fetch("SELECT attempts FROM extraction_jobs")
        contents = await conn.fetch("SELECT content FROM memories ORDER BY content")

    assert left == 0
    assert len(fake.messages.calls) == 5
    assert all(r["attempts"] == 1 for r in attempts)
    assert [r["content"] for r in contents] == [f"turn {i}" for i in range(5)]


async def test_worker_survives_failed_iteration(client, monkeypatch):
    await post_turn(client)
    real_process_one = worker.process_one
    calls = 0

    async def flaky_process_one(pool, fake):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("postgres restarted")
        return await real_process_one(pool, fake)

    monkeypatch.setattr(worker, "process_one", flaky_process_one)
    monkeypatch.setattr(worker, "POLL_SECONDS", 0.01)

    stop = asyncio.Event()
    task = asyncio.create_task(run_worker(app.state.pool, FakeClient(), stop))

    async with app.state.pool.acquire() as conn:
        for _ in range(100):
            job = await get_job(conn)
            if job["status"] == "done":
                break
            await asyncio.sleep(0.05)

    stop.set()
    await asyncio.wait_for(task, timeout=2)

    assert calls >= 2
    assert job["status"] == "done"


async def test_done_job_logs_one_line(client, caplog):
    caplog.set_level(logging.INFO, logger="smriti.worker")
    await post_turn(client)

    await process_one(app.state.pool, FakeClient())

    lines = [r.getMessage() for r in caplog.records if "outcome=done" in r.getMessage()]
    assert len(lines) == 1
    assert "in_tokens=120" in lines[0]
    assert "out_tokens=40" in lines[0]
    assert "memories=1" in lines[0]


async def test_failed_job_logs_retryable(client, caplog):
    caplog.set_level(logging.INFO, logger="smriti.worker")
    await post_turn(client)
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.RateLimitError(
        "slow down", response=httpx.Response(429, request=req), body=None
    )

    await process_one(app.state.pool, FakeClient(error=error))

    assert "outcome=failed" in caplog.text
    assert "retryable=True" in caplog.text
    assert "error=RateLimitError" in caplog.text


async def test_logs_never_contain_turn_text(client, caplog):
    caplog.set_level(logging.INFO, logger="smriti.worker")
    await post_turn(client)

    await process_one(app.state.pool, FakeClient())

    assert caplog.records  # something was logged
    assert "main Pune mein rehti hoon" not in caplog.text


async def test_crash_message_stays_out_of_logs(client, caplog):
    caplog.set_level(logging.DEBUG, logger="smriti.worker")
    await post_turn(client)

    secret = "main Pune mein rehti hoon"
    await process_one(app.state.pool, FakeClient(error=RuntimeError(secret)))

    async with app.state.pool.acquire() as conn:
        job = await get_job(conn)

    assert job["status"] == "dead"
    assert secret in job["last_error"]  # detail kept in the DB
    assert "error=RuntimeError" in caplog.text  # class name in the log
    assert secret not in caplog.text  # but never the user text
