import asyncpg
import pytest

from smriti import db
from smriti.db import with_database
from smriti.main import DATABASE_URL


def test_with_database_only_swaps_the_database():
    url = "postgresql://smriti:smriti@localhost:5433/smriti"

    assert with_database(url, "smriti_test") == (
        "postgresql://smriti:smriti@localhost:5433/smriti_test"
    )


def test_tests_never_touch_the_dev_database():
    assert DATABASE_URL.endswith("/smriti_test")


async def test_pool_retries_until_the_database_is_up(monkeypatch):
    calls, sleeps = [], []
    sentinel = object()

    async def flaky_create_pool(url, **kwargs):
        calls.append(url)
        if len(calls) < 3:
            raise ConnectionRefusedError
        return sentinel

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(db.asyncpg, "create_pool", flaky_create_pool)
    monkeypatch.setattr(db.asyncio, "sleep", fake_sleep)

    pool = await db.create_pool_with_retry("postgresql://x/y", attempts=5)

    assert pool is sentinel
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]


async def test_pool_gives_up_with_a_clear_error(monkeypatch):
    sleeps = []

    async def down(url, **kwargs):
        raise ConnectionRefusedError

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(db.asyncpg, "create_pool", down)
    monkeypatch.setattr(db.asyncio, "sleep", fake_sleep)

    with pytest.raises(db.DatabaseUnavailable, match="after 6 attempts"):
        await db.create_pool_with_retry("postgresql://x/y", attempts=6)

    assert sleeps == [0.5, 1.0, 2.0, 4.0, 4.0]


async def test_pool_does_not_retry_a_wrong_password(monkeypatch):
    calls = []

    async def bad_password(url, **kwargs):
        calls.append(url)
        raise asyncpg.InvalidPasswordError("password authentication failed")

    monkeypatch.setattr(db.asyncpg, "create_pool", bad_password)

    with pytest.raises(asyncpg.InvalidPasswordError):
        await db.create_pool_with_retry("postgresql://x/y")

    assert len(calls) == 1
