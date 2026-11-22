import asyncio
import os

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from smriti.db import DEFAULT_DATABASE_URL, recreate_database, with_database

TEST_DB = "smriti_test"
DEV_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

os.environ["DATABASE_URL"] = with_database(DEV_URL, TEST_DB)
os.environ["VOYAGE_API_KEY"] = "test-key"


@pytest.fixture(scope="session", autouse=True)
def test_database():
    asyncio.run(recreate_database(DEV_URL, TEST_DB))


@pytest_asyncio.fixture
async def client():
    from smriti.main import app

    async with LifespanManager(app):
        async with app.state.pool.acquire() as conn:
            await conn.execute(
                "TRUNCATE turns, extraction_jobs, memories, forgotten_memories "
                "RESTART IDENTITY"
            )

        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as c:
            yield c
