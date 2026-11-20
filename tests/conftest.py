import os

import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from smriti.main import app

os.environ["VOYAGE_API_KEY"] = "test-key"


@pytest_asyncio.fixture
async def client():
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
