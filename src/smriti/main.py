import os
import asyncpg
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException, Response, Request, Query
from pydantic import AwareDatetime, BaseModel, Field

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://smriti:smriti@localhost:5433/smriti",
)

INSERT_TURN = """
INSERT INTO turns (user_id, external_id, role, text, sent_at)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (user_id, external_id) DO NOTHING
RETURNING id, user_id, external_id, role, text, sent_at
"""
SELECT_TURN = """
SELECT id, user_id, external_id, role, text, sent_at
FROM turns
WHERE user_id = $1 AND external_id = $2
"""
LIST_TURNS_FIRST_PAGE = """
SELECT id, user_id, external_id, role, text, sent_at
FROM turns
WHERE user_id = $1
ORDER BY sent_at DESC, id DESC
LIMIT $2
"""

LIST_TURNS_AFTER_CURSOR = """
SELECT id, user_id, external_id, role, text, sent_at
FROM turns
WHERE user_id = $1
AND (sent_at, id) < ($2, $3)
ORDER BY sent_at DESC, id DESC
LIMIT $4
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=10,
    )
    yield
    await app.state.pool.close()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request):
    try:
        pool = request.app.state.pool

        async with pool.acquire(timeout=2) as conn:
            await conn.fetchval("SELECT 1", timeout=2)

        return {"status": "ready"}

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="database unavailable",
        )


class TurnIn(BaseModel):
    external_id: str = Field(min_length=1, max_length=128)
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=4000)
    sent_at: AwareDatetime


@app.post("/v1/users/{user_id}/turns")
async def create_turn(user_id: str, turn: TurnIn, request: Request, response: Response):

    pool = request.app.state.pool
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            INSERT_TURN,
            user_id,
            turn.external_id,
            turn.role,
            turn.text,
            turn.sent_at,
        )

        if row is not None:
            response.status_code = 201
            return dict(row)

        existing = await conn.fetchrow(
            SELECT_TURN,
            user_id,
            turn.external_id,
        )

        # Return 409 instead of recreating a turn that another request deleted
        if existing is None:
            raise HTTPException(
                status_code=409,
                detail="turn was deleted; it will not be re-created",
            )
        if (
            existing["role"] == turn.role
            and existing["text"] == turn.text
            and existing["sent_at"] == turn.sent_at
        ):
            response.status_code = 200
            return dict(existing)

        raise HTTPException(
            status_code=409,
            detail="Conflicting turn with same external_id",
        )


@app.get("/v1/users/{user_id}/turns")
async def get_turns(
    user_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    before_sent_at: AwareDatetime | None = None,
    before_id: int | None = None,
):
    if (before_sent_at is None) != (before_id is None):
        raise HTTPException(
            status_code=422,
            detail="before_sent_at and before_id must be provided together",
        )

    pool = request.app.state.pool
    async with pool.acquire() as conn:
        if before_sent_at is None:
            rows = await conn.fetch(
                LIST_TURNS_FIRST_PAGE,
                user_id,
                limit + 1,
            )
        else:
            rows = await conn.fetch(
                LIST_TURNS_AFTER_CURSOR,
                user_id,
                before_sent_at,
                before_id,
                limit + 1,
            )
    has_more = len(rows) > limit
    items = [dict(row) for row in rows[:limit]]

    if has_more:
        last = items[-1]
        next_cursor = {
            "before_sent_at": last["sent_at"].isoformat().replace("+00:00", "Z"),
            "before_id": last["id"],
        }
    else:
        next_cursor = None

    return {
        "items": items,
        "next_cursor": next_cursor,
    }
