import os
import asyncpg
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException, Response, Request
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

turns = {}

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

        if existing is None:
            raise HTTPException(
                status_code=409,
                detail="turn was deleted; it will not be re-created",
            )
        if (existing["role"] == turn.role and existing["text"] == turn.text and existing["sent_at"] == turn.sent_at):
            response.status_code = 200
            return dict(existing)
        
        raise HTTPException(
            status_code=409,
            detail="Conflicting turn with same external_id",
        )
    


@app.get("/v1/users/{user_id}/turns")
def get_turns(user_id: str):
    user_turns = turns.get(user_id, {})

    return sorted(user_turns.values(), key=lambda turn: (turn["sent_at"], turn["id"]))
