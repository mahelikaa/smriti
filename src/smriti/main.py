import os
from contextlib import asynccontextmanager
from datetime import UTC
from typing import Literal

import asyncpg
from fastapi import FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, Field

from smriti import embed as embedding
from smriti.db import DEFAULT_DATABASE_URL
from smriti.embed import EMBED_MODEL, EmbeddingError, to_pgvector

DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

DB_UNAVAILABLE_ERRORS = (
    OSError,
    TimeoutError,
    asyncpg.CannotConnectNowError,
    asyncpg.InterfaceError,
)

INSERT_TURN = """
INSERT INTO turns (user_id, external_id, role, text, sent_at)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (user_id, external_id) DO NOTHING
RETURNING id, user_id, external_id, role, text, sent_at
"""
ENQUEUE_EXTRACTION_JOB = """
INSERT INTO extraction_jobs(turn_id)
VALUES($1)
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

SEARCH_MEMORIES = """
SELECT id, kind, content, importance, created_at,
       1 - (embedding <=> $2::vector) AS score
FROM memories
WHERE user_id = $1 AND embedding_model = $3 AND superseded_by IS NULL
ORDER BY embedding <=> $2::vector
LIMIT $4
"""

GET_MEMORY = """
SELECT id, kind, content, importance, created_at, superseded_by
FROM memories
WHERE id = $1 AND user_id = $2
"""

IS_FORGOTTEN = """
SELECT EXISTS (
    SELECT 1 FROM forgotten_memories WHERE memory_id = $1 AND user_id = $2
)
"""

FORGET_MEMORY = """
WITH RECURSIVE chain AS (
    SELECT id FROM memories WHERE id = $1 AND user_id = $2
    UNION
    SELECT m.id FROM memories m JOIN chain c ON m.superseded_by = c.id
),
gone AS (
    DELETE FROM memories WHERE id IN (SELECT id FROM chain)
    RETURNING id, user_id
)
INSERT INTO forgotten_memories (memory_id, user_id)
SELECT id, user_id FROM gone
RETURNING memory_id
"""

TOMBSTONE_USER_MEMORIES = """
INSERT INTO forgotten_memories (memory_id, user_id)
SELECT id, user_id FROM memories WHERE user_id = $1
ON CONFLICT DO NOTHING
"""

DELETE_USER_TURNS = """
DELETE FROM turns WHERE user_id = $1
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=10,
    )
    app.state.embed_client = embedding.make_client()
    yield
    await app.state.embed_client.aclose()
    await app.state.pool.close()


app = FastAPI(lifespan=lifespan)


async def db_unavailable_handler(
    request: Request,
    exc: Exception,
):
    return JSONResponse(
        status_code=503,
        content={"detail": "database unavailable"},
        headers={"Retry-After": "1"},
    )


class DatabaseUnavailable(Exception):
    """The database could not be reached. The only exception mapped to 503."""


@asynccontextmanager
async def db_conn(request: Request):
    # Translate connection-level errors into DatabaseUnavailable ONLY around DB
    # access, so an OSError/TimeoutError from anywhere else surfaces as a 500.
    try:
        async with request.app.state.pool.acquire(timeout=2) as conn:
            yield conn
    except DB_UNAVAILABLE_ERRORS as e:
        raise DatabaseUnavailable() from e


async def embedding_error_handler(request: Request, exc: EmbeddingError):
    if exc.retryable:
        return JSONResponse(
            status_code=503,
            content={"detail": "embedding service unavailable"},
            headers={"Retry-After": "1"},
        )
    return JSONResponse(status_code=502, content={"detail": "embedding service error"})


app.add_exception_handler(DatabaseUnavailable, db_unavailable_handler)
app.add_exception_handler(EmbeddingError, embedding_error_handler)


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

    except (*DB_UNAVAILABLE_ERRORS, asyncpg.PostgresError):
        raise HTTPException(
            status_code=503,
            detail="database unavailable",
        )


Role = Literal["user", "assistant"]


class TurnIn(BaseModel):
    external_id: str = Field(min_length=1, max_length=128)
    role: Role
    text: str = Field(min_length=1, max_length=4000)
    sent_at: AwareDatetime


class TurnOut(BaseModel):
    id: int
    user_id: str
    external_id: str
    role: Role
    text: str
    sent_at: AwareDatetime


class Cursor(BaseModel):
    before_sent_at: str
    before_id: int


class TurnPage(BaseModel):
    items: list[TurnOut]
    next_cursor: Cursor | None


@app.post("/v1/users/{user_id}/turns", response_model=TurnOut)
async def create_turn(
    turn: TurnIn,
    request: Request,
    response: Response,
    user_id: str = Path(min_length=1, max_length=128),
):

    async with db_conn(request) as conn, conn.transaction():
        row = await conn.fetchrow(
            INSERT_TURN,
            user_id,
            turn.external_id,
            turn.role,
            turn.text,
            turn.sent_at,
        )

        if row is not None:
            await conn.execute(ENQUEUE_EXTRACTION_JOB, row["id"])
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


@app.get("/v1/users/{user_id}/turns", response_model=TurnPage)
async def get_turns(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    before_sent_at: AwareDatetime | None = None,
    before_id: int | None = None,
    user_id: str = Path(min_length=1, max_length=128),
):
    if (before_sent_at is None) != (before_id is None):
        raise HTTPException(
            status_code=422,
            detail="before_sent_at and before_id must be provided together",
        )

    async with db_conn(request) as conn:
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
            "before_sent_at": last["sent_at"]
            .astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "before_id": last["id"],
        }
    else:
        next_cursor = None

    return {
        "items": items,
        "next_cursor": next_cursor,
    }


class MemoryHit(BaseModel):
    id: int
    kind: str
    content: str
    importance: int
    created_at: AwareDatetime
    score: float


class MemorySearch(BaseModel):
    items: list[MemoryHit]


@app.get("/v1/users/{user_id}/memories/search", response_model=MemorySearch)
async def search_memories(
    request: Request,
    q: str = Query(min_length=1, max_length=500),
    k: int = Query(5, ge=1, le=20),
    user_id: str = Path(min_length=1, max_length=128),
):
    [query_vector] = await embedding.embed(request.app.state.embed_client, [q], "query")
    async with db_conn(request) as conn:
        rows = await conn.fetch(
            SEARCH_MEMORIES, user_id, to_pgvector(query_vector), EMBED_MODEL, k
        )
    return {"items": [dict(row) for row in rows]}


class MemoryOut(BaseModel):
    id: int
    kind: str
    content: str
    importance: int
    created_at: AwareDatetime
    superseded_by: int | None


def memory_gone() -> HTTPException:
    return HTTPException(status_code=410, detail="memory was forgotten")


def memory_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="memory not found")


@app.get("/v1/users/{user_id}/memories/{memory_id}", response_model=MemoryOut)
async def get_memory(
    request: Request,
    memory_id: int,
    user_id: str = Path(min_length=1, max_length=128),
):
    async with db_conn(request) as conn:
        row = await conn.fetchrow(GET_MEMORY, memory_id, user_id)
        if row is not None:
            return dict(row)
        if await conn.fetchval(IS_FORGOTTEN, memory_id, user_id):
            raise memory_gone()
    raise memory_not_found()


@app.delete("/v1/users/{user_id}/memories/{memory_id}", status_code=204)
async def forget_memory(
    request: Request,
    memory_id: int,
    user_id: str = Path(min_length=1, max_length=128),
):
    async with db_conn(request) as conn, conn.transaction():
        forgotten = await conn.fetch(FORGET_MEMORY, memory_id, user_id)
        if forgotten:
            return Response(status_code=204)
        if await conn.fetchval(IS_FORGOTTEN, memory_id, user_id):
            raise memory_gone()
    raise memory_not_found()


@app.delete("/v1/users/{user_id}", status_code=204)
async def forget_user(
    request: Request,
    user_id: str = Path(min_length=1, max_length=128),
):
    async with db_conn(request) as conn, conn.transaction():
        await conn.execute(TOMBSTONE_USER_MEMORIES, user_id)
        await conn.execute(DELETE_USER_TURNS, user_id)
    return Response(status_code=204)
