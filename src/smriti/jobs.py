import asyncpg

LEASE_SECONDS = 60
MAX_ATTEMPTS = 8
BACKOFF_BASE = 2
BACKOFF_CAP = 300

CLAIM_JOB = """
UPDATE extraction_jobs
SET status = 'running',
    attempts = attempts + 1,
    locked_until = now() + make_interval(secs => $1)
WHERE id = (
    SELECT id FROM extraction_jobs
    WHERE (status = 'pending' AND run_after <= now())
       OR (status = 'running' AND locked_until < now())
    ORDER BY id
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING id, turn_id, attempts
"""

COMPLETE_JOB = """
UPDATE extraction_jobs
SET status = 'done',
    locked_until = NULL,
    last_error = NULL
WHERE id = $1 AND attempts = $2 AND status = 'running'
RETURNING turn_id
"""

DELETE_TURN_MEMORIES = """
DELETE FROM memories WHERE turn_id = $1
"""

INSERT_MEMORY = """
INSERT INTO memories (user_id, turn_id, kind, content, importance)
SELECT user_id, id, $2, $3, $4 FROM turns WHERE id = $1
"""

RETRY_JOB = """
UPDATE extraction_jobs
SET status = 'pending',
    run_after = now() + make_interval(secs => $3),
    locked_until = NULL,
    last_error = $4
WHERE id = $1 AND attempts = $2 AND status = 'running'
"""

KILL_JOB = """
UPDATE extraction_jobs
SET status = 'dead',
    locked_until = NULL,
    last_error = $3
WHERE id = $1 AND attempts = $2 AND status = 'running'
"""


def backoff_delay(attempts: int, jitter: float) -> float:
    return (
        min(
            BACKOFF_CAP,
            BACKOFF_BASE * 2 ** (attempts - 1),
        )
        * jitter
    )


async def claim_job(conn) -> asyncpg.Record | None:
    return await conn.fetchrow(CLAIM_JOB, LEASE_SECONDS)


async def complete_job(
    conn,
    job_id: int,
    attempts: int,
    memories: list[dict],
) -> bool:
    async with conn.transaction():
        row = await conn.fetchrow(COMPLETE_JOB, job_id, attempts)
        if row is None:
            return False

        turn_id = row["turn_id"]
        await conn.execute(DELETE_TURN_MEMORIES, turn_id)
        await conn.executemany(
            INSERT_MEMORY,
            [(turn_id, m["kind"], m["content"], m["importance"]) for m in memories],
        )
        return True


async def fail_job(
    conn,
    job_id: int,
    attempts: int,
    error: str,
    retryable: bool,
    jitter: float,
) -> bool:
    if not retryable or attempts >= MAX_ATTEMPTS:
        result = await conn.execute(KILL_JOB, job_id, attempts, error)
    else:
        delay = backoff_delay(attempts, jitter)
        result = await conn.execute(RETRY_JOB, job_id, attempts, delay, error)
    return result == "UPDATE 1"
