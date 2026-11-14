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
