import asyncio
import logging
import random
import signal

import asyncpg

from smriti.extract import ExtractionError, extract_memories, make_client
from smriti.jobs import claim_job, complete_job, fail_job
from smriti.main import DATABASE_URL

log = logging.getLogger("smriti.worker")

POLL_SECONDS = 1.0

GET_TURN_TEXT = """
SELECT text FROM turns WHERE id = $1
"""


async def process_one(pool, client) -> bool:
    """Handle one job. Returns False if the queue was empty."""
    async with pool.acquire() as conn:
        job = await claim_job(conn)
        if job is None:
            return False
        text = await conn.fetchval(GET_TURN_TEXT, job["turn_id"])
    # connection is back in the pool during the slow Claude call

    try:
        memories = await extract_memories(client, text)
    except ExtractionError as e:
        error, retryable = str(e), e.retryable
    except Exception as e:
        log.exception("job %s crashed", job["id"])
        error, retryable = f"bug: {type(e).__name__}: {e}", False
    else:
        async with pool.acquire() as conn:
            saved = await complete_job(conn, job["id"], job["attempts"], memories)
        if not saved:
            log.warning("job %s: lost the job before saving", job["id"])
        return True

    async with pool.acquire() as conn:
        await fail_job(
            conn,
            job["id"],
            job["attempts"],
            error,
            retryable=retryable,
            jitter=random.uniform(0.5, 1.0),
        )
    return True


async def run_worker(pool, client, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            did_work = await process_one(pool, client)
        except Exception:
            # e.g. Postgres restarted mid-claim: log, back off, keep the worker alive
            log.exception("worker iteration failed")
            did_work = False
        if not did_work:
            try:
                await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
            except TimeoutError:
                pass


async def main(concurrency: int = 4) -> None:
    logging.basicConfig(level=logging.INFO)
    # each worker holds at most one connection at a time, plus one spare
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=concurrency + 1)
    client = make_client()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)  # Ctrl+C / deploy: finish job, exit
    try:
        await asyncio.gather(
            *(run_worker(pool, client, stop) for _ in range(concurrency))
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
