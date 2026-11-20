import asyncio
import logging
import random
import signal
import time

import asyncpg

from smriti import embed as embedding
from smriti.embed import EMBED_MODEL, EmbeddingError
from smriti.extract import ExtractionError, cost_usd, extract_memories, make_client
from smriti.jobs import claim_job, complete_job, fail_job
from smriti.main import DATABASE_URL

log = logging.getLogger("smriti.worker")

POLL_SECONDS = 1.0

GET_TURN_TEXT = """
SELECT text FROM turns WHERE id = $1
"""


def ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


async def process_one(pool, client, embed_client) -> bool:
    """Handle one job. Returns False if the queue was empty."""
    async with pool.acquire() as conn:
        job = await claim_job(conn)
        if job is None:
            return False
        text = await conn.fetchval(GET_TURN_TEXT, job["turn_id"])

    stage = "extract"
    claude_ms = embed_ms = 0.0
    in_tokens = out_tokens = 0
    start = time.perf_counter()
    try:
        result = await extract_memories(client, text)
        claude_ms = ms_since(start)
        in_tokens, out_tokens = result.input_tokens, result.output_tokens

        stage = "embed"
        start = time.perf_counter()
        contents = [m["content"] for m in result.memories]
        vectors = await embedding.embed(embed_client, contents, "document")
        embed_ms = ms_since(start)
    except (ExtractionError, EmbeddingError) as e:
        error, retryable = f"{stage}: {e}", e.retryable
        error_type = type(e.__cause__ or e).__name__
    except Exception as e:  # noqa: BLE001 - recorded via fail_job + log line
        # no log.exception: its message could contain user text. The class name
        # goes in the log line below; the full message is kept in last_error (DB).
        error, retryable = f"bug: {type(e).__name__}: {e}", False
        error_type = type(e).__name__
    else:
        memories = [
            {**m, "embedding": v, "embedding_model": EMBED_MODEL}
            for m, v in zip(result.memories, vectors, strict=True)
        ]
        async with pool.acquire() as conn:
            saved = await complete_job(conn, job["id"], job["attempts"], memories)
        log.info(
            "job=%s outcome=%s attempts=%s claude_ms=%.0f embed_ms=%.0f "
            "in_tokens=%s out_tokens=%s cost_usd=%.6f memories=%s",
            job["id"],
            "done" if saved else "lost",
            job["attempts"],
            claude_ms,
            embed_ms,
            in_tokens,
            out_tokens,
            cost_usd(in_tokens, out_tokens),
            len(memories),
        )
        return True

    if stage == "extract":
        claude_ms = ms_since(start)
    else:
        embed_ms = ms_since(start)
    async with pool.acquire() as conn:
        await fail_job(
            conn,
            job["id"],
            job["attempts"],
            error,
            retryable=retryable,
            jitter=random.uniform(0.5, 1.0),
        )
    log.info(
        "job=%s outcome=failed stage=%s retryable=%s error=%s attempts=%s "
        "claude_ms=%.0f embed_ms=%.0f in_tokens=%s out_tokens=%s cost_usd=%.6f "
        "memories=0",
        job["id"],
        stage,
        retryable,
        error_type,
        job["attempts"],
        claude_ms,
        embed_ms,
        in_tokens,
        out_tokens,
        cost_usd(in_tokens, out_tokens),
    )
    return True


async def run_worker(pool, client, embed_client, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            did_work = await process_one(pool, client, embed_client)
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
    embed_client = embedding.make_client()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)  # Ctrl+C / deploy: finish job, exit
    try:
        await asyncio.gather(
            *(run_worker(pool, client, embed_client, stop) for _ in range(concurrency))
        )
    finally:
        await embed_client.aclose()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
