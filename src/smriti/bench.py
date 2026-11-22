import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg

from smriti.embed import EMBED_DIMS, EMBED_MODEL, to_pgvector
from smriti.main import DATABASE_URL, SEARCH_MEMORIES

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = ROOT / "migrations"
RESULTS_PATH = ROOT / "eval" / "search_latency.json"
BENCH_DB = "smriti_bench"
HEAVY_USER = "bench-heavy"
TYPICAL_USER = "bench-0"
MEMORIES_PER_USER = 100
HEAVY_SHARE = 0.1
K = 5
WARMUP = 20
QUERIES = 500

LOAD = """
WITH owners AS (
    SELECT i,
           CASE WHEN i <= $2 THEN 'bench-heavy'
                ELSE 'bench-' || ((i - $2 - 1) / $3) END AS user_id
    FROM generate_series(1, $1) AS i
), users AS (
    INSERT INTO turns (user_id, external_id, role, text, sent_at)
    SELECT DISTINCT user_id, 'bench', 'user', 'bench', now() FROM owners
    RETURNING id, user_id
)
INSERT INTO memories (user_id, turn_id, kind, content, importance,
                      embedding, embedding_model)
SELECT o.user_id, u.id, 'fact', 'bench memory ' || o.i, 3,
       (SELECT array_agg(random() * 2 - 1)::vector
        FROM generate_series(1, $4) WHERE o.i > 0),
       $5
FROM owners o JOIN users u USING (user_id)
"""

CREATE_HNSW = """
CREATE INDEX memories_embedding_hnsw_idx
    ON memories USING hnsw (embedding vector_cosine_ops)
"""

CONFIGS = {
    "exact": [],
    "hnsw": ["SET enable_sort = off"],
    "hnsw_iterative": [
        "SET enable_sort = off",
        "SET hnsw.iterative_scan = relaxed_order",
    ],
}


def bench_url(url: str, db: str = BENCH_DB) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{db}"))


def summarize(latencies_ms: list[float]) -> dict:
    cuts = statistics.quantiles(latencies_ms, n=100, method="inclusive")
    return {
        "p50_ms": round(cuts[49], 2),
        "p99_ms": round(cuts[98], 2),
    }


def recall(got: list[list[int]], exact: list[list[int]]) -> float:
    found = sum(len(set(g) & set(e)) for g, e in zip(got, exact, strict=True))
    return round(found / sum(len(e) for e in exact), 3)


def random_queries(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [
        to_pgvector([rng.uniform(-1, 1) for _ in range(EMBED_DIMS)]) for _ in range(n)
    ]


async def load(conn, total: int, heavy: int, per_user: int = MEMORIES_PER_USER):
    await conn.execute(LOAD, total, heavy, per_user, EMBED_DIMS, EMBED_MODEL)


async def plan_for(conn, user_id: str, query: str) -> str:
    rows = await conn.fetch(
        "EXPLAIN " + SEARCH_MEMORIES, user_id, query, EMBED_MODEL, K
    )
    plan = "\n".join(r["QUERY PLAN"] for r in rows)
    if "hnsw" in plan:
        return "hnsw"
    if "memories_user_model_idx" in plan:
        return "user index + sort"
    return "seq scan"


async def time_searches(conn, user_id: str, queries: list[str]):
    for q in queries[:WARMUP]:
        await conn.fetch(SEARCH_MEMORIES, user_id, q, EMBED_MODEL, K)
    latencies, results = [], []
    for q in queries:
        start = time.perf_counter()
        rows = await conn.fetch(SEARCH_MEMORIES, user_id, q, EMBED_MODEL, K)
        latencies.append((time.perf_counter() - start) * 1000)
        results.append([r["id"] for r in rows])
    return latencies, results


async def measure_size(conn, total: int, queries: list[str]) -> dict:
    await conn.execute(
        "TRUNCATE turns, extraction_jobs, memories RESTART IDENTITY CASCADE"
    )
    heavy = int(total * HEAVY_SHARE)
    start = time.perf_counter()
    await load(conn, total, heavy)
    await conn.execute("ANALYZE memories")
    report = {
        "memories": total,
        "users": {HEAVY_USER: heavy, TYPICAL_USER: MEMORIES_PER_USER},
        "load_s": round(time.perf_counter() - start, 1),
    }

    exact = {}
    for user_id in (TYPICAL_USER, HEAVY_USER):
        latencies, exact[user_id] = await time_searches(conn, user_id, queries)
        report[f"exact/{user_id}"] = summarize(latencies)

    start = time.perf_counter()
    await conn.execute("SET maintenance_work_mem = '1GB'")
    await conn.execute("SET max_parallel_maintenance_workers = 0")
    await conn.execute(CREATE_HNSW)
    await conn.execute("ANALYZE memories")
    report["hnsw_build_s"] = round(time.perf_counter() - start, 1)

    for user_id in (TYPICAL_USER, HEAVY_USER):
        report[f"planner_picks/{user_id}"] = await plan_for(conn, user_id, queries[0])

    for name in ("hnsw", "hnsw_iterative"):
        for setting in CONFIGS[name]:
            await conn.execute(setting)
        for user_id in (TYPICAL_USER, HEAVY_USER):
            latencies, got = await time_searches(conn, user_id, queries)
            report[f"{name}/{user_id}"] = {
                **summarize(latencies),
                "rows_returned": round(statistics.mean(map(len, got)), 2),
                f"recall@{K}": recall(got, exact[user_id]),
            }
        await conn.execute("RESET ALL")

    await conn.execute("DROP INDEX memories_embedding_hnsw_idx")
    return report


async def apply_migrations(conn) -> None:
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        await conn.execute(path.read_text())


async def run(sizes: list[int]) -> list[dict]:
    admin = await asyncpg.connect(DATABASE_URL)
    await admin.execute(f"DROP DATABASE IF EXISTS {BENCH_DB} WITH (FORCE)")
    await admin.execute(f"CREATE DATABASE {BENCH_DB}")
    try:
        conn = await asyncpg.connect(bench_url(DATABASE_URL))
        try:
            await apply_migrations(conn)
            queries = random_queries(QUERIES)
            reports = []
            for total in sizes:
                report = await measure_size(conn, total, queries)
                print(json.dumps(report, indent=2))
                reports.append(report)
            return reports
        finally:
            await conn.close()
    finally:
        await admin.execute(f"DROP DATABASE IF EXISTS {BENCH_DB} WITH (FORCE)")
        await admin.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    args = parser.parse_args()
    reports = asyncio.run(run(args.sizes))
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "k": K,
                "queries": QUERIES,
                "dims": EMBED_DIMS,
                "sizes": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
