import pytest

from smriti.bench import load, recall, summarize
from smriti.main import app


def test_summarize_reads_p50_and_p99():
    latencies = [float(ms) for ms in range(1, 101)]

    summary = summarize(latencies)

    assert summary == {"p50_ms": pytest.approx(50.5), "p99_ms": pytest.approx(99.01)}


def test_recall_counts_overlap_with_exact_results():
    exact = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    got = [[1, 2, 3, 4, 99], [6, 7]]

    assert recall(got, exact) == 0.6


async def test_load_gives_heavy_user_its_share(client):
    async with app.state.pool.acquire() as conn:
        await load(conn, total=250, heavy=50, per_user=100)

        counts = dict(
            await conn.fetch("SELECT user_id, count(*) FROM memories GROUP BY user_id")
        )
        distinct = await conn.fetchval(
            "SELECT count(DISTINCT embedding::text) FROM memories"
        )
        dims = await conn.fetchval(
            "SELECT DISTINCT vector_dims(embedding) FROM memories"
        )

    assert counts == {"bench-heavy": 50, "bench-0": 100, "bench-1": 100}
    assert distinct == 250
    assert dims == 1024
