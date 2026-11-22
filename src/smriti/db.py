import asyncio
import logging
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg

DEFAULT_DATABASE_URL = "postgresql://smriti:smriti@localhost:5433/smriti"
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
CONNECT_ATTEMPTS = 8
FIRST_DELAY_S = 0.5
MAX_DELAY_S = 4.0

DB_UNAVAILABLE_ERRORS = (
    OSError,
    TimeoutError,
    asyncpg.CannotConnectNowError,
    asyncpg.InterfaceError,
)

log = logging.getLogger("smriti.db")


class DatabaseUnavailable(RuntimeError):
    pass


async def create_pool_with_retry(
    url: str, attempts: int = CONNECT_ATTEMPTS, **pool_kwargs
) -> asyncpg.Pool:
    delay = FIRST_DELAY_S
    for attempt in range(1, attempts + 1):
        try:
            return await asyncpg.create_pool(url, **pool_kwargs)
        except DB_UNAVAILABLE_ERRORS as e:
            if attempt == attempts:
                raise DatabaseUnavailable(
                    f"database unreachable after {attempts} attempts: "
                    f"{type(e).__name__}"
                ) from e
            log.warning(
                "database not ready (attempt %d/%d, %s), retrying in %.1fs",
                attempt,
                attempts,
                type(e).__name__,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_DELAY_S)


def with_database(url: str, db: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{db}"))


async def apply_migrations(conn) -> None:
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        await conn.execute(path.read_text())


async def drop_database(admin_url: str, db: str) -> None:
    admin = await asyncpg.connect(admin_url)
    try:
        await admin.execute(f"DROP DATABASE IF EXISTS {db} WITH (FORCE)")
    finally:
        await admin.close()


async def recreate_database(admin_url: str, db: str) -> None:
    await drop_database(admin_url, db)
    admin = await asyncpg.connect(admin_url)
    try:
        await admin.execute(f"CREATE DATABASE {db}")
    finally:
        await admin.close()
    conn = await asyncpg.connect(with_database(admin_url, db))
    try:
        await apply_migrations(conn)
    finally:
        await conn.close()
