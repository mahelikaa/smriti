from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg

DEFAULT_DATABASE_URL = "postgresql://smriti:smriti@localhost:5433/smriti"
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


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
