from smriti.db import with_database
from smriti.main import DATABASE_URL


def test_with_database_only_swaps_the_database():
    url = "postgresql://smriti:smriti@localhost:5433/smriti"

    assert with_database(url, "smriti_test") == (
        "postgresql://smriti:smriti@localhost:5433/smriti_test"
    )


def test_tests_never_touch_the_dev_database():
    assert DATABASE_URL.endswith("/smriti_test")
