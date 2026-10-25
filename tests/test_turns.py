import pytest


def make_payload(**overrides):
    payload = {
        "external_id": "msg1",
        "role": "user",
        "text": "Hello",
        "sent_at": "2026-10-14T10:00:00Z",
    }
    payload.update(overrides)
    return payload


async def test_create_turn(client):
    response = await client.post("/v1/users/alice/turns", json=make_payload())

    assert response.status_code == 201
    assert isinstance(response.json()["id"], int)
    assert response.json()["text"] == "Hello"


async def test_identical_replay(client):
    first = await client.post("/v1/users/alice/turns", json=make_payload())
    second = await client.post("/v1/users/alice/turns", json=make_payload())

    assert first.status_code == 201
    assert second.status_code == 200

    assert first.json()["id"] == second.json()["id"]

    result = await client.get("/v1/users/alice/turns")

    assert result.status_code == 200
    assert len(result.json()["items"]) == 1


@pytest.mark.asyncio
async def test_conflicting_replay(client):
    first = await client.post("/v1/users/alice/turns", json=make_payload())
    second = await client.post(
        "/v1/users/alice/turns", json=make_payload(text="Goodbye")
    )

    assert first.status_code == 201
    assert second.status_code == 409

    result = await client.get("/v1/users/alice/turns")
    assert len(result.json()["items"]) == 1
    assert result.json()["items"][0]["text"] == "Hello"


async def test_same_external_id_different_user(client):
    first = await client.post("/v1/users/alice/turns", json=make_payload())
    second = await client.post("/v1/users/bob/turns", json=make_payload())

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]

    alice_turns = await client.get("/v1/users/alice/turns")
    bob_turns = await client.get("/v1/users/bob/turns")
    assert alice_turns.status_code == 200
    assert bob_turns.status_code == 200
    assert len(alice_turns.json()["items"]) == 1
    assert len(bob_turns.json()["items"]) == 1


async def test_naive_sent_at_returns_422(client):
    response = await client.post(
        "/v1/users/alice/turns", json=make_payload(sent_at="2026-10-14T10:00:00")
    )
    assert response.status_code == 422


async def test_get_returns_turns_sorted_by_timestamp(client):
    first = await client.post("/v1/users/alice/turns", json=make_payload())
    second = await client.post(
        "/v1/users/alice/turns",
        json=make_payload(sent_at="2026-10-14T10:05:56Z", external_id="msg2"),
    )
    third = await client.post(
        "/v1/users/alice/turns",
        json=make_payload(sent_at="2026-10-14T10:02:05Z", external_id="msg3"),
    )
    assert [r.status_code for r in (first, second, third)] == [201, 201, 201]

    response = await client.get("/v1/users/alice/turns")
    assert response.status_code == 200
    turns = response.json()["items"]

    assert turns[0]["external_id"] == "msg2"
    assert turns[1]["external_id"] == "msg3"
    assert turns[2]["external_id"] == "msg1"
    assert len(turns) == 3


async def test_two_msgs_with_identical_timestamps(client):
    first = await client.post("/v1/users/alice/turns", json=make_payload())
    second = await client.post(
        "/v1/users/alice/turns", json=make_payload(external_id="msg3")
    )
    third = await client.post(
        "/v1/users/alice/turns", json=make_payload(external_id="msg2")
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 201

    response = await client.get("/v1/users/alice/turns")
    assert response.status_code == 200
    turns = response.json()["items"]

    assert len(turns) == 3

    assert turns[0]["external_id"] == "msg2"
    assert turns[1]["external_id"] == "msg3"
    assert turns[2]["external_id"] == "msg1"

    ids = [turn["id"] for turn in turns]
    assert ids == sorted(ids, reverse=True)


async def test_pagination_walk_returns_every_turn_once(client):
    timestamps = [
        "2026-10-14T10:01:00Z",
        "2026-10-14T10:02:00Z",
        "2026-10-14T10:03:00Z",
        "2026-10-14T10:03:00Z",
        "2026-10-14T10:05:00Z",
    ]

    for i, timestamp in enumerate(timestamps, start=1):
        response = await client.post(
            "/v1/users/alice/turns",
            json=make_payload(
                external_id=f"msg{i}",
                sent_at=timestamp,
            ),
        )
        assert response.status_code == 201

    cursor = {}
    collected_ids = []
    request_count = 0

    while True:
        response = await client.get(
            "/v1/users/alice/turns",
            params={"limit": 2, **cursor},
        )
        assert response.status_code == 200

        data = response.json()
        request_count += 1

        collected_ids.extend(item["external_id"] for item in data["items"])

        cursor = data["next_cursor"]

        if cursor is None:
            break

    assert collected_ids == ["msg5", "msg4", "msg3", "msg2", "msg1"]
    assert len(collected_ids) == len(set(collected_ids))
    assert request_count == 3


async def test_exact_limit_returns_no_cursor(client):
    for i in range(1, 3):
        response = await client.post(
            "/v1/users/alice/turns",
            json=make_payload(external_id=f"msg{i}"),
        )
        assert response.status_code == 201

    response = await client.get(
        "/v1/users/alice/turns",
        params={"limit": 2},
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) == 2
    assert data["next_cursor"] is None


async def test_single_cursor_param_returns_422(client):
    response = await client.get(
        "/v1/users/alice/turns",
        params={"before_id": 5},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("limit", [0, 101])
async def test_limit_out_of_range_returns_422(client, limit):
    response = await client.get(
        "/v1/users/alice/turns",
        params={"limit": limit},
    )
    assert response.status_code == 422


async def test_readyz_returns_200_when_db_up(client):
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
