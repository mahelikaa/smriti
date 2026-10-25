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


@pytest.mark.asyncio
async def test_create_turn(client):
    response = await client.post("/v1/users/alice/turns", json=make_payload())

    assert response.status_code == 201
    assert isinstance(response.json()["id"], int)
    assert response.json()["text"] == "Hello"


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_naive_sent_at_returns_422(client):
    response = await client.post(
        "/v1/users/alice/turns", json=make_payload(sent_at="2026-10-14T10:00:00")
    )

    assert response.status_code == 422


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
