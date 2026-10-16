import pytest
from fastapi.testclient import TestClient

from smriti import main

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def reset_storage():
    main.turns.clear()
    main.next_id = 1


def make_payload(**overrides):
    payload = {
        "external_id": "msg1",
        "role": "user",
        "text": "Hello",
        "sent_at": "2026-10-14T10:00:00Z",
    }
    payload.update(overrides)
    return payload


def test_create_turn():
    response = client.post("/v1/users/alice/turns", json=make_payload())

    assert response.status_code == 201
    assert response.json()["id"] == 1
    assert response.json()["text"] == "Hello"


def test_identical_replay():
    first = client.post("/v1/users/alice/turns", json=make_payload())
    second = client.post("/v1/users/alice/turns", json=make_payload())

    assert first.status_code == 201
    assert second.status_code == 200

    assert first.json()["id"] == second.json()["id"]

    result = client.get("/v1/users/alice/turns")

    assert result.status_code == 200
    assert len(result.json()) == 1


def test_conflicting_replay():
    first = client.post("/v1/users/alice/turns", json=make_payload())
    second = client.post("/v1/users/alice/turns", json=make_payload(text="Goodbye"))

    assert first.status_code == 201
    assert second.status_code == 409

    result = client.get("/v1/users/alice/turns")
    assert len(result.json()) == 1
    assert result.json()[0]["text"] == "Hello"


def test_same_external_id_different_user():
    first = client.post("/v1/users/alice/turns", json=make_payload())
    second = client.post("/v1/users/bob/turns", json=make_payload())

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]

    alice_turns = client.get("/v1/users/alice/turns")
    bob_turns = client.get("/v1/users/bob/turns")
    assert alice_turns.status_code == 200
    assert bob_turns.status_code == 200
    assert len(alice_turns.json()) == 1
    assert len(bob_turns.json()) == 1


def test_naive_sent_at_returns_422():
    response = client.post(
        "/v1/users/alice/turns", json=make_payload(sent_at="2026-10-14T10:00:00")
    )

    assert response.status_code == 422


def test_get_returns_turns_sorted_by_timestamp():
    first = client.post("/v1/users/alice/turns", json=make_payload())
    second = client.post(
        "/v1/users/alice/turns",
        json=make_payload(sent_at="2026-10-14T10:05:56Z", external_id="msg2"),
    )
    third = client.post(
        "/v1/users/alice/turns",
        json=make_payload(sent_at="2026-10-14T10:02:05Z", external_id="msg3"),
    )
    assert [r.status_code for r in (first, second, third)] == [201, 201, 201]

    response = client.get("/v1/users/alice/turns")
    assert response.status_code == 200
    turns = response.json()

    assert turns[0]["external_id"] == "msg1"
    assert turns[1]["external_id"] == "msg3"
    assert turns[2]["external_id"] == "msg2"
    assert len(turns) == 3


def test_two_msgs_with_identical_timestamps():
    first = client.post("/v1/users/alice/turns", json=make_payload())
    second = client.post("/v1/users/alice/turns", json=make_payload(external_id="msg3"))
    third = client.post("/v1/users/alice/turns", json=make_payload(external_id="msg2"))

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 201

    response = client.get("/v1/users/alice/turns")
    assert response.status_code == 200
    turns = response.json()

    assert len(turns) == 3

    assert turns[0]["external_id"] == "msg1"
    assert turns[1]["external_id"] == "msg3"
    assert turns[2]["external_id"] == "msg2"

    ids = [turn["id"] for turn in turns]
    assert ids == sorted(ids)
