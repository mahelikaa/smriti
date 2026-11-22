from datetime import UTC, datetime
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from pydantic import ValidationError

from smriti.extract import (
    MODEL,
    Extraction,
    ExtractionError,
    Memory,
    cost_usd,
    extract_memories,
)


class FakeMessages:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, **kw):
        self.messages = FakeMessages(**kw)


def fake_response(stop_reason="end_turn", memories=None):
    parsed = None if memories is None else Extraction(memories=memories)
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=parsed,
        usage=SimpleNamespace(input_tokens=120, output_tokens=40),
    )


REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=REQ), body=None)


def make_validation_error():
    try:
        Memory(kind="fact", content="x", importance=9)
    except ValidationError as e:
        return e


async def test_returns_memories_as_dicts():
    client = FakeClient(
        result=fake_response(
            memories=[Memory(kind="fact", content="lives in Pune", importance=3)]
        )
    )

    result = await extract_memories(client, "main Pune mein rehti hoon")

    assert result.memories == [
        {"kind": "fact", "content": "lives in Pune", "importance": 3, "supersedes": []}
    ]


async def test_sends_text_and_schema():
    client = FakeClient(result=fake_response(memories=[]))

    await extract_memories(client, "hello")

    call = client.messages.calls[0]
    assert call["model"] == MODEL
    assert call["output_format"] is Extraction
    assert call["messages"] == [{"role": "user", "content": "hello"}]


async def test_nothing_to_remember_returns_empty_list():
    client = FakeClient(result=fake_response(memories=[]))

    result = await extract_memories(client, "hi")

    assert result.memories == []


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
async def test_bad_stop_reason_is_not_retryable(stop_reason):
    client = FakeClient(result=fake_response(stop_reason=stop_reason, memories=[]))

    with pytest.raises(ExtractionError) as exc:
        await extract_memories(client, "hello")

    assert exc.value.retryable is False
    assert stop_reason in str(exc.value)


async def test_missing_parsed_output_is_not_retryable():
    client = FakeClient(result=fake_response(memories=None))

    with pytest.raises(ExtractionError) as exc:
        await extract_memories(client, "hello")

    assert exc.value.retryable is False


@pytest.mark.parametrize(
    "error, retryable",
    [
        (status_error(anthropic.RateLimitError, 429), True),
        (status_error(anthropic.InternalServerError, 529), True),
        (anthropic.APITimeoutError(request=REQ), True),
        (status_error(anthropic.BadRequestError, 400), False),
        (make_validation_error(), False),
    ],
)
async def test_api_errors_are_sorted(error, retryable):
    client = FakeClient(error=error)

    with pytest.raises(ExtractionError) as exc:
        await extract_memories(client, "hello")

    assert exc.value.retryable is retryable


def test_cost_one_million_input_tokens():
    assert cost_usd(1_000_000, 0, "claude-opus-5-5") == 4.0


def test_cost_mixed():
    assert cost_usd(1000, 500, "claude-opus-5-5") == pytest.approx(0.014)


def test_cost_uses_the_model_price():
    assert cost_usd(1_000_000, 1_000_000, "claude-haiku-5-5") == pytest.approx(0.60)


async def test_sends_the_requested_model():
    client = FakeClient(result=fake_response(memories=[]))

    await extract_memories(client, "hello", model="claude-opus-5-5")

    assert client.messages.calls[0]["model"] == "claude-opus-5-5"


async def test_usage_passes_through():
    client = FakeClient(result=fake_response(memories=[]))

    result = await extract_memories(client, "hello")

    assert result.input_tokens == 120
    assert result.output_tokens == 40


async def test_sends_known_memories_with_ids():
    client = FakeClient(result=fake_response(memories=[]))
    known = [{"id": 7, "content": "Lives in Gurgaon"}]

    await extract_memories(client, "ab Bangalore mein hoon", known)

    content = client.messages.calls[0]["messages"][0]["content"]
    assert content == (
        "KNOWN MEMORIES:\n[7] Lives in Gurgaon\n\nNEW MESSAGE:\nab Bangalore mein hoon"
    )


async def test_supersedes_passes_through():
    memory = Memory(
        kind="fact", content="Lives in Bangalore", importance=3, supersedes=[7]
    )
    client = FakeClient(result=fake_response(memories=[memory]))

    result = await extract_memories(client, "ab Bangalore mein hoon")

    assert result.memories[0]["supersedes"] == [7]


@pytest.mark.parametrize(
    "sent_at, expected",
    [
        (datetime(2026, 11, 21, 18, 0, tzinfo=UTC), "2026-11-21 (Saturday) 23:30 IST"),
        (datetime(2026, 11, 21, 19, 0, tzinfo=UTC), "2026-11-22 (Sunday) 00:30 IST"),
    ],
)
async def test_sends_sent_at_in_ist(sent_at, expected):
    client = FakeClient(result=fake_response(memories=[]))

    await extract_memories(client, "kal doctor ke paas jaana hai", sent_at=sent_at)

    content = client.messages.calls[0]["messages"][0]["content"]
    assert content == (
        f"MESSAGE SENT AT: {expected}\n\nNEW MESSAGE:\nkal doctor ke paas jaana hai"
    )


async def test_sent_at_comes_before_known_memories():
    client = FakeClient(result=fake_response(memories=[]))
    known = [{"id": 7, "content": "Lives in Gurgaon"}]
    sent_at = datetime(2026, 11, 21, 6, 30, tzinfo=UTC)

    await extract_memories(client, "hi", known, sent_at)

    content = client.messages.calls[0]["messages"][0]["content"]
    assert content == (
        "MESSAGE SENT AT: 2026-11-21 (Saturday) 12:00 IST\n\n"
        "KNOWN MEMORIES:\n[7] Lives in Gurgaon\n\n"
        "NEW MESSAGE:\nhi"
    )
