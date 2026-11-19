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
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed)


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

    assert result == [{"kind": "fact", "content": "lives in Pune", "importance": 3}]


async def test_sends_text_and_schema():
    client = FakeClient(result=fake_response(memories=[]))

    await extract_memories(client, "hello")

    call = client.messages.calls[0]
    assert call["model"] == MODEL
    assert call["output_format"] is Extraction
    assert call["messages"] == [{"role": "user", "content": "hello"}]


async def test_nothing_to_remember_returns_empty_list():
    client = FakeClient(result=fake_response(memories=[]))

    assert await extract_memories(client, "hi") == []


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
