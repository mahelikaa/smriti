from dataclasses import dataclass
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

MODEL = "claude-opus-5-5"
MAX_TOKENS = 4096

PRICE_IN_PER_MTOK = 4.00  # USD per million input tokens (claude-opus-5-5)
PRICE_OUT_PER_MTOK = 20.00  # USD per million output tokens, thinking included

SYSTEM_PROMPT = """\
You extract long-term memories about the USER from one chat message.
The message may be in English, Hindi, or Hinglish (Hindi in Latin script).
Write each memory in short English, in third person ("lives in Pune").
Only keep things worth remembering in a future conversation:
facts, preferences, events, relationships. Skip greetings, small talk,
and anything about the assistant. If nothing is worth remembering,
return an empty list. importance: 1 = trivia, 5 = core to who they are.
"""

RETRYABLE_ERRORS = (
    anthropic.RateLimitError,  # 429
    anthropic.InternalServerError,  # 5xx
    anthropic.APIConnectionError,
)


class Memory(BaseModel):
    kind: Literal["fact", "preference", "event", "relationship"]
    content: str = Field(min_length=1, max_length=500)
    importance: int = Field(ge=1, le=5)


class Extraction(BaseModel):
    memories: list[Memory]


@dataclass
class ExtractResult:
    memories: list[dict]
    input_tokens: int
    output_tokens: int


class ExtractionError(Exception):
    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * PRICE_IN_PER_MTOK + output_tokens * PRICE_OUT_PER_MTOK
    ) / 1_000_000


def make_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(max_retries=0, timeout=30.0)


async def extract_memories(client, text: str) -> ExtractResult:
    try:
        response = await client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_format=Extraction,
            output_config={"effort": "low"},
        )
    except RETRYABLE_ERRORS as e:
        raise ExtractionError(f"{type(e).__name__}: {e}", retryable=True) from e
    except anthropic.APIStatusError as e:
        raise ExtractionError(f"{type(e).__name__}: {e}", retryable=False) from e
    except ValidationError as e:
        # Claude's reply didn't match the schema (e.g. importance: 7)
        raise ExtractionError(f"invalid output: {e}", retryable=False) from e

    if response.stop_reason != "end_turn":
        raise ExtractionError(f"stop_reason={response.stop_reason}", retryable=False)

    if response.parsed_output is None:
        raise ExtractionError("no parsed output", retryable=False)

    return ExtractResult(
        memories=[m.model_dump() for m in response.parsed_output.memories],
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
