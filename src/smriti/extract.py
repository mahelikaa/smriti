from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import anthropic
from pydantic import BaseModel, Field, ValidationError

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 4096
USER_TZ = ZoneInfo("Asia/Kolkata")

PRICES_PER_MTOK = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

SYSTEM_PROMPT = """\
You extract long-term memories about the USER from one chat message.
The message may be in English, Hindi, or Hinglish (Hindi in Latin script).
Write each memory in short English, in third person ("lives in Pune").
Only keep things worth remembering in a future conversation:
facts, preferences, events, relationships. Plans that have a date are
worth remembering. Skip greetings, small talk,
and anything about the assistant. If nothing is worth remembering,
return an empty list. importance: 1 = trivia, 5 = core to who they are.
You may also get KNOWN MEMORIES, each with an id. If a new memory makes a
known one no longer true (they moved, changed jobs, broke up), put the old
id in supersedes. Only use ids from the list. Do not repeat known memories.
Only save what the user actually said. Do not guess gender, age, job,
studies, city or relationships from grammar or context: "office mein
meeting thi" does not say where they work, and verb forms are not a memory.
Do not save that the user knows someone unless they say how (sister,
friend, manager).
Save each fact once. When something changes (moved, new job), write one
memory with the new state, e.g. "lives in Kochi (moved from Indore in
Mar 2026)", not one for the change and another for the new state.
Use MESSAGE SENT AT to turn relative times (kal, parso, aaj, next week,
tomorrow) into dates, e.g. "starts a new job on 19 Nov 2026". Never write
"tomorrow" or "next week" in a memory. "kal" can mean yesterday or
tomorrow: decide from the verb tense.
Do not save passing moods or states (nervous today, tired, bored, hungry).
Only save a feeling if it lasts, e.g. "is afraid of flying".
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
    supersedes: list[int] = Field(default_factory=list)


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


def cost_usd(input_tokens: int, output_tokens: int, model: str = MODEL) -> float:
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def format_sent_at(sent_at: datetime) -> str:
    return sent_at.astimezone(USER_TZ).strftime("%Y-%m-%d (%A) %H:%M IST")


def build_prompt(text: str, known: list[dict], sent_at: datetime | None = None) -> str:
    sections = []
    if sent_at is not None:
        sections.append(f"MESSAGE SENT AT: {format_sent_at(sent_at)}")
    if known:
        lines = "\n".join(f"[{m['id']}] {m['content']}" for m in known)
        sections.append(f"KNOWN MEMORIES:\n{lines}")
    if not sections:
        return text
    sections.append(f"NEW MESSAGE:\n{text}")
    return "\n\n".join(sections)


def make_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(max_retries=0, timeout=30.0)


async def extract_memories(
    client,
    text: str,
    known: list[dict] | None = None,
    sent_at: datetime | None = None,
    model: str = MODEL,
) -> ExtractResult:
    try:
        response = await client.messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_prompt(text, known or [], sent_at)}
            ],
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
