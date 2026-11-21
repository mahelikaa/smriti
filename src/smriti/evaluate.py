import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import anthropic
from pydantic import BaseModel

from smriti.extract import MODEL, cost_usd, extract_memories

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "eval" / "cases.jsonl"
RESULTS_PATH = ROOT / "eval" / "results.json"
JUDGE_MODEL = "claude-opus-5-5"
CONCURRENCY = 4
DEFAULT_SENT_AT = datetime(2026, 11, 21, 4, 30, tzinfo=UTC)

JUDGE_SYSTEM = """\
You grade a memory extractor. You get EXPECTED facts (numbered from 0) and
SAVED memories (numbered from 0). For every saved memory, in order, return:
covers: the numbers of the expected facts it states. Same meaning is enough;
wording and language may differ. Names, places, numbers and dates must agree.
A date matches if it is the same day, or falls inside the week or month the
expected fact gives. A memory may cover several expected facts, or none.
duplicate_of: the number of an EARLIER saved memory that says the same thing,
else null.
"""


class Case(BaseModel):
    id: str
    category: str
    text: str
    sent_at: datetime = DEFAULT_SENT_AT
    known: list[dict] = []
    expect: list[str]
    supersedes: list[int] | None = None


class Verdict(BaseModel):
    covers: list[int]
    duplicate_of: int | None


class Judgement(BaseModel):
    verdicts: list[Verdict]


@dataclass
class CaseScore:
    id: str
    category: str
    expected: int
    found: int
    saved: int
    correct: int
    skip_ok: bool | None
    supersede_ok: bool | None


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    cases = [Case.model_validate_json(line) for line in lines]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case id")
    return cases


def score_case(case: Case, memories: list[dict], verdicts: list[Verdict]) -> CaseScore:
    covered = set()
    correct = 0
    for i, v in enumerate(verdicts):
        hits = {e for e in v.covers if 0 <= e < len(case.expect)}
        duplicate = v.duplicate_of is not None and 0 <= v.duplicate_of < i
        if hits and not duplicate:
            correct += 1
        covered |= hits

    supersede_ok = None
    if case.supersedes is not None:
        superseded = {i for m in memories for i in m.get("supersedes", [])}
        supersede_ok = superseded == set(case.supersedes)

    return CaseScore(
        id=case.id,
        category=case.category,
        expected=len(case.expect),
        found=len(covered),
        saved=len(memories),
        correct=correct,
        skip_ok=None if case.expect else not memories,
        supersede_ok=supersede_ok,
    )


def rate(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None


def summarize(scores: list[CaseScore]) -> dict:
    skips = [s.skip_ok for s in scores if s.skip_ok is not None]
    supersedes = [s.supersede_ok for s in scores if s.supersede_ok is not None]
    return {
        "cases": len(scores),
        "recall": rate(sum(s.found for s in scores), sum(s.expected for s in scores)),
        "precision": rate(sum(s.correct for s in scores), sum(s.saved for s in scores)),
        "skip": rate(sum(skips), len(skips)),
        "supersede": rate(sum(supersedes), len(supersedes)),
    }


def build_judge_prompt(expect: list[str], memories: list[dict]) -> str:
    expected = "\n".join(f"[{i}] {e}" for i, e in enumerate(expect))
    saved = "\n".join(f"[{i}] {m['content']}" for i, m in enumerate(memories))
    return f"EXPECTED:\n{expected}\n\nSAVED:\n{saved}"


async def judge(
    client, case: Case, memories: list[dict]
) -> tuple[list[Verdict], int, int]:
    if not memories:
        return [], 0, 0
    if not case.expect:
        return [Verdict(covers=[], duplicate_of=None) for _ in memories], 0, 0

    response = await client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=2048,
        system=JUDGE_SYSTEM,
        messages=[
            {"role": "user", "content": build_judge_prompt(case.expect, memories)}
        ],
        output_format=Judgement,
        output_config={"effort": "low"},
    )
    verdicts = response.parsed_output.verdicts
    if len(verdicts) != len(memories):
        raise ValueError(
            f"{case.id}: judge gave {len(verdicts)} verdicts for {len(memories)} memories"
        )
    return verdicts, response.usage.input_tokens, response.usage.output_tokens


async def run_case(client, sem: asyncio.Semaphore, case: Case) -> dict:
    async with sem:
        result = await extract_memories(client, case.text, case.known, case.sent_at)
        verdicts, judge_in, judge_out = await judge(client, case, result.memories)
    score = score_case(case, result.memories, verdicts)
    return {
        "score": score,
        "memories": result.memories,
        "verdicts": [v.model_dump() for v in verdicts],
        "cost": cost_usd(
            result.input_tokens + judge_in, result.output_tokens + judge_out
        ),
    }


def print_report(cases: list[Case], runs: list[dict]) -> dict:
    scores = [r["score"] for r in runs]
    categories = sorted({s.category for s in scores})
    print(f"{'category':<12}{'recall':>8}{'precision':>11}{'skip':>7}{'supersede':>11}")
    by_category = {}
    for cat in categories + ["TOTAL"]:
        group = scores if cat == "TOTAL" else [s for s in scores if s.category == cat]
        summary = summarize(group)
        by_category[cat] = summary
        cells = [summary[k] for k in ("recall", "precision", "skip", "supersede")]
        shown = ["-" if c is None else f"{c:.0%}" for c in cells]
        print(f"{cat:<12}{shown[0]:>8}{shown[1]:>11}{shown[2]:>7}{shown[3]:>11}")

    print("\nmisses:")
    for case, run in zip(cases, runs):
        s = run["score"]
        if s.found < s.expected or s.correct < s.saved or s.supersede_ok is False:
            saved = [m["content"] for m in run["memories"]] or ["(nothing)"]
            print(f"  {case.id}: expected {case.expect} | saved {saved}")

    total_cost = sum(r["cost"] for r in runs)
    print(f"\ncost: ${total_cost:.3f}")
    return by_category


async def run_eval(cases: list[Case]) -> list[dict]:
    client = anthropic.AsyncAnthropic(max_retries=4, timeout=60.0)
    sem = asyncio.Semaphore(CONCURRENCY)
    try:
        return await asyncio.gather(*(run_case(client, sem, c) for c in cases))
    finally:
        await client.close()


def main() -> None:
    cases = load_cases()
    runs = asyncio.run(run_eval(cases))
    summary = print_report(cases, runs)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "model": MODEL,
                "judge_model": JUDGE_MODEL,
                "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "summary": summary,
                "cases": [
                    {
                        **asdict(r["score"]),
                        "memories": r["memories"],
                        "verdicts": r["verdicts"],
                    }
                    for r in runs
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
