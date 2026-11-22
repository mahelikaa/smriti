import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import anthropic
from pydantic import BaseModel

from smriti.extract import MODEL, PRICES_PER_MTOK, cost_usd, extract_memories

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "eval" / "cases.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
JUDGE_MODEL = "claude-opus-5-5"
CONCURRENCY = 4
METRICS = ("recall", "precision", "skip", "supersede")
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


async def run_case(client, sem: asyncio.Semaphore, case: Case, model: str) -> dict:
    async with sem:
        result = await extract_memories(
            client, case.text, case.known, case.sent_at, model
        )
        verdicts, judge_in, judge_out = await judge(client, case, result.memories)
    return {
        "score": score_case(case, result.memories, verdicts),
        "memories": result.memories,
        "verdicts": [v.model_dump() for v in verdicts],
        "extract_cost": cost_usd(result.input_tokens, result.output_tokens, model),
        "judge_cost": cost_usd(judge_in, judge_out, JUDGE_MODEL),
    }


def by_category(scores: list[CaseScore]) -> dict:
    categories = sorted({s.category for s in scores})
    out = {c: summarize([s for s in scores if s.category == c]) for c in categories}
    out["TOTAL"] = summarize(scores)
    return out


def mean_of(tables: list[dict]) -> dict:
    out = {}
    for cat in tables[0]:
        out[cat] = {}
        for metric in METRICS:
            values = [t[cat][metric] for t in tables if t[cat][metric] is not None]
            out[cat][metric] = round(sum(values) / len(values), 3) if values else None
    return out


def is_miss(s: CaseScore) -> bool:
    return (
        s.found < s.expected
        or s.correct < s.saved
        or s.skip_ok is False
        or s.supersede_ok is False
    )


def cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def print_report(cases: list[Case], runs: list[list[dict]], model: str) -> dict:
    tables = [by_category([r["score"] for r in run]) for run in runs]
    mean = mean_of(tables)

    print(f"model {model}, {len(runs)} run(s), mean:")
    print(f"{'category':<12}{'recall':>8}{'precision':>11}{'skip':>7}{'supersede':>11}")
    for cat, row in mean.items():
        cells = [cell(row[m]) for m in METRICS]
        print(f"{cat:<12}{cells[0]:>8}{cells[1]:>11}{cells[2]:>7}{cells[3]:>11}")

    print("\nper run:")
    for i, t in enumerate(tables, 1):
        total = t["TOTAL"]
        print(f"  run {i}: " + "  ".join(f"{m} {cell(total[m])}" for m in METRICS))

    print("\nmisses (runs missed / runs):")
    for i, case in enumerate(cases):
        bad = [run[i] for run in runs if is_miss(run[i]["score"])]
        if bad:
            saved = [m["content"] for m in bad[-1]["memories"]] or ["(nothing)"]
            print(f"  {case.id} {len(bad)}/{len(runs)}: saved {saved}")

    extract = sum(r["extract_cost"] for run in runs for r in run) / len(runs)
    judge_cost = sum(r["judge_cost"] for run in runs for r in run) / len(runs)
    print(f"\ncost per run: extraction ${extract:.4f}, judge ${judge_cost:.4f}")
    print(f"extraction per case: ${extract / len(cases):.5f}")
    return mean


async def run_eval(cases: list[Case], model: str, runs: int) -> list[list[dict]]:
    client = anthropic.AsyncAnthropic(max_retries=4, timeout=60.0)
    sem = asyncio.Semaphore(CONCURRENCY)
    try:
        results = []
        for _ in range(runs):
            results.append(
                await asyncio.gather(*(run_case(client, sem, c, model) for c in cases))
            )
        return results
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, choices=sorted(PRICES_PER_MTOK))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    runs = asyncio.run(run_eval(cases, args.model, args.runs))
    mean = print_report(cases, runs, args.model)

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "" if args.cases == CASES_PATH else f"-{args.cases.stem}"
    (RESULTS_DIR / f"{args.model}{suffix}.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "judge_model": JUDGE_MODEL,
                "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "mean": mean,
                "runs": [
                    {
                        "summary": by_category([r["score"] for r in run])["TOTAL"],
                        "extract_cost": round(sum(r["extract_cost"] for r in run), 5),
                        "misses": [
                            {**asdict(r["score"]), "memories": r["memories"]}
                            for r in run
                            if is_miss(r["score"])
                        ],
                    }
                    for run in runs
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
