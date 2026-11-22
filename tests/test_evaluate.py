import pytest

from smriti.evaluate import (
    Case,
    Verdict,
    judge,
    load_cases,
    mean_of,
    score_case,
    summarize,
)


def make_case(expect, supersedes=None):
    return Case(
        id="c", category="facts", text="x", expect=expect, supersedes=supersedes
    )


def memory(content, supersedes=()):
    return {"content": content, "supersedes": list(supersedes)}


def test_cases_file_is_valid():
    cases = load_cases()
    assert len(cases) >= 40
    assert {"facts", "dates", "moods", "supersede", "context"} <= {
        c.category for c in cases
    }
    for case in cases:
        known_ids = {m["id"] for m in case.known}
        assert set(case.supersedes or []) <= known_ids, case.id


def test_duplicate_memory_is_not_correct():
    case = make_case(["Attended Riya's mehendi on 20 Nov 2026"])
    memories = [memory("Attended Riya's mehendi"), memory("Knows Riya's mehendi")]
    verdicts = [
        Verdict(covers=[0], duplicate_of=None),
        Verdict(covers=[0], duplicate_of=0),
    ]

    score = score_case(case, memories, verdicts)

    assert (score.found, score.correct, score.saved) == (1, 1, 2)


def test_one_memory_can_cover_two_facts():
    case = make_case(["Has a sister named Riya", "Riya's wedding is in December"])
    verdicts = [Verdict(covers=[0, 1], duplicate_of=None)]

    score = score_case(case, [memory("Sister Riya's wedding is in December")], verdicts)

    assert (score.found, score.correct) == (2, 1)


def test_judge_indexes_out_of_range_are_ignored():
    case = make_case(["Lives in Pune"])
    verdicts = [Verdict(covers=[5], duplicate_of=None)]

    score = score_case(case, [memory("Lives in Pune")], verdicts)

    assert (score.found, score.correct) == (0, 0)


def test_saving_anything_fails_a_skip_case():
    case = make_case([])
    verdicts = [Verdict(covers=[], duplicate_of=None)]

    score = score_case(case, [memory("Is tired")], verdicts)

    assert score.skip_ok is False
    assert (score.correct, score.saved) == (0, 1)


@pytest.mark.parametrize(
    ("saved", "ok"),
    [([11], True), ([], False), ([11, 12], False)],
)
def test_supersede_must_match_exactly(saved, ok):
    case = make_case(["Lives in Bangalore"], supersedes=[11])

    score = score_case(
        case,
        [memory("Lives in Bangalore", supersedes=saved)],
        [Verdict(covers=[0], duplicate_of=None)],
    )

    assert score.supersede_ok is ok


def test_supersede_not_checked_when_case_has_none():
    case = make_case(["Lives in Bangalore"])

    score = score_case(
        case,
        [memory("Lives in Bangalore", supersedes=[11])],
        [Verdict(covers=[0], duplicate_of=None)],
    )

    assert score.supersede_ok is None


def test_summarize_adds_up_across_cases():
    a = score_case(
        make_case(["x", "y"]),
        [memory("x")],
        [Verdict(covers=[0], duplicate_of=None)],
    )
    b = score_case(make_case([]), [], [])

    summary = summarize([a, b])

    assert summary["recall"] == 0.5
    assert summary["precision"] == 1.0
    assert summary["skip"] == 1.0
    assert summary["supersede"] is None


class NoCallClient:
    @property
    def messages(self):
        raise AssertionError("judge should not call Claude")


async def test_judge_skips_claude_when_nothing_expected():
    verdicts, _, _ = await judge(NoCallClient(), make_case([]), [memory("Is tired")])

    assert verdicts == [Verdict(covers=[], duplicate_of=None)]


async def test_judge_skips_claude_when_nothing_saved():
    verdicts, _, _ = await judge(NoCallClient(), make_case(["Lives in Pune"]), [])

    assert verdicts == []


def test_mean_of_averages_runs_and_skips_missing_values():
    run1 = {"TOTAL": {"recall": 1.0, "precision": 0.8, "skip": None, "supersede": 1.0}}
    run2 = {"TOTAL": {"recall": 0.5, "precision": 0.9, "skip": None, "supersede": None}}

    mean = mean_of([run1, run2])

    assert mean["TOTAL"] == {
        "recall": 0.75,
        "precision": 0.85,
        "skip": None,
        "supersede": 1.0,
    }
