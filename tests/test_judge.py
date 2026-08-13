"""Tests for the pinned, rubric-hashed judge.

These tests are the argument that a number produced by :class:`PinnedJudge` can
be compared to a number it produced last month: the model id could not have been
an alias, the rubric could not have changed unannounced, and every verdict --
plus every response that failed to parse -- is on the record exactly once.

The adapter here is a local scripted double rather than ``rigor.adapters.fake``:
the judge's contract is the ``Adapter`` protocol, and these tests must hold for
any object that satisfies it.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from rigor.errors import JudgeOutputError, ModelPinError, RubricDriftError
from rigor.evidence import (
    EVENT_JUDGE_INIT,
    EVENT_JUDGE_PARSE_FAILURE,
    EVENT_JUDGE_VERDICT,
    EVENT_RUBRIC_CHANGE_ACCEPTED,
    EvidenceLog,
)
from rigor.judge import (
    OUTPUT_FORMAT_INSTRUCTION,
    SCORE_MAX,
    SCORE_MIN,
    PinnedJudge,
    Verdict,
)

PINNED_MODEL = "claude-sonnet-4-5-20250929"
ALIASED_MODEL = "claude-3-5-sonnet-latest"

RUBRIC_LF = "# Rubric\n\nPass if the summary is faithful.\n"
RUBRIC_CRLF = RUBRIC_LF.replace("\n", "\r\n")

PASS_JSON = '{"pass": true, "score": 5, "reason": "faithful and complete"}'
FAIL_JSON = '{"pass": false, "score": 2, "reason": "dropped the stated caveat"}'


class ScriptedAdapter:
    """Adapter double that returns queued strings in order.

    Satisfies the ``Adapter`` protocol and nothing else, so a test that passes
    here cannot be relying on anything a real provider would not give us.
    """

    def __init__(self, model_id: str, responses: list[str] | None = None) -> None:
        self._model_id = model_id
        self._responses = list(responses or [])
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("scripted adapter ran out of responses")
        return self._responses.pop(0)


def write_rubric(tmp_path: Path, text: str = RUBRIC_LF, name: str = "rubric.md") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def make_judge(
    tmp_path: Path,
    *,
    responses: list[str] | None = None,
    model_id: str = PINNED_MODEL,
    rubric: Path | None = None,
    evidence: EvidenceLog | None = None,
    name: str = "summariser",
    accept_rubric_change: bool = False,
) -> tuple[PinnedJudge, ScriptedAdapter, EvidenceLog]:
    adapter = ScriptedAdapter(model_id, responses)
    log = evidence if evidence is not None else EvidenceLog(tmp_path / "evidence.jsonl")
    judge = PinnedJudge(
        adapter,
        rubric if rubric is not None else write_rubric(tmp_path),
        log,
        name=name,
        accept_rubric_change=accept_rubric_change,
    )
    return judge, adapter, log


def records(log: EvidenceLog, event_type: str) -> list:
    return [record for record in log.read() if record.event_type == event_type]


# --------------------------------------------------------------------------- #
# pinning
# --------------------------------------------------------------------------- #


def test_aliased_model_id_is_refused_at_construction(tmp_path: Path) -> None:
    # Construction time, not analysis time: an alias re-points silently, so the
    # cost of finding out late is every verdict recorded in between.
    log = EvidenceLog(tmp_path / "evidence.jsonl")

    with pytest.raises(ModelPinError) as excinfo:
        PinnedJudge(ScriptedAdapter(ALIASED_MODEL), write_rubric(tmp_path), log, name="summariser")

    assert ALIASED_MODEL in str(excinfo.value)
    assert log.read() == []  # a judge that refused to exist recorded nothing


def test_pinned_model_id_is_accepted_and_recorded(tmp_path: Path) -> None:
    judge, _, log = make_judge(tmp_path)

    (init,) = records(log, EVENT_JUDGE_INIT)

    assert judge.model_id == PINNED_MODEL
    assert init.payload["model_id"] == PINNED_MODEL
    assert init.payload["judge"] == "summariser"


# --------------------------------------------------------------------------- #
# rubric hashing
# --------------------------------------------------------------------------- #


def test_logged_rubric_hash_is_the_sha256_of_the_rubric(tmp_path: Path) -> None:
    rubric = write_rubric(tmp_path)
    judge, _, log = make_judge(tmp_path, rubric=rubric)

    expected = hashlib.sha256(RUBRIC_LF.encode("utf-8")).hexdigest()

    assert judge.rubric_hash == expected
    assert records(log, EVENT_JUDGE_INIT)[0].payload["rubric_hash"] == expected
    assert records(log, EVENT_JUDGE_INIT)[0].payload["rubric_path"] == str(rubric)


def test_crlf_and_lf_rubrics_hash_identically(tmp_path: Path) -> None:
    # Windows checkout vs Linux CI runner. Without newline normalisation every
    # cross-platform run would look like rubric drift for a file nobody touched.
    lf_judge, _, _ = make_judge(
        tmp_path,
        rubric=write_rubric(tmp_path, RUBRIC_LF, "lf.md"),
        evidence=EvidenceLog(tmp_path / "lf.jsonl"),
    )
    crlf_judge, _, _ = make_judge(
        tmp_path,
        rubric=write_rubric(tmp_path, RUBRIC_CRLF, "crlf.md"),
        evidence=EvidenceLog(tmp_path / "crlf.jsonl"),
    )

    assert (tmp_path / "crlf.md").read_bytes() != (tmp_path / "lf.md").read_bytes()
    assert crlf_judge.rubric_hash == lf_judge.rubric_hash


def test_shipped_rubric_ends_with_the_output_format_the_judge_parses(tmp_path: Path) -> None:
    # The rubric is the prompt's tail. If the two drift apart, the judge asks for
    # one format and the rubric documents another, and parsing fails in the field.
    shipped = Path(__file__).resolve().parents[1] / "rubrics" / "example-rubric.md"
    text = shipped.read_text(encoding="utf-8")

    assert text.rstrip("\n").endswith(OUTPUT_FORMAT_INSTRUCTION)

    judge, adapter, _ = make_judge(tmp_path, rubric=shipped, responses=[PASS_JSON])
    judge.evaluate("Summarise the memo.", "The memo says X.")

    assert OUTPUT_FORMAT_INSTRUCTION in adapter.prompts[0]


# --------------------------------------------------------------------------- #
# rubric drift
# --------------------------------------------------------------------------- #


def test_unchanged_rubric_reconstructs_without_error(tmp_path: Path) -> None:
    rubric = write_rubric(tmp_path)
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    first, _, _ = make_judge(tmp_path, rubric=rubric, evidence=log)

    second, _, _ = make_judge(tmp_path, rubric=rubric, evidence=log)

    assert second.rubric_hash == first.rubric_hash
    assert len(records(log, EVENT_JUDGE_INIT)) == 2
    assert records(log, EVENT_RUBRIC_CHANGE_ACCEPTED) == []


def test_changed_rubric_raises_drift_naming_both_hashes(tmp_path: Path) -> None:
    rubric = write_rubric(tmp_path)
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    first, _, _ = make_judge(tmp_path, rubric=rubric, evidence=log)
    rubric.write_bytes(b"# Rubric\n\nPass if the summary is short.\n")

    with pytest.raises(RubricDriftError) as excinfo:
        make_judge(tmp_path, rubric=rubric, evidence=log)

    error = excinfo.value
    new_hash = hashlib.sha256(rubric.read_bytes()).hexdigest()
    assert error.recorded_hash == first.rubric_hash
    assert error.current_hash == new_hash
    message = str(error)
    assert first.rubric_hash in message
    assert new_hash in message
    assert "summariser" in message
    # The refused construction logged no new init: the drift is the whole story.
    assert len(records(log, EVENT_JUDGE_INIT)) == 1


def test_drift_is_scoped_to_the_judge_that_recorded_the_hash(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    make_judge(tmp_path, rubric=write_rubric(tmp_path, name="a.md"), evidence=log, name="a")

    # A different judge with a different rubric on the same shared log is not drift.
    other, _, _ = make_judge(
        tmp_path,
        rubric=write_rubric(tmp_path, "# Other rubric\n", "b.md"),
        evidence=log,
        name="b",
    )

    assert other.name == "b"
    assert len(records(log, EVENT_JUDGE_INIT)) == 2


def test_accepted_rubric_change_is_recorded_with_both_hashes(tmp_path: Path) -> None:
    rubric = write_rubric(tmp_path)
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    first, _, _ = make_judge(tmp_path, rubric=rubric, evidence=log)
    rubric.write_bytes(b"# Rubric\n\nPass if the summary is faithful and short.\n")

    second, _, _ = make_judge(tmp_path, rubric=rubric, evidence=log, accept_rubric_change=True)

    (accepted,) = records(log, EVENT_RUBRIC_CHANGE_ACCEPTED)
    assert accepted.payload["recorded_hash"] == first.rubric_hash
    assert accepted.payload["current_hash"] == second.rubric_hash
    assert second.rubric_hash != first.rubric_hash
    assert accepted.payload["judge"] == "summariser"
    # The acceptance precedes the init that now carries the new hash.
    event_types = [record.event_type for record in log.read()]
    assert event_types == [
        EVENT_JUDGE_INIT,
        EVENT_RUBRIC_CHANGE_ACCEPTED,
        EVENT_JUDGE_INIT,
    ]
    assert records(log, EVENT_JUDGE_INIT)[-1].payload["rubric_hash"] == second.rubric_hash


def test_accepting_a_change_that_did_not_happen_records_nothing(tmp_path: Path) -> None:
    rubric = write_rubric(tmp_path)
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    make_judge(tmp_path, rubric=rubric, evidence=log)

    make_judge(tmp_path, rubric=rubric, evidence=log, accept_rubric_change=True)

    assert records(log, EVENT_RUBRIC_CHANGE_ACCEPTED) == []


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #


def test_passing_and_failing_verdicts_both_parse(tmp_path: Path) -> None:
    judge, _, log = make_judge(tmp_path, responses=[PASS_JSON, FAIL_JSON])

    good = judge.evaluate("Summarise the memo.", "Faithful summary.")
    bad = judge.evaluate("Summarise the memo.", "Invented a number.")

    assert (good.passed, good.score) == (True, 5.0)
    assert (bad.passed, bad.score) == (False, 2.0)
    assert good.raw == PASS_JSON
    assert good.model_id == PINNED_MODEL
    assert good.rubric_hash == judge.rubric_hash
    assert [record.payload["passed"] for record in records(log, EVENT_JUDGE_VERDICT)] == [
        True,
        False,
    ]


def test_verdict_is_frozen(tmp_path: Path) -> None:
    # A verdict is evidence: if a caller could mutate it after it was logged, the
    # object in memory and the fact on disk would disagree with no trace.
    judge, _, _ = make_judge(tmp_path, responses=[PASS_JSON])
    verdict = judge.evaluate("Summarise the memo.", "Faithful summary.")

    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.passed = False  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.score = 1.0  # type: ignore[misc]


def test_verdict_field_names_are_part_of_the_contract() -> None:
    fields = {field.name for field in dataclasses.fields(Verdict)}

    assert {"passed", "score", "raw"} <= fields


def test_exactly_one_verdict_record_per_successful_evaluate(tmp_path: Path) -> None:
    # Never zero (an unrecorded verdict is not evidence), never two (a
    # double-counted verdict skews every rate computed from the log).
    judge, _, log = make_judge(tmp_path, responses=[PASS_JSON, FAIL_JSON, PASS_JSON])

    for output in ("first", "second", "third"):
        judge.evaluate("Summarise the memo.", output)

    verdicts = records(log, EVENT_JUDGE_VERDICT)
    assert len(verdicts) == 3
    assert [record.payload["model_id"] for record in verdicts] == [PINNED_MODEL] * 3
    assert [record.payload["rubric_hash"] for record in verdicts] == [judge.rubric_hash] * 3


def test_omitted_score_is_none_rather_than_zero(tmp_path: Path) -> None:
    judge, _, _ = make_judge(tmp_path, responses=['{"pass": true}'])

    verdict = judge.evaluate("Summarise the memo.", "Faithful summary.")

    assert verdict.passed is True
    assert verdict.score is None


def test_null_score_is_none(tmp_path: Path) -> None:
    judge, _, _ = make_judge(tmp_path, responses=['{"pass": false, "score": null}'])

    assert judge.evaluate("in", "out").score is None


def test_prompt_carries_the_rubric_and_both_sides_of_the_exchange(tmp_path: Path) -> None:
    judge, adapter, _ = make_judge(tmp_path, responses=[PASS_JSON])

    judge.evaluate("Summarise the quarterly memo.", "The memo recommends X.")

    prompt = adapter.prompts[0]
    assert "Pass if the summary is faithful." in prompt
    assert "Summarise the quarterly memo." in prompt
    assert "The memo recommends X." in prompt
    assert prompt.rstrip().endswith(OUTPUT_FORMAT_INSTRUCTION)


# --------------------------------------------------------------------------- #
# parsing: tolerated shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        PASS_JSON,
        f"```json\n{PASS_JSON}\n```",
        f"```\n{PASS_JSON}\n```",
        f"Here is my assessment.\n\n{PASS_JSON}\n\nLet me know if you need more.",
        f"  \n{PASS_JSON}\n  ",
    ],
    ids=["bare", "fenced-json", "fenced-plain", "prose-wrapped", "whitespace"],
)
def test_common_real_world_wrappers_parse(tmp_path: Path, raw: str) -> None:
    judge, _, log = make_judge(tmp_path, responses=[raw])

    verdict = judge.evaluate("in", "out")

    assert verdict.passed is True
    assert verdict.score == 5.0
    assert verdict.raw == raw  # the raw response is preserved exactly as received
    assert len(records(log, EVENT_JUDGE_VERDICT)) == 1


def test_nested_objects_inside_the_verdict_do_not_confuse_the_parser(tmp_path: Path) -> None:
    raw = '{"pass": true, "score": 4, "detail": {"faithfulness": "ok", "coverage": "ok"}}'
    judge, _, _ = make_judge(tmp_path, responses=[raw])

    verdict = judge.evaluate("in", "out")

    assert (verdict.passed, verdict.score) == (True, 4.0)


def test_passed_spelling_of_the_pass_field_is_accepted(tmp_path: Path) -> None:
    judge, _, _ = make_judge(tmp_path, responses=['{"passed": true, "score": 4}'])

    assert judge.evaluate("in", "out").passed is True


# --------------------------------------------------------------------------- #
# parsing: refusals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        "Yes, this summary passes -- it is faithful and complete.",
        "PASS",
        "",
        "   ",
        "{not json at all}",
        '{"score": 5, "reason": "faithful"}',
        '{"pass": "true", "score": 5}',
        '{"pass": "yes"}',
        '{"pass": 1}',
        '{"pass": null}',
        '{"pass": true, "score": "five"}',
        '{"pass": true, "score": true}',
    ],
    ids=[
        "prose-yes",
        "bare-word",
        "empty",
        "whitespace-only",
        "not-json",
        "missing-pass",
        "string-pass",
        "yes-string",
        "int-pass",
        "null-pass",
        "string-score",
        "bool-score",
    ],
)
def test_unparseable_responses_raise_rather_than_failing_the_verdict(
    tmp_path: Path, raw: str
) -> None:
    # Never passed=False. An unparseable answer is missing data, and folding it
    # into the failure bucket biases the pass rate by the judge's own flakiness.
    judge, _, log = make_judge(tmp_path, responses=[raw])

    with pytest.raises(JudgeOutputError):
        judge.evaluate("in", "out")

    assert records(log, EVENT_JUDGE_VERDICT) == []


def test_parse_failure_logs_the_raw_response_and_no_verdict(tmp_path: Path) -> None:
    raw = "I think it's fine, honestly. Call it a pass?"
    judge, _, log = make_judge(tmp_path, responses=[raw])

    with pytest.raises(JudgeOutputError) as excinfo:
        judge.evaluate("Summarise the memo.", "A summary.")

    assert excinfo.value.raw == raw
    (failure,) = records(log, EVENT_JUDGE_PARSE_FAILURE)
    assert failure.payload["raw"] == raw
    assert failure.payload["judge"] == "summariser"
    assert failure.payload["model_id"] == PINNED_MODEL
    assert failure.payload["rubric_hash"] == judge.rubric_hash
    # Explicitly: the failure is on the record, and no verdict was invented.
    assert records(log, EVENT_JUDGE_VERDICT) == []
    assert [record.event_type for record in log.read()] == [
        EVENT_JUDGE_INIT,
        EVENT_JUDGE_PARSE_FAILURE,
    ]


def test_two_conflicting_verdict_objects_are_ambiguous_and_raise(tmp_path: Path) -> None:
    # Picking one would be a coin flip recorded as a measurement.
    raw = f"First thought:\n{PASS_JSON}\nOn reflection:\n{FAIL_JSON}"
    judge, _, log = make_judge(tmp_path, responses=[raw])

    with pytest.raises(JudgeOutputError, match="conflicting"):
        judge.evaluate("in", "out")

    assert records(log, EVENT_JUDGE_PARSE_FAILURE)[0].payload["raw"] == raw
    assert records(log, EVENT_JUDGE_VERDICT) == []


def test_a_restated_identical_verdict_is_not_ambiguous(tmp_path: Path) -> None:
    judge, _, _ = make_judge(tmp_path, responses=[f"{PASS_JSON}\n\nAgain:\n{PASS_JSON}"])

    assert judge.evaluate("in", "out").passed is True


@pytest.mark.parametrize("score", [0, 0.9, 6, 100, -1])
def test_score_outside_the_rubric_range_raises_rather_than_clamping(
    tmp_path: Path, score: float
) -> None:
    # A judge that answers 9 on a 1-5 scale misread the rubric; clamping to 5
    # manufactures a measurement nobody made.
    judge, _, log = make_judge(tmp_path, responses=[f'{{"pass": true, "score": {score}}}'])

    with pytest.raises(JudgeOutputError, match="outside the rubric's range"):
        judge.evaluate("in", "out")

    assert records(log, EVENT_JUDGE_VERDICT) == []


@pytest.mark.parametrize("score", [SCORE_MIN, SCORE_MAX])
def test_scores_at_the_boundaries_are_accepted(tmp_path: Path, score: float) -> None:
    judge, _, _ = make_judge(tmp_path, responses=[f'{{"pass": true, "score": {score}}}'])

    assert judge.evaluate("in", "out").score == score
