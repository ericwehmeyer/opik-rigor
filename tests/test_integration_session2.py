"""The library eating its own cooking.

Everything else in the suite tests one module against a double. This file uses
opik_rigor the way a caller would: a stochastic judge, sampled n times, gated by the
statistical assertions, with the whole run recorded in an evidence log and the
result stored as a hash-verified baseline.

It exists because a library that gates stochastic systems should be willing to
gate one itself. If ``assert_pass_rate`` is awkward to use, or the evidence it
leaves is not enough to reconstruct what happened, that shows up here first.

Determinism note: the judge under test is genuinely stochastic -- it returns
different verdicts on identical input -- but its randomness comes from a seeded
``FakeAdapter``, so these tests are exactly reproducible. That is the same
discipline the library asks of its users, and the reason ``FakeAdapter`` takes a
seed at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opik_rigor import (
    Baseline,
    EvidenceLog,
    FakeAdapter,
    PassRateError,
    PinnedJudge,
    RegressionError,
    assert_no_regression,
    assert_pass_rate,
    assert_score_distribution,
    sample,
)
from opik_rigor.evidence import EVENT_ASSERTION, EVENT_JUDGE_VERDICT, EVENT_SAMPLE_COMPLETED

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBRIC = REPO_ROOT / "rubrics" / "example-rubric.md"

#: Seeds are fixed and named so a failure can be reproduced exactly. Changing one
#: changes the observed pass rate, which is the point: these numbers are measured
#: from the fixture, not chosen to make an assertion pass.
GOOD_SEED = 20260813
POOR_SEED = 424242

# A judge that mostly passes: four of five scripted verdicts are a pass, drawn
# with replacement, so the observed rate varies run to run around 0.8.
MOSTLY_PASSES = [
    '{"pass": true, "score": 5, "reason": "faithful and complete"}',
    '{"pass": true, "score": 4, "reason": "minor secondary point missing"}',
    '{"pass": true, "score": 5, "reason": "faithful, tight, correctly attributed"}',
    '{"pass": true, "score": 4, "reason": "wording looser than needed"}',
    '{"pass": false, "score": 2, "reason": "omits the stated caveat"}',
]

# The same judge after something regressed: the same five verdicts, reweighted.
MOSTLY_FAILS = [
    '{"pass": false, "score": 2, "reason": "omits the stated caveat"}',
    '{"pass": false, "score": 1, "reason": "states a figure the source does not"}',
    '{"pass": false, "score": 2, "reason": "presents a disputed claim as settled"}',
    '{"pass": true, "score": 4, "reason": "faithful if loose"}',
    '{"pass": false, "score": 3, "reason": "attribution is vague"}',
]


def judged(tmp_path: Path, script: list[str], seed: int) -> tuple[PinnedJudge, EvidenceLog]:
    """A real judge over a real rubric, backed by a seeded stochastic adapter."""
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    adapter = FakeAdapter(responses=script, seed=seed)
    return PinnedJudge(adapter, RUBRIC, log, name="summariser"), log


def run_eval(judge: PinnedJudge, n: int, **kwargs: object) -> object:
    return sample(lambda: judge.evaluate("Summarise the memo.", "A summary."), n, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# the full loop
# --------------------------------------------------------------------------- #


def test_a_stochastic_judge_is_sampled_and_gated_end_to_end(tmp_path: Path) -> None:
    judge, log = judged(tmp_path, MOSTLY_PASSES, GOOD_SEED)

    result = run_eval(judge, 100, evidence=log, label="summariser-nightly")
    report = assert_pass_rate(result, 0.6, evidence=log, label="summariser-nightly")

    # The judge really did vary -- otherwise this file would be testing a constant.
    assert 0 < result.successes < result.n
    assert report["passed"] is True
    assert report["lower_bound"] <= result.pass_rate

    # And the whole run is reconstructable from the log alone: 100 verdicts, the
    # sample that collected them, and the gate that judged them.
    events = [record.event_type for record in log.read()]
    assert events.count(EVENT_JUDGE_VERDICT) == 100
    assert events.count(EVENT_SAMPLE_COMPLETED) == 1
    assert events.count(EVENT_ASSERTION) == 1


def test_the_gate_fails_a_judge_that_is_genuinely_worse(tmp_path: Path) -> None:
    judge, log = judged(tmp_path, MOSTLY_FAILS, POOR_SEED)

    result = run_eval(judge, 100)

    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate(result, 0.8, evidence=log)

    # The failure message is the statistical report, not "assert 0.2 >= 0.8".
    message = str(excinfo.value)
    assert str(result.n) in message
    assert "0.8" in message
    assert excinfo.value.stats["lower_bound"] < 0.8


def test_a_small_sample_cannot_sneak_past_the_gate_that_a_large_one_clears(
    tmp_path: Path,
) -> None:
    # The argument for the whole library, run against a live judge rather than
    # asserted about hand-written counts: the same judge, the same true quality,
    # gated at the same bar -- and only the adequately sampled run is allowed
    # through, because only it is evidence.
    small_judge, _ = judged(tmp_path / "small", MOSTLY_PASSES, GOOD_SEED)
    large_judge, _ = judged(tmp_path / "large", MOSTLY_PASSES, GOOD_SEED)

    small = run_eval(small_judge, 10)
    large = run_eval(large_judge, 400)

    bar = 0.6
    with pytest.raises(PassRateError):
        assert_pass_rate(small, bar)
    assert assert_pass_rate(large, bar)["passed"] is True


def test_score_distribution_gates_the_same_sampled_run(tmp_path: Path) -> None:
    judge, log = judged(tmp_path, MOSTLY_PASSES, GOOD_SEED)

    result = run_eval(judge, 120)
    scores = result.scores()

    assert len(scores) == 120  # every verdict in this script carries a score
    assert assert_score_distribution(result, min_mean=3.0, evidence=log)["passed"] is True

    # A bar the judge genuinely does not clear.
    with pytest.raises(Exception) as excinfo:
        assert_score_distribution(result, min_mean=4.9)
    assert "mean" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# baselines, across a simulated regression
# --------------------------------------------------------------------------- #


def test_a_recorded_baseline_detects_a_later_regression(tmp_path: Path) -> None:
    good_judge, _ = judged(tmp_path / "good", MOSTLY_PASSES, GOOD_SEED)
    before = run_eval(good_judge, 150)
    path = Baseline.from_sample("summariser", before).save(tmp_path / "baseline.json")

    # ... time passes, something regresses ...
    poor_judge, log = judged(tmp_path / "poor", MOSTLY_FAILS, POOR_SEED)
    after = run_eval(poor_judge, 150)

    loaded = Baseline.load(path)
    with pytest.raises(RegressionError) as excinfo:
        assert_no_regression(after.scores(), loaded.scores, evidence=log)

    assert excinfo.value.stats["p_value"] < 0.05
    assert [r.event_type for r in log.read()].count(EVENT_ASSERTION) == 1


def test_the_same_judge_re_run_is_not_a_regression_against_its_own_baseline(
    tmp_path: Path,
) -> None:
    # The false-alarm direction, which matters more in practice than detection: a
    # regression gate that fires on an unchanged system gets switched off, and
    # then it is not protecting anything.
    first_judge, _ = judged(tmp_path / "first", MOSTLY_PASSES, GOOD_SEED)
    baseline = Baseline.from_sample("summariser", run_eval(first_judge, 200))

    second_judge, _ = judged(tmp_path / "second", MOSTLY_PASSES, POOR_SEED)
    rerun = run_eval(second_judge, 200)

    report = assert_no_regression(rerun.scores(), baseline.scores)

    assert report["passed"] is True
    assert report["p_value"] >= 0.05


def test_a_tampered_baseline_is_refused_before_it_can_hide_a_regression(
    tmp_path: Path,
) -> None:
    # The attack the digest exists to stop: make the regression go away by
    # lowering the bar it is compared against.
    good_judge, _ = judged(tmp_path / "good", MOSTLY_PASSES, GOOD_SEED)
    path = Baseline.from_sample("summariser", run_eval(good_judge, 60)).save(
        tmp_path / "baseline.json"
    )
    document = path.read_text(encoding="utf-8")
    path.write_text(document.replace('"scores": [', '"scores": [1.0, '), encoding="utf-8")

    with pytest.raises(Exception) as excinfo:
        Baseline.load(path)
    assert "digest" in str(excinfo.value).lower()


# --------------------------------------------------------------------------- #
# reproducibility of this file itself
# --------------------------------------------------------------------------- #


def test_the_same_seed_reproduces_the_same_verdicts_and_the_same_gate_decision(
    tmp_path: Path,
) -> None:
    # If this ever fails, every other assertion in this file is a coin flip.
    def once(where: Path) -> tuple[int, float]:
        judge, _ = judged(where, MOSTLY_PASSES, GOOD_SEED)
        result = run_eval(judge, 80)
        return result.successes, assert_pass_rate(result, 0.5)["lower_bound"]

    assert once(tmp_path / "a") == once(tmp_path / "b")


def test_concurrent_sampling_gives_the_same_gate_decision_as_serial(tmp_path: Path) -> None:
    # Concurrency must change only the wall clock. The multiset of draws from a
    # seeded adapter is fixed; only which thread gets which draw is not.
    serial_judge, _ = judged(tmp_path / "serial", MOSTLY_PASSES, GOOD_SEED)
    pooled_judge, _ = judged(tmp_path / "pooled", MOSTLY_PASSES, GOOD_SEED)

    serial = run_eval(serial_judge, 120)
    pooled = run_eval(pooled_judge, 120, concurrency=8)

    assert serial.successes == pooled.successes
    assert sorted(serial.scores()) == sorted(pooled.scores())
    assert assert_pass_rate(serial, 0.6)["passed"] == assert_pass_rate(pooled, 0.6)["passed"]
