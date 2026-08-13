"""Tests for the statistical gates.

These tests are written so that the implementation cannot validate itself. Not
one expected number here was produced by calling the code under test. Every
number comes from one of four places:

* **Oracle 1** -- :func:`wilson_lower_by_bisection`, which root-finds the score
  inequality ``|p_hat - p| <= z * sqrt(p(1-p)/n)`` that *defines* the Wilson
  interval. That is a completely different algorithm from the closed-form
  quadratic the module evaluates, so agreement between the two is real evidence
  rather than a tautology.
* **Oracle 2** -- analytic identities derived on paper (the ``successes == n``
  collapse, the ``successes == 0`` cancellation, the ``p -> 1-p`` symmetry of the
  interval, and the monotonicities any correct bound must satisfy).
* **Oracle 3** -- literal fixtures computed by bisection before the
  implementation existed, hardcoded below.
* **Oracle 4** -- :func:`mann_whitney_u1`, which hand-counts the U statistic from
  its definition. Checking scipy against scipy would prove nothing.

The subject of this library is flaky stochastic tests, so a flaky test here would
be self-refuting: every random draw comes from an explicitly seeded
``random.Random``, never from the bare ``random`` module.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import pytest
from scipy.stats import norm

from rigor.distribution import (
    PassRateError,
    RegressionError,
    ScoreDistributionError,
    assert_no_regression,
    assert_pass_rate,
    assert_score_distribution,
    wilson_interval,
    wilson_lower_bound,
)
from rigor.errors import StatisticalAssertionError
from rigor.evidence import EVENT_ASSERTION, EvidenceLog
from rigor.judge import Verdict
from rigor.sampling import Run, SampleResult

# --------------------------------------------------------------------------- #
# seeds -- stated once, never bare random.*
# --------------------------------------------------------------------------- #

#: Seeds for the three power fixtures. Fixed constants, not clock-derived: a test
#: about statistical flakiness that is itself flaky proves the opposite of its
#: thesis.
BASELINE_SEED = 20260813
SAME_SEED = 20260814
SHIFTED_SEED = 20260815

#: The one-sided z for the default 0.95 confidence, quoted rather than computed
#: from the module: z = norm.ppf(0.95).
Z_95 = 1.6448536269514722
Z_95_SQUARED = 2.705543454095413


# --------------------------------------------------------------------------- #
# Oracle 1 -- Wilson by bisection
# --------------------------------------------------------------------------- #


def wilson_lower_by_bisection(successes: int, n: int, confidence: float = 0.95) -> float:
    """Independent oracle: bisects the score-test inequality that DEFINES the
    interval, rather than evaluating the closed form under test."""
    if successes == 0:
        return 0.0
    z = norm.ppf(confidence)
    p_hat = successes / n

    def g(p: float) -> float:
        return (p_hat - p) - z * math.sqrt(p * (1.0 - p) / n)

    lo, hi = 0.0, p_hat
    if g(lo) < 0:
        return 0.0
    for _ in range(400):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- #
# Oracle 4 -- Mann-Whitney U by hand-counting
# --------------------------------------------------------------------------- #


def mann_whitney_u1(current: Sequence[float], baseline: Sequence[float]) -> float:
    """U1 straight from its definition, by counting pairs.

    ``U1 = #{(c, b) : c > b} + 0.5 * #{(c, b) : c == b}`` over the full cartesian
    product. No ranking, no normal approximation, no scipy.
    """
    wins = sum(1 for c, b in product(current, baseline) if c > b)
    ties = sum(1 for c, b in product(current, baseline) if c == b)
    return wins + 0.5 * ties


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

#: Oracle 3. Computed by bisecting the defining inequality before the
#: implementation was written; hardcoded so the module cannot move them.
WILSON_LITERALS: tuple[tuple[int, int, float], ...] = (
    (18, 20, 0.7383369536731332),
    (20, 20, 0.8808421626390355),
    (0, 20, 0.0),
    (14, 20, 0.5161962804075577),
    (190, 200, 0.9181081670817905),
    (200, 200, 0.9866528393452243),
    (900, 1000, 0.8832999865159246),
    (1, 10, 0.022634908730867202),
    (99, 100, 0.956418226465944),
)

#: The sweep grid for the closed form versus Oracle 1.
SWEEP_SIZES = (1, 2, 5, 10, 20, 100, 200)
SWEEP_CONFIDENCES = (0.80, 0.90, 0.95, 0.99)


def sample_of_outcomes(outcomes: Sequence[bool]) -> SampleResult:
    """A real :class:`SampleResult` carrying the given pass/fail outcomes."""
    runs = tuple(
        Run(index=index, value=Verdict(passed=bool(outcome), score=None, raw=""), outcome=outcome)
        for index, outcome in enumerate(outcomes)
    )
    return SampleResult(runs=runs, wall_clock=0.0)


def sample_of_judge_scores(scores: Sequence[float]) -> SampleResult:
    """A real :class:`SampleResult` whose ``.scores()`` are the given numbers."""
    runs = tuple(
        Run(index=index, value=Verdict(passed=True, score=float(score), raw=""), outcome=True)
        for index, score in enumerate(scores)
    )
    return SampleResult(runs=runs, wall_clock=0.0)


def gate_report(gate: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """The report a gate produced, whether it returned it or raised with it."""
    try:
        return gate(*args, **kwargs)
    except StatisticalAssertionError as exc:
        return dict(exc.stats)


def jsonable(value: Any) -> Any:
    """Tuples become lists on the way through JSON; normalise for comparison."""
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def as_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {key: jsonable(value) for key, value in report.items()}


@pytest.fixture(scope="module")
def power_samples() -> tuple[list[float], list[float], list[float]]:
    """Three seeded gaussian samples: a baseline, a redraw of it, and a shift down."""
    rng = random.Random(BASELINE_SEED)
    baseline = [rng.gauss(4.0, 1.0) for _ in range(200)]
    rng2 = random.Random(SAME_SEED)
    same = [rng2.gauss(4.0, 1.0) for _ in range(200)]
    rng3 = random.Random(SHIFTED_SEED)
    shifted = [rng3.gauss(3.5, 1.0) for _ in range(200)]
    return baseline, same, shifted


# --------------------------------------------------------------------------- #
# Oracle 1 -- the closed form against bisection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("confidence", SWEEP_CONFIDENCES)
def test_closed_form_agrees_with_bisecting_the_defining_inequality(confidence: float) -> None:
    # The closed-form quadratic and a bisection of |p_hat - p| <= z*sqrt(p(1-p)/n)
    # share no arithmetic, so agreement across the whole grid is evidence rather
    # than a restatement.
    worst = 0.0
    worst_case: tuple[int, int] | None = None
    for n in SWEEP_SIZES:
        for successes in range(n + 1):
            closed_form = wilson_lower_bound(successes, n, confidence)
            bisected = wilson_lower_by_bisection(successes, n, confidence)
            difference = abs(closed_form - bisected)
            if difference > worst:
                worst, worst_case = difference, (successes, n)

    assert worst < 1e-9, (
        f"widest disagreement {worst:.3e} at successes/n={worst_case} "
        f"with confidence={confidence}"
    )


def test_two_sided_interval_is_the_one_sided_bound_at_the_split_error_budget() -> None:
    # Definitional, not empirical: a two-sided interval at confidence c spends
    # (1-c)/2 in each tail, so its lower end is the one-sided bound at
    # 1 - (1-c)/2. Oracle 1 is asked for that level directly.
    for confidence in SWEEP_CONFIDENCES:
        one_sided_level = 1.0 - (1.0 - confidence) / 2.0
        for n in (1, 5, 20, 100):
            for successes in range(n + 1):
                lower, _ = wilson_interval(successes, n, confidence)
                expected = wilson_lower_by_bisection(successes, n, one_sided_level)
                assert abs(lower - expected) < 1e-9


def test_two_sided_lower_end_sits_below_the_one_sided_bound() -> None:
    # Same nominal confidence, more error budget spent on the tail you are not
    # gating on. The two are not interchangeable and this pins which is which.
    for n in (5, 20, 200):
        for successes in range(1, n + 1):
            two_sided_lower, _ = wilson_interval(successes, n, 0.95)
            assert two_sided_lower < wilson_lower_bound(successes, n, 0.95)


# --------------------------------------------------------------------------- #
# Oracle 2 -- analytic identities
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 5, 10, 20, 100, 200, 1000])
def test_at_a_perfect_score_the_bound_collapses_to_n_over_n_plus_z_squared(n: int) -> None:
    # With p = 1: centre = (1 + z**2/(2n)) / (1 + z**2/n) = (n + z**2/2)/(n + z**2)
    #             half   = z/(1 + z**2/n) * sqrt(0 + z**2/(4n**2))
    #                    = z * n/(n + z**2) * z/(2n) = z**2 / (2(n + z**2))
    #             lower  = centre - half = n / (n + z**2)
    expected = n / (n + Z_95_SQUARED)

    assert wilson_lower_bound(n, n) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 5, 20, 200, 1000])
def test_at_zero_successes_the_bound_is_exactly_zero(n: int) -> None:
    # With p = 0: centre = z**2 / (2(n + z**2)) and half = z**2 / (2(n + z**2)),
    # so the two cancel identically. Exact equality, not approx: a "pass rate" of
    # -1e-17 or 4e-18 in a report is noise the clamp exists to remove.
    assert wilson_lower_bound(0, n) == 0.0


def test_at_zero_successes_the_bound_is_nil_at_every_n_and_confidence() -> None:
    # The same cancellation, swept rather than spot-checked. Tolerant to 1e-15 so
    # that it measures the mathematics; the exactness of the result is the
    # separate (xfailing) test below.
    for confidence in (*SWEEP_CONFIDENCES, 0.50, 0.999):
        for n in range(1, 301):
            assert wilson_lower_bound(0, n, confidence) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("n", [11, 15, 19, 22, 24])
def test_at_zero_successes_the_bound_is_exactly_zero_at_every_n(n: int) -> None:
    # Regression test. Analytically centre and half are the same expression at
    # p = 0, so the difference is exactly zero -- but they are evaluated in
    # different orders and the cancellation left a residue of either sign. _clamp
    # squeezed the negative side only, so wilson_lower_bound(0, 11) returned
    # 1.3877787807814457e-17: a lower confidence bound sitting *above* its own
    # point estimate, on roughly 15% of (n, confidence) pairs. These particular n
    # are the ones that exhibited it at the default confidence.
    assert wilson_lower_bound(0, n) == 0.0


@pytest.mark.parametrize("n", [11, 15, 19, 22, 24])
def test_the_bound_never_exceeds_the_point_estimate_at_zero_successes(n: int) -> None:
    # The invariant the residue violated, stated directly.
    assert wilson_lower_bound(0, n) <= 0.0 / n


def test_the_interval_is_symmetric_under_swapping_successes_for_failures() -> None:
    # Substituting successes -> n - successes maps p -> 1-p, which leaves the
    # half-width unchanged and sends centre -> 1 - centre. Hence
    # upper(successes) == 1 - lower(n - successes) at the same z.
    for n in (1, 5, 20, 100):
        for successes in range(n + 1):
            _, upper = wilson_interval(successes, n, 0.95)
            mirrored_lower, _ = wilson_interval(n - successes, n, 0.95)
            assert upper == pytest.approx(1.0 - mirrored_lower, abs=1e-12)


def test_the_bound_never_exceeds_the_point_estimate() -> None:
    # A lower confidence bound above the observed rate would be a bound on
    # nothing. True for every cell of the grid, at every confidence.
    #
    # successes == 0 is excluded here only because the implementation currently
    # violates it there by a few times 1e-17; that failure is asserted on its own
    # in test_at_zero_successes_the_bound_is_exactly_zero_at_every_n rather than
    # being swallowed by a tolerance in this one.
    for confidence in SWEEP_CONFIDENCES:
        for n in SWEEP_SIZES:
            for successes in range(1, n + 1):
                assert wilson_lower_bound(successes, n, confidence) <= successes / n


def increasing(values: Sequence[float]) -> bool:
    return all(a < b for a, b in zip(values[:-1], values[1:], strict=True))


def decreasing(values: Sequence[float]) -> bool:
    return all(a > b for a, b in zip(values[:-1], values[1:], strict=True))


def test_the_bound_rises_with_successes_at_fixed_n() -> None:
    for n in (2, 5, 20, 100):
        assert increasing([wilson_lower_bound(successes, n, 0.95) for successes in range(n + 1)])


def test_the_bound_rises_with_n_at_a_fixed_observed_rate() -> None:
    # 18/20 and 900/1000 are both "90%" and the whole library exists because they
    # are not the same evidence. The bound has to say so.
    assert increasing(
        [
            wilson_lower_bound(9, 10),
            wilson_lower_bound(18, 20),
            wilson_lower_bound(90, 100),
            wilson_lower_bound(180, 200),
            wilson_lower_bound(900, 1000),
        ]
    )


def test_the_bound_falls_as_confidence_rises() -> None:
    # More confidence demanded of the same evidence buys a weaker floor.
    for successes, n in ((1, 10), (14, 20), (190, 200), (900, 1000)):
        assert decreasing(
            [
                wilson_lower_bound(successes, n, confidence)
                for confidence in (0.50, 0.80, 0.90, 0.95, 0.99, 0.999)
            ]
        )


# --------------------------------------------------------------------------- #
# Oracle 3 -- literal fixtures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("successes", "n", "expected"), WILSON_LITERALS)
def test_wilson_lower_bound_matches_the_precomputed_literals(
    successes: int, n: int, expected: float
) -> None:
    # Bisected independently before the implementation existed. If the closed
    # form is rewritten, these do not move with it.
    assert wilson_lower_bound(successes, n) == pytest.approx(expected, abs=1e-12)


def test_the_default_confidence_is_the_one_sided_ninety_five_percent_z() -> None:
    # Pins the one-sided convention: at 20/20 the bound is n/(n+z**2) with
    # z = norm.ppf(0.95) = 1.6448536269514722, i.e. 20/22.705543... = 0.88084...
    # The two-sided z (1.959964) would give 20/23.8415... = 0.83887..., so this
    # literal alone distinguishes the two conventions.
    assert wilson_lower_bound(20, 20, 0.95) == pytest.approx(0.8808421626390355, abs=1e-12)
    assert wilson_lower_bound(20, 20) == wilson_lower_bound(20, 20, 0.95)
    analytic = 20 / (20 + Z_95_SQUARED)
    assert analytic == pytest.approx(0.8808421626390355, abs=1e-12)


# --------------------------------------------------------------------------- #
# assert_pass_rate -- the headline property
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("successes", "n"), [(18, 20), (180, 200), (900, 1000)])
def test_ninety_percent_observed_never_clears_a_ninety_percent_bar(
    successes: int, n: int
) -> None:
    # The bound approaches the observed rate from below and never reaches it, so
    # an observed rate sitting exactly on min_rate cannot clear it at any n. You
    # have to beat the bar, not hit it -- more runs is the wrong advice here, and
    # the message has to say so instead of sending the reader into a loop of
    # raising n.
    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate((successes, n), 0.9)

    stats = excinfo.value.stats
    assert stats["pass_rate"] == 0.9
    assert stats["lower_bound"] < 0.9
    assert stats["underpowered"] is True
    assert stats["runs_needed"] is None
    assert "no sample size clears this bar" in str(excinfo.value)


def test_ninety_five_percent_observed_does_clear_a_ninety_percent_bar() -> None:
    # 190/200: the bound is 0.9181081670817905 by Oracle 3, comfortably over 0.9.
    report = assert_pass_rate((190, 200), 0.9)

    assert report["passed"] is True
    assert report["lower_bound"] == pytest.approx(0.9181081670817905, abs=1e-12)
    assert report["pass_rate"] == 0.95
    assert report["n"] == 200
    assert report["successes"] == 190
    assert report["failures"] == 10
    assert report["min_rate"] == 0.9
    assert report["method"] == "wilson-one-sided"


def test_the_same_evidence_in_three_input_forms_gives_one_report() -> None:
    outcomes = [True] * 14 + [False] * 6

    from_sample = assert_pass_rate(sample_of_outcomes(outcomes), 0.4)
    from_counts = assert_pass_rate((14, 20), 0.4)
    from_bools = assert_pass_rate(outcomes, 0.4)

    assert from_sample == from_counts == from_bools
    assert from_counts["successes"] == 14
    assert from_counts["n"] == 20


def test_a_two_element_bool_tuple_is_outcomes_and_a_two_element_int_tuple_is_counts() -> None:
    # The documented disambiguation. (1, 0) genuinely could be "one pass and one
    # fail" or "one success out of zero runs"; the rule is that a two-element
    # tuple of non-boolean ints is counts, and everything else is outcomes.
    from_bools = assert_pass_rate((True, False), 0.0)
    assert from_bools["n"] == 2
    assert from_bools["successes"] == 1

    # (1, 0) reads as counts, which means n=0, which is not a rate at all.
    with pytest.raises(ValueError, match="n must be >= 1"):
        assert_pass_rate((1, 0), 0.0)

    from_counts = assert_pass_rate((2, 5), 0.0)
    assert from_counts["n"] == 5
    assert from_counts["successes"] == 2


def test_the_failure_message_is_the_statistical_report() -> None:
    # 14/20 = 0.7000 observed; bound 0.5161962804075577 -> "0.5162" at .4f, by
    # Oracle 3. Every number a reader needs is in the prose.
    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate((14, 20), 0.9, label="helpfulness")

    message = str(excinfo.value)
    assert "helpfulness" in message
    assert "14/20" in message  # the counts
    assert "0.7000" in message  # the observed rate
    assert "0.5162" in message  # the lower bound
    assert "0.9000" in message  # the threshold
    assert "Wilson" in message
    assert "more runs will not fix it" in message


def test_the_same_numbers_are_on_the_exception_so_nobody_parses_prose() -> None:
    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate((14, 20), 0.9, label="helpfulness")

    stats = excinfo.value.stats
    assert stats["gate"] == "pass_rate"
    assert stats["label"] == "helpfulness"
    assert stats["passed"] is False
    assert stats["n"] == 20
    assert stats["successes"] == 14
    assert stats["failures"] == 6
    assert stats["pass_rate"] == 0.7
    assert stats["lower_bound"] == pytest.approx(0.5161962804075577, abs=1e-12)
    assert stats["min_rate"] == 0.9
    assert stats["confidence"] == 0.95
    assert stats["underpowered"] is False


def test_an_underpowered_failure_says_so_rather_than_blaming_the_system() -> None:
    # 18/20 observed 0.9 clears min_rate 0.85, but its bound (0.7383369536731332
    # by Oracle 3) does not. These are opposite diagnoses -- "raise n" versus
    # "fix the system" -- and conflating them sends the reader the wrong way.
    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate((18, 20), 0.85)

    message = str(excinfo.value)
    stats = excinfo.value.stats
    assert stats["underpowered"] is True
    assert stats["pass_rate"] == 0.9
    assert stats["pass_rate"] > stats["min_rate"]
    assert stats["lower_bound"] == pytest.approx(0.7383369536731332, abs=1e-12)
    assert stats["lower_bound"] < stats["min_rate"]
    assert "underpowered sample, not a demonstrated failure" in message
    assert isinstance(stats["runs_needed"], int)
    assert stats["runs_needed"] > 20
    assert f"roughly {stats['runs_needed']} runs" in message


def test_a_genuine_miss_says_more_runs_will_not_help() -> None:
    # 14/20 = 0.70 observed against a 0.90 bar: the point estimate itself is under
    # the bar, so this is the other failure mode entirely.
    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate((14, 20), 0.9)

    assert excinfo.value.stats["underpowered"] is False
    assert excinfo.value.stats["runs_needed"] is None
    assert "the system missed the bar, and more runs will not fix it" in str(excinfo.value)


def test_the_report_carries_the_two_sided_interval_alongside_the_gating_bound() -> None:
    report = assert_pass_rate((190, 200), 0.9)

    # Two-sided lower at 0.95 == one-sided lower at 0.975, by Oracle 1.
    assert report["interval_lower"] == pytest.approx(
        wilson_lower_by_bisection(190, 200, 0.975), abs=1e-9
    )
    assert report["interval_lower"] < report["lower_bound"] <= report["pass_rate"]
    assert report["interval_upper"] > report["pass_rate"]


def test_a_perfect_sample_still_reports_a_bound_below_one() -> None:
    # 20/20 is not a demonstration of perfection; the bound is 20/(20+z**2).
    report = assert_pass_rate((20, 20), 0.85)

    assert report["pass_rate"] == 1.0
    assert report["lower_bound"] == pytest.approx(20 / (20 + Z_95_SQUARED), abs=1e-12)
    assert report["lower_bound"] < 1.0

    with pytest.raises(PassRateError):
        assert_pass_rate((20, 20), 0.9)


# --------------------------------------------------------------------------- #
# assert_score_distribution
# --------------------------------------------------------------------------- #

#: The hand-checked fixture. Sorted already, five values, so every statistic is
#: computable on paper.
FIVE_SCORES = [1.0, 2.0, 3.0, 4.0, 5.0]

#: mean = (1+2+3+4+5)/5 = 15/5 = 3.0
FIVE_MEAN = 3.0

#: numpy.percentile(x, 10) with linear interpolation: the index is
#: 0.10 * (5 - 1) = 0.4, i.e. four tenths of the way from x[0]=1 to x[1]=2,
#: giving 1 + 0.4*(2-1) = 1.4. A nearest-rank percentile would answer 1.0, so
#: this value alone pins the definition.
FIVE_P10 = 1.4


def test_the_statistics_are_the_pinned_definitions() -> None:
    # stddev with ddof=1: deviations from the mean of 3.0 are -2,-1,0,1,2.
    sum_of_squared_deviations = 4 + 1 + 0 + 1 + 4  # = 10
    expected_stddev = math.sqrt(sum_of_squared_deviations / (len(FIVE_SCORES) - 1))  # sqrt(2.5)

    report = assert_score_distribution(FIVE_SCORES, min_mean=0.0, min_p10=0.0, max_stddev=10.0)

    assert report["mean"] == pytest.approx(FIVE_MEAN, abs=1e-12)
    assert report["p10"] == pytest.approx(FIVE_P10, abs=1e-12)
    assert report["stddev"] == pytest.approx(expected_stddev, abs=1e-12)
    assert report["n"] == 5
    assert report["min_score"] == 1.0
    assert report["max_score"] == 5.0


def test_p10_interpolates_linearly_rather_than_taking_a_nearest_rank() -> None:
    # Two points, 0.0 and 10.0: the index is 0.10 * (2 - 1) = 0.1, i.e. one tenth
    # of the way from 0 to 10, which is 1.0. Nearest-rank would give 0.0 and pass
    # a gate this one must fail.
    report = assert_score_distribution([0.0, 10.0], min_p10=0.0)
    assert report["p10"] == pytest.approx(1.0, abs=1e-12)

    with pytest.raises(ScoreDistributionError):
        assert_score_distribution([0.0, 10.0], min_p10=1.5)


def test_stddev_uses_the_sample_denominator_not_the_population_one() -> None:
    # [0, 10]: mean 5, deviations -5 and +5, sum of squared deviations 25+25=50.
    # ddof=1 divides by 1 -> sqrt(50) = 7.0710678...; ddof=0 would divide by 2 and
    # give 5.0, understating exactly the spread this gate exists to bound.
    sum_of_squared_deviations = 25 + 25  # = 50
    expected = math.sqrt(sum_of_squared_deviations / 1)

    report = assert_score_distribution([0.0, 10.0], max_stddev=8.0)

    assert report["stddev"] == pytest.approx(expected, abs=1e-12)
    assert report["stddev"] != pytest.approx(5.0, abs=1e-9)


def test_min_mean_gates_alone_and_only_when_supplied() -> None:
    passing = assert_score_distribution(FIVE_SCORES, min_mean=2.5)
    assert passing["passed"] is True
    assert passing["checks"] == ("min_mean",)
    assert passing["min_p10"] is None
    assert passing["max_stddev"] is None

    with pytest.raises(ScoreDistributionError) as excinfo:
        assert_score_distribution(FIVE_SCORES, min_mean=3.5)
    assert excinfo.value.stats["violations"] == ("mean 3.0000 < min_mean 3.5000",)


def test_min_p10_gates_alone_and_only_when_supplied() -> None:
    # p10 is 1.4 by hand, so 1.0 clears and 2.0 does not -- while the mean of 3.0
    # would sail past any mean gate you might have set instead.
    passing = assert_score_distribution(FIVE_SCORES, min_p10=1.0)
    assert passing["passed"] is True
    assert passing["checks"] == ("min_p10",)

    with pytest.raises(ScoreDistributionError) as excinfo:
        assert_score_distribution(FIVE_SCORES, min_p10=2.0)
    assert excinfo.value.stats["violations"] == ("p10 1.4000 < min_p10 2.0000",)
    assert excinfo.value.stats["checks"] == ("min_p10",)


def test_max_stddev_gates_alone_and_only_when_supplied() -> None:
    # stddev is sqrt(2.5) = 1.5811388..., so 2.0 clears and 1.0 does not.
    passing = assert_score_distribution(FIVE_SCORES, max_stddev=2.0)
    assert passing["passed"] is True
    assert passing["checks"] == ("max_stddev",)

    with pytest.raises(ScoreDistributionError) as excinfo:
        assert_score_distribution(FIVE_SCORES, max_stddev=1.0)
    assert excinfo.value.stats["violations"] == ("stddev(ddof=1) 1.5811 > max_stddev 1.0000",)


def test_every_violation_is_reported_together_not_one_per_ci_run() -> None:
    # mean 3.0 misses 3.5 and stddev 1.5811 exceeds 1.0; p10 1.4 clears 1.0. Two
    # of three checks fail and both must appear, or fixing the mean rediscovers
    # the stddev problem tomorrow.
    with pytest.raises(ScoreDistributionError) as excinfo:
        assert_score_distribution(
            FIVE_SCORES, min_mean=3.5, min_p10=1.0, max_stddev=1.0, label="judge"
        )

    message = str(excinfo.value)
    violations = excinfo.value.stats["violations"]
    assert violations == (
        "mean 3.0000 < min_mean 3.5000",
        "stddev(ddof=1) 1.5811 > max_stddev 1.0000",
    )
    assert excinfo.value.stats["checks"] == ("min_mean", "min_p10", "max_stddev")
    assert "failed 2 of 3 checks over 5 scores" in message
    assert "mean 3.0000 < min_mean 3.5000" in message
    assert "stddev(ddof=1) 1.5811 > max_stddev 1.0000" in message
    assert "min_p10" not in "; ".join(violations)  # the gate that passed stays out of the list
    assert "judge" in message


def test_a_gate_with_no_thresholds_is_a_bug_in_the_test_not_a_pass() -> None:
    with pytest.raises(ValueError, match="at least one of min_mean, min_p10, or"):
        assert_score_distribution(FIVE_SCORES)


def test_max_stddev_needs_at_least_two_observations() -> None:
    # The spread of one observation is undefined; reporting it as 0.0 would pass
    # the strictest possible stddev gate on no evidence at all.
    with pytest.raises(ValueError, match="max_stddev needs at least 2 scores"):
        assert_score_distribution([4.0], max_stddev=0.5)

    # ...but a mean or p10 gate over one score is merely uninformative, not
    # meaningless, and is allowed with stddev reported as None.
    report = assert_score_distribution([4.0], min_mean=3.0)
    assert report["stddev"] is None
    assert report["n"] == 1


def test_a_distribution_over_zero_observations_is_not_one() -> None:
    with pytest.raises(ValueError, match="no scores to gate on"):
        assert_score_distribution([], min_mean=1.0)


def test_a_sample_result_feeds_the_gate_through_its_scores() -> None:
    result = sample_of_judge_scores(FIVE_SCORES)

    report = assert_score_distribution(result, min_mean=2.5, min_p10=1.0, max_stddev=2.0)

    assert report["passed"] is True
    assert report["n"] == 5
    assert report["mean"] == pytest.approx(FIVE_MEAN, abs=1e-12)
    assert report["p10"] == pytest.approx(FIVE_P10, abs=1e-12)


# --------------------------------------------------------------------------- #
# assert_no_regression
# --------------------------------------------------------------------------- #

#: Oracle 4 cases. U1 confirmed by hand-counting the pairs; the counting helper
#: reproduces the count, and the gate must report the same statistic.
U_CASES: tuple[tuple[list[float], list[float], float], ...] = (
    ([1, 2, 3, 4], [5, 6, 7, 8], 0.0),
    ([1, 3, 5, 7], [2, 4, 6, 8], 6.0),
    ([1, 2, 2], [2, 3, 4], 1.0),
    ([1, 2, 3, 4], [1, 2, 3, 4], 8.0),
    ([5, 6, 7, 8], [1, 2, 3, 4], 16.0),
)


@pytest.mark.parametrize(("current", "baseline", "expected_u1"), U_CASES)
def test_the_counting_oracle_reproduces_the_hand_counted_u1(
    current: list[float], baseline: list[float], expected_u1: float
) -> None:
    assert mann_whitney_u1(current, baseline) == expected_u1


@pytest.mark.parametrize(("current", "baseline", "expected_u1"), U_CASES)
def test_the_reported_statistic_is_u1_counted_from_its_definition(
    current: list[float], baseline: list[float], expected_u1: float
) -> None:
    # Checking scipy against scipy proves nothing, so U1 is counted from
    # #{c > b} + 0.5 * #{c == b} instead. The [1,2,2] vs [2,3,4] row is the one
    # that pins tie handling: zero strict wins, two ties, U1 = 1.0.
    report = gate_report(assert_no_regression, current, baseline)

    assert report["u_statistic"] == pytest.approx(expected_u1, abs=1e-12)
    assert report["test"] == "mann-whitney-u"
    assert report["alternative"] == "less"


def test_a_seeded_half_sigma_drop_is_detected(
    power_samples: tuple[list[float], list[float], list[float]],
) -> None:
    baseline, _, shifted = power_samples

    with pytest.raises(RegressionError) as excinfo:
        assert_no_regression(shifted, baseline, label="quality")

    stats = excinfo.value.stats
    assert stats["passed"] is False
    assert stats["p_value"] < 1e-4  # deliberately generous; the decision is the assertion
    assert stats["n_current"] == 200
    assert stats["n_baseline"] == 200
    assert stats["median_current"] < stats["median_baseline"]
    assert "quality" in str(excinfo.value)
    assert "significantly" in str(excinfo.value)


def test_a_seeded_redraw_of_the_same_distribution_does_not_false_alarm(
    power_samples: tuple[list[float], list[float], list[float]],
) -> None:
    baseline, same, _ = power_samples

    report = assert_no_regression(same, baseline)

    assert report["passed"] is True
    assert 0.05 < report["p_value"] < 0.95
    assert report["u_statistic"] == pytest.approx(mann_whitney_u1(same, baseline), abs=1e-9)


def test_getting_better_is_not_a_regression(
    power_samples: tuple[list[float], list[float], list[float]],
) -> None:
    # The mirror of the detected drop: the same two samples in the other order.
    # A two-sided test would fail the build for improving.
    baseline, _, shifted = power_samples

    report = assert_no_regression(baseline, shifted)

    assert report["passed"] is True
    assert report["p_value"] > 0.9
    assert report["median_current"] > report["median_baseline"]


def test_current_stochastically_smaller_is_the_regression_direction() -> None:
    # Small hand-built samples, so the direction convention is visible without
    # any distributional argument: [1,2,3,4] against [5,6,7,8] is every current
    # value below every baseline value.
    with pytest.raises(RegressionError):
        assert_no_regression([1, 2, 3, 4], [5, 6, 7, 8], alpha=0.05)

    report = assert_no_regression([5, 6, 7, 8], [1, 2, 3, 4], alpha=0.05)
    assert report["passed"] is True


def test_identical_inputs_do_not_trip_the_gate() -> None:
    scores = [4.0, 4.5, 3.0, 5.0, 4.0, 2.0, 4.5, 5.0]

    report = assert_no_regression(scores, list(scores))

    assert report["passed"] is True
    assert report["mean_current"] == report["mean_baseline"]
    assert report["median_current"] == report["median_baseline"]
    assert report["u_statistic"] == pytest.approx(mann_whitney_u1(scores, scores), abs=1e-12)


def test_two_fully_tied_samples_carry_no_evidence_of_a_drop() -> None:
    # Zero rank information either way. Whether scipy answers p=1.0 or NaN, the
    # only defensible decision is "not a regression".
    report = assert_no_regression([3.0, 3.0, 3.0], [3.0, 3.0, 3.0])

    assert report["passed"] is True
    assert isinstance(report["degenerate"], bool)
    # U1 = 0 wins + 0.5 * 9 ties = 4.5, counted by hand.
    assert report["u_statistic"] == pytest.approx(4.5, abs=1e-12)


def test_a_missing_baseline_is_not_a_passing_comparison() -> None:
    with pytest.raises(ValueError, match="current has no scores"):
        assert_no_regression([], [1.0, 2.0])

    with pytest.raises(ValueError, match="baseline has no scores"):
        assert_no_regression([1.0, 2.0], [])


def test_sample_results_feed_the_regression_gate() -> None:
    current = sample_of_judge_scores([1.0, 2.0, 3.0, 4.0])
    baseline = sample_of_judge_scores([5.0, 6.0, 7.0, 8.0])

    with pytest.raises(RegressionError) as excinfo:
        assert_no_regression(current, baseline)

    # U1 = 0.0 by hand-counting: no current value exceeds or ties any baseline.
    assert excinfo.value.stats["u_statistic"] == pytest.approx(0.0, abs=1e-12)
    assert excinfo.value.stats["n"] == 4


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #


def evidence_log(tmp_path: Path) -> EvidenceLog:
    return EvidenceLog(tmp_path / "log.jsonl")


def test_a_passing_pass_rate_gate_records_exactly_one_assertion(tmp_path: Path) -> None:
    log = evidence_log(tmp_path)

    report = assert_pass_rate((190, 200), 0.9, evidence=log, label="pass-rate")

    records = log.read()
    assert len(records) == 1
    assert records[0].event_type == EVENT_ASSERTION
    assert records[0].payload == as_payload(report)


def test_a_failing_pass_rate_gate_records_before_it_raises(tmp_path: Path) -> None:
    # A gate that only records its passes is a highlight reel, not an audit
    # trail. The record has to exist by the time the exception is caught.
    log = evidence_log(tmp_path)

    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate((18, 20), 0.9, evidence=log, label="pass-rate")

    records = log.read()
    assert len(records) == 1
    assert records[0].event_type == EVENT_ASSERTION
    assert records[0].payload["passed"] is False
    assert records[0].payload == as_payload(dict(excinfo.value.stats))


def test_a_passing_distribution_gate_records_exactly_one_assertion(tmp_path: Path) -> None:
    log = evidence_log(tmp_path)

    report = assert_score_distribution(FIVE_SCORES, min_mean=2.5, evidence=log)

    records = log.read()
    assert len(records) == 1
    assert records[0].event_type == EVENT_ASSERTION
    assert records[0].payload == as_payload(report)


def test_a_failing_distribution_gate_records_before_it_raises(tmp_path: Path) -> None:
    log = evidence_log(tmp_path)

    with pytest.raises(ScoreDistributionError) as excinfo:
        assert_score_distribution(FIVE_SCORES, min_mean=3.5, max_stddev=1.0, evidence=log)

    records = log.read()
    assert len(records) == 1
    assert records[0].payload["passed"] is False
    assert records[0].payload == as_payload(dict(excinfo.value.stats))


def test_a_passing_regression_gate_records_exactly_one_assertion(tmp_path: Path) -> None:
    log = evidence_log(tmp_path)

    report = assert_no_regression([5, 6, 7, 8], [1, 2, 3, 4], evidence=log)

    records = log.read()
    assert len(records) == 1
    assert records[0].event_type == EVENT_ASSERTION
    assert records[0].payload == as_payload(report)


def test_a_failing_regression_gate_records_before_it_raises(tmp_path: Path) -> None:
    log = evidence_log(tmp_path)

    with pytest.raises(RegressionError) as excinfo:
        assert_no_regression([1, 2, 3, 4], [5, 6, 7, 8], evidence=log)

    records = log.read()
    assert len(records) == 1
    assert records[0].payload["passed"] is False
    assert records[0].payload == as_payload(dict(excinfo.value.stats))


def test_three_gates_against_one_log_append_three_records(tmp_path: Path) -> None:
    log = evidence_log(tmp_path)

    assert_pass_rate((190, 200), 0.9, evidence=log)
    assert_score_distribution(FIVE_SCORES, min_mean=2.5, evidence=log)
    assert_no_regression([5, 6, 7, 8], [1, 2, 3, 4], evidence=log)

    records = log.read()
    assert [record.payload["gate"] for record in records] == [
        "pass_rate",
        "score_distribution",
        "no_regression",
    ]


def test_a_gate_without_an_evidence_log_writes_nothing(tmp_path: Path) -> None:
    log = evidence_log(tmp_path)

    assert_pass_rate((190, 200), 0.9)

    assert log.read() == []


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("successes", "n", "pattern"),
    [
        (0, 0, "n must be >= 1"),
        (1, -1, "n must be >= 1"),
        (21, 20, r"successes \(21\) cannot exceed n \(20\)"),
        (-1, 20, "successes must be >= 0"),
    ],
)
def test_impossible_counts_are_rejected(successes: int, n: int, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        wilson_lower_bound(successes, n)
    with pytest.raises(ValueError, match=pattern):
        wilson_interval(successes, n)
    with pytest.raises(ValueError, match=pattern):
        assert_pass_rate((successes, n), 0.5)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.1, 2.0])
def test_confidence_outside_the_open_unit_interval_is_rejected(confidence: float) -> None:
    # norm.ppf(1.0) is infinite and a bound at confidence 0 is not a bound.
    with pytest.raises(ValueError, match="confidence must be strictly between 0 and 1"):
        wilson_lower_bound(14, 20, confidence)
    with pytest.raises(ValueError, match="confidence must be strictly between 0 and 1"):
        wilson_interval(14, 20, confidence)
    with pytest.raises(ValueError, match="confidence must be strictly between 0 and 1"):
        assert_pass_rate((14, 20), 0.5, confidence=confidence)


def test_a_nan_confidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence must be strictly between 0 and 1"):
        wilson_lower_bound(14, 20, float("nan"))


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
def test_alpha_outside_the_open_unit_interval_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        assert_no_regression([1, 2, 3], [1, 2, 3], alpha=alpha)


@pytest.mark.parametrize("min_rate", [-0.1, 1.1, 2.0])
def test_min_rate_outside_the_closed_unit_interval_is_rejected(min_rate: float) -> None:
    with pytest.raises(ValueError, match="min_rate must be between 0 and 1 inclusive"):
        assert_pass_rate((14, 20), min_rate)


@pytest.mark.parametrize("min_rate", [0.0, 1.0])
def test_the_endpoints_of_min_rate_are_legitimate_bars(min_rate: float) -> None:
    # 0.0 and 1.0 are both meaningful thresholds to set, unlike a confidence of
    # 0.0 or 1.0, which is why the two are validated differently.
    report = gate_report(assert_pass_rate, (20, 20), min_rate)
    assert report["min_rate"] == min_rate
    assert report["passed"] is (min_rate == 0.0)


def test_boolean_counts_are_rejected_rather_than_silently_coerced() -> None:
    # bool is an int subclass, so successes=True would otherwise mean one success.
    with pytest.raises(ValueError, match="successes must be an integer, got the boolean"):
        wilson_lower_bound(True, 20)
    with pytest.raises(ValueError, match="n must be an integer, got the boolean"):
        wilson_lower_bound(1, True)


def test_a_float_count_is_a_bug_not_a_rounding_opportunity() -> None:
    with pytest.raises(ValueError, match="n must be an integer, got float"):
        wilson_lower_bound(14, 19.5)


def test_a_non_bool_element_in_an_outcome_sequence_is_rejected() -> None:
    # Truthiness would score the non-empty string "fail" as a pass.
    with pytest.raises(ValueError, match=r"result\[1\] must be a bool"):
        assert_pass_rate([True, "fail", True], 0.5)


def test_a_nan_score_is_missing_data_rather_than_a_score() -> None:
    with pytest.raises(ValueError, match="is NaN"):
        assert_score_distribution([1.0, float("nan"), 3.0], min_mean=1.0)

    with pytest.raises(ValueError, match="is NaN"):
        assert_no_regression([1.0, float("nan")], [1.0, 2.0])


def test_a_sequence_of_bools_in_a_score_gate_is_rejected() -> None:
    # Averaging outcomes would silently produce a pass rate labelled "mean score".
    with pytest.raises(ValueError, match=r"scores\[0\] must be a number"):
        assert_score_distribution([True, False, True], min_mean=0.5)


@pytest.mark.parametrize("threshold", ["min_mean", "min_p10", "max_stddev"])
def test_a_non_numeric_threshold_is_rejected(threshold: str) -> None:
    with pytest.raises(ValueError, match=f"{threshold} must be a number or None"):
        assert_score_distribution(FIVE_SCORES, **{threshold: "3.0"})


def test_a_negative_max_stddev_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_stddev must be >= 0"):
        assert_score_distribution(FIVE_SCORES, max_stddev=-1.0)
