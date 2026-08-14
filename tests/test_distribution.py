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
import warnings
from collections.abc import Callable, Sequence
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from opik_rigor.distribution import (
    PassRateError,
    RegressionError,
    ScoreDistributionError,
    _runs_needed,
    _wilson,
    assert_no_regression,
    assert_pass_rate,
    assert_score_distribution,
    wilson_interval,
    wilson_lower_bound,
)
from opik_rigor.errors import StatisticalAssertionError
from opik_rigor.evidence import EVENT_ASSERTION, EvidenceLog
from opik_rigor.judge import Verdict
from opik_rigor.sampling import Run, SampleResult

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
# Oracle 5 -- the gate predicate decided algebraically, and scanned
# --------------------------------------------------------------------------- #


def bound_clears(successes: int, n: int, min_rate: float, z: float = Z_95) -> bool:
    """Oracle 5: is ``wilson_lower(successes, n) >= min_rate``, without a bound?

    The Wilson interval *is* the set ``{p : |p_hat - p| <= z*sqrt(p(1-p)/n)}``, so
    its lower end sits at or above ``min_rate`` exactly when ``min_rate`` falls
    below that set -- which is one inequality, evaluated once::

        (p_hat - min_rate) >= z * sqrt(min_rate*(1-min_rate)/n)

    No root-finding, no quadratic, no closed form. That makes it both the cheapest
    oracle in this file and the most independent of the code: it is the defining
    inequality itself, and it lets the brute-force scans below cover thousands of
    n without costing anything.
    """
    return (successes / n - min_rate) >= z * math.sqrt(min_rate * (1.0 - min_rate) / n)


def gate_clears_at(p: float, min_rate: float, n: int, z: float = Z_95) -> bool:
    """Oracle 5: the predicate ``_runs_needed`` searches, evaluated directly."""
    return bound_clears(min(round(p * n), n), n, min_rate, z)


def last_failing_n(p: float, min_rate: float, ceiling: int, z: float = Z_95) -> int:
    """Oracle 5: brute force. Largest ``n <= ceiling`` at which the gate still fails.

    One past this is the n from which the gate clears *and keeps clearing*, which
    is what ``_runs_needed`` owes its reader. Returns 0 when nothing fails.
    """
    failures = [n for n in range(1, ceiling + 1) if not gate_clears_at(p, min_rate, n, z)]
    return failures[-1] if failures else 0


def minimum_successes_by_scan(min_rate: float, n: int, z: float = Z_95) -> int:
    """Oracle 5: fewest passes out of n that clear the bar, by trying every count."""
    for successes in range(n + 1):
        if bound_clears(successes, n, min_rate, z):
            return successes
    return n + 1


# --------------------------------------------------------------------------- #
# Oracle 6 -- binomial tails in exact rational arithmetic
# --------------------------------------------------------------------------- #


def binomial_tail_exact(k: int, n: int, p: float) -> float:
    """Oracle 6: ``P(X >= k)`` for ``X ~ Binomial(n, p)``, summed as Fractions.

    Every term is ``comb(n, i) * p**i * (1-p)**(n-i)`` in exact rational
    arithmetic over the binary value of ``p``, summed across the whole upper tail
    with no window and nothing rounded until the end. No logarithms, no gamma
    functions, no scipy -- which is the point, since the implementation reaches
    the same number from a log-gamma seed and a multiplicative recurrence over a
    truncated window.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    probability = Fraction(p)
    complement = 1 - probability
    total = sum(
        (math.comb(n, i) * probability**i * complement ** (n - i) for i in range(k, n + 1)),
        Fraction(0),
    )
    return float(total)


def gate_power_exact(p: float, min_rate: float, n: int, z: float = Z_95) -> float:
    """Oracle 6: how often a fresh sample of n runs clears the gate, if the rate is p."""
    return binomial_tail_exact(minimum_successes_by_scan(min_rate, n, z), n, p)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def assert_pass_rate_report_for(result: Any, min_rate: float) -> dict[str, Any]:
    """The report dict from a pass-rate gate that is expected to fail."""
    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate(result, min_rate)
    return dict(excinfo.value.stats)


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


def test_the_wilson_docstring_worked_example_is_a_number_the_code_produces() -> None:
    # The docstring's 20/20 example read "roughly [0.86, 1.0]". 0.86 is neither
    # end of anything: two-sided 95% is [0.8389, 1.0] and the one-sided 95% lower
    # bound is 0.8808. A worked example in a statistics library is a claim, and
    # this one was checkable and wrong -- so it is checked, by Oracle 1.
    documentation = _wilson.__doc__ or ""
    two_sided_lower = wilson_lower_by_bisection(20, 20, 0.975)
    one_sided_lower = wilson_lower_by_bisection(20, 20, 0.95)

    assert "0.86" not in documentation
    assert f"``[{two_sided_lower:.4f}, 1.0]`` two-sided" in documentation
    assert f"lower bound of ``{one_sided_lower:.4f}``" in documentation


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
    # 0.50 dropped from the sweep: the one-sided bound refuses it now, and why
    # is pinned in test_a_one_sided_confidence_at_or_below_a_half_is_refused.
    for confidence in (*SWEEP_CONFIDENCES, 0.51, 0.999):
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
                for confidence in (0.51, 0.80, 0.90, 0.95, 0.99, 0.999)
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
    assert f"the bound clears from {stats['runs_needed']} runs on" in message


# --------------------------------------------------------------------------- #
# how many more runs -- the number, and what it is not
# --------------------------------------------------------------------------- #

#: 45 (min_rate, observed) pairs spanning the bars an eval suite actually sets and
#: margins from a fifth of a point to twenty points.
RUNS_NEEDED_GRID: tuple[tuple[float, float], ...] = tuple(
    (min_rate, round(min(min_rate + margin, 0.999), 4))
    for min_rate in (0.70, 0.75, 0.80, 0.85, 0.90)
    for margin in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20)
)


def test_the_runs_needed_predicate_oscillates_so_its_minimum_is_not_a_budget() -> None:
    # The premise of the whole item. successes = round(p*n) rounds the observed
    # rate up at some n and down at others, so the predicate does not switch on
    # once and stay on -- and a binary search, which assumes it does, lands
    # wherever the oscillation happens to put it.
    holds = {n for n in range(80, 130) if gate_clears_at(0.95, 0.90, n)}

    assert min(holds) == 86
    assert not (set(range(91, 100)) & holds)
    assert set(range(100, 110)) <= holds
    assert not (set(range(110, 113)) & holds)
    assert set(range(113, 130)) <= holds

    # So 86 -- "the smallest n that clears" -- is a trap. It clears only because
    # round(0.95 * 86) = 82 rounds the rate up to 0.9535; a reader told "86 runs"
    # who runs 91 for luck fails, which is the opposite of what a budget is for.
    assert round(0.95 * 86) / 86 > 0.95
    assert not gate_clears_at(0.95, 0.90, 91)


@pytest.mark.parametrize(("min_rate", "observed"), RUNS_NEEDED_GRID)
def test_runs_needed_is_one_past_the_last_n_that_fails(min_rate: float, observed: float) -> None:
    # The claim being tested is not "some n that clears" but "the n past which the
    # answer stops depending on where round() lands". Oracle 5 scans every n up to
    # a ceiling set from the closed form n* = z^2*m(1-m)/(p-m)^2 -- generously,
    # since the ratio of the true answer to n* runs as high as 1.3 -- and reports
    # the last one that fails. One past that is the whole answer.
    n_star = Z_95_SQUARED * min_rate * (1.0 - min_rate) / (observed - min_rate) ** 2
    ceiling = int(1.8 * n_star) + 40

    assert _runs_needed(observed, min_rate, 0.95) == last_failing_n(observed, min_rate, ceiling) + 1


def test_the_documented_cap_is_the_cap() -> None:
    # Approaching the cap by doubling from 1 topped out at 2**23 = 8,388,608:
    # 16,777,216 exceeded the documented 10,000,000 and fell out of the loop, so
    # every answer in the 8.4M-10M band came back as None -- "no sample size can
    # do this" -- for margins where a finite, in-cap answer exists. The closed
    # form here is about 9.0e6, comfortably inside the documented cap.
    answer = _runs_needed(0.9001645, 0.9, 0.95)

    assert answer is not None
    assert 8_388_608 < answer <= 10_000_000
    assert gate_clears_at(0.9001645, 0.9, answer)
    assert not gate_clears_at(0.9001645, 0.9, answer - 1)

    # And past the cap the answer really is None, rather than a number above it.
    assert _runs_needed(0.9001645, 0.9, 0.95, cap=1_000_000) is None


def test_the_underpowered_message_refuses_to_be_read_as_a_power_calculation() -> None:
    # 19/20 against a 0.90 bar. runs_needed is 113, and the exact binomial chance
    # that a fresh 113 runs from a system whose true rate really is 0.95 clears
    # this gate is 0.66 -- a coin flip. Quoting 113 on its own, as "roughly 113
    # runs would clear the bar" did, is read as a budget and is wrong a third of
    # the time. The message now carries the power next to the number.
    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate((19, 20), 0.90)

    message = str(excinfo.value)
    stats = excinfo.value.stats

    assert stats["runs_needed"] == 113
    assert stats["power_at_runs_needed"] == pytest.approx(
        gate_power_exact(0.95, 0.90, 113), abs=1e-12
    )
    assert 0.6 < stats["power_at_runs_needed"] < 0.7
    assert "not a power calculation" in message
    assert "clears this gate only 66% of the time" in message

    # And a number that *is* a budget is offered beside it.
    assert stats["target_power"] == 0.80
    assert stats["runs_for_target_power"] == 188
    assert "Budget 188 runs to clear it 80% of the time" in message


def test_the_powered_recommendation_reaches_the_target_and_stays_there() -> None:
    # Derived, not asserted: Oracle 6 recomputes the exact binomial power from
    # Fractions at the recommended n and at the n below it.
    powered = assert_pass_rate_report_for((19, 20), 0.90)["runs_for_target_power"]

    assert gate_power_exact(0.95, 0.90, powered) >= 0.80
    assert gate_power_exact(0.95, 0.90, powered - 1) < 0.80

    # It is emphatically not "the first n that reaches 80%": power oscillates for
    # the same lattice reason the bound does, and 164 is the first crossing while
    # 165-187 fall back below it. Reporting 164 would under-budget by 13%.
    first_crossing = next(n for n in range(1, 300) if gate_power_exact(0.95, 0.90, n) >= 0.80)
    assert first_crossing == 164
    assert powered == 188
    assert any(gate_power_exact(0.95, 0.90, n) < 0.80 for n in range(165, 188))
    assert all(gate_power_exact(0.95, 0.90, n) >= 0.80 for n in range(188, 210))


@pytest.mark.parametrize(
    ("successes", "n", "min_rate"),
    [(19, 20, 0.90), (18, 20, 0.85), (45, 50, 0.85), (95, 100, 0.92)],
)
def test_the_reported_power_matches_exact_rational_arithmetic(
    successes: int, n: int, min_rate: float
) -> None:
    # The implementation sums a log-gamma-seeded recurrence over a 14-sigma
    # window; Oracle 6 sums Fractions over the whole tail. Nothing is shared.
    report = assert_pass_rate_report_for((successes, n), min_rate)
    observed = successes / n

    assert report["power_at_runs_needed"] == pytest.approx(
        gate_power_exact(observed, min_rate, report["runs_needed"]), abs=1e-12
    )


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


def test_a_sample_of_text_is_told_its_data_is_the_wrong_shape_not_that_it_is_absent() -> None:
    # Roadmap item 14. `.scores()` harvests getattr(run.value, "score", None), so
    # a sample of plain completions harvests nothing -- and "current has no
    # scores" reads as "you have no data" to a caller holding 200 of them. The
    # message has to name the type it found and the first value that caused it.
    runs = tuple(
        Run(index=index, value=value, outcome=True)
        for index, value in enumerate(("Paris", "Berlin", "Rome"))
    )
    current = SampleResult(runs=runs, wall_clock=0.0)

    with pytest.raises(ValueError) as excinfo:
        assert_no_regression(current, [4.0, 4.5, 5.0])

    message = str(excinfo.value)
    assert "current has no scores" in message  # the old opening survives
    assert "SampleResult" in message  # what it actually is
    assert "3 runs" in message  # it is not empty
    assert "str" in message and "'Paris'" in message  # the first offending value


def test_a_sample_whose_runs_all_raised_is_told_that_rather_than_told_it_is_empty() -> None:
    # The other way a SampleResult yields no scores: nothing completed at all.
    # Naming the first error is the difference between "wrong shape" and "the
    # provider was down", which call for opposite responses.
    runs = tuple(
        Run(index=index, error=RuntimeError(f"upstream 503 #{index}")) for index in range(4)
    )
    current = SampleResult(runs=runs, wall_clock=0.0)

    with pytest.raises(ValueError) as excinfo:
        assert_no_regression(current, [4.0, 4.5, 5.0])

    message = str(excinfo.value)
    assert "4 runs" in message
    assert "every one of them raised" in message
    assert "RuntimeError: upstream 503 #0" in message


def test_a_genuinely_empty_input_still_gets_the_plain_message() -> None:
    # The diagnosis must not be bolted onto the case where "you have no data" is
    # simply true; an empty list is exactly what the original sentence describes.
    with pytest.raises(ValueError) as excinfo:
        assert_no_regression([], [1.0, 2.0])

    assert str(excinfo.value) == (
        "current has no scores; there is nothing to compare against baseline."
    )


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


@pytest.mark.parametrize("confidence", [0.0001, 0.2896, 0.49, 0.5])
def test_a_one_sided_confidence_at_or_below_a_half_is_refused(confidence: float) -> None:
    # z = ppf(c) is zero at 0.5 and negative below it, so the "lower bound" stops
    # being a floor: at 0.2896 it comes out *above* the observed rate, and it
    # *falls* as the sample grows -- more evidence, worse bound. The arithmetic is
    # right; the domain was wrong, and the failure was silent and backwards, since
    # a gate written confidence=0.3 reads as caution and is looser than comparing
    # the raw rate. Refused rather than documented, because nobody wants it.
    with pytest.raises(ValueError, match="confidence must be greater than 0.5"):
        wilson_lower_bound(157, 200, confidence)
    with pytest.raises(ValueError, match=f"got {confidence!r}"):
        assert_pass_rate((157, 200), 0.79, confidence=confidence)

    # The two-sided interval is immune and keeps the whole open interval: its z is
    # ppf((1 + c) / 2), which is non-negative everywhere in (0, 1).
    lower, upper = wilson_interval(157, 200, confidence)
    assert lower <= 157 / 200 <= upper


def test_at_half_confidence_the_bound_would_have_been_the_raw_rate() -> None:
    # Why 0.5 itself is refused and not merely warned about. z = 0 there, so the
    # "bound" is successes/n exactly -- and the module's opening paragraph exists
    # to say that 20/20 must never yield 1.0. It did, and a gate at min_rate=1.0
    # passed on twenty runs. Oracle 5 confirms the arithmetic that used to run.
    assert bound_clears(20, 20, 1.0, z=0.0)
    assert bound_clears(157, 200, 157 / 200, z=0.0)

    with pytest.raises(ValueError, match="confidence must be greater than 0.5"):
        assert_pass_rate((20, 20), 1.0, confidence=0.5)


def test_a_numpy_integer_count_pair_is_read_as_counts() -> None:
    # (np.int64(190), np.int64(200)) is a count pair by every reading of the rule
    # -- a two-element tuple of non-boolean integers -- but numpy integers are not
    # `int`, so it fell through to the outcome branch and was refused with
    # "result[0] must be a bool (or 0/1), got int64". That sent the reader to look
    # at their data when the answer was their dtype, and _as_count's own docstring
    # already says numpy integers arrive routinely from array code.
    report = assert_pass_rate((np.int64(190), np.int64(200)), 0.9)

    assert report["successes"] == 190
    assert report["n"] == 200
    assert report["lower_bound"] == pytest.approx(
        wilson_lower_by_bisection(190, 200, 0.95), abs=1e-9
    )

    # np.bool_ is still outcomes, not counts: it is not an np.integer.
    assert assert_pass_rate((np.bool_(True), np.bool_(True)), 0.1)["n"] == 2


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


@pytest.mark.parametrize(
    ("scores", "gate", "statistic"),
    [
        ([1.0, 1.0, math.inf], {"max_stddev": 0.001}, "stddev"),
        ([1.0, 1.0, -math.inf], {"min_p10": 4.9}, "p10"),
        ([1.0, 1.0, math.inf], {"min_mean": 4.9}, "mean"),
    ],
)
def test_an_infinite_score_is_refused_rather_than_passed(
    scores: list[float], gate: dict[str, float], statistic: str
) -> None:
    # This is the failure the library exists to prevent, so it gets its own test.
    # An infinite score makes every statistic inf or nan; a nan loses every
    # comparison, so no violation is recorded and the gate returns *green* -- on a
    # 0.001 stddev bar, over a sample containing infinity. The only sign anything
    # happened was a bare numpy RuntimeWarning on stderr, which CI discards.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any leaked RuntimeWarning fails here
        with pytest.raises(ValueError) as excinfo:
            assert_score_distribution(scores, **gate)

    message = str(excinfo.value)
    assert "scores[2] is" in message
    assert ("-inf" if any(value == -math.inf for value in scores) else "inf") in message
    assert statistic in ("mean", "p10", "stddev")  # the gate that would have passed


def test_an_infinite_score_is_refused_by_the_regression_gate_too() -> None:
    # Same hole, other gate: mean_current came back inf and the comparison passed.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"current\[1\] is inf"):
            assert_no_regression([1.0, math.inf], [1.0, 2.0])

        with pytest.raises(ValueError, match=r"baseline\[0\] is -inf"):
            assert_no_regression([1.0, 2.0], [-math.inf, 2.0])


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
