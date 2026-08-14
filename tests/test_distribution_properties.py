"""Property-based tests for the statistical gates.

The example-based suite in ``test_distribution.py`` checks the numbers against
four independent oracles at inputs a human chose. This file checks *invariants*
at inputs a human did not choose: the things that have to be true of every
``(successes, n, confidence)`` and every pair of samples, not just of the nine
hardcoded fixtures. The two are complementary -- an oracle catches a wrong
formula, a property catches a formula that is right in the middle and wrong at
the edge, which is where a statistics library gets used.

**No hypothesis.** It is not installed and this library ships one runtime
dependency on purpose, so the generator below is written out: a few hundred lines
of draw-shrink-report rather than a new entry in ``dev``. It does the three
things that matter -- generates biased-toward-the-edges input, shrinks a
counterexample before reporting it, and prints the seed.

**Every case is reproducible.** Each property states its seed as a literal, the
draw is a ``random.Random(seed)`` and nothing else, and the failure message
carries the seed, the case index, the original input and the shrunk input. Two
runs of this file on the same machine produce byte-identical results; there is no
clock, no ``random.random``, no set iteration order and no unordered dict in any
draw. A test suite about flaky statistics that was itself flaky would be
self-refuting.

**Where a property is marked xfail(strict=True)** the property is wrong and the
code is right; the reason string says which, and the property is kept running so
that the day the behaviour changes, this file says so.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest

from opik_rigor.distribution import (
    PassRateError,
    RegressionError,
    ScoreDistributionError,
    assert_no_regression,
    assert_pass_rate,
    assert_score_distribution,
    wilson_interval,
    wilson_lower_bound,
)
from opik_rigor.judge import Verdict
from opik_rigor.sampling import Run, SampleResult

# --------------------------------------------------------------------------- #
# budgets -- stated once, so the cost of this file is one number
# --------------------------------------------------------------------------- #

#: Cases per property, by how expensive one case is. The file draws 6,720 cases
#: in all (11 x 400 + 8 x 200 + 6 x 120) and runs in about seven seconds; that is
#: deliberately inside the inner loop of a pull request rather than a nightly.
#: Rejection rates were measured rather than assumed -- the worst is 36% (only a
#: failing gate has a message to parse), so no property is quietly testing thirty
#: cases while claiming four hundred. ``for_all`` fails outright above 90%.
CASES_ARITHMETIC = 400  # closed-form Wilson only: microseconds each
CASES_GATE = 200  # a full gate: coercion, report dict, message formatting
CASES_SCIPY = 120  # mannwhitneyu, sometimes the exact permutation test

#: Ceiling on candidate evaluations while shrinking one counterexample. Shrinking
#: is a convenience, not the test; a pathological case must not turn a two-second
#: file into a two-minute one.
SHRINK_BUDGET = 400


# --------------------------------------------------------------------------- #
# the generator
# --------------------------------------------------------------------------- #


class Rejected(Exception):
    """Raised by a property to say "this input is out of scope", not "this failed".

    Needed because shrinking walks off the constraint surface constantly: shrink
    ``n`` in ``(successes=3, n=10)`` toward 0 and you get ``successes > n``, which
    the module rightly refuses. A rejected candidate is discarded rather than
    counted as a smaller counterexample.
    """


def assume(condition: bool) -> None:
    """Skip this case unless ``condition`` holds."""
    if not condition:
        raise Rejected


def _shrink_candidates(value: Any) -> Iterator[Any]:
    """Structurally simpler versions of ``value``, simplest first.

    The convention this file follows, so that one generic shrinker serves every
    property: **a tuple is a fixed-arity case record** (its components shrink, its
    length does not) and **a list is a data sample** (elements can be dropped).
    That is exactly how the draws below are written.
    """
    if isinstance(value, bool):
        if value:
            yield False
    elif isinstance(value, int):
        for candidate in (0, value // 2, value - 1 if value > 0 else value + 1):
            if abs(candidate) < abs(value):
                yield candidate
    elif isinstance(value, float):
        for candidate in (0.0, float(round(value)), round(value, 2), value / 2.0):
            if candidate != value and abs(candidate) <= abs(value):
                yield candidate
    elif isinstance(value, list):
        for index in range(len(value)):
            yield value[:index] + value[index + 1 :]
        for index, item in enumerate(value):
            for smaller in _shrink_candidates(item):
                yield value[:index] + [smaller] + value[index + 1 :]
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            for smaller in _shrink_candidates(item):
                yield value[:index] + (smaller,) + value[index + 1 :]


def _failure(holds: Callable[[Any], None], case: Any) -> str | None:
    """The failure message ``case`` produces, or None if it passes or is rejected."""
    try:
        holds(case)
    except Rejected:
        return None
    except AssertionError as exc:
        return str(exc) or "assertion failed"
    return None


def _shrink(case: Any, holds: Callable[[Any], None]) -> tuple[Any, str]:
    """Greedily walk to a smaller case that still fails.

    Plain hill-climbing with no backtracking: it does not find *the* minimum, it
    finds a small local one, which is the difference between a counterexample a
    reader can hold in their head and one they cannot.
    """
    best = case
    reason = _failure(holds, case) or "assertion failed"
    budget = SHRINK_BUDGET
    improved = True
    while improved and budget > 0:
        improved = False
        for candidate in _shrink_candidates(best):
            budget -= 1
            if budget <= 0:
                break
            candidate_reason = _failure(holds, candidate)
            if candidate_reason is not None:
                best, reason, improved = candidate, candidate_reason, True
                break
    return best, reason


def for_all(
    name: str,
    draw: Callable[[random.Random], Any],
    holds: Callable[[Any], None],
    *,
    seed: int,
    cases: int,
) -> None:
    """Check ``holds`` on ``cases`` inputs from ``draw``, seeded by ``seed``.

    On the first counterexample: shrink it, then fail with everything needed to
    reproduce it without re-running the generator -- the seed, the case index, the
    input as drawn, and the shrunk input, which is a literal that can be pasted
    straight into a REPL.
    """
    rng = random.Random(seed)
    rejected = 0
    for index in range(cases):
        case = draw(rng)
        try:
            holds(case)
        except Rejected:
            rejected += 1
            continue
        except AssertionError:
            # The original message is not reported: _shrink recomputes it on the
            # smaller case, and that is the one a reader wants.
            minimal, reason = _shrink(case, holds)
            raise AssertionError(
                f"property {name!r} is false.\n"
                f"  seed={seed} case #{index} of {cases}\n"
                f"  as drawn: {case!r}\n"
                f"  shrunk to: {minimal!r}\n"
                f"  because: {reason}\n"
                f"  reproduce: random.Random({seed}) then the case above"
            ) from None
    # A generator that rejects nearly everything is testing nothing, and would do
    # it silently. This is the tripwire.
    assert rejected < cases * 0.9, (
        f"property {name!r} rejected {rejected} of {cases} draws (seed={seed}); "
        f"the generator is out of step with the property's preconditions"
    )


# --------------------------------------------------------------------------- #
# draws -- biased toward the edges, because that is where closed forms break
# --------------------------------------------------------------------------- #

#: Sample sizes worth drawing: 1 and 2 (where the closed form degenerates), the
#: sizes an eval suite can afford, and a couple of large ones. Weighted toward
#: the small end deliberately -- 200 is where the arithmetic is boring.
SIZES = (1, 1, 2, 2, 3, 4, 5, 7, 10, 11, 19, 20, 24, 50, 97, 100, 200, 1000)

#: Confidences: the conventional four, the extremes at both ends, and the
#: not-round values that catch anything special-casing 0.95.
CONFIDENCES = (0.5, 0.6827, 0.8, 0.9, 0.95, 0.99, 0.999, 0.51, 0.9999, 0.0001)


def draw_confidence(rng: random.Random) -> float:
    """A valid confidence: from the table half the time, uniform the other half."""
    if rng.random() < 0.5:
        return rng.choice(CONFIDENCES)
    return rng.uniform(0.001, 0.999)


def draw_gating_confidence(rng: random.Random) -> float:
    """A confidence at or above 0.5 -- the half of the domain a gate is set in.

    Below 0.5 the one-sided z goes negative and the "lower bound" comes out
    *above* the observed rate. That is not a defect of the closed form (see
    :func:`test_a_confidence_below_a_half_turns_the_bound_upside_down`, which pins
    it) but it inverts every orientation property, so the properties about
    orientation draw from here and say so.
    """
    confidence = draw_confidence(rng)
    return confidence if confidence >= 0.5 else 1.0 - confidence


def assume_valid_counts(successes: Any, n: Any, confidence: Any) -> None:
    """Restrict to the module's documented domain.

    The draws never leave it, so this matters only during shrinking, which walks
    off the constraint surface constantly -- shrink ``n`` toward 0 and the module
    rightly raises ``ValueError``. Without this the shrinker turns a real
    counterexample into a ValueError from three frames down.
    """
    assume(isinstance(n, int) and n >= 1)
    assume(isinstance(successes, int) and 0 <= successes <= n)
    assume(isinstance(confidence, float) and 0.0 < confidence < 1.0)


def draw_counts(rng: random.Random) -> tuple[int, int, float]:
    """``(successes, n, confidence)``, with the two endpoints over-represented.

    0/n and n/n are a third of all draws: they are the two cases where the two
    terms of the closed form cancel analytically and do not cancel in floating
    point, which is a bug this module has already had once.
    """
    n = rng.choice(SIZES)
    roll = rng.random()
    if roll < 0.17:
        successes = 0
    elif roll < 0.34:
        successes = n
    else:
        successes = rng.randint(0, n)
    return successes, n, draw_confidence(rng)


def draw_scores(rng: random.Random, *, minimum: int = 1, maximum: int = 30) -> list[float]:
    """A sample of judge scores: a 1-5 scale, a 0-1 scale, or arbitrary floats.

    Includes all-identical samples (zero variance, where a stddev or a
    rank test degenerates) on purpose.
    """
    size = rng.randint(minimum, maximum)
    shape = rng.random()
    if shape < 0.3:
        return [float(rng.randint(1, 5)) for _ in range(size)]
    if shape < 0.5:
        return [rng.choice([0.0, 1.0]) for _ in range(size)]
    if shape < 0.6:
        return [float(rng.randint(1, 5))] * size
    if shape < 0.7:
        return [rng.uniform(-1e6, 1e6) for _ in range(size)]
    return [round(rng.uniform(0.0, 5.0), 3) for _ in range(size)]


def outcomes_for(successes: int, n: int) -> list[bool]:
    """A pass/fail list with exactly ``successes`` passes out of ``n``."""
    return [True] * successes + [False] * (n - successes)


def sample_of_outcomes(outcomes: Sequence[bool]) -> SampleResult:
    """A real :class:`SampleResult` carrying the given pass/fail outcomes."""
    runs = tuple(
        Run(index=index, value=Verdict(passed=bool(outcome), score=None, raw=""), outcome=outcome)
        for index, outcome in enumerate(outcomes)
    )
    return SampleResult(runs=runs, wall_clock=0.0)


def report_of(gate: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """The report a gate produced, whether it returned it or raised carrying it."""
    try:
        return gate(*args, **kwargs)
    except (PassRateError, RegressionError, ScoreDistributionError) as exc:
        return dict(exc.stats)


def finite_values(report: dict[str, Any]) -> list[tuple[str, float]]:
    """Every numeric leaf of a report, for the no-NaN sweep."""
    return [
        (key, float(value))
        for key, value in report.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


# --------------------------------------------------------------------------- #
# Wilson: the interval lives in [0, 1] and brackets its own point estimate
# --------------------------------------------------------------------------- #


def test_every_wilson_bound_lies_inside_the_unit_interval() -> None:
    # A confidence bound on a probability that is not itself a probability is not
    # a bound on anything, and the closed form does stray outside at the endpoints
    # -- _clamp exists for exactly this. Swept rather than spot-checked because
    # which (successes, n, confidence) strays is a property of the float unit in
    # the last place, not of any case a human would pick.
    def holds(case: tuple[int, int, float]) -> None:
        successes, n, confidence = case
        assume_valid_counts(successes, n, confidence)
        lower, upper = wilson_interval(successes, n, confidence)
        one_sided = wilson_lower_bound(successes, n, confidence)
        assert 0.0 <= lower <= 1.0, f"two-sided lower {lower!r} outside [0, 1]"
        assert 0.0 <= upper <= 1.0, f"two-sided upper {upper!r} outside [0, 1]"
        assert 0.0 <= one_sided <= 1.0, f"one-sided bound {one_sided!r} outside [0, 1]"
        assert lower <= upper, f"interval is inverted: [{lower!r}, {upper!r}]"

    for_all(
        "wilson bounds lie in [0, 1]",
        draw_counts,
        holds,
        seed=2026081401,
        cases=CASES_ARITHMETIC,
    )


def test_the_interval_brackets_the_observed_rate() -> None:
    # lower <= p_hat <= upper is definitional: the Wilson interval is the set of p
    # satisfying |p_hat - p| <= z*sqrt(p(1-p)/n), and p = p_hat makes the left side
    # zero. A bound that excluded the number actually observed would be reporting
    # on some other experiment. The one-sided bound is checked against p_hat too --
    # a lower confidence bound above the observed rate is the bug that produced
    # wilson_lower_bound(0, 11) == 1.39e-17.
    #
    # The one-sided half of this is true only at confidence >= 0.5, and that is a
    # fact about confidence bounds rather than a concession: at c < 0.5 the z is
    # negative and the bound legitimately sits above the observed rate. The case
    # this file first drew, (157, 200, 0.2896), is pinned below in its own test.
    def holds(case: tuple[int, int, float]) -> None:
        successes, n, confidence = case
        assume_valid_counts(successes, n, confidence)
        observed = successes / n
        lower, upper = wilson_interval(successes, n, confidence)
        assert lower <= observed, f"two-sided lower {lower!r} above observed {observed!r}"
        assert observed <= upper, f"observed {observed!r} above two-sided upper {upper!r}"
        if confidence >= 0.5:
            one_sided = wilson_lower_bound(successes, n, confidence)
            assert one_sided <= observed, (
                f"one-sided bound {one_sided!r} above observed {observed!r} at "
                f"confidence {confidence!r}"
            )

    for_all(
        "the interval brackets the observed rate",
        draw_counts,
        holds,
        seed=2026081402,
        cases=CASES_ARITHMETIC,
    )


def test_the_one_sided_bound_is_never_below_the_two_sided_lower_limit() -> None:
    # The two are different quantities at the same nominal confidence: the
    # two-sided interval spends (1-c)/2 on the tail nobody is gating on, so its
    # lower end sits below the one-sided bound. Strictly below everywhere except
    # successes == 0, where both are pinned to exactly 0 -- hence >= rather than >.
    def holds(case: tuple[int, int, float]) -> None:
        successes, n, confidence = case
        assume_valid_counts(successes, n, confidence)
        one_sided = wilson_lower_bound(successes, n, confidence)
        two_sided_lower, _ = wilson_interval(successes, n, confidence)
        assert one_sided >= two_sided_lower, (
            f"one-sided {one_sided!r} < two-sided lower {two_sided_lower!r}"
        )
        if successes > 0:
            assert one_sided > two_sided_lower, (
                f"one-sided {one_sided!r} did not exceed two-sided lower "
                f"{two_sided_lower!r} at successes > 0"
            )

    for_all(
        "one-sided bound >= two-sided lower limit",
        draw_counts,
        holds,
        seed=2026081403,
        cases=CASES_ARITHMETIC,
    )


def test_a_confidence_below_a_half_turns_the_one_sided_bound_upside_down() -> None:
    # Found by the generator, at (157, 200, confidence=0.2896), while checking
    # "the bound never exceeds the observed rate". Pinned here rather than swept
    # into a precondition, because it is the sharpest edge on this API.
    #
    # THE CODE IS RIGHT AND THE PROPERTY WAS TOO BROAD. z = norm.ppf(c) is
    # negative for c < 0.5, and a lower bound you are only 29% confident in
    # genuinely sits *above* the point estimate -- P(true rate >= L) = 0.29 has no
    # solution below p_hat. The closed form is the correct bound at that level.
    #
    # It is still a footgun, and the reason to pin it: _validate_unit accepts any
    # confidence in (0, 1), so assert_pass_rate(result, min_rate, confidence=0.3)
    # is a gate that is *easier to pass than comparing the raw rate*, while
    # reading in the test source like a deliberate act of statistical caution.
    # Nothing in the docstrings says confidence < 0.5 inverts the comparison.
    observed = 157 / 200

    assert wilson_lower_bound(157, 200, 0.95) < observed
    assert wilson_lower_bound(157, 200, 0.5) == pytest.approx(observed, abs=1e-12)
    assert wilson_lower_bound(157, 200, 0.2896) > observed

    # The gate inherits it whole: a bar the sample plainly misses is cleared by
    # asking for less confidence, and the report says "passed" without qualification.
    strict = report_of(assert_pass_rate, (157, 200), 0.79, confidence=0.95)
    lax = report_of(assert_pass_rate, (157, 200), 0.79, confidence=0.2896)
    assert not strict["passed"]
    assert lax["passed"], "expected the inverted bound to clear a bar above the observed rate"
    assert lax["lower_bound"] > lax["pass_rate"]

    # The two-sided interval is immune: its z is norm.ppf((1 + c) / 2), which is
    # non-negative for every c in (0, 1). Only the one-sided bound flips.
    for confidence in (0.0001, 0.2896, 0.5, 0.95):
        low, high = wilson_interval(157, 200, confidence)
        assert low <= observed <= high

    # And the boundary case, which is the one that bites hardest: at exactly 0.5,
    # z = 0 and the bound *is* the point estimate. The module's opening argument is
    # that 20/20 must never yield a bound of 1.0 -- at confidence=0.5 it does, and
    # a gate at min_rate=1.0 passes on twenty runs.
    assert wilson_lower_bound(20, 20, 0.5) == 1.0
    assert report_of(assert_pass_rate, (20, 20), 1.0, confidence=0.5)["passed"]
    assert not report_of(assert_pass_rate, (20, 20), 1.0, confidence=0.95)["passed"]


# --------------------------------------------------------------------------- #
# Wilson: monotonicity in n and in confidence
# --------------------------------------------------------------------------- #


def draw_scaled_counts(rng: random.Random) -> tuple[int, int, int, float]:
    """``(successes, n, multiplier, confidence)`` -- the same rate, more data."""
    successes, n, confidence = draw_counts(rng)
    return successes, n, rng.choice([2, 3, 5, 10, 100]), confidence


def draw_scaled_gating_counts(rng: random.Random) -> tuple[int, int, int, float]:
    """As :func:`draw_scaled_counts`, restricted to confidences a gate would use."""
    successes, n, multiplier, _ = draw_scaled_counts(rng)
    return successes, n, multiplier, draw_gating_confidence(rng)


def test_more_data_at_the_same_rate_never_widens_the_interval() -> None:
    # (k*successes, k*n) is the identical observed rate seen k times over. The
    # half-width is z/(1 + z**2/n) * sqrt(p(1-p)/n + z**2/(4n**2)), monotonically
    # decreasing in n at fixed p, so the interval can only tighten. If more
    # evidence ever widened the interval the gate would reward under-sampling.
    def holds(case: tuple[int, int, int, float]) -> None:
        successes, n, multiplier, confidence = case
        assume_valid_counts(successes, n, confidence)
        assume(multiplier >= 1)
        narrow_lower, narrow_upper = wilson_interval(successes, n, confidence)
        wide_lower, wide_upper = wilson_interval(successes * multiplier, n * multiplier, confidence)
        small_width = narrow_upper - narrow_lower
        large_width = wide_upper - wide_lower
        assert large_width <= small_width + 1e-15, (
            f"width grew from {small_width!r} at n={n} to {large_width!r} at "
            f"n={n * multiplier} at the same rate"
        )

    for_all(
        "more data never widens the interval",
        draw_scaled_counts,
        holds,
        seed=2026081404,
        cases=CASES_ARITHMETIC,
    )


def test_more_data_at_the_same_rate_never_lowers_the_one_sided_bound() -> None:
    # The gating half of the same fact, stated on the number the gate actually
    # compares: at a fixed observed rate the lower bound climbs toward the rate as
    # n grows. This is the entire premise of the "you need more runs" diagnosis --
    # if it were false, _runs_needed would be searching for something that does
    # not exist.
    #
    # Confidence >= 0.5 only: below that the bound approaches the observed rate
    # from *above*, so more data lowers it, correctly. First drawn as
    # (89, 100, x10, confidence=0.0001), where the bound falls 0.9615 -> 0.9216.
    def holds(case: tuple[int, int, int, float]) -> None:
        successes, n, multiplier, confidence = case
        assume_valid_counts(successes, n, confidence)
        assume(confidence >= 0.5)
        assume(multiplier >= 1)
        small = wilson_lower_bound(successes, n, confidence)
        large = wilson_lower_bound(successes * multiplier, n * multiplier, confidence)
        assert large >= small - 1e-15, (
            f"bound fell from {small!r} at n={n} to {large!r} at n={n * multiplier} "
            f"at the same observed rate"
        )

    for_all(
        "more data never lowers the one-sided bound",
        draw_scaled_gating_counts,
        holds,
        seed=2026081405,
        cases=CASES_ARITHMETIC,
    )


def draw_two_confidences(rng: random.Random) -> tuple[int, int, float, float]:
    """``(successes, n, lower_confidence, higher_confidence)``."""
    successes, n, first = draw_counts(rng)
    second = draw_confidence(rng)
    return (successes, n, min(first, second), max(first, second))


def test_higher_confidence_never_narrows_the_interval() -> None:
    # Wanting to be surer cannot be free. z is increasing in the confidence level
    # and the half-width is increasing in z, so a 99% interval contains the 95% one.
    def holds(case: tuple[int, int, float, float]) -> None:
        successes, n, low_confidence, high_confidence = case
        assume_valid_counts(successes, n, low_confidence)
        assume_valid_counts(successes, n, high_confidence)
        assume(low_confidence <= high_confidence)
        loose_lower, loose_upper = wilson_interval(successes, n, low_confidence)
        tight_lower, tight_upper = wilson_interval(successes, n, high_confidence)
        assert (tight_upper - tight_lower) >= (loose_upper - loose_lower) - 1e-15, (
            f"raising confidence from {low_confidence!r} to {high_confidence!r} "
            f"narrowed the interval from {loose_upper - loose_lower!r} to "
            f"{tight_upper - tight_lower!r}"
        )

    for_all(
        "higher confidence never narrows the interval",
        draw_two_confidences,
        holds,
        seed=2026081406,
        cases=CASES_ARITHMETIC,
    )


def test_higher_confidence_never_raises_the_one_sided_bound() -> None:
    # The same trade, on the gating number: demanding more confidence can only
    # push the floor down. A gate that got easier to pass by asking for 99%
    # instead of 95% would be reporting the opposite of what its argument says.
    def holds(case: tuple[int, int, float, float]) -> None:
        successes, n, low_confidence, high_confidence = case
        assume_valid_counts(successes, n, low_confidence)
        assume_valid_counts(successes, n, high_confidence)
        assume(low_confidence <= high_confidence)
        loose = wilson_lower_bound(successes, n, low_confidence)
        tight = wilson_lower_bound(successes, n, high_confidence)
        assert tight <= loose + 1e-15, (
            f"raising confidence from {low_confidence!r} to {high_confidence!r} "
            f"raised the bound from {loose!r} to {tight!r}"
        )

    for_all(
        "higher confidence never raises the one-sided bound",
        draw_two_confidences,
        holds,
        seed=2026081407,
        cases=CASES_ARITHMETIC,
    )


# --------------------------------------------------------------------------- #
# Wilson: the endpoints, exactly
# --------------------------------------------------------------------------- #


def test_a_shut_out_pins_the_lower_end_to_exactly_zero_and_leaves_the_upper_free() -> None:
    # At p = 0 the centre and the half-width are the same expression, so the lower
    # end is exactly 0 -- but they are evaluated in different orders and the
    # cancellation leaves float residue, which is why the implementation special-
    # cases it. Exact equality, swept over every drawn (n, confidence), because
    # "approximately zero" was the bug. The upper end must stay strictly positive:
    # zero failures out of n is not proof that the true rate is 0.
    def holds(case: tuple[int, int, float]) -> None:
        _, n, confidence = case
        assume_valid_counts(0, n, confidence)
        lower, upper = wilson_interval(0, n, confidence)
        one_sided = wilson_lower_bound(0, n, confidence)
        assert lower == 0.0, f"two-sided lower at 0/{n} was {lower!r}, not exactly 0.0"
        assert one_sided == 0.0, f"one-sided bound at 0/{n} was {one_sided!r}, not exactly 0.0"
        assert upper > 0.0, f"upper at 0/{n} was {upper!r}; 0/{n} does not prove a rate of 0"

    for_all(
        "successes=0 pins the lower end to exactly 0",
        draw_counts,
        holds,
        seed=2026081408,
        cases=CASES_ARITHMETIC,
    )


def test_a_clean_sweep_pins_the_upper_end_to_exactly_one_and_leaves_the_lower_free() -> None:
    # The mirror image, and the headline claim of the module docstring: 20/20 must
    # not produce a lower bound of 1.0. The upper end is exactly 1 by the same
    # cancellation; the lower end is n/(n + z**2), which is strictly below 1 at
    # every finite n.
    def holds(case: tuple[int, int, float]) -> None:
        _, n, confidence = case
        assume_valid_counts(n, n, confidence)
        lower, upper = wilson_interval(n, n, confidence)
        assert upper == 1.0, f"upper at {n}/{n} was {upper!r}, not exactly 1.0"
        assert lower < 1.0, f"lower at {n}/{n} was {lower!r}; {n} runs do not prove perfection"
        # The one-sided bound is n/(n + z**2), which is below 1 for every z != 0 --
        # and z is exactly 0 at confidence 0.5, where the "bound" collapses onto
        # the point estimate and 1/1 does report 1.0. Pinned separately; see
        # test_a_confidence_below_a_half_turns_the_one_sided_bound_upside_down.
        if confidence > 0.5:
            one_sided = wilson_lower_bound(n, n, confidence)
            assert one_sided < 1.0, (
                f"one-sided bound at {n}/{n} was {one_sided!r} at confidence "
                f"{confidence!r}; {n} runs do not prove perfection"
            )

    for_all(
        "successes=n pins the upper end to exactly 1",
        draw_counts,
        holds,
        seed=2026081409,
        cases=CASES_ARITHMETIC,
    )


def test_the_interval_mirrors_about_a_half_when_successes_and_failures_swap() -> None:
    # Substituting successes -> n - successes maps p -> 1-p, which leaves the
    # half-width unchanged and sends the centre to 1 - centre. So the interval for
    # (n-s, n) is the reflection of the interval for (s, n) about 0.5. Any
    # asymmetry here would mean the arithmetic treats passes and failures
    # differently, which nothing in the derivation licenses.
    def holds(case: tuple[int, int, float]) -> None:
        successes, n, confidence = case
        assume_valid_counts(successes, n, confidence)
        lower, upper = wilson_interval(successes, n, confidence)
        mirror_lower, mirror_upper = wilson_interval(n - successes, n, confidence)
        assert lower == pytest.approx(1.0 - mirror_upper, abs=1e-12), (
            f"lower {lower!r} at {successes}/{n} is not the mirror of upper "
            f"{mirror_upper!r} at {n - successes}/{n}"
        )
        assert upper == pytest.approx(1.0 - mirror_lower, abs=1e-12), (
            f"upper {upper!r} at {successes}/{n} is not the mirror of lower "
            f"{mirror_lower!r} at {n - successes}/{n}"
        )

    for_all(
        "the interval mirrors about 0.5 under s -> n-s",
        draw_counts,
        holds,
        seed=2026081410,
        cases=CASES_ARITHMETIC,
    )


# --------------------------------------------------------------------------- #
# assert_pass_rate
# --------------------------------------------------------------------------- #


def draw_pass_rate_case(rng: random.Random) -> tuple[int, int, float, float]:
    """``(successes, n, min_rate, confidence)``.

    ``min_rate`` is drawn at the observed rate itself a fifth of the time: that is
    the one bar no sample size can ever clear, and the gate has a special
    diagnosis for it.
    """
    successes, n, confidence = draw_counts(rng)
    roll = rng.random()
    if roll < 0.2:
        min_rate = successes / n
    elif roll < 0.35:
        min_rate = rng.choice([0.0, 1.0])
    else:
        min_rate = rng.choice([0.5, 0.7, 0.8, 0.9, 0.95, 0.99, rng.random()])
    return successes, n, min_rate, confidence


def assume_valid_gate(successes: Any, n: Any, min_rate: Any, confidence: Any) -> None:
    """The pass-rate gate's documented domain, for the shrinker's benefit."""
    assume_valid_counts(successes, n, confidence)
    assume(isinstance(min_rate, float) and 0.0 <= min_rate <= 1.0)


def test_the_pass_rate_verdict_is_a_pure_function_of_its_inputs() -> None:
    # Three things at once, all of them "the answer depends on the data and
    # nothing else": calling twice gives the identical report; the counts spelled
    # as a tuple, as a list of outcomes, and as a SampleResult give the identical
    # verdict and numbers; and permuting the outcomes changes nothing, because a
    # rate has no memory of the order the runs came in.
    def holds(case: tuple[int, int, float, float]) -> None:
        successes, n, min_rate, confidence = case
        assume_valid_gate(successes, n, min_rate, confidence)
        outcomes = outcomes_for(successes, n)
        shuffled = list(outcomes)
        random.Random(successes * 1000 + n).shuffle(shuffled)
        spellings = [(successes, n), outcomes, shuffled, sample_of_outcomes(outcomes)]

        collected = [
            report_of(assert_pass_rate, spelling, min_rate, confidence=confidence)
            for spelling in spellings
        ]
        repeated = report_of(assert_pass_rate, spellings[0], min_rate, confidence=confidence)
        assert collected[0] == repeated, "two identical calls produced different reports"
        for spelling, report in zip(spellings[1:], collected[1:], strict=True):
            assert report == collected[0], (
                f"{type(spelling).__name__} spelling of {successes}/{n} gave "
                f"{report!r}, tuple spelling gave {collected[0]!r}"
            )

    for_all(
        "the pass-rate verdict is a pure function of its inputs",
        draw_pass_rate_case,
        holds,
        seed=2026081411,
        cases=CASES_GATE,
    )


def test_the_pass_rate_gate_is_exactly_the_lower_bound_against_the_bar() -> None:
    # The claim in the module docstring, checked as an identity rather than
    # trusted: the verdict is lower_bound >= min_rate and never the point
    # estimate. The second assertion is the one with teeth -- it fails the moment
    # anyone "fixes" a flaky gate by comparing successes/n.
    def holds(case: tuple[int, int, float, float]) -> None:
        successes, n, min_rate, confidence = case
        assume_valid_gate(successes, n, min_rate, confidence)
        report = report_of(assert_pass_rate, (successes, n), min_rate, confidence=confidence)
        expected = wilson_lower_bound(successes, n, confidence) >= min_rate
        assert report["passed"] is expected, (
            f"verdict {report['passed']!r} but lower bound {report['lower_bound']!r} "
            f"vs min_rate {report['min_rate']!r}"
        )
        assert report["lower_bound"] == pytest.approx(
            wilson_lower_bound(successes, n, confidence), abs=0.0
        )
        assert report["pass_rate"] == successes / n
        assert report["failures"] == n - successes

    for_all(
        "the gate is the lower bound against the bar",
        draw_pass_rate_case,
        holds,
        seed=2026081412,
        cases=CASES_GATE,
    )


def test_a_passing_pass_rate_report_never_carries_a_runs_needed() -> None:
    # "How many more runs would clear the bar" is a question only a failure has.
    # A passing report carrying one would read as advice to sample more when the
    # gate has already cleared, and a CI dashboard that surfaces the key would
    # nag on green builds.
    def holds(case: tuple[int, int, float, float]) -> None:
        successes, n, min_rate, confidence = case
        assume_valid_gate(successes, n, min_rate, confidence)
        report = report_of(assert_pass_rate, (successes, n), min_rate, confidence=confidence)
        if report["passed"]:
            assert "runs_needed" not in report, (
                f"passing report carried runs_needed={report['runs_needed']!r}"
            )
            assert "underpowered" not in report, (
                f"passing report carried underpowered={report['underpowered']!r}"
            )
        else:
            # The converse, which is the part that makes the first half meaningful:
            # a failure always says which of the two failure modes it was.
            assert "underpowered" in report and "runs_needed" in report
            assert report["underpowered"] is (report["pass_rate"] >= report["min_rate"])
            if report["runs_needed"] is not None:
                # And the number, when offered, has to be true: at that many runs
                # and this observed rate the bound really does clear the bar.
                needed = report["runs_needed"]
                rounded = min(round(report["pass_rate"] * needed), needed)
                assert wilson_lower_bound(rounded, needed, confidence) >= min_rate, (
                    f"runs_needed={needed} does not actually clear min_rate={min_rate!r}"
                )

    for_all(
        "a passing report never carries runs_needed",
        draw_pass_rate_case,
        holds,
        seed=2026081413,
        cases=CASES_GATE,
    )


#: The numbers the failure message prints, in the order it prints them. Parsed
#: back out rather than reconstructed with the same format string the code uses,
#: which would only prove that format() is deterministic.
_MESSAGE_NUMBERS = re.compile(
    r"failed: (?P<successes>\d+)/(?P<n>\d+) passed "
    r"\(observed (?P<observed>[\d.]+)\); "
    r"one-sided (?P<confidence>[\d.]+)% Wilson lower bound "
    r"(?P<lower>[\d.]+) < min_rate (?P<min_rate>[\d.]+)\. "
    r"Two-sided [\d.]+% interval \[(?P<interval_lower>[\d.]+), (?P<interval_upper>[\d.]+)\]"
)


def test_the_numbers_in_the_failure_message_are_the_numbers_in_its_stats() -> None:
    # The module's whole bet is that the failure message *is* the statistical
    # report, and that a caller who prefers the dict gets the same facts. If the
    # two ever drift, one of the two audiences is being lied to and neither can
    # tell. Parsed with a regex and compared to .stats at the precision printed:
    # four decimals means a half-ulp of 5e-5.
    def holds(case: tuple[int, int, float, float]) -> None:
        successes, n, min_rate, confidence = case
        assume_valid_gate(successes, n, min_rate, confidence)
        try:
            assert_pass_rate((successes, n), min_rate, confidence=confidence)
        except PassRateError as exc:
            message = str(exc)
            stats = exc.stats
        else:
            raise Rejected  # passing gates have no message to check

        match = _MESSAGE_NUMBERS.search(message)
        assert match is not None, f"failure message did not parse: {message!r}"
        assert int(match["successes"]) == stats["successes"]
        assert int(match["n"]) == stats["n"]
        assert float(match["observed"]) == pytest.approx(stats["pass_rate"], abs=5e-5)
        assert float(match["lower"]) == pytest.approx(stats["lower_bound"], abs=5e-5)
        assert float(match["min_rate"]) == pytest.approx(stats["min_rate"], abs=5e-5)
        assert float(match["interval_lower"]) == pytest.approx(stats["interval_lower"], abs=5e-5)
        assert float(match["interval_upper"]) == pytest.approx(stats["interval_upper"], abs=5e-5)
        # The message asserts the inequality that produced the failure; if the
        # printed numbers do not satisfy it, the reader is looking at a
        # contradiction and will not trust the next one either.
        assert float(match["lower"]) <= float(match["min_rate"])

    for_all(
        "the failure message's numbers are its stats",
        draw_pass_rate_case,
        holds,
        seed=2026081414,
        cases=CASES_GATE,
    )


def test_lowering_the_bar_never_turns_a_pass_into_a_failure() -> None:
    # Monotone in min_rate: the bound does not know what it is being compared to,
    # so a gate that passes at 0.9 has to pass at 0.8. Cheap to state, and it is
    # the invariant that would break if anyone made the threshold feed back into
    # the estimate.
    def holds(case: tuple[int, int, float, float]) -> None:
        successes, n, min_rate, confidence = case
        assume_valid_gate(successes, n, min_rate, confidence)
        strict = report_of(assert_pass_rate, (successes, n), min_rate, confidence=confidence)
        lenient_bar = min_rate / 2.0
        lenient = report_of(assert_pass_rate, (successes, n), lenient_bar, confidence=confidence)
        if strict["passed"]:
            assert lenient["passed"], (
                f"passed at min_rate={min_rate!r} but failed at {lenient_bar!r}"
            )

    for_all(
        "lowering the bar never turns a pass into a failure",
        draw_pass_rate_case,
        holds,
        seed=2026081415,
        cases=CASES_GATE,
    )


# --------------------------------------------------------------------------- #
# assert_no_regression
# --------------------------------------------------------------------------- #


def draw_sample_pair(rng: random.Random) -> tuple[list[float], list[float], float]:
    """``(current, baseline, alpha)`` -- two independently shaped samples."""
    return draw_scores(rng, maximum=15), draw_scores(rng, maximum=15), rng.choice([0.01, 0.05, 0.1])


def test_swapping_current_and_baseline_mirrors_the_statistic() -> None:
    # U1(A, B) + U1(B, A) = |A|*|B| exactly, by the definition of U as a count over
    # the cartesian product -- every pair contributes 1 to one side or 0.5 to each.
    # Exact equality, not approx: it is a count of pairs. The means and medians
    # have to swap places too, which is the cheap way to catch the argument order
    # being crossed somewhere in the report.
    def holds(case: tuple[list[float], list[float], float]) -> None:
        current, baseline, alpha = case
        assume(bool(current) and bool(baseline))
        forward = report_of(assert_no_regression, current, baseline, alpha=alpha)
        backward = report_of(assert_no_regression, baseline, current, alpha=alpha)
        pairs = len(current) * len(baseline)
        assert forward["u_statistic"] + backward["u_statistic"] == pairs, (
            f"U({len(current)}x{len(baseline)}) pair counts sum to "
            f"{forward['u_statistic'] + backward['u_statistic']!r}, not {pairs}"
        )
        assert forward["mean_current"] == backward["mean_baseline"]
        assert forward["median_current"] == backward["median_baseline"]

    for_all(
        "swapping the arguments mirrors the U statistic",
        draw_sample_pair,
        holds,
        seed=2026081416,
        cases=CASES_SCIPY,
    )


def test_two_samples_cannot_each_be_a_regression_against_the_other() -> None:
    # The symmetric reading of a one-sided test: "current is worse than baseline"
    # and "baseline is worse than current" cannot both be established at any alpha
    # below 0.5. If both directions ever fired, the direction convention
    # (alternative="less") would not be doing what its docstring says and every
    # verdict this gate has ever returned would be a coin flip.
    def holds(case: tuple[list[float], list[float], float]) -> None:
        current, baseline, alpha = case
        assume(bool(current) and bool(baseline))
        assume(alpha < 0.5)
        forward = report_of(assert_no_regression, current, baseline, alpha=alpha)
        backward = report_of(assert_no_regression, baseline, current, alpha=alpha)
        assert forward["passed"] or backward["passed"], (
            f"both directions flagged a regression at alpha={alpha!r}: "
            f"p_forward={forward['p_value']!r}, p_backward={backward['p_value']!r}"
        )

    for_all(
        "two samples cannot each regress against the other",
        draw_sample_pair,
        holds,
        seed=2026081417,
        cases=CASES_SCIPY,
    )


def test_a_sample_never_regresses_against_itself() -> None:
    # The gate's floor: rerun the identical numbers and the build stays green.
    # Includes the degenerate all-identical sample, where the rank test has zero
    # variance and scipy may report NaN -- the module writes its comparison so a
    # NaN passes, because two indistinguishable samples are not evidence of a drop.
    def holds(case: tuple[list[float], list[float], float]) -> None:
        current, _, alpha = case
        assume(bool(current))
        report = report_of(assert_no_regression, current, list(current), alpha=alpha)
        assert report["passed"], (
            f"a sample regressed against a copy of itself: p={report['p_value']!r} "
            f"at alpha={alpha!r}"
        )

    for_all(
        "a sample never regresses against itself",
        draw_sample_pair,
        holds,
        seed=2026081418,
        cases=CASES_SCIPY,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "THE PROPERTY IS WRONG, NOT THE CODE. Appending the same observation to both "
        "samples is not a no-op: it adds a pair to the comparison and raises n, and a "
        "rank test with more data has more power. A p-value just above alpha can and "
        "does cross below it. Minimal counterexample found by the shrinker: "
        "current=[0, 0] vs baseline=[1, 2, 2] gives p=0.0641 and passes; append 1 to "
        "both and p=0.0461 fails at alpha=0.05. The jump is amplified by scipy's "
        "method='auto', which uses the exact permutation distribution when there are "
        "no ties and switches to the tie-corrected normal approximation when the "
        "appended value creates one -- so the p-value moves for two reasons at once. "
        "Kept and run rather than deleted: it pins the behaviour, and if a future "
        "scipy makes it monotone this test starts passing and says so."
    ),
)
def test_appending_the_same_observation_to_both_samples_never_creates_a_regression() -> None:
    # Stated as the task stated it, and left to fail. The intuition behind it --
    # "adding the same thing to both sides is neutral" -- is an intuition about
    # arithmetic, not about evidence: the appended pair is one more comparison,
    # and a tie in a rank test is not nothing.
    def holds(case: tuple[list[float], list[float], float]) -> None:
        current, baseline, alpha = case
        assume(bool(current) and bool(baseline))
        before = report_of(assert_no_regression, current, baseline, alpha=alpha)
        assume(before["passed"])
        extra = current[0]
        after = report_of(
            assert_no_regression, [*current, extra], [*baseline, extra], alpha=alpha
        )
        assert after["passed"], (
            f"appending {extra!r} to both samples flipped the verdict: "
            f"p went {before['p_value']!r} -> {after['p_value']!r} at alpha={alpha!r}"
        )

    for_all(
        "appending an identical observation to both never creates a regression",
        draw_sample_pair,
        holds,
        seed=2026081419,
        cases=CASES_SCIPY,
    )


def test_the_shrunk_counterexample_to_the_appending_property_is_pinned() -> None:
    # The minimal case the shrinker found for the xfail above, written out so that
    # the finding survives independently of the generator, the seed, and anyone's
    # decision to retune the draws. Two observations against three, one appended
    # value, and the verdict flips.
    current = [0.0, 0.0]
    baseline = [1.0, 2.0, 2.0]
    before = report_of(assert_no_regression, current, baseline, alpha=0.05)
    after = report_of(assert_no_regression, [*current, 1.0], [*baseline, 1.0], alpha=0.05)

    assert before["passed"], f"expected the two-sample case to pass, got p={before['p_value']!r}"
    assert not after["passed"], f"expected the appended case to fail, got p={after['p_value']!r}"
    assert before["p_value"] > 0.05 > after["p_value"]


# --------------------------------------------------------------------------- #
# assert_score_distribution
# --------------------------------------------------------------------------- #


def mean_oracle(values: Sequence[float]) -> float:
    """The mean by exact summation -- fsum, not numpy's pairwise accumulation."""
    return math.fsum(values) / len(values)


def p10_oracle(values: Sequence[float]) -> float:
    """The 10th percentile by linear interpolation between order statistics.

    Written from the definition the docstring pins ("numpy's default *linear*
    interpolation"), in plain Python, so it is a check on numpy's behaviour rather
    than a restatement of the call under test.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * 0.10
    below = math.floor(position)
    above = math.ceil(position)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def stddev_oracle(values: Sequence[float]) -> float:
    """The sample standard deviation, ddof=1, by the two-pass definition."""
    centre = mean_oracle(values)
    return math.sqrt(math.fsum((value - centre) ** 2 for value in values) / (len(values) - 1))


def draw_distribution_case(rng: random.Random) -> tuple[list[float], float, float, float]:
    """``(scores, min_mean, min_p10, max_stddev)``, thresholds drawn near the data."""
    scores = draw_scores(rng, minimum=2, maximum=40)
    centre = sum(scores) / len(scores)
    return (
        scores,
        centre + rng.choice([-1.0, 0.0, 1.0]),
        min(scores) + rng.choice([-1.0, 0.0, 1.0]),
        abs(rng.choice([0.0, 0.5, 1.0, 10.0])),
    )


def test_the_reported_statistics_match_an_independent_computation() -> None:
    # numpy is not the oracle here; the definitions in the docstring are. p10 by
    # linear interpolation between order statistics and stddev at ddof=1 are both
    # choices the module argues for in prose, and prose is not enforcement -- a
    # ddof=0 stddev understates exactly the spread this gate exists to bound, and
    # would still look plausible in every failure message it ever printed.
    def holds(case: tuple[list[float], float, float, float]) -> None:
        scores, min_mean, min_p10, max_stddev = case
        assume(len(scores) >= 2)
        report = report_of(
            assert_score_distribution,
            scores,
            min_mean=min_mean,
            min_p10=min_p10,
            max_stddev=max_stddev,
        )
        assert report["mean"] == pytest.approx(mean_oracle(scores), rel=1e-12, abs=1e-12)
        assert report["p10"] == pytest.approx(p10_oracle(scores), rel=1e-12, abs=1e-12)
        assert report["stddev"] == pytest.approx(stddev_oracle(scores), rel=1e-9, abs=1e-12)
        assert report["min_score"] == min(scores)
        assert report["max_score"] == max(scores)
        assert report["n"] == len(scores)

    for_all(
        "the reported statistics match an independent computation",
        draw_distribution_case,
        holds,
        seed=2026081420,
        cases=CASES_GATE,
    )


def test_adding_a_value_equal_to_the_mean_leaves_the_mean_alone() -> None:
    # (S + m)/(n+1) == m when m == S/n, so the mean is a fixed point of its own
    # sample. Only the mean: adding a point at the centre lowers the stddev and
    # can move p10, and this test deliberately does not claim otherwise. Compared
    # at 1e-9 relative rather than exactly, because the two sums are accumulated in
    # different orders and float addition is not associative.
    def holds(case: tuple[list[float], float, float, float]) -> None:
        scores, min_mean, min_p10, max_stddev = case
        assume(len(scores) >= 2)
        thresholds = {"min_mean": min_mean, "min_p10": min_p10, "max_stddev": max_stddev}
        before = report_of(assert_score_distribution, scores, **thresholds)
        after = report_of(assert_score_distribution, [*scores, before["mean"]], **thresholds)
        assert after["mean"] == pytest.approx(before["mean"], rel=1e-9, abs=1e-12), (
            f"mean moved from {before['mean']!r} to {after['mean']!r} when its own "
            f"value was appended to a sample of {len(scores)}"
        )

    for_all(
        "appending the mean does not move the mean",
        draw_distribution_case,
        holds,
        seed=2026081421,
        cases=CASES_GATE,
    )


def test_the_distribution_verdict_is_the_conjunction_of_the_gates_it_was_given() -> None:
    # Every supplied threshold is checked independently and all violations are
    # reported together -- the docstring's promise, which exists so that a fixed
    # mean does not reveal a stddev problem on the next CI run. Checked as an
    # identity against the three comparisons done by hand.
    def holds(case: tuple[list[float], float, float, float]) -> None:
        scores, min_mean, min_p10, max_stddev = case
        assume(len(scores) >= 2)
        report = report_of(
            assert_score_distribution,
            scores,
            min_mean=min_mean,
            min_p10=min_p10,
            max_stddev=max_stddev,
        )
        expected = sum(
            [
                mean_oracle(scores) < min_mean,
                p10_oracle(scores) < min_p10,
                stddev_oracle(scores) > max_stddev,
            ]
        )
        # A violation can sit within float noise of its threshold, in which case
        # the hand computation and numpy's can legitimately disagree by one. Only
        # an unambiguous disagreement is a failure.
        assert abs(len(report["violations"]) - expected) <= 1, (
            f"{len(report['violations'])} violations reported, {expected} computed by "
            f"hand: {report['violations']!r}"
        )
        assert report["passed"] is (len(report["violations"]) == 0)
        assert set(report["checks"]) == {"min_mean", "min_p10", "max_stddev"}

    for_all(
        "the verdict is the conjunction of the gates given",
        draw_distribution_case,
        holds,
        seed=2026081422,
        cases=CASES_GATE,
    )


# --------------------------------------------------------------------------- #
# cross-cutting
# --------------------------------------------------------------------------- #


def test_no_gate_ever_reports_a_nan_or_an_infinity() -> None:
    # A NaN in a report is worse than an exception: it propagates through a
    # dashboard, compares false against every threshold, and serialises into an
    # evidence log as `NaN`, which is not JSON. Every numeric field of every report
    # from all three gates, over generated input.
    def holds(case: tuple[list[float], list[float], float]) -> None:
        current, baseline, alpha = case
        assume(bool(current) and bool(baseline))
        successes = sum(1 for value in current if value > 0)
        reports = [
            report_of(assert_pass_rate, (successes, len(current)), 0.5, confidence=0.95),
            report_of(assert_no_regression, current, baseline, alpha=alpha),
        ]
        if len(current) >= 2:
            reports.append(
                report_of(
                    assert_score_distribution,
                    current,
                    min_mean=0.0,
                    min_p10=0.0,
                    max_stddev=1.0,
                )
            )
        for report in reports:
            for key, value in finite_values(report):
                assert math.isfinite(value), f"{report['gate']}.{key} is {value!r}"

    for_all(
        "no gate reports a NaN or an infinity",
        draw_sample_pair,
        holds,
        seed=2026081423,
        cases=CASES_SCIPY,
    )


def test_no_gate_mutates_the_data_it_was_handed() -> None:
    # A gate is a measurement. Sorting the caller's list in place to find a
    # percentile, or consuming an iterator, would be invisible until the second
    # gate in the same test read different data from the first -- which is the
    # kind of bug that gets blamed on the model.
    def holds(case: tuple[list[float], list[float], float]) -> None:
        current, baseline, alpha = case
        assume(len(current) >= 2 and bool(baseline))
        current_before = list(current)
        baseline_before = list(baseline)
        outcomes = [value > 0 for value in current]
        outcomes_before = list(outcomes)

        report_of(assert_no_regression, current, baseline, alpha=alpha)
        report_of(assert_score_distribution, current, min_mean=0.0)
        report_of(assert_pass_rate, outcomes, 0.5)

        assert current == current_before, f"current was mutated: {current!r}"
        assert baseline == baseline_before, f"baseline was mutated: {baseline!r}"
        assert outcomes == outcomes_before, f"outcomes were mutated: {outcomes!r}"

    for_all(
        "no gate mutates its input",
        draw_sample_pair,
        holds,
        seed=2026081424,
        cases=CASES_SCIPY,
    )


def draw_invalid_counts(rng: random.Random) -> tuple[Any, Any, Any]:
    """``(successes, n, confidence)`` with exactly one thing wrong with it."""
    successes, n, confidence = draw_counts(rng)
    fault = rng.randrange(5)
    if fault == 0:
        return successes, rng.choice([0, -1, -7]), confidence
    if fault == 1:
        return rng.choice([-1, -5, -100]), n, confidence
    if fault == 2:
        return n + rng.choice([1, 2, 50]), n, confidence
    if fault == 3:
        return successes, n, rng.choice([0.0, 1.0, -0.5, 1.5, 42.0])
    return successes, n, float("nan")


def test_every_malformed_count_is_refused_by_value_error_naming_the_value() -> None:
    # A gate that accepts nonsense reports nonsense. Every refusal has to be a
    # ValueError -- not a TypeError from numpy three frames down, not a
    # ZeroDivisionError -- and the message has to contain the offending number,
    # because the caller is looking at a stack trace and needs to know which of the
    # three arguments they got wrong.
    def holds(case: tuple[Any, Any, Any]) -> None:
        successes, n, confidence = case
        offender = next(
            (
                value
                for value, ok in (
                    (n, isinstance(n, int) and n >= 1),
                    (successes, isinstance(successes, int) and 0 <= successes <= max(n, 0)),
                    (confidence, isinstance(confidence, float) and 0.0 < confidence < 1.0),
                )
                if not ok
            ),
            None,
        )
        assume(offender is not None)
        for gate in (wilson_lower_bound, wilson_interval):
            with pytest.raises(ValueError) as raised:
                gate(successes, n, confidence)
            message = str(raised.value)
            assert str(offender) in message or repr(offender) in message, (
                f"{gate.__name__} refused {case!r} without naming {offender!r}: {message}"
            )

    for_all(
        "malformed counts are refused by ValueError naming the value",
        draw_invalid_counts,
        holds,
        seed=2026081425,
        cases=CASES_ARITHMETIC,
    )


#: Refusals that cannot be generated from the count grid: wrong *types*, not wrong
#: numbers. Each row is (id, thunk, the value the message ought to name).
REFUSALS: tuple[Any, ...] = (
    pytest.param(lambda: wilson_lower_bound(3, 5, "0.95"), "0.95", id="confidence-as-string"),
    pytest.param(lambda: wilson_lower_bound(True, 5), "True", id="successes-as-bool"),
    pytest.param(lambda: wilson_lower_bound(3, 5.5), "5.5", id="n-as-float"),
    pytest.param(lambda: assert_pass_rate((3, 5), 1.5), "1.5", id="min_rate-out-of-range"),
    pytest.param(lambda: assert_pass_rate(["yes", "no"], 0.5), "'yes'", id="outcome-as-string"),
    pytest.param(
        lambda: assert_score_distribution([1.0, float("nan")], min_mean=1.0),
        "NaN",
        id="score-is-nan",
    ),
    pytest.param(
        lambda: assert_score_distribution([1.0, 2.0], max_stddev=-1.0), "-1.0", id="negative-stddev"
    ),
    pytest.param(
        lambda: assert_score_distribution([1.0, "2"], min_mean=1.0), "'2'", id="score-as-string"
    ),
    pytest.param(
        lambda: assert_score_distribution([1.0, 2.0], min_mean="x"),
        "'x'",
        id="min_mean-as-string",
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "THE CODE IS WRONG, NOT THE PROPERTY -- mildly. The refusal names only "
                "the type: 'min_mean must be a number or None, got str'. Every other "
                "refusal in this module quotes the offending value, and a caller who "
                "passed a threshold through a config file needs to see which string it "
                "was. One-word fix: add {value!r} to the message."
            ),
        ),
    ),
    pytest.param(
        lambda: assert_score_distribution("abc", min_mean=1.0),
        "'abc'",
        id="scores-as-string",
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "THE CODE IS WRONG, NOT THE PROPERTY -- mildly. 'scores must be a "
                "sequence of numbers, got str' does not say which str. Same one-word "
                "fix as min_mean-as-string; the two share a habit, not a code path."
            ),
        ),
    ),
)


@pytest.mark.parametrize(("thunk", "offender"), REFUSALS)
def test_every_type_refusal_names_the_offending_value(
    thunk: Callable[[], Any], offender: str
) -> None:
    # The generated test above covers wrong numbers; this covers wrong types,
    # which cannot be drawn from a grid of ints and floats. Same standard: a
    # ValueError, and the value in the message.
    with pytest.raises(ValueError) as raised:
        thunk()
    assert offender in str(raised.value), (
        f"refusal did not name {offender}: {raised.value}"
    )


def test_the_generator_is_deterministic() -> None:
    # The claim the whole file rests on: same seed, same cases, byte for byte. If
    # this ever fails, every "reproduce with seed=..." message above is a lie and
    # the failures reported by this file cannot be trusted.
    first = [draw_pass_rate_case(random.Random(2026081499)) for _ in range(5)]
    second = [draw_pass_rate_case(random.Random(2026081499)) for _ in range(5)]
    assert first == second

    stream = random.Random(2026081499)
    sequence_one = [draw_counts(stream) for _ in range(20)]
    stream = random.Random(2026081499)
    sequence_two = [draw_counts(stream) for _ in range(20)]
    assert sequence_one == sequence_two


def test_the_shrinker_actually_shrinks() -> None:
    # The harness is code too, and an assertion about a "minimal reproducing
    # input" is only as good as this. A property false for every n >= 3 must
    # shrink to n == 3, not report whatever n the generator happened to draw.
    def holds(case: tuple[int, int, float]) -> None:
        _, n, _ = case
        assert n < 3, f"n={n} is at least 3"

    minimal, reason = _shrink((7, 97, 0.95), holds)
    assert minimal[1] == 3, f"shrank to {minimal!r}, expected n=3"
    assert "n=3" in reason

    # And it must respect Rejected: a candidate that is out of scope is not a
    # smaller counterexample, so the walk stops above it rather than through it.
    def holds_with_precondition(case: tuple[int, int, float]) -> None:
        _, n, _ = case
        assume(n >= 5)
        assert n < 3, f"n={n} is at least 3"

    minimal, _ = _shrink((7, 97, 0.95), holds_with_precondition)
    assert minimal[1] == 5, f"shrank past the precondition to {minimal!r}"
