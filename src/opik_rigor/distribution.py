"""Statistical gates: assertions that know they are looking at a sample.

An LLM test suite that asserts ``pass_rate >= 0.9`` is not testing the model, it
is testing one draw from the model. 18/20 and 900/1000 are both "90%", and only
one of them is evidence: the first is consistent with a true rate anywhere from
about 70% to 98%, the second with 88% to 92%. A gate that cannot tell those two
situations apart will flap on the small sample and call it flakiness.

Every gate here therefore reports three things a bare comparison cannot:

* **what was observed** -- the point estimate, which is never what is gated on;
* **how well it is pinned down** -- an interval or a p-value, which is;
* **which of the two failure modes occurred** -- "you failed the bar" versus
  "you did not sample enough to tell". Those call for opposite responses (fix
  the system versus raise n), and a message that conflates them sends the reader
  in the wrong direction half the time.

The failure message *is* the statistical report, and the same numbers are
attached to the exception as a dict so a caller does not have to parse prose.
Anything given an :class:`~opik_rigor.evidence.EvidenceLog` writes exactly one
``assertion.evaluated`` record whether it passed or failed -- a gate that only
records its passes is a highlight reel, not an audit trail.

**What this module imports.** NumPy at module scope, for the score summaries and
for accepting the numpy scalar types that array code and CSV round-trips produce.
SciPy only inside :func:`assert_no_regression`, which is the sole caller that
needs it: importing ``scipy.stats`` eagerly cost every suite about a second of
warm import for a gate most of them never call.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np

from .errors import StatisticalAssertionError
from .evidence import EVENT_ASSERTION, EvidenceLog
from .sampling import SampleResult, _short_repr

#: Default confidence for the Wilson bound. 0.95 one-sided, which is the honest
#: default for a gate: you care only about the floor, not about how high the rate
#: might be, so spending half the error budget on an upper bound you never read
#: costs you sample size for nothing.
DEFAULT_CONFIDENCE = 0.95

#: Default significance level for the regression gate. A regression is flagged
#: when p < alpha, so this is the rate at which an unchanged system is expected
#: to trip the gate anyway -- 1 run in 20.
DEFAULT_ALPHA = 0.05

#: What the pass-rate gate accepts: a sample, a ``(successes, n)`` pair, or the
#: raw per-run outcomes.
PassData = SampleResult | tuple[int, int] | Sequence[bool]

#: What the distribution and regression gates accept: a sample (its ``.scores()``
#: are used) or the numbers themselves.
ScoreData = SampleResult | Sequence[float]

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CONFIDENCE",
    "PassRateError",
    "RegressionError",
    "ScoreDistributionError",
    "assert_no_regression",
    "assert_pass_rate",
    "assert_score_distribution",
    "wilson_interval",
    "wilson_lower_bound",
]


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


class PassRateError(StatisticalAssertionError):
    """Raised when the lower confidence bound on the pass rate is below the bar.

    Note what this does *not* mean: it does not mean the system's true pass rate
    is below ``min_rate``. It means the evidence does not establish that it is
    above it, which is the only thing a finite sample can ever establish. The
    message says which of the two situations produced the failure.
    """


class ScoreDistributionError(StatisticalAssertionError):
    """Raised when a judge-score distribution violates one or more shape gates.

    Carries *every* violated gate, not the first one found: fixing a mean and
    then rediscovering a stddev problem on the next CI run is two round trips
    for one piece of information.
    """


class RegressionError(StatisticalAssertionError):
    """Raised when current scores are significantly worse than the baseline.

    "Significantly" in the strict sense: a shift this large or larger would occur
    with probability less than ``alpha`` if the two samples came from the same
    distribution. It is a statement about the evidence, not about the size of the
    drop -- a tiny but very consistent regression trips this, and a large one
    seen in four runs does not.
    """


# --------------------------------------------------------------------------- #
# Wilson score interval
# --------------------------------------------------------------------------- #


#: The standard normal, used only for its quantile function. ``NormalDist.inv_cdf``
#: is CPython's implementation of Wichura's AS241, stdlib since 3.8, and this
#: package requires >= 3.10 -- so ``scipy.stats.norm.ppf`` bought nothing here but
#: the import of all of ``scipy.stats``. Swept against ``norm.ppf`` over 6.4M points
#: of the open interval (0, 1): max relative deviation 1.22e-15, max 8 ULP.
#: Propagated through :func:`_wilson` that shrinks to at most 3.3e-16 in the
#: reported bound, and across 1,443,519 combinations of confidence x successes x n
#: x min_rate it flips no gate verdict and changes no :func:`_runs_needed` answer.
#:
#: Constructed once at import: ``NormalDist()`` is cheap, but this sits on the
#: pass-rate path, which is the most-called gate in the package.
_STANDARD_NORMAL = NormalDist()


def _z(probability: float) -> float:
    """The standard normal quantile (inverse CDF) at ``probability``.

    Split out so the three call sites read alike, and so that the one place which
    would change if the quantile source ever moves again has a name.
    """
    return _STANDARD_NORMAL.inv_cdf(probability)


def _clamp(value: float) -> float:
    """Squeeze a bound into ``[0, 1]``.

    The closed form can stray a hair outside the unit interval at extreme counts
    (0/n and n/n especially). A "pass rate" of -1e-17 is a distraction in a
    failure message and a nonsense in a report, so it is clipped here rather than
    explained everywhere downstream.
    """
    return min(1.0, max(0.0, value))


def _as_count(value: Any, name: str) -> int:
    """Coerce to a plain ``int``, rejecting bools and non-integers.

    ``operator.index`` accepts numpy integers (which arrive routinely from array
    code) and rejects floats, which is the right split: ``n=19.5`` is a bug, not
    a rounding opportunity. Bools are rejected explicitly because ``bool`` is an
    ``int`` subclass and ``successes=True`` almost certainly means someone passed
    an outcome where a count belongs.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got the boolean {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__} {value!r}"
        ) from None


def _validate_unit(value: Any, name: str, *, exclusive: bool) -> float:
    """Validate a probability-valued argument.

    ``exclusive`` distinguishes a confidence or alpha (which must be strictly
    inside ``(0, 1)`` -- the normal quantile at 1.0 is infinite, and a gate at
    alpha 0 can never fire) from a threshold like ``min_rate``, where 0.0 and 1.0
    are both meaningful bars to set.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__} {value!r}")
    numeric = float(value)
    ok = 0.0 < numeric < 1.0 if exclusive else 0.0 <= numeric <= 1.0
    if not ok:  # also catches NaN, which fails every comparison
        bounds = "strictly between 0 and 1" if exclusive else "between 0 and 1 inclusive"
        raise ValueError(f"{name} must be {bounds}, got {value!r}")
    return numeric


def _validate_gating_confidence(value: Any) -> float:
    """Validate a *one-sided* confidence, and refuse the half of the range that inverts it.

    ``z = ppf(c)`` is negative below 0.5, so the "lower bound" comes out *above*
    the observed rate and gets **worse** as the sample grows:
    ``wilson_lower_bound(89, 100, 0.0001)`` is 0.9615 and the same rate over 1000
    runs gives 0.9216. At exactly 0.5, ``z`` is 0 and the bound *is*
    ``successes / n`` -- so ``wilson_lower_bound(20, 20, 0.5)`` returns 1.0 and
    ``assert_pass_rate((20, 20), 1.0, confidence=0.5)`` passes, which is the exact
    claim this module's opening paragraph exists to refuse.

    Both numbers are arithmetically correct, which is why this is a refusal rather
    than a bug fix in the formula. Neither is a thing a caller wants, and the
    failure is silent and backwards: ``confidence=0.3`` reads in a test file like
    an act of statistical caution and produces a gate *looser* than comparing the
    raw rate. A one-sided bound is a floor you are willing to defend, and there is
    no level of belief below a coin flip that anyone defends.

    The two-sided :func:`wilson_interval` keeps the full open interval and is not
    routed through here: its ``z`` is ``ppf((1 + c) / 2)``, non-negative for every
    ``c`` in ``(0, 1)``, so it never inverts.
    """
    numeric = _validate_unit(value, "confidence", exclusive=True)
    if numeric <= 0.5:
        raise ValueError(
            f"confidence must be greater than 0.5 for a one-sided bound, got {value!r}. "
            f"Below 0.5 the z is negative, so the 'lower bound' sits above the observed "
            f"rate and falls as n grows -- more evidence, worse bound -- and a gate set "
            f"there is looser than comparing the raw rate. At exactly 0.5 the bound is "
            f"the raw rate. Use wilson_interval if you want the full range two-sided"
        )
    return numeric


def _validate_counts(successes: Any, n: Any) -> tuple[int, int]:
    """Check the two counts against each other, not just individually."""
    n_int = _as_count(n, "n")
    successes_int = _as_count(successes, "successes")
    if n_int < 1:
        raise ValueError(f"n must be >= 1, got {n_int}; a rate over zero runs is not a rate")
    if successes_int < 0:
        raise ValueError(f"successes must be >= 0, got {successes_int}")
    if successes_int > n_int:
        raise ValueError(f"successes ({successes_int}) cannot exceed n ({n_int})")
    return successes_int, n_int


def _wilson(successes: int, n: int, z: float) -> tuple[float, float]:
    """The Wilson score interval in closed form, both ends.

    Implemented directly rather than pulled from statsmodels: it is four lines of
    arithmetic, and a statistics library exists to make its own numbers auditable
    on the page.

    With ``p = successes / n``::

        centre = (p + z**2 / (2n)) / (1 + z**2 / n)
        half   = z / (1 + z**2 / n) * sqrt(p(1-p)/n + z**2 / (4n**2))

    Wilson rather than the textbook normal approximation
    ``p +- z*sqrt(p(1-p)/n)`` because the latter is exactly wrong where an eval
    suite lives: at 20/20 it gives the interval ``[1.0, 1.0]``, claiming certainty
    of perfection from twenty runs. Wilson gives ``[0.8389, 1.0]`` two-sided at
    95%, or a one-sided 95% lower bound of ``0.8808`` -- the number a gate reads.
    That is what twenty runs actually buy you.

    **Wilson rather than Clopper-Pearson**, the other standard choice. Both are
    defensible; the trade is coverage against power. Clopper-Pearson is *exact* in
    the sense that its coverage is guaranteed to be at least the nominal level, but
    it buys that guarantee by being conservative -- at the small n an eval suite
    actually runs (20, 50, 100 samples, because each one costs a model call) its
    intervals are noticeably wider, so a system that genuinely meets the bar gets
    failed for want of samples nobody can afford. Wilson's coverage oscillates
    around the nominal level rather than sitting above it, which is the honest
    trade for a gate whose job is to be run on every pull request. It is also
    closed-form, so the number in a failure message can be recomputed on paper by
    whoever is disputing it -- Clopper-Pearson requires inverting a beta
    distribution, which makes the same message an appeal to authority.

    If your setting genuinely requires guaranteed coverage -- a regulated
    submission rather than a CI gate -- Clopper-Pearson is the better choice and
    this is the function to swap.
    """
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z / denominator * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    lower, upper = centre - half, centre + half

    # At the endpoints the two terms are analytically equal and the difference is
    # exactly 0 or 1, but they are evaluated in different orders, so the
    # cancellation leaves a float residue of either sign. Clamping alone only
    # catches the negative side, which let wilson_lower_bound(0, 11) return
    # 1.39e-17 -- a lower confidence bound sitting *above* its own point estimate.
    # Substituting p = 0 into the formulae above gives
    # centre = half = z**2 / (2*(n + z**2)), hence lower = 0 exactly; p = 1 is the
    # mirror image, giving upper = 1 exactly.
    if successes == 0:
        lower = 0.0
    if successes == n:
        upper = 1.0
    return _clamp(lower), _clamp(upper)


def wilson_lower_bound(
    successes: int,
    n: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """One-sided lower confidence bound on the underlying success probability.

    **One-sided**: ``z = NormalDist().inv_cdf(confidence)``, not
    ``inv_cdf(1 - (1 - confidence) / 2)``.
    This is the number the pass-rate gate compares against, and a gate only ever
    asks "is the true rate at least X?" -- there is no upper bar to defend, so
    the full error budget goes to the floor. Using the two-sided z here would
    quietly make the gate stricter than the stated confidence, at 97.5% one-sided.

    Args:
        successes: Runs that passed. ``0 <= successes <= n``.
        n: Total runs in the denominator. Must be >= 1.
        confidence: Strictly between 0.5 and 1. Default 0.95. **Not** the full
            unit interval: at or below 0.5 the one-sided z is zero or negative and
            the bound stops being a floor, so it is refused rather than returned.
            See :func:`_validate_gating_confidence`.

    Returns:
        The lower bound, clamped into ``[0.0, 1.0]``.

    Raises:
        ValueError: On ``n < 1``, ``successes`` outside ``[0, n]``, or a
            confidence outside the open interval ``(0.5, 1)``.
    """
    successes, n = _validate_counts(successes, n)
    level = _validate_gating_confidence(confidence)
    lower, _ = _wilson(successes, n, _z(level))
    return lower


def wilson_interval(
    successes: int,
    n: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Two-sided Wilson score interval, for reporting rather than gating.

    **Two-sided**: ``z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)``, so a 95%
    interval splits its 5% error budget across both tails. Its lower end is therefore
    *below* :func:`wilson_lower_bound` at the same nominal confidence, and the two
    are not interchangeable -- this one is what you print when a human asks "what
    do we actually know about the rate?", the other is what a gate compares to.

    Returns:
        ``(lower, upper)``, both clamped into ``[0.0, 1.0]``.
    """
    successes, n = _validate_counts(successes, n)
    level = _validate_unit(confidence, "confidence", exclusive=True)
    z = _z(1.0 - (1.0 - level) / 2.0)
    return _wilson(successes, n, z)


def _runs_needed(p: float, min_rate: float, confidence: float, cap: int = 10_000_000) -> int | None:
    """Smallest n from which an observed rate of ``p`` clears ``min_rate`` and stays clear.

    Only meaningful when ``p > min_rate``: the bound converges upward to ``p`` as
    n grows, so if the observed rate is at or below the bar no amount of sampling
    will clear it and the answer is "fix the system", not "run it more".

    **Not the smallest n that happens to clear**, which is a different and much
    worse number. The predicate is ``wilson_lower(round(p*n), n) >= min_rate``,
    and ``round`` makes it *oscillate* rather than switch on once: at ``p=0.95,
    min_rate=0.90, confidence=0.95`` it holds at n = 86-90, fails at 91-99, holds
    at 100-109, fails at 110-112, and holds from 113 on. 86 is the smallest n
    satisfying it and is useless as a budget -- it clears only because
    ``round(0.95 * 86) = 82`` rounds the rate *up* to 0.9535, and a reader told
    "86 runs" who runs 91 fails anyway. What is reported is 113: the point past
    which the answer no longer depends on where the rounding happens to land.
    That is the property the caller actually needs, and the only one that survives
    the reader adding a few runs for luck.

    The number is well defined and cheap to find because there is a genuinely
    monotone predicate underneath the oscillating one. ``round(p*n)`` is never
    below ``floor(p*n - 0.5)``, so the Wilson bound at *that* count is a floor for
    the real one; it rises with n (both its rate, ``>= p - 1/(2n)``, and its n
    increase, and the bound rises in each), so a binary search on it is valid
    where a binary search on the oscillating predicate is not -- the old one
    returned an arbitrary clearing n, up to 31% above the minimum on one grid case
    and below the stable point on others. The search finds the n from which
    clearing is guaranteed whatever the rounding does, and a short walk downward
    finds the last n that still failed. The walk is bounded by the width of the
    oscillating band, which is ``O(sqrt(n))``: 12 steps or fewer on every case of
    a 45-point grid, and 229 steps at ``n ~ 9e6``.

    Args:
        cap: Largest n considered. Searched directly rather than approached by
            doubling, so the documented cap is the real one -- doubling from 1
            topped out at 2**23 = 8,388,608 and returned None for answers that sat
            comfortably inside the cap.

    Returns:
        The n described above, or None when no n at or below ``cap`` qualifies.
    """
    if not p > min_rate:
        return None
    z = _z(confidence)

    def guaranteed(n: int) -> bool:
        """Does the bound clear at the least favourable rounding of ``p * n``?"""
        successes = max(0, math.floor(p * n - 0.5))
        return _wilson(min(successes, n), n, z)[0] >= min_rate

    def clears(n: int) -> bool:
        """Does the bound clear at the rounding that actually occurs?"""
        successes = round(p * n)
        return _wilson(min(successes, n), n, z)[0] >= min_rate

    if not guaranteed(cap):
        return None
    low, high = 1, cap
    while low < high:
        mid = (low + high) // 2
        if guaranteed(mid):
            high = mid
        else:
            low = mid + 1
    while low > 1 and clears(low - 1):
        low -= 1
    return low


#: Power the second recommendation aims at. 0.80 is the conventional floor, and
#: the point of reporting it is that the *first* recommendation does not reach it.
_POWER_TARGET = 0.80

#: Largest n the power search will consider. Past this the exact figure is noise
#: -- nobody budgets four thousand model calls off a rounding -- and the search is
#: a linear scan, so the cap is what keeps a failing assertion from stalling.
_POWER_SEARCH_CAP = 2_000


def _binomial_tail(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``, exactly, using only the stdlib.

    Summed with the recurrence ``pmf(i+1) = pmf(i) * (n-i)/(i+1) * p/(1-p)`` from a
    single log-gamma seed, over the 14-sigma window around the mean. Everything
    outside that window contributes less than 1e-40 -- far below the resolution of
    a probability printed to two figures -- and restricting to it makes the sum
    ``O(sqrt(n))`` instead of ``O(n)``, which is what lets the caller scan.

    A normal approximation would have been one line, but this function exists to
    say how often a *small* sample clears a gate, and the normal approximation to a
    binomial is least trustworthy exactly there.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p >= 1.0:
        return 1.0
    if p <= 0.0:
        return 0.0
    mean = n * p
    sd = math.sqrt(n * p * (1.0 - p))
    low = max(k, int(mean - 14.0 * sd) - 1)
    high = min(n, int(mean + 14.0 * sd) + 1)
    if k > high:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_factorial_n = math.lgamma(n + 1.0)
    term = math.exp(
        log_factorial_n
        - math.lgamma(low + 1.0)
        - math.lgamma(n - low + 1.0)
        + low * log_p
        + (n - low) * log_q
    )
    total = term
    ratio = p / (1.0 - p)
    for i in range(low, high):
        term *= (n - i) / (i + 1.0) * ratio
        total += term
    return min(1.0, total)


def _minimum_successes(min_rate: float, n: int, z: float) -> int:
    """Fewest passes out of ``n`` that clear the gate, or ``n + 1`` if none do.

    Takes ``z`` rather than a confidence because the power search calls this tens
    of thousands of times: routing each one through :func:`wilson_lower_bound`,
    with its argument validation and a fresh :func:`_z`, turned a failing
    assertion into a three-second pause.

    Found by bisection on :func:`_wilson` itself rather than by inverting the
    formula, so the power figure is computed against the very function the gate
    calls -- an independently derived critical count that drifted from the gate by
    one would misreport the power of a gate nobody had changed. Bisection is valid
    here (unlike in :func:`_runs_needed`) because the bound really is monotone in
    ``successes`` at fixed ``n``: no rounding sits between them.
    """
    if _wilson(n, n, z)[0] < min_rate:
        return n + 1
    low, high = 0, n
    while low < high:
        mid = (low + high) // 2
        if _wilson(mid, n, z)[0] >= min_rate:
            high = mid
        else:
            low = mid + 1
    return low


def _gate_power(p: float, min_rate: float, n: int, confidence: float) -> float:
    """How often a *fresh* sample of ``n`` runs clears the gate, if the rate is ``p``.

    Exact binomial power, not a normal approximation: the count of passes is
    ``Binomial(n, p)``, the gate clears iff that count reaches
    :func:`_minimum_successes`, and this is the probability of that.

    This is the number that turns :func:`_runs_needed` from a promise into an
    estimate. ``_runs_needed`` answers "at what n does an observed rate of p clear
    the bar?", which is a fact about one arithmetic identity; it says nothing about
    whether the *next* n runs will reproduce that rate, and they land above or
    below it at random. At ``p=0.95, min_rate=0.90``, ``_runs_needed`` returns 113
    and the power there is 0.66 -- a coin flip dressed as a budget.
    """
    return _binomial_tail(_minimum_successes(min_rate, n, _z(confidence)), n, p)


def _runs_for_power(
    p: float,
    min_rate: float,
    confidence: float,
    target: float = _POWER_TARGET,
    cap: int = _POWER_SEARCH_CAP,
) -> int | None:
    """Smallest n from which the gate clears at least ``target`` of the time.

    The honest companion to :func:`_runs_needed`: not "how many runs make this
    rate clear the bar" but "how many runs make it *likely* that a fresh sample
    clears the bar", which is the question a reader budgeting runs is actually
    asking. It comes out 1.5-2.2x larger.

    ``_gate_power`` oscillates in n for the same lattice reason ``_runs_needed``
    does, so the same discipline applies: what is returned is the point past which
    the power stays at or above the target, found by scanning up from n=1 and
    remembering the last n that fell short. The scan stops once it has seen
    ``4*sqrt(n)`` consecutive sizes hold -- the oscillating band is ``O(sqrt(n))``
    wide, and this window reproduces an unbounded brute-force scan on every case
    tested. A plain "first n that reaches the target" would report 164 where the
    answer is 188, with 165-187 falling short in between.

    Returns None when no n at or below ``cap`` qualifies, in which case the caller
    simply does not offer a powered budget -- an unreachable number is worse than
    no number.
    """
    if not p > min_rate:
        return None
    z = _z(confidence)
    last_failure = 0
    n = 1
    while n <= cap:
        if _binomial_tail(_minimum_successes(min_rate, n, z), n, p) < target:
            last_failure = n
        elif n - last_failure > 4.0 * math.sqrt(n):
            return last_failure + 1
        n += 1
    return None


# --------------------------------------------------------------------------- #
# input coercion
# --------------------------------------------------------------------------- #


def _looks_like_counts(data: Any) -> bool:
    """Whether ``data`` is the ``(successes, n)`` form rather than outcomes.

    The two accepted sequence forms genuinely overlap -- ``(1, 0)`` could be one
    pass and one fail, or one success out of zero runs -- so the rule is pinned
    here and stated in the docstring rather than guessed per call site: a
    two-element **tuple** of non-boolean integers is a count pair; anything else,
    including ``(True, False)``, is a sequence of outcomes.

    "Integer" includes ``numpy.integer``, for the same reason :func:`_as_count`
    accepts it: numpy integers arrive routinely from array code, and
    ``(np.int64(3), np.int64(5))`` used to fall through to the outcome branch and
    be refused with "result[0] must be a bool (or 0/1)" -- a misdiagnosis that
    sent the reader looking at their data instead of their dtype. ``np.bool_`` is
    excluded alongside ``bool``: it is not an ``np.integer``, so the same rule
    holds without a second check.
    """
    if not isinstance(data, tuple) or len(data) != 2:
        return False
    return all(
        not isinstance(item, bool) and isinstance(item, (int, np.integer)) for item in data
    )


def _coerce_pass_data(data: Any, argument: str = "result") -> tuple[int, int]:
    """Normalise the three accepted pass/fail inputs to ``(successes, n)``."""
    if isinstance(data, SampleResult):
        return data.successes, data.n
    # Duck-typed so a caller's own result object works without importing ours.
    if hasattr(data, "successes") and hasattr(data, "n") and not isinstance(data, (tuple, list)):
        return _as_count(data.successes, "successes"), _as_count(data.n, "n")
    if _looks_like_counts(data):
        return int(data[0]), int(data[1])
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        try:
            outcomes = list(data)
        except TypeError:
            raise ValueError(
                f"{argument} must be a SampleResult, a (successes, n) tuple, or a "
                f"sequence of bools; got {type(data).__name__}"
            ) from None
    else:
        outcomes = list(data)

    successes = 0
    for index, outcome in enumerate(outcomes):
        if isinstance(outcome, (bool, np.bool_)):
            successes += int(outcome)
            continue
        # 0/1 ints are accepted because numpy and CSV round-trips produce them;
        # anything else is rejected rather than run through truthiness, which
        # would score the non-empty string "fail" as a pass.
        if isinstance(outcome, (int, np.integer)) and int(outcome) in (0, 1):
            successes += int(outcome)
            continue
        raise ValueError(
            f"{argument}[{index}] must be a bool (or 0/1), got "
            f"{type(outcome).__name__} {outcome!r}"
        )
    return successes, len(outcomes)


def _coerce_scores(data: Any, argument: str) -> tuple[float, ...]:
    """Normalise a :class:`SampleResult` or a sequence of numbers to floats."""
    if isinstance(data, SampleResult):
        return data.scores()
    # Resolved once into a local rather than `hasattr(data, "scores") and
    # callable(data.scores)`, which looked the attribute up twice and so ran a
    # property with a side effect twice.
    #
    # `produced` is annotated because `callable()` narrows to a callable whose
    # return type is `object`, and `object` is not iterable -- pyright reports
    # exactly that in strict mode, inside a package whose py.typed marker
    # promises it will not. The duck-typed protocol here is genuinely dynamic
    # (anything with a `.scores()` returning numbers), so `Any` is the honest
    # annotation rather than a cast that claims to know more than we do.
    scores_attr = getattr(data, "scores", None)
    if callable(scores_attr):
        produced: Any = scores_attr()
        return tuple(float(value) for value in produced)
    if isinstance(data, (str, bytes)):
        raise ValueError(
            f"{argument} must be a sequence of numbers, got {type(data).__name__} "
            f"{_short_repr(data)}"
        )
    try:
        values = list(data)
    except TypeError:
        raise ValueError(
            f"{argument} must be a SampleResult or a sequence of numbers, got "
            f"{type(data).__name__}"
        ) from None

    scores: list[float] = []
    for index, value in enumerate(values):
        # bool is an int subclass; a sequence of bools is an outcome list that
        # wandered into a score gate, and averaging it would silently produce a
        # pass rate labelled "mean score".
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(
                f"{argument}[{index}] must be a number, got {type(value).__name__} {value!r}"
            )
        numeric = float(value)
        if math.isnan(numeric):
            raise ValueError(
                f"{argument}[{index}] is NaN; a missing score is not a score, drop it "
                f"deliberately rather than letting it poison the mean"
            )
        # Infinity is refused for the same reason as NaN, and more urgently. It
        # does not merely poison the mean, it turns the gate green: inf - inf is
        # nan, so the variance is nan, so `stddev > max_stddev` is False, so no
        # violation is recorded and assert_score_distribution *passes* -- on a
        # single unbounded score, with a bare numpy RuntimeWarning on stderr as the
        # only sign anything happened. A gate that returns green on garbage is
        # precisely what this library exists to prevent, and the same argument the
        # n<2 stddev check makes ("reporting it as 0.0 would pass the strictest
        # possible stddev gate on no evidence at all") applies here with the sign
        # reversed: infinite evidence is not evidence either.
        if math.isinf(numeric):
            raise ValueError(
                f"{argument}[{index}] is {numeric}; an infinite score is not a score. "
                f"Every statistic over it comes back inf or nan, and a nan fails every "
                f"comparison, so the gate would find no violation and pass. Clamp it to "
                f"the ends of your scale, drop it deliberately, or fix whatever produced "
                f"it -- but do not let it decide a build"
            )
        scores.append(numeric)
    return tuple(scores)


def _no_scores_detail(data: Any) -> str:
    """Why ``data`` yielded no scores, in terms of the object the caller passed.

    "current has no scores" reads as *you have no data*, and the caller who sees
    it is usually holding several hundred completions. What actually happened is
    that :meth:`~opik_rigor.sampling.SampleResult.scores` harvests
    ``getattr(run.value, "score", None)``, so a sample of plain strings -- the
    most common thing an adapter returns -- harvests nothing. The shape is wrong,
    not the quantity, and the message has to say which.

    Returns a sentence to append, or ``""`` when the input really is empty and the
    plain message is already the truth.
    """
    runs = getattr(data, "runs", None)
    if not isinstance(runs, (tuple, list)) or not runs:
        return ""

    errored = [run for run in runs if getattr(run, "error", None) is not None]
    if len(errored) == len(runs):
        first = getattr(errored[0], "error", None)
        return (
            f" It is a {type(data).__name__} of {len(runs)} runs and every one of them "
            f"raised, so there was nothing to harvest a score from; the first error is "
            f"{type(first).__name__}: {first}"
        )

    def unscored(run: Any) -> bool:
        score = getattr(getattr(run, "value", None), "score", None)
        return isinstance(score, bool) or not isinstance(score, (int, float))

    completed = [run for run in runs if getattr(run, "error", None) is None]
    offender = next((run for run in completed if unscored(run)), None)
    detail = (
        f" It is a {type(data).__name__} of {len(runs)} runs, {len(completed)} of which "
        f"completed, but none of them carried a numeric .score"
    )
    if offender is not None:
        # getattr rather than attribute access: this runs while building an error
        # message, and a duck-typed run object that raised here would replace the
        # caller's real problem with an AttributeError from inside a statistics call.
        value = getattr(offender, "value", None)
        detail += (
            f": run {getattr(offender, 'index', '?')} returned "
            f"{type(value).__name__} {_short_repr(value)}"
        )
    return (
        f"{detail}. This gate compares judge scores, so pass the verdicts -- or a "
        f"sequence of numbers -- rather than the raw completions."
    )


def _record(evidence: EvidenceLog | None, report: dict[str, Any]) -> None:
    """Append exactly one assertion record, on the way past -- pass or fail."""
    if evidence is not None:
        evidence.append(EVENT_ASSERTION, report)


def _labelled(label: str | None) -> str:
    return f" {label!r}" if label else ""


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def assert_pass_rate(
    result: PassData,
    min_rate: float,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    evidence: EvidenceLog | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Assert that the true pass rate is at least ``min_rate``, with confidence.

    The gate is ``wilson_lower_bound(successes, n, confidence) >= min_rate``.

    Args:
        result: A :class:`~opik_rigor.sampling.SampleResult`, a ``(successes, n)``
            tuple, or a sequence of per-run bools. A two-element tuple of plain
            ints is read as counts; every other sequence is read as outcomes.
        min_rate: The bar, in ``[0.0, 1.0]``.
        confidence: One-sided confidence for the bound. Strictly between 0.5 and
            1; default 0.95. At or below 0.5 the bound inverts and the gate would
            be looser than a raw comparison, so it is refused.
        evidence: If given, one ``assertion.evaluated`` record is appended --
            including when the gate fails, before the exception is raised.
        label: Free-text name for this gate, carried into the message and record.

    Returns:
        A report dict with ``passed=True``, the counts, the observed rate, the
        lower bound, the two-sided interval, the threshold, and ``n``.

    Raises:
        PassRateError: If the lower bound is below ``min_rate``.
        ValueError: On malformed input (see :func:`wilson_lower_bound`).
    """
    successes, n = _coerce_pass_data(result)
    successes, n = _validate_counts(successes, n)
    level = _validate_gating_confidence(confidence)
    threshold = _validate_unit(min_rate, "min_rate", exclusive=False)

    # The whole point of the library, in one line: the gate is the *lower bound*,
    # never `successes / n`. 18/20 and 900/1000 are both 90%; gating on the point
    # estimate treats them as the same evidence, which is how a suite ends up
    # passing on 20 runs and then "flaking" for a fortnight. The point estimate is
    # reported because a reader wants it, and compared against nothing.
    observed = successes / n
    lower = wilson_lower_bound(successes, n, level)
    interval = wilson_interval(successes, n, level)
    passed = lower >= threshold

    report: dict[str, Any] = {
        "gate": "pass_rate",
        "label": label,
        "passed": passed,
        "n": n,
        "successes": successes,
        "failures": n - successes,
        "pass_rate": observed,
        "lower_bound": lower,
        "interval_lower": interval[0],
        "interval_upper": interval[1],
        "min_rate": threshold,
        "confidence": level,
        "method": "wilson-one-sided",
    }

    if passed:
        _record(evidence, report)
        return report

    underpowered = observed >= threshold
    needed_power: float | None = None
    powered: int | None = None
    if underpowered:
        needed = _runs_needed(observed, threshold, level)
        diagnosis = (
            f"The observed rate {observed:.4f} clears min_rate {threshold:.4f} but the "
            f"lower bound does not: this is an underpowered sample, not a demonstrated "
            f"failure. {n} runs cannot distinguish a system at {observed:.1%} from one at "
            f"{lower:.1%}."
        )
        if needed is not None:
            # Two numbers, because one of them alone has been read as the other in
            # every project that has shipped it. `needed` is arithmetic on this one
            # rate: hold the rate at exactly `observed` and the bound clears from
            # there on. It is not a power calculation, and quoting it alone invites
            # the reader to budget `needed` runs and then be surprised a third of
            # the time -- so the power at `needed` is stated next to it, and a
            # genuinely powered n offered after it.
            needed_power = _gate_power(observed, threshold, needed, level)
            powered = _runs_for_power(observed, threshold, level)
            diagnosis += (
                f" Hold the rate at exactly {observed:.4f} and the bound clears from "
                f"{needed} runs on. That is arithmetic on this one rate, not a power "
                f"calculation: a fresh sample of {needed} runs from a system whose true "
                f"rate is {observed:.1%} lands above or below {observed:.1%} at random, "
                f"and clears this gate only {needed_power:.0%} of the time."
            )
            if powered is not None:
                diagnosis += (
                    f" Budget {powered} runs to clear it {_POWER_TARGET:.0%} of the time; "
                    f"that is the number to plan against."
                )
        elif observed == threshold:
            # p == min_rate exactly. The bound rises toward p as n grows but never
            # reaches it, so "sample more" is the wrong advice here and printing it
            # would send the reader into an unbounded loop of raising n.
            diagnosis += (
                f" The observed rate sits exactly on min_rate, and the lower bound "
                f"approaches the observed rate from below without ever reaching it: no "
                f"sample size clears this bar at exactly {observed:.4f}. The system needs "
                f"real headroom above min_rate, or min_rate has to come down."
            )
        else:
            diagnosis += (
                " The margin over min_rate is so thin that clearing the bar would take "
                "an impractical number of runs."
            )
    else:
        needed = None
        diagnosis = (
            f"The observed rate {observed:.4f} is itself below min_rate {threshold:.4f}: "
            f"the system missed the bar, and more runs will not fix it."
        )
    report["underpowered"] = underpowered
    report["runs_needed"] = needed
    report["power_at_runs_needed"] = needed_power
    report["target_power"] = _POWER_TARGET
    report["runs_for_target_power"] = powered

    message = (
        f"pass rate gate{_labelled(label)} failed: {successes}/{n} passed "
        f"(observed {observed:.4f}); one-sided {level:.0%} Wilson lower bound "
        f"{lower:.4f} < min_rate {threshold:.4f}. "
        f"Two-sided {level:.0%} interval [{interval[0]:.4f}, {interval[1]:.4f}]. "
        f"{diagnosis}"
    )
    _record(evidence, report)
    raise PassRateError(message, **report)


def assert_score_distribution(
    scores: ScoreData,
    *,
    min_mean: float | None = None,
    min_p10: float | None = None,
    max_stddev: float | None = None,
    evidence: EvidenceLog | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Assert the shape of a judge-score distribution, not just its average.

    A mean gate alone passes a system that is excellent four times in five and
    unusable the fifth time -- which is precisely the failure mode users notice.
    ``min_p10`` bounds the bad tail and ``max_stddev`` bounds the swing, and each
    of the three is checked **independently and only if not None**. All supplied
    gates are evaluated and every violation appears in one message: discovering
    them one CI run at a time is a waste of round trips.

    Definitions, pinned so that a reader can reproduce the numbers exactly:

    * **mean** is ``numpy.mean(scores)``.
    * **p10** is ``numpy.percentile(scores, 10)`` with numpy's default *linear*
      interpolation between order statistics -- not a nearest-rank percentile,
      which would give a different answer on small samples.
    * **stddev** is the **sample** standard deviation, ``numpy.std(scores,
      ddof=1)``. The population form (``ddof=0``) divides by n instead of n-1 and
      so understates exactly the spread this gate exists to bound -- most
      severely at the small n where an eval suite operates.

    Args:
        scores: A :class:`~opik_rigor.sampling.SampleResult` (its :meth:`scores` are
            used) or a sequence of numbers.
        min_mean: Lower bound on the mean, or None to skip.
        min_p10: Lower bound on the 10th percentile, or None to skip.
        max_stddev: Upper bound on the sample stddev, or None to skip.
        evidence: If given, one record is appended on pass and on fail.
        label: Free-text name for this gate.

    Returns:
        A report dict with ``passed=True``, all three statistics (``stddev`` is
        None for a single score), the thresholds, and ``n``.

    Raises:
        ScoreDistributionError: If any supplied gate is violated.
        ValueError: If all three thresholds are None (a gate that gates nothing
            is a bug in the test, not a pass), if there are no scores, or if
            ``max_stddev`` is set with fewer than 2 scores -- the spread of one
            observation is undefined, and reporting it as 0.0 would pass the
            strictest possible stddev gate on no evidence at all.
    """
    if min_mean is None and min_p10 is None and max_stddev is None:
        raise ValueError(
            "assert_score_distribution needs at least one of min_mean, min_p10, or "
            "max_stddev; a gate with no thresholds always passes, which is a bug in "
            "the test rather than a fact about the system"
        )
    values = _coerce_scores(scores, "scores")
    n = len(values)
    if n == 0:
        raise ValueError("no scores to gate on; a distribution over zero observations is not one")
    if max_stddev is not None and n < 2:
        raise ValueError(
            f"max_stddev needs at least 2 scores to have a meaning, got {n}"
        )
    for name, value in (("min_mean", min_mean), ("min_p10", min_p10), ("max_stddev", max_stddev)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(
                f"{name} must be a number or None, got {type(value).__name__} "
                f"{_short_repr(value)}"
            )
    if max_stddev is not None and max_stddev < 0:
        raise ValueError(f"max_stddev must be >= 0, got {max_stddev!r}")

    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    p10 = float(np.percentile(array, 10))
    stddev = float(np.std(array, ddof=1)) if n >= 2 else None

    checks: list[str] = []
    violations: list[str] = []
    if min_mean is not None:
        checks.append("min_mean")
        if mean < float(min_mean):
            violations.append(f"mean {mean:.4f} < min_mean {float(min_mean):.4f}")
    if min_p10 is not None:
        checks.append("min_p10")
        if p10 < float(min_p10):
            violations.append(f"p10 {p10:.4f} < min_p10 {float(min_p10):.4f}")
    if max_stddev is not None:
        checks.append("max_stddev")
        # stddev is not None here: n >= 2 was enforced above.
        if stddev is not None and stddev > float(max_stddev):
            violations.append(
                f"stddev(ddof=1) {stddev:.4f} > max_stddev {float(max_stddev):.4f}"
            )

    report: dict[str, Any] = {
        "gate": "score_distribution",
        "label": label,
        "passed": not violations,
        "n": n,
        "mean": mean,
        "p10": p10,
        "stddev": stddev,
        "min_mean": None if min_mean is None else float(min_mean),
        "min_p10": None if min_p10 is None else float(min_p10),
        "max_stddev": None if max_stddev is None else float(max_stddev),
        "checks": tuple(checks),
        "violations": tuple(violations),
        "min_score": float(np.min(array)),
        "max_score": float(np.max(array)),
    }

    if not violations:
        _record(evidence, report)
        return report

    observed = (
        f"n={n}, mean={mean:.4f}, p10={p10:.4f}, "
        f"stddev(ddof=1)={'n/a' if stddev is None else format(stddev, '.4f')}, "
        f"range=[{report['min_score']:.4f}, {report['max_score']:.4f}]"
    )
    detail = "; ".join(violations)
    message = (
        f"score distribution gate{_labelled(label)} failed "
        f"{len(violations)} of {len(checks)} checks over {n} scores: {detail}. "
        f"Observed distribution: {observed}. "
        f"These are properties of the sample, and a sample this size pins them down "
        f"only so far -- but the violated thresholds were missed on the evidence you have."
    )
    _record(evidence, report)
    raise ScoreDistributionError(message, **report)


def _mannwhitneyu() -> Any:
    """Import ``scipy.stats.mannwhitneyu`` on first use, not at module import.

    This is the *only* thing in the package that needs SciPy, and importing it at
    module scope made ``import opik_rigor`` pay for ``scipy.stats`` -- which drags
    in ``scipy.optimize``, ``scipy.spatial``, ``scipy.sparse`` and
    ``scipy.linalg`` -- on every suite, including the overwhelming majority that
    only ever call :func:`assert_pass_rate`. Measured warm and interleaved, that
    was 1018.6 ms of import against a ~40 ms interpreter floor; deferring it here
    takes it to 247.2 ms. The regression gate still pays the full cost on first
    call, which is the one caller that should.

    **This deferral is only half the fix, and half does not work.** Deferring
    ``mannwhitneyu`` while the Wilson bound still called ``scipy.stats.norm.ppf``
    was measured at 213.4 ms to import and then 1096.7 ms to run
    :func:`assert_pass_rate` -- no better than the 1070.6 ms it replaced, because
    the first gate call imported the SciPy the module had just avoided importing.
    The bound uses :class:`statistics.NormalDist` for exactly that reason; if it
    is ever moved back onto SciPy, this deferral silently stops being one.

    SciPy remains a hard, declared dependency, so the failure below is not an
    expected path -- it is reachable only if a user's environment has lost it, or
    if they installed the package's dependencies by hand. The message says so,
    because a deferred import turns what used to be an import-time traceback into
    one raised from the middle of a gate.
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError as exc:  # pragma: no cover -- needs a scipy-free environment
        raise ModuleNotFoundError(
            "assert_no_regression needs SciPy, which is not importable here. It is the "
            "only gate in opik_rigor that does: it runs the Mann-Whitney U test via "
            "scipy.stats.mannwhitneyu, which is deliberately not reimplemented in this "
            "package (scipy's carries the tie-corrected ranks and the exact null "
            "distribution that make the p-value trustworthy). SciPy is a hard dependency "
            "of opik-rigor, so this normally cannot happen -- install it with "
            "`pip install scipy` or reinstall opik-rigor. assert_pass_rate and "
            "assert_score_distribution do not need SciPy and keep working without it."
        ) from exc
    return mannwhitneyu


def assert_no_regression(
    current: ScoreData,
    baseline: ScoreData,
    *,
    alpha: float = DEFAULT_ALPHA,
    evidence: EvidenceLog | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Assert that ``current`` is not significantly worse than ``baseline``.

    Uses ``scipy.stats.mannwhitneyu(current, baseline, alternative="less")`` and
    raises iff ``p < alpha``. This is the only gate in the package that touches
    SciPy, and it imports it on first call rather than at module import -- so a
    suite that never calls this function never pays for ``scipy.stats``.

    **Direction convention.** ``alternative="less"`` asks exactly one question:
    is ``current`` stochastically *smaller* than ``baseline`` -- i.e. is a
    randomly drawn current score likely to fall below a randomly drawn baseline
    score? A small p-value is evidence that today's scores are worse than the
    recorded ones. The test is deliberately one-sided: an improvement is not a
    regression, and a two-sided test would fail the build for getting better.
    Passing the arguments in the wrong order inverts the meaning silently, so the
    order is ``(current, baseline)`` everywhere, matching the sentence "current
    regressed against baseline".

    **Why Mann-Whitney rather than a t-test.** Judge scores are ordinal (the gap
    between a 4 and a 5 is not the gap between a 1 and a 2), bounded (1-5 with no
    tail beyond), routinely non-normal and often multi-modal -- a judge on a 1-5
    scale piles most of its mass on 4 and 5 and leaves a thin spike at 1. A
    t-test on that data is testing an assumption the data does not meet, and its
    p-value is a number about a normal distribution nobody sampled. Mann-Whitney
    assumes only that the values can be ranked, which is the one thing an ordinal
    scale genuinely provides.

    Args:
        current: Scores from the run under test -- a
            :class:`~opik_rigor.sampling.SampleResult` or a sequence of numbers.
        baseline: Scores from the recorded baseline, same forms accepted.
        alpha: Significance level. Strictly between 0 and 1. Default 0.05.
        evidence: If given, one record is appended on pass and on fail.
        label: Free-text name for this gate.

    Returns:
        A report dict with ``passed=True``, the U statistic, the p-value, both
        sample sizes, both medians and means, ``alpha``, and ``n``
        (``len(current)``, the sample actually under test).

    Raises:
        RegressionError: If ``p < alpha``.
        ValueError: If either input is empty -- there is nothing to compare, and
            silently passing would turn a lost baseline into a green build.
    """
    current_scores = _coerce_scores(current, "current")
    baseline_scores = _coerce_scores(baseline, "baseline")
    level = _validate_unit(alpha, "alpha", exclusive=True)
    if not current_scores:
        raise ValueError(
            "current has no scores; there is nothing to compare against baseline."
            + _no_scores_detail(current)
        )
    if not baseline_scores:
        raise ValueError(
            "baseline has no scores; a missing baseline is not a passing comparison."
            + _no_scores_detail(baseline)
        )

    result = _mannwhitneyu()(current_scores, baseline_scores, alternative="less")
    u_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    # Fully tied samples (every value identical in both) carry no rank information;
    # current scipy reports p = 1.0 for them, but a zero-variance normal
    # approximation is one implementation detail away from NaN. The comparison is
    # written so that a NaN passes -- two indistinguishable samples are not evidence
    # of a drop -- and the fact is recorded rather than left to a silent `<`.
    degenerate = math.isnan(p_value)
    passed = not (p_value < level)

    current_array = np.asarray(current_scores, dtype=float)
    baseline_array = np.asarray(baseline_scores, dtype=float)
    report: dict[str, Any] = {
        "gate": "no_regression",
        "label": label,
        "passed": passed,
        "n": len(current_scores),
        "n_current": len(current_scores),
        "n_baseline": len(baseline_scores),
        "u_statistic": u_statistic,
        "p_value": p_value,
        "alpha": level,
        "mean_current": float(np.mean(current_array)),
        "mean_baseline": float(np.mean(baseline_array)),
        "median_current": float(np.median(current_array)),
        "median_baseline": float(np.median(baseline_array)),
        "test": "mann-whitney-u",
        "alternative": "less",
        "degenerate": degenerate,
    }

    if passed:
        _record(evidence, report)
        return report

    delta = report["median_current"] - report["median_baseline"]
    message = (
        f"regression gate{_labelled(label)} failed: current scores are significantly "
        f"lower than baseline. Mann-Whitney U = {u_statistic:.1f}, "
        f"p = {p_value:.6g} < alpha {level:g} (one-sided, alternative='less': current "
        f"stochastically smaller than baseline). "
        f"current n={len(current_scores)}, median={report['median_current']:.4f}, "
        f"mean={report['mean_current']:.4f}; "
        f"baseline n={len(baseline_scores)}, median={report['median_baseline']:.4f}, "
        f"mean={report['mean_baseline']:.4f}; median delta {delta:+.4f}. "
        f"A shift at least this large would arise by chance in {p_value:.2%} of runs if "
        f"the two samples came from the same distribution, so this is a real drop rather "
        f"than an unlucky sample -- note that it says nothing about the drop being large."
    )
    _record(evidence, report)
    raise RegressionError(message, **report)
