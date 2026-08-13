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
Anything given an :class:`~rigor.evidence.EvidenceLog` writes exactly one
``assertion.evaluated`` record whether it passed or failed -- a gate that only
records its passes is a highlight reel, not an audit trail.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu, norm

from .errors import StatisticalAssertionError
from .evidence import EVENT_ASSERTION, EvidenceLog
from .sampling import SampleResult

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
    inside ``(0, 1)`` -- ``norm.ppf(1.0)`` is infinite and a gate at alpha 0 can
    never fire) from a threshold like ``min_rate``, where 0.0 and 1.0 are both
    meaningful bars to set.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__} {value!r}")
    numeric = float(value)
    ok = 0.0 < numeric < 1.0 if exclusive else 0.0 <= numeric <= 1.0
    if not ok:  # also catches NaN, which fails every comparison
        bounds = "strictly between 0 and 1" if exclusive else "between 0 and 1 inclusive"
        raise ValueError(f"{name} must be {bounds}, got {value!r}")
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
    of perfection from twenty runs. Wilson gives roughly ``[0.86, 1.0]``, which is
    what twenty runs actually buy you.

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

    **One-sided**: ``z = norm.ppf(confidence)``, not ``norm.ppf(1 - (1-c)/2)``.
    This is the number the pass-rate gate compares against, and a gate only ever
    asks "is the true rate at least X?" -- there is no upper bar to defend, so
    the full error budget goes to the floor. Using the two-sided z here would
    quietly make the gate stricter than the stated confidence, at 97.5% one-sided.

    Args:
        successes: Runs that passed. ``0 <= successes <= n``.
        n: Total runs in the denominator. Must be >= 1.
        confidence: Strictly between 0 and 1. Default 0.95.

    Returns:
        The lower bound, clamped into ``[0.0, 1.0]``.

    Raises:
        ValueError: On ``n < 1``, ``successes`` outside ``[0, n]``, or a
            confidence outside the open interval ``(0, 1)``.
    """
    successes, n = _validate_counts(successes, n)
    level = _validate_unit(confidence, "confidence", exclusive=True)
    lower, _ = _wilson(successes, n, float(norm.ppf(level)))
    return lower


def wilson_interval(
    successes: int,
    n: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Two-sided Wilson score interval, for reporting rather than gating.

    **Two-sided**: ``z = norm.ppf(1 - (1 - confidence) / 2)``, so a 95% interval
    splits its 5% error budget across both tails. Its lower end is therefore
    *below* :func:`wilson_lower_bound` at the same nominal confidence, and the two
    are not interchangeable -- this one is what you print when a human asks "what
    do we actually know about the rate?", the other is what a gate compares to.

    Returns:
        ``(lower, upper)``, both clamped into ``[0.0, 1.0]``.
    """
    successes, n = _validate_counts(successes, n)
    level = _validate_unit(confidence, "confidence", exclusive=True)
    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    return _wilson(successes, n, z)


def _runs_needed(p: float, min_rate: float, confidence: float, cap: int = 10_000_000) -> int | None:
    """Smallest n at which an observed rate of ``p`` would clear ``min_rate``.

    Only meaningful when ``p > min_rate``: the bound converges upward to ``p`` as
    n grows, so if the observed rate is at or below the bar no amount of sampling
    will clear it and the answer is "fix the system", not "run it more". Binary
    search rather than a scan because the bound is monotone in n for fixed p.

    Returned in the failure message purely as a courtesy -- it answers the first
    question a reader of an underpowered failure asks.
    """
    if not p > min_rate:
        return None
    z = float(norm.ppf(confidence))
    low, high = 1, 1
    while high <= cap:
        successes = round(p * high)
        if _wilson(min(successes, high), high, z)[0] >= min_rate:
            break
        low = high + 1
        high *= 2
    else:
        return None
    while low < high:
        mid = (low + high) // 2
        successes = round(p * mid)
        if _wilson(min(successes, mid), mid, z)[0] >= min_rate:
            high = mid
        else:
            low = mid + 1
    return low


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
    """
    if not isinstance(data, tuple) or len(data) != 2:
        return False
    return all(not isinstance(item, bool) and isinstance(item, int) for item in data)


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
    if hasattr(data, "scores") and callable(data.scores):
        return tuple(float(value) for value in data.scores())
    if isinstance(data, (str, bytes)):
        raise ValueError(f"{argument} must be a sequence of numbers, got {type(data).__name__}")
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
        scores.append(numeric)
    return tuple(scores)


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
        result: A :class:`~rigor.sampling.SampleResult`, a ``(successes, n)``
            tuple, or a sequence of per-run bools. A two-element tuple of plain
            ints is read as counts; every other sequence is read as outcomes.
        min_rate: The bar, in ``[0.0, 1.0]``.
        confidence: One-sided confidence for the bound. Default 0.95.
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
    level = _validate_unit(confidence, "confidence", exclusive=True)
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
    if underpowered:
        needed = _runs_needed(observed, threshold, level)
        diagnosis = (
            f"The observed rate {observed:.4f} clears min_rate {threshold:.4f} but the "
            f"lower bound does not: this is an underpowered sample, not a demonstrated "
            f"failure. {n} runs cannot distinguish a system at {observed:.1%} from one at "
            f"{lower:.1%}."
        )
        if needed is not None:
            diagnosis += f" At this observed rate roughly {needed} runs would clear the bar."
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
        scores: A :class:`~rigor.sampling.SampleResult` (its :meth:`scores` are
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
            raise ValueError(f"{name} must be a number or None, got {type(value).__name__}")
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
    raises iff ``p < alpha``.

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
            :class:`~rigor.sampling.SampleResult` or a sequence of numbers.
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
        raise ValueError("current has no scores; there is nothing to compare against baseline")
    if not baseline_scores:
        raise ValueError("baseline has no scores; a missing baseline is not a passing comparison")

    result = mannwhitneyu(current_scores, baseline_scores, alternative="less")
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
