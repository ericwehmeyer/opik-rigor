"""Run a stochastic function n times and keep everything that happened.

A single call to an LLM tells you almost nothing: the same prompt answers
differently on the next call, so one sample of a stochastic system is an anecdote.
Everything downstream of this module -- every statistical gate -- needs n
observations plus an honest account of the ones that did not produce an
observation at all.

The distinction this module exists to preserve is between a **failure** (the
system ran and its output did not meet the bar) and an **exception** (the system
did not run). They are different facts about different things, and collapsing them
is how a provider outage comes to look like a quality regression. They are stored
separately here; ``errors_as_failures`` decides only how they are *counted*, and
the underlying separation survives either way.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .evidence import EVENT_SAMPLE_COMPLETED, EvidenceLog


class SampleTimeout(Exception):
    """Raised inside a run that exceeded the per-run timeout.

    Stored on the :class:`Run` like any other exception rather than raised to the
    caller: a timeout is one observation going missing, not the sample failing.
    """


@dataclass(frozen=True)
class Run:
    """One execution of the sampled function."""

    index: int
    value: Any = None
    outcome: bool | None = None
    error: BaseException | None = None
    duration: float = 0.0

    @property
    def raised(self) -> bool:
        return self.error is not None


def _short_repr(value: Any, limit: int = 60) -> str:
    """``repr(value)`` clipped, so a 4kB completion cannot become the message."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a broken __repr__ must not replace the real error
        return f"<unreprable {type(value).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def default_outcome(value: Any) -> bool:
    """Decide whether one returned value counts as a pass.

    Understands the two things a caller actually returns -- a
    :class:`~opik_rigor.judge.Verdict` (use ``.passed``) or a plain ``bool`` -- and
    refuses anything else. Truthiness is not used on purpose: a non-empty string
    of judge prose is truthy, and silently scoring it as a pass would manufacture
    a measurement out of an unparsed response.

    The refusal is right and the message has to carry its own consequences,
    because of where it lands. :func:`sample` stores it on ``Run.error``, the same
    field a provider outage lands in, and :attr:`SampleResult.completed` filters
    those runs out -- so an adapter that answered every prompt correctly reports
    ``pass_rate=0.0`` beside ``failures=0`` and empty ``.values``. The message
    therefore names the value, says where the runs went, and gives the one-line
    fix rather than only stating the rule.
    """
    if isinstance(value, bool):
        return value
    passed = getattr(value, "passed", None)
    if isinstance(passed, bool):
        return passed
    raise TypeError(
        f"cannot decide pass/fail from {type(value).__name__} {_short_repr(value)}; "
        f"sample() records this run as errored, which drops it from .values, "
        f".outcomes, .successes and .completed -- so a whole sample of these reads "
        f"as pass_rate=0.0 beside failures=0, which looks like an outage rather "
        f"than an unanswered question. Return a bool or an object with a boolean "
        f".passed attribute, or pass an explicit outcome=... callable to sample(). "
        f"If you have no pass/fail question yet and only want the values back, say "
        f"so with outcome=lambda value: True."
    )


@dataclass(frozen=True)
class SampleResult:
    """Everything that happened across n runs.

    Carries the per-run durations and wall clock even though no assertion needs
    them yet -- they are free at collection time and impossible to recover later,
    and they are what a cost or latency gate would be built from.
    """

    runs: tuple[Run, ...]
    wall_clock: float
    errors_as_failures: bool = True
    concurrency: int = 1

    @property
    def n(self) -> int:
        """Runs in the denominator.

        With ``errors_as_failures=False`` the exceptions leave the denominator
        entirely: you are then estimating the pass rate *conditional on the system
        responding*, which is a different quantity and should be reported as one.
        """
        return len(self.runs) if self.errors_as_failures else len(self.completed)

    @property
    def completed(self) -> tuple[Run, ...]:
        return tuple(run for run in self.runs if not run.raised)

    @property
    def errored_runs(self) -> tuple[Run, ...]:
        """The runs that raised -- :class:`Run` objects, not exceptions.

        The exception itself is ``run.error``, so the line a caller usually wants
        is ``[str(run.error) for run in result.errored_runs]``.
        """
        return tuple(run for run in self.runs if run.raised)

    @property
    def exceptions(self) -> tuple[Run, ...]:
        """Deprecated alias of :attr:`errored_runs`, kept working forever.

        The name is wrong and always was: it returns :class:`Run` objects, so
        ``[str(e) for e in result.exceptions]`` yields run reprs rather than error
        messages, silently. Prefer :attr:`errored_runs`, which says what it hands
        back.

        No :class:`DeprecationWarning` is emitted. This attribute is read inside
        loops and inside every consumer's assertions, and a warning there buys a
        wall of test output rather than a fix; the two names are the same tuple,
        so nothing is at risk while a caller migrates.
        """
        return self.errored_runs

    @property
    def successes(self) -> int:
        return sum(1 for run in self.completed if run.outcome)

    @property
    def failures(self) -> int:
        """Runs that produced an output which did not pass.

        Excludes exceptions regardless of ``errors_as_failures`` -- that flag
        changes the denominator, not what the word "failure" means.
        """
        return sum(1 for run in self.completed if run.outcome is False)

    @property
    def pass_rate(self) -> float:
        """Observed pass rate. A point estimate -- never gate on it directly."""
        return self.successes / self.n if self.n else 0.0

    @property
    def outcomes(self) -> tuple[bool, ...]:
        return tuple(bool(run.outcome) for run in self.completed)

    @property
    def values(self) -> tuple[Any, ...]:
        return tuple(run.value for run in self.completed)

    @property
    def durations(self) -> tuple[float, ...]:
        return tuple(run.duration for run in self.runs)

    def scores(self) -> tuple[float, ...]:
        """Numeric scores from returned verdicts, for the distribution gates.

        Verdicts carrying ``score=None`` are omitted rather than zero-filled: a
        judge that declined to score did not score zero, and averaging in a
        fabricated zero drags the mean toward a number nobody measured.
        """
        found: list[float] = []
        for run in self.completed:
            score = getattr(run.value, "score", None)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                found.append(float(score))
        return tuple(found)

    def summary(self) -> dict[str, Any]:
        """The dict written to the evidence log, also useful in failure messages."""
        return {
            "n": self.n,
            "runs": len(self.runs),
            "successes": self.successes,
            "failures": self.failures,
            "exceptions": len(self.exceptions),
            "pass_rate": self.pass_rate,
            "errors_as_failures": self.errors_as_failures,
            "concurrency": self.concurrency,
            "wall_clock": self.wall_clock,
        }


def sample(
    fn: Callable[[], Any],
    n: int,
    *,
    concurrency: int = 1,
    timeout: float | None = None,
    errors_as_failures: bool = True,
    outcome: Callable[[Any], bool] | None = None,
    evidence: EvidenceLog | None = None,
    label: str | None = None,
) -> SampleResult:
    """Call ``fn`` ``n`` times and return everything that happened.

    Args:
        fn: Zero-argument callable. Called exactly ``n`` times.
        n: Number of runs. Must be >= 1.
        concurrency: Thread-pool width. ``1`` runs serially in this thread.
        timeout: Per-run wall-clock budget in seconds, measured from the moment
            that run begins executing -- not from when it was queued. An
            over-running run is recorded as a :class:`SampleTimeout` exception.
            Note that a Python thread cannot be killed, so the budget is detected
            rather than enforced: an over-running call still runs to completion
            and is then recorded as having missed its budget. ``sample`` therefore
            does not bound its own wall clock, and a genuinely hung ``fn`` will
            hang the caller.
        errors_as_failures: Whether exceptions stay in the denominator.
        outcome: Maps a returned value to pass/fail. Defaults to
            :func:`default_outcome`.
        evidence: If given, one ``sample.completed`` record is appended.
        label: Free-text name for the sample, recorded with the evidence.

    Never raises because ``fn`` raised -- an exception from the system under test
    is data about that system, and turning it into a traceback here would lose the
    other n-1 observations.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError(f"concurrency must be an integer >= 1, got {concurrency!r}")
    if timeout is not None and timeout <= 0:
        raise ValueError(f"timeout must be > 0 seconds or None, got {timeout!r}")

    decide = outcome or default_outcome
    started = time.perf_counter()

    if concurrency == 1:
        runs = [_run_once(fn, index, decide, timeout) for index in range(n)]
    else:
        runs = _run_pool(fn, n, decide, concurrency, timeout)

    result = SampleResult(
        runs=tuple(sorted(runs, key=lambda run: run.index)),
        wall_clock=time.perf_counter() - started,
        errors_as_failures=errors_as_failures,
        concurrency=concurrency,
    )
    if evidence is not None:
        evidence.append(EVENT_SAMPLE_COMPLETED, {"label": label, **result.summary()})
    return result


def _run_once(
    fn: Callable[[], Any],
    index: int,
    decide: Callable[[Any], bool],
    timeout: float | None,
) -> Run:
    """Serial path. A timeout is detected after the fact, not enforced."""
    started = time.perf_counter()
    try:
        value = fn()
    except BaseException as exc:  # noqa: BLE001 - the system under test may raise anything
        return Run(index=index, error=exc, duration=time.perf_counter() - started)
    elapsed = time.perf_counter() - started
    if timeout is not None and elapsed > timeout:
        return Run(
            index=index,
            value=value,
            error=SampleTimeout(f"run {index} took {elapsed:.3f}s, over the {timeout:.3f}s budget"),
            duration=elapsed,
        )
    return _decide(index, value, decide, elapsed)


def _run_pool(
    fn: Callable[[], Any],
    n: int,
    decide: Callable[[Any], bool],
    concurrency: int,
    timeout: float | None,
) -> list[Run]:
    """Concurrent path. Runs the *same* ``_run_once`` the serial path runs.

    The timeout is enforced inside the worker rather than by the collector, for
    two reasons that both showed up as bugs in the first version of this module.

    First, ``future.result(timeout=...)`` raises ``concurrent.futures.TimeoutError``,
    which since Python 3.11 *is* ``builtins.TimeoutError`` -- the same class a
    provider's socket timeout raises. Catching it here made a provider outage
    indistinguishable from our own budget expiring, and silently replaced the
    original exception with a fabricated one.

    Second, a collector that waits on futures in submission order accrues budget:
    it only starts the k-th run's clock when it reaches the k-th future, so a run
    that queued behind others got several budgets' worth of wall clock. Timing the
    run from inside the worker makes the budget genuinely per-run.

    Nothing is lost by detecting the overrun after the fact rather than
    abandoning the wait early: ``ThreadPoolExecutor.__exit__`` joins every worker
    regardless, so the old early return never actually shortened anything.
    """
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run_once, fn, index, decide, timeout) for index in range(n)]
        return [future.result() for future in futures]


def _decide(index: int, value: Any, decide: Callable[[Any], bool], elapsed: float) -> Run:
    """Classify a returned value, recording a bad classifier as a failed run."""
    try:
        passed = decide(value)
    except BaseException as exc:  # noqa: BLE001 - a bad outcome() must not lose the run
        return Run(index=index, value=value, error=exc, duration=elapsed)
    return Run(index=index, value=value, outcome=bool(passed), duration=elapsed)


def sample_of(values: Sequence[Any], **kwargs: Any) -> SampleResult:
    """Wrap already-collected values in a :class:`SampleResult`.

    For the common case of comparing against results you gathered elsewhere --
    notably feeding a stored baseline into the regression gate without pretending
    to re-run it.
    """
    iterator = iter(values)
    return sample(lambda: next(iterator), len(values), **kwargs)
