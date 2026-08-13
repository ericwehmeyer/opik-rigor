"""Tests for the sampler.

The sampler is where one anecdote becomes n observations, so these tests are the
argument that the counts it hands to the statistical gates are trustworthy: no
run lost or double-counted under a thread pool, no exception silently promoted to
a failure (or demoted out of the denominator by accident), and no value scored as
a pass because it happened to be truthy.

Determinism is a constraint here rather than a nicety -- a flaky test in a library
about flaky tests proves nothing. Every stochastic script below is driven by an
explicitly seeded :class:`random.Random`, and every timing assertion is one-sided
and generous: it can only fail if concurrency is not happening at all, never
because CI was busy.
"""

from __future__ import annotations

import random
import threading
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import pytest

from rigor.adapters.fake import FakeAdapter
from rigor.evidence import EVENT_SAMPLE_COMPLETED, EvidenceLog
from rigor.judge import Verdict
from rigor.sampling import (
    Run,
    SampleResult,
    SampleTimeout,
    default_outcome,
    sample,
    sample_of,
)

#: One seed for every stochastic script in this file. Held on a private
#: ``random.Random`` rather than the global module state, so no test can shift
#: what another test measures.
SEED = 20260812

#: Per-call sleep for the timing tests. Small enough that the whole suite stays
#: fast, large enough to dwarf thread start-up.
LATENCY = 0.05

#: How long a deliberately blocked call takes to return. Used only by the pool
#: timeout test, which needs a margin no scheduling hiccup can close: a
#: ``future.result(timeout=...)`` wakeup is granular to roughly 16ms on Windows,
#: so a 50ms call is not comfortably slower than a 5ms budget. Nothing waits on
#: this except the pool's own shutdown join, which is why it stays short.
BLOCKED = 0.3


class Boom(Exception):
    """An ordinary failure raised by the system under test."""


class Script:
    """Hands out one scripted item per call; raises items that are exceptions.

    Thread-safe, because the concurrent tests drive it from a pool. The mapping
    from run index to script item is then non-deterministic, so concurrent tests
    assert aggregate counts rather than per-index outcomes.
    """

    def __init__(self, items: Sequence[Any]) -> None:
        self._items = list(items)
        self._lock = threading.Lock()
        self._cursor = 0
        self.calls = 0

    def __call__(self) -> Any:
        with self._lock:
            item = self._items[self._cursor]
            self._cursor += 1
            self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


def verdict(passed: bool, score: float | None = None) -> Verdict:
    return Verdict(passed=passed, score=score, raw="")


def always_pass(value: Any) -> bool:
    """Outcome for tests about plumbing rather than about pass/fail."""
    return True


def blocked() -> str:
    """Return only after :data:`BLOCKED` seconds -- nothing ever sets the event.

    It must return eventually: a timed-out worker is still joined when the pool
    shuts down, so a call that never finished would hang ``sample`` forever.
    """
    threading.Event().wait(BLOCKED)
    return "ok"


# --------------------------------------------------------------------------- #
# counts
# --------------------------------------------------------------------------- #


def test_serial_sample_produces_n_runs_indexed_zero_to_n_minus_one() -> None:
    script = Script([True] * 5)

    result = sample(script, 5)

    assert isinstance(result, SampleResult)
    assert len(result.runs) == 5
    assert all(isinstance(run, Run) for run in result.runs)
    assert [run.index for run in result.runs] == [0, 1, 2, 3, 4]
    assert script.calls == 5
    assert result.concurrency == 1


def test_concurrent_sample_produces_n_runs_indexed_zero_to_n_minus_one() -> None:
    rng = random.Random(SEED)
    script = Script([rng.random() < 0.7 for _ in range(24)])

    result = sample(script, 24, concurrency=4)

    assert len(result.runs) == 24
    assert [run.index for run in result.runs] == list(range(24))
    assert script.calls == 24
    assert result.concurrency == 4


def test_calling_n_times_is_exact_and_does_not_depend_on_the_pool_width() -> None:
    serial = Script([True] * 12)
    pooled = Script([True] * 12)

    sample(serial, 12)
    sample(pooled, 12, concurrency=6)

    assert serial.calls == pooled.calls == 12


# --------------------------------------------------------------------------- #
# the success / failure / exception split
# --------------------------------------------------------------------------- #


def test_successes_failures_exceptions_and_pass_rate_are_counted_independently() -> None:
    items = [True] * 5 + [False] * 3 + [Boom("a"), Boom("b")]

    result = sample(Script(items), 10)

    assert result.successes == 5
    assert result.failures == 3
    assert len(result.exceptions) == 2
    assert result.n == 10
    assert result.pass_rate == 0.5
    assert [run.index for run in result.exceptions] == [8, 9]
    assert result.outcomes == (True,) * 5 + (False,) * 3
    assert len(result.completed) == 8


def test_failures_exclude_exceptions_under_both_error_accounting_settings() -> None:
    # errors_as_failures moves the denominator; it must never change what the
    # word "failure" means. A provider outage is not a quality regression, and
    # the two counts stay separate no matter how they are totalled.
    items = [True, False] + [Boom(f"down {index}") for index in range(3)]

    counted = sample(Script(items), 5, errors_as_failures=True)
    conditional = sample(Script(items), 5, errors_as_failures=False)

    for result in (counted, conditional):
        assert result.failures == 1
        assert result.successes == 1
        assert len(result.exceptions) == 3
        assert len(result.runs) == 5

    assert counted.n == 5
    assert conditional.n == 2


def test_failures_exclude_exceptions_on_the_concurrent_path_too() -> None:
    items = [True] * 4 + [False] * 2 + [Boom(f"down {index}") for index in range(3)]

    counted = sample(Script(items), 9, concurrency=4, errors_as_failures=True)
    conditional = sample(Script(items), 9, concurrency=4, errors_as_failures=False)

    for result in (counted, conditional):
        assert result.successes == 4
        assert result.failures == 2
        assert len(result.exceptions) == 3

    assert counted.n == 9
    assert conditional.n == 6


def test_errors_as_failures_false_removes_exceptions_from_the_denominator() -> None:
    items = [True] * 6 + [False] * 2 + [Boom(f"down {index}") for index in range(2)]

    counted = sample(Script(items), 10, errors_as_failures=True)
    conditional = sample(Script(items), 10, errors_as_failures=False)

    assert counted.n == 10
    assert counted.pass_rate == 0.6
    assert conditional.n == 8
    assert conditional.pass_rate == 0.75
    # Same ten observations either way -- only the quantity being estimated moved.
    assert len(counted.runs) == len(conditional.runs) == 10
    assert counted.successes == conditional.successes == 6


def test_pass_rate_is_zero_when_every_run_raised_and_errors_leave_the_denominator() -> None:
    items = [Boom(f"down {index}") for index in range(4)]

    result = sample(Script(items), 4, errors_as_failures=False)

    assert result.n == 0
    assert result.pass_rate == 0.0  # guarded division, not a measured rate
    assert result.failures == 0


# --------------------------------------------------------------------------- #
# exceptions never propagate
# --------------------------------------------------------------------------- #


def test_sample_returns_normally_when_fn_raises_an_ordinary_exception() -> None:
    result = sample(Script([True, Boom("provider 500"), True]), 3)

    assert len(result.runs) == 3
    assert result.successes == 2
    assert isinstance(result.exceptions[0].error, Boom)
    assert result.exceptions[0].raised is True


def test_base_exception_from_fn_is_captured_and_the_other_runs_survive() -> None:
    # sample() catches BaseException on purpose: losing n-1 observations to a
    # traceback from the system under test would defeat the point of sampling.
    items = [True, KeyboardInterrupt("interrupted"), True, False]

    result = sample(Script(items), 4)

    assert len(result.runs) == 4
    assert result.successes == 2
    assert result.failures == 1
    (interrupted,) = result.exceptions
    assert isinstance(interrupted.error, KeyboardInterrupt)
    assert interrupted.index == 1


def test_base_exception_is_captured_on_the_concurrent_path_too() -> None:
    items = [True, KeyboardInterrupt("interrupted"), True, True]

    result = sample(Script(items), 4, concurrency=4)

    assert len(result.runs) == 4
    assert result.successes == 3
    (interrupted,) = result.exceptions
    assert isinstance(interrupted.error, KeyboardInterrupt)


def test_every_run_raising_still_returns_a_result_rather_than_raising() -> None:
    items = [Boom(f"down {index}") for index in range(6)]

    result = sample(Script(items), 6, concurrency=3)

    assert len(result.exceptions) == 6
    assert result.completed == ()
    assert result.successes == 0
    assert result.pass_rate == 0.0


# --------------------------------------------------------------------------- #
# timeouts
# --------------------------------------------------------------------------- #


def test_slow_run_is_recorded_as_a_sample_timeout_exception_on_the_serial_path() -> None:
    # Serial detection is post-hoc: the call is allowed to finish and is then
    # judged over budget, so this can only fail if 50ms of sleep took under 5ms.
    adapter = FakeAdapter(responses=["ok"], cycle=True, latency=LATENCY)

    result = sample(lambda: adapter.complete("p"), 2, timeout=0.005, outcome=always_pass)

    assert len(result.exceptions) == 2
    assert all(isinstance(run.error, SampleTimeout) for run in result.exceptions)
    # A timeout is a missing observation, not a failing one.
    assert result.failures == 0
    assert result.successes == 0
    assert result.n == 2


def test_slow_run_is_recorded_as_a_sample_timeout_exception_on_the_concurrent_path() -> None:
    # Both paths now run the same _run_once, so the budget is measured from inside
    # the worker on either. This stays as its own test because the two paths reach
    # that code differently, and a future refactor could easily split them again.
    result = sample(blocked, 1, concurrency=2, timeout=0.005, outcome=always_pass)

    (timed_out,) = result.exceptions
    assert isinstance(timed_out.error, SampleTimeout)
    # Same rule as the serial path: a missing observation, not a failing one.
    assert result.failures == 0
    assert result.successes == 0
    assert result.n == 1


def test_every_concurrent_run_gets_its_own_budget_rather_than_an_accruing_one() -> None:
    # Regression test. The collector used to wait on futures in submission order,
    # starting each run's clock only when it *reached* that future -- so a run
    # queued behind others was granted several budgets' worth of wall clock and
    # quietly escaped a budget it had blown. Every one of these three runs blocks
    # for BLOCKED seconds against a budget 30x smaller, so all three must be
    # recorded as timeouts. The margin is what keeps this from testing scheduling
    # luck: no plausible jitter turns 0.3s into under 0.01s.
    result = sample(blocked, 3, concurrency=3, timeout=BLOCKED / 30, outcome=always_pass)

    assert len(result.exceptions) == 3
    assert all(isinstance(run.error, SampleTimeout) for run in result.exceptions)
    assert result.successes == 0


def test_a_run_inside_a_generous_timeout_budget_is_not_a_timeout() -> None:
    result = sample(Script([True, True, False]), 3, timeout=30.0)

    assert result.exceptions == ()
    assert result.successes == 2
    assert result.failures == 1


def test_timeout_error_raised_by_fn_is_preserved_on_the_serial_path() -> None:
    raised = TimeoutError("provider socket timeout")

    result = sample(Script([True, raised]), 2)

    (failed,) = result.exceptions
    assert failed.error is raised
    assert not isinstance(failed.error, SampleTimeout)


def test_timeout_error_raised_by_fn_is_preserved_on_the_concurrent_path() -> None:
    # Regression test. _run_pool used to collect with future.result(timeout=...),
    # whose concurrent.futures.TimeoutError has *been* builtins.TimeoutError since
    # Python 3.11 -- so a provider's own socket timeout was caught as if it were
    # rigor's budget expiring, and the original exception was replaced with a
    # fabricated SampleTimeout. A provider outage must never be misreported as our
    # own timeout: they are facts about different systems.
    raised = TimeoutError("provider socket timeout")

    result = sample(Script([True, raised]), 2, concurrency=2)

    (failed,) = result.exceptions
    assert failed.error is raised
    assert not isinstance(failed.error, SampleTimeout)


# --------------------------------------------------------------------------- #
# outcome=
# --------------------------------------------------------------------------- #


def test_outcome_callable_overrides_default_outcome() -> None:
    def looks_ok(value: Any) -> bool:
        return "ok" in value

    result = sample(Script(["ok great", "nope", "ok fine"]), 3, outcome=looks_ok)

    assert result.successes == 2
    assert result.failures == 1
    assert result.outcomes == (True, False, True)
    assert result.exceptions == ()


def test_outcome_callable_that_raises_records_the_run_as_an_exception() -> None:
    def picky(value: Any) -> bool:
        if value == "bad":
            raise Boom("classifier blew up")
        return True

    result = sample(Script(["good", "bad", "good"]), 3, outcome=picky)

    assert result.successes == 2
    (lost,) = result.exceptions
    assert isinstance(lost.error, Boom)
    # The run keeps its value: a broken classifier must not delete an observation.
    assert lost.value == "bad"
    assert lost.index == 1


def test_outcome_callable_result_is_coerced_to_a_plain_bool() -> None:
    result = sample(Script(["a", "b"]), 2, outcome=lambda value: value == "a")

    assert result.outcomes == (True, False)
    assert all(isinstance(run.outcome, bool) for run in result.completed)


# --------------------------------------------------------------------------- #
# default_outcome
# --------------------------------------------------------------------------- #


def test_default_outcome_accepts_a_bool() -> None:
    assert default_outcome(True) is True
    assert default_outcome(False) is False


def test_default_outcome_reads_a_boolean_passed_attribute() -> None:
    assert default_outcome(verdict(True, 5.0)) is True
    assert default_outcome(verdict(False, 1.0)) is False


def test_default_outcome_rejects_a_truthy_non_bool() -> None:
    # The anti-truthiness rule, and the reason this function exists at all: a
    # non-empty string of judge prose is truthy, so bool() would score an
    # unparsed response as a pass and manufacture a measurement out of nothing.
    # Refusing is the design decision -- do not "fix" this by falling back to
    # truthiness.
    with pytest.raises(TypeError, match="cannot decide pass/fail from str"):
        default_outcome("YES -- the answer is clearly correct")


@pytest.mark.parametrize(
    "value",
    [1, 0, 0.5, "", "pass", [True], (), {"passed": True}, None, object()],
)
def test_default_outcome_rejects_anything_that_is_not_a_bool_or_a_verdict(value: Any) -> None:
    with pytest.raises(TypeError, match="cannot decide pass/fail"):
        default_outcome(value)


def test_default_outcome_rejects_an_object_whose_passed_is_not_a_bool() -> None:
    class Sloppy:
        passed = "yes"

    with pytest.raises(TypeError, match="cannot decide pass/fail"):
        default_outcome(Sloppy())


def test_value_the_default_outcome_cannot_read_is_recorded_as_an_exception_run() -> None:
    result = sample(Script([True, "looks good to me"]), 2)

    assert result.successes == 1
    (undecidable,) = result.exceptions
    assert isinstance(undecidable.error, TypeError)
    assert undecidable.value == "looks good to me"


# --------------------------------------------------------------------------- #
# scores()
# --------------------------------------------------------------------------- #


def test_scores_collects_numeric_scores_from_returned_verdicts() -> None:
    values = [verdict(True, 4), verdict(False, 2.5), verdict(True, 1.0)]

    result = sample_of(values)

    assert result.scores() == (4.0, 2.5, 1.0)
    assert all(isinstance(score, float) for score in result.scores())


def test_scores_omits_none_rather_than_zero_filling() -> None:
    # A judge that declined to score did not score zero. Here zero-filling would
    # drag the mean from 5.0 to 3.0 -- a number nobody measured -- so the missing
    # scores are dropped and the sample size shrinks honestly instead.
    values = [
        verdict(True, 5.0),
        verdict(True, None),
        verdict(True, 5.0),
        verdict(True, None),
        verdict(True, 5.0),
    ]

    result = sample_of(values)

    assert result.scores() == (5.0, 5.0, 5.0)
    assert fmean(result.scores()) == 5.0
    assert fmean([value.score or 0.0 for value in values]) == 3.0
    assert len(result.completed) == 5  # the runs themselves are all still there


def test_scores_ignores_values_without_a_numeric_score() -> None:
    values = [verdict(True, 3.0), verdict(True, True), object(), verdict(True, "4")]

    result = sample(Script(values), 4, outcome=always_pass)

    # score=True is a bool, not a measurement of 1.0; the others carry no score.
    assert result.scores() == (3.0,)


def test_scores_ignores_runs_that_raised() -> None:
    items = [verdict(True, 5.0), Boom("provider 500"), verdict(True, 1.0)]

    result = sample(Script(items), 3)

    assert result.scores() == (5.0, 1.0)


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_sample_finishes_below_the_serial_lower_bound() -> None:
    n = 16
    adapter = FakeAdapter(responses=["ok"], cycle=True, latency=LATENCY)

    result = sample(lambda: adapter.complete("p"), n, concurrency=8, outcome=always_pass)

    assert result.successes == n
    # One-sided and deliberately loose: eight workers should need about two
    # rounds of 50ms, so anything under the 800ms a serial run could not beat
    # proves the pool is real. A loaded box cannot turn this red.
    assert result.wall_clock < n * LATENCY


def test_concurrent_runs_consume_each_scripted_response_exactly_once() -> None:
    script = [f"response-{index}" for index in range(24)]
    adapter = FakeAdapter(responses=script, latency=0.001)

    result = sample(lambda: adapter.complete("p"), len(script), concurrency=8, outcome=always_pass)

    # Running out of script raises AdapterError, so a duplicated hand-out would
    # show up as both a missing response and an exception run.
    assert result.exceptions == ()
    assert sorted(result.values) == sorted(script)
    assert adapter.call_count == len(script)
    assert len(result.runs) == len(script)


def test_seeded_adapter_under_concurrency_reproduces_the_scripted_draws() -> None:
    script = ["alpha", "beta", "gamma", "delta"]
    n = 20
    # The adapter draws from its private Random under its own lock, so the
    # sequence of draws is fixed by the seed even though which worker receives
    # which draw is not. The multiset is therefore an exact assertion.
    rng = random.Random(SEED)
    expected = Counter(rng.choice(script) for _ in range(n))
    adapter = FakeAdapter(responses=script, seed=SEED)

    result = sample(lambda: adapter.complete("p"), n, concurrency=8, outcome=always_pass)

    assert Counter(result.values) == expected
    assert adapter.call_count == n
    assert result.exceptions == ()


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #


def test_evidence_receives_exactly_one_record_whose_payload_matches_the_summary(
    tmp_path: Path,
) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    items = [True, False, Boom("provider 500")]

    result = sample(Script(items), 3, evidence=log, label="smoke")

    (record,) = log.read()
    assert record.event_type == EVENT_SAMPLE_COMPLETED
    assert record.payload == {"label": "smoke", **result.summary()}
    assert record.payload["successes"] == 1
    assert record.payload["failures"] == 1
    assert record.payload["exceptions"] == 1


def test_evidence_label_defaults_to_none_and_is_still_recorded(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")

    sample(Script([True]), 1, evidence=log)

    (record,) = log.read()
    assert record.payload["label"] is None


def test_no_record_is_written_when_no_evidence_log_is_passed(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")

    sample(Script([True, False]), 2)

    assert log.read() == []
    assert not log.path.exists()


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"n": -1},
        {"n": 1.0},
        {"n": "3"},
        {"n": None},
        {"n": 2, "concurrency": 0},
        {"n": 2, "concurrency": -4},
        {"n": 2, "concurrency": 2.0},
        {"n": 2, "timeout": 0},
        {"n": 2, "timeout": -0.5},
    ],
)
def test_invalid_arguments_raise_value_error_before_fn_is_called(kwargs: dict[str, Any]) -> None:
    kwargs = dict(kwargs)
    n = kwargs.pop("n")
    script = Script([True] * 4)

    with pytest.raises(ValueError):
        sample(script, n, **kwargs)

    assert script.calls == 0


@pytest.mark.parametrize("field", ["n", "concurrency"])
def test_a_bool_is_rejected_even_though_bool_is_an_int_subclass(field: str) -> None:
    # isinstance(True, int) is True, so a plain int check would silently accept
    # n=True as "one run" -- almost certainly a variable that lost its value
    # somewhere. Rejecting it is deliberate.
    script = Script([True] * 4)
    kwargs: dict[str, Any] = {"n": 2, field: True}
    n = kwargs.pop("n")

    with pytest.raises(ValueError, match="True"):
        sample(script, n, **kwargs)

    assert script.calls == 0


def test_timeout_none_means_no_budget_rather_than_an_instant_one() -> None:
    result = sample(Script([True, True]), 2, timeout=None)

    assert result.exceptions == ()
    assert result.successes == 2


# --------------------------------------------------------------------------- #
# sample_of
# --------------------------------------------------------------------------- #


def test_sample_of_preserves_order_and_length() -> None:
    values = [True, False, True, True, False]

    result = sample_of(values)

    assert len(result.runs) == 5
    assert [run.index for run in result.runs] == [0, 1, 2, 3, 4]
    assert result.values == (True, False, True, True, False)
    assert result.outcomes == (True, False, True, True, False)
    assert result.successes == 3
    assert result.failures == 2


def test_sample_of_accepts_the_same_options_as_sample(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")

    result = sample_of([verdict(True, 5.0), verdict(False, 1.0)], evidence=log, label="stored")

    assert result.scores() == (5.0, 1.0)
    assert result.pass_rate == 0.5
    (record,) = log.read()
    assert record.payload["label"] == "stored"


def test_sample_of_refuses_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="n must be an integer >= 1"):
        sample_of([])


# --------------------------------------------------------------------------- #
# durations
# --------------------------------------------------------------------------- #


def test_durations_are_recorded_for_every_run_including_the_ones_that_raised() -> None:
    items = [True, Boom("provider 500"), False, KeyboardInterrupt("interrupted")]

    result = sample(Script(items), 4)

    assert len(result.durations) == len(result.runs) == 4
    assert all(duration >= 0.0 for duration in result.durations)
    assert all(run.duration >= 0.0 for run in result.exceptions)


def test_durations_are_recorded_on_the_concurrent_path_too() -> None:
    items = [True] * 3 + [Boom("provider 500")]

    result = sample(Script(items), 4, concurrency=4)

    assert len(result.durations) == 4
    assert all(duration >= 0.0 for duration in result.durations)
    assert result.wall_clock >= 0.0
