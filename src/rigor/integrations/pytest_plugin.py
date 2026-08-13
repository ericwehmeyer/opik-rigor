"""pytest integration: a statistical gate expressed as a marker.

The point of this module is to make the *statistical* discipline the path of
least resistance inside an ordinary test suite. Without it, a team testing a
stochastic system reaches for a retry plugin -- rerun until it goes green -- which
measures nothing and hides everything. ``@pytest.mark.rigor_repeat`` runs the same
body n times and gates the result with
:func:`rigor.distribution.assert_pass_rate`, so the reported outcome is a
confidence bound rather than one lucky draw.

Three properties are deliberate.

**Failure and exception stay apart.** A body that returns normally is a pass; a
body that raises ``AssertionError`` is a **failure** -- the system under test ran
and missed the bar; any other exception is an **exception** -- the harness broke.
That distinction is :mod:`rigor.sampling`'s central thesis (a provider outage must
not read as a quality regression) and this plugin is the place where it would be
easiest to flatten, since pytest itself calls both "a failing test". It is not
flattened: assertions become ``outcome=False`` observations, everything else is
recorded as ``Run.error`` and lands in the exception bucket, and both counts are
printed before the gate runs so the two are distinguishable in the output.

**Nothing here imports Opik.** Opik ships its own ``pytest11`` entry point named
``opik``; ours is named ``rigor``, uses a marker rather than a function decorator,
and prefixes every marker, fixture and ini option with ``rigor``. Loading both
changes nothing about either. See ``COMPATIBILITY.md``.

**Importing this module does no work.** It defines hooks and fixtures and nothing
else -- no file is opened, no config is read, no network is touched. An entry-point
plugin is imported in every pytest process on the machine, including ones that
never use it, so anything expensive at import time is a tax on unrelated suites.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from rigor.distribution import DEFAULT_CONFIDENCE, assert_pass_rate
from rigor.evidence import EVENT_SAMPLE_COMPLETED, EvidenceLog
from rigor.judge import PinnedJudge
from rigor.sampling import sample

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import cost at zero
    import os

    from rigor.adapters.base import Adapter
    from rigor.sampling import SampleResult

#: The marker, the fixtures and the ini option are all ``rigor``-prefixed. The
#: plugin shares a process with whatever else the user has installed, and an
#: unprefixed name like ``repeat`` or ``evidence`` is a collision waiting for the
#: one suite that already had its own.
MARKER_NAME = "rigor_repeat"
INI_EVIDENCE_PATH = "rigor_evidence_path"

#: Where ``rigor_evidence`` writes when no path is configured. One file per test,
#: inside pytest's own ``tmp_path``, so the log is per-test and self-cleaning.
DEFAULT_EVIDENCE_FILENAME = "rigor-evidence.jsonl"

_MARKER_HELP = (
    "rigor_repeat(n, min_rate, confidence=0.95, errors_as_failures=True): run this "
    "test n times and gate the pass rate with rigor.assert_pass_rate. A body that "
    "returns is a pass, one that raises AssertionError is a failure, any other "
    "exception is an exception (harness broke, not the system under test)."
)

#: Positional spelling of the marker, so ``rigor_repeat(20, 0.9)`` works as well
#: as the keyword form.
_POSITIONAL = ("n", "min_rate", "confidence", "errors_as_failures")

#: Exceptions that are pytest's own flow control rather than a fact about the
#: system under test. A ``pytest.skip()`` or ``pytest.xfail()`` inside the body
#: means "this test does not apply", which is a statement about all n runs at
#: once -- counting it as an observation, in either bucket, would be a fabricated
#: measurement. They abort the repeat and propagate unchanged. ``XFailed`` is
#: listed explicitly because it subclasses ``Failed``, which is otherwise treated
#: as a failure below.
_CONTROL_FLOW: tuple[type[BaseException], ...] = (
    pytest.skip.Exception,
    pytest.xfail.Exception,
    pytest.exit.Exception,
    KeyboardInterrupt,
    SystemExit,
)

#: Raising these means "the system under test did not meet the bar" -- an
#: observation of a failure, not a broken harness. ``pytest.fail()`` is included
#: because it is the explicit spelling of exactly what a bare ``assert`` says.
_FAILURE: tuple[type[BaseException], ...] = (AssertionError, pytest.fail.Exception)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the ini options.

    ``rigor_evidence_path`` is a string rather than a ``path`` type so that a
    relative value can be anchored on ``rootpath`` here, rather than on whatever
    the working directory happened to be when pytest started.
    """
    parser.addini(
        INI_EVIDENCE_PATH,
        help=(
            "Path for the rigor_evidence log. Relative paths resolve against the "
            "rootdir. When unset, each test gets its own log inside tmp_path. The "
            "log is append-only, so pointing several tests at one path builds a "
            "single audit trail rather than overwriting."
        ),
        default="",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker so ``--strict-markers`` accepts it.

    This repo runs ``--strict-markers`` itself, which is the correct setting: an
    unregistered marker is almost always a typo, and a typo'd ``rigor_repeat``
    would silently run the body once and gate nothing.
    """
    config.addinivalue_line("markers", _MARKER_HELP)


# --------------------------------------------------------------------------- #
# the marker
# --------------------------------------------------------------------------- #


def _marker_arguments(marker: pytest.Mark) -> dict[str, Any]:
    """Normalise the marker's args and kwargs into validated keyword arguments.

    Unknown or duplicated arguments are refused rather than ignored. A silently
    dropped ``min_rate=0.9`` would leave the test passing on a gate nobody set,
    which is the one failure mode this plugin exists to prevent.
    """
    if len(marker.args) > len(_POSITIONAL):
        raise TypeError(
            f"{MARKER_NAME} takes at most {len(_POSITIONAL)} positional arguments "
            f"{_POSITIONAL}, got {len(marker.args)}"
        )
    # strict=False on purpose: the positional form is a prefix, so fewer args than
    # names is the ordinary case rather than a mismatch.
    supplied = dict(zip(_POSITIONAL, marker.args, strict=False))
    duplicated = sorted(set(supplied) & set(marker.kwargs))
    if duplicated:
        raise TypeError(f"{MARKER_NAME} got repeated argument(s): {', '.join(duplicated)}")
    supplied.update(marker.kwargs)

    unknown = sorted(set(supplied) - set(_POSITIONAL))
    if unknown:
        raise TypeError(
            f"{MARKER_NAME} got unexpected argument(s): {', '.join(unknown)}; "
            f"accepted arguments are {', '.join(_POSITIONAL)}"
        )
    missing = [name for name in ("n", "min_rate") if name not in supplied]
    if missing:
        raise TypeError(f"{MARKER_NAME} requires {' and '.join(missing)}")

    supplied.setdefault("confidence", DEFAULT_CONFIDENCE)
    supplied.setdefault("errors_as_failures", True)
    if not isinstance(supplied["errors_as_failures"], bool):
        raise TypeError(
            f"{MARKER_NAME} errors_as_failures must be a bool, got "
            f"{type(supplied['errors_as_failures']).__name__}"
        )
    return supplied


def _repeat_once(testfunction: Callable[..., Any], testargs: dict[str, Any]) -> bool:
    """Run the body once and classify the result. Never returns for an exception.

    The classification is the whole point of the plugin, so it is stated in one
    place:

    * returns normally -> ``True``, a pass;
    * raises ``AssertionError`` (or ``pytest.fail()``) -> ``False``, a **failure**:
      the system under test ran and its output missed the bar;
    * raises anything else -> propagates, so :func:`rigor.sampling.sample` records
      it as an **exception**: the harness broke and this run produced no
      observation at all.

    A returned value is deliberately not inspected. pytest already treats a
    non-``None`` return from a test as an error, so making the body's return value
    meaningful here would put this plugin at odds with the runner it lives in;
    ``assert`` is how a pytest test states its verdict.
    """
    try:
        testfunction(**testargs)
    except _CONTROL_FLOW:
        # Re-raised before the failure clause because XFailed subclasses Failed:
        # matched the other way round, an xfail inside the body would be recorded
        # as an observed failure of the system under test.
        raise
    except _FAILURE:
        return False
    return True


def _summary_line(nodeid: str, result: SampleResult, options: dict[str, Any]) -> str:
    """One greppable line separating failures from exceptions, printed every run.

    ``assert_pass_rate`` reports ``failures = n - successes``, which is the right
    denominator arithmetic for a pass-rate gate but blurs exactly the distinction
    this plugin promises to keep. So the sample's own breakdown is printed before
    the gate runs: pytest shows captured stdout for a failing test, so the reader
    of a red build sees whether 20 runs missed the bar or 20 runs never happened.
    """
    summary = result.summary()
    return (
        f"[{MARKER_NAME}] {nodeid} runs={summary['runs']} successes={summary['successes']} "
        f"failures={summary['failures']} exceptions={summary['exceptions']} "
        f"pass_rate={summary['pass_rate']:.4f} n={summary['n']} "
        f"min_rate={float(options['min_rate']):.4f} confidence={float(options['confidence']):.4f}"
    )


@pytest.hookimpl
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run a ``rigor_repeat``-marked test n times instead of once.

    Returns ``True`` to tell pytest the call was handled, and ``None`` for every
    unmarked test so the default implementation runs untouched -- which is what
    makes this plugin invisible to a suite that does not use it.

    Registered without ``tryfirst``: plugins such as pytest-asyncio claim the same
    firstresult hook for the test styles they own, and jumping the queue would
    take those tests away from them. A coroutine function is refused outright
    rather than called n times and never awaited, which would score n meaningless
    passes.

    Fixtures work across repeats because the body is called with the same resolved
    ``funcargs`` each time. That is the correct scoping -- a function-scoped
    fixture is set up once per *test*, and re-running its setup n times would be a
    different (and much slower) contract -- but it does mean a body that mutates
    its fixture sees the mutation on the next repeat, exactly as it would inside a
    hand-written loop.
    """
    marker = pyfuncitem.get_closest_marker(MARKER_NAME)
    if marker is None:
        return None

    options = _marker_arguments(marker)
    testfunction = pyfuncitem.obj
    if _is_coroutine_function(testfunction):
        raise TypeError(
            f"{MARKER_NAME} cannot repeat the async test {pyfuncitem.name!r}: another "
            f"plugin owns the call for coroutine tests, so rigor would run the body "
            f"n times without ever awaiting it and score n meaningless passes"
        )

    # _fixtureinfo.argnames is what pytest's own pytest_pyfunc_call uses to pick
    # the requested fixtures out of funcargs; there is no public equivalent.
    testargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    evidence = testargs.get("rigor_evidence")
    if not isinstance(evidence, EvidenceLog):
        evidence = None

    # Control-flow exceptions abort the whole repeat. sample() catches
    # BaseException by design (an exception from the system under test is data),
    # so the only way out is to remember the first one, make the remaining runs
    # re-raise it immediately, and re-raise it here once sampling is over.
    aborted: list[BaseException] = []

    def once() -> bool:
        if aborted:
            raise aborted[0]
        try:
            return _repeat_once(testfunction, testargs)
        except _CONTROL_FLOW as exc:
            aborted.append(exc)
            raise

    # The log is handed to sample() only after the fact, not through it: an
    # aborted repeat would otherwise leave a sample.completed record claiming n
    # runs when the body ran once and asked to be skipped. A record of runs that
    # did not happen is worse than no record.
    result = sample(
        once,
        options["n"],
        errors_as_failures=options["errors_as_failures"],
        label=pyfuncitem.nodeid,
    )
    if aborted:
        raise aborted[0]
    if evidence is not None:
        evidence.append(EVENT_SAMPLE_COMPLETED, {"label": pyfuncitem.nodeid, **result.summary()})

    print(_summary_line(pyfuncitem.nodeid, result, options))

    # PassRateError propagates untouched: its message *is* the statistical report,
    # and wrapping it in a pytest.fail() would replace a confidence bound with a
    # one-line summary of one.
    assert_pass_rate(
        result,
        options["min_rate"],
        confidence=options["confidence"],
        evidence=evidence,
        label=pyfuncitem.nodeid,
    )
    return True


def _is_coroutine_function(function: Any) -> bool:
    """Local import so the module-level import stays free of ``inspect``."""
    import inspect

    return inspect.iscoroutinefunction(function)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _evidence_path(config: pytest.Config, tmp_path: Path) -> Path:
    """Resolve the configured evidence path, falling back to ``tmp_path``."""
    configured = str(config.getini(INI_EVIDENCE_PATH) or "").strip()
    if not configured:
        return tmp_path / DEFAULT_EVIDENCE_FILENAME
    path = Path(configured).expanduser()
    return path if path.is_absolute() else Path(config.rootpath) / path


@pytest.fixture
def rigor_evidence(request: pytest.FixtureRequest, tmp_path: Path) -> EvidenceLog:
    """An :class:`~rigor.evidence.EvidenceLog` for this test.

    Defaults to a per-test file under pytest's ``tmp_path``, which is the right
    default for a fixture: evidence from one test cannot be confused with
    another's, and nothing is left behind in the working tree.

    Setting the ``rigor_evidence_path`` ini option redirects every test to one
    file instead. Records are **appended**, never truncated -- the log is
    append-only by construction, and a fixture that cleared it per test would turn
    a suite-wide audit trail into whatever the last test happened to write.
    """
    return EvidenceLog(_evidence_path(request.config, tmp_path))


@pytest.fixture
def rigor_judge(rigor_evidence: EvidenceLog) -> Callable[..., PinnedJudge]:
    """Factory building a :class:`~rigor.judge.PinnedJudge` on this test's log.

    Deliberately thin: it wires the evidence log and forwards everything else
    verbatim, so the judge's real contract -- model pinning, rubric hashing,
    drift detection -- stays documented in one place instead of being re-explained
    (and eventually re-implemented) here. Pass ``evidence=`` to override the log
    for the rare test that needs a second one.
    """

    def build(
        adapter: Adapter,
        rubric_path: str | os.PathLike[str],
        *,
        name: str = "default",
        accept_rubric_change: bool = False,
        evidence: EvidenceLog | None = None,
    ) -> PinnedJudge:
        return PinnedJudge(
            adapter,
            rubric_path,
            evidence if evidence is not None else rigor_evidence,
            name=name,
            accept_rubric_change=accept_rubric_change,
        )

    return build
