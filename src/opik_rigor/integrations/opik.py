"""Publish a opik_rigor sample -- and the gates run over it -- to Opik.

Two functions, no framework. :func:`log_sample_to_opik` maps one
:class:`~opik_rigor.sampling.SampleResult` to one Opik trace with one span per run;
:func:`log_assertion_to_opik` maps a gate's report dict to feedback scores on a
trace that already exists. Everything else -- datasets, experiments, a "opik_rigor
project" -- is left to the caller, because opik_rigor does not own your Opik workspace
and creating things in it behind your back is not a feature. The reasoning, and
the exact Opik surface these two functions depend on, is written down in
``COMPATIBILITY.md``; if an Opik release moves something, that file is the
shortest path back.

Three properties this module is built to preserve, each of which breaks something
real if it is dropped:

* **Nothing imports Opik at module scope.** ``import opik_rigor.integrations.opik``
  must succeed with the extra uninstalled, so that a caller can import it, catch
  :class:`OpikIntegrationError`, and carry on. An import at module scope would
  turn a missing optional dependency into a collection error for the whole test
  session.
* **A run that raised stays distinguishable from a run that failed.** That
  distinction is the entire thesis of :mod:`opik_rigor.sampling` -- a provider outage
  is not a quality regression -- and a mapping that flattened both into "span
  with a falsey output" would destroy it at exactly the moment it is most needed
  (the dashboard someone looks at during an incident). Here a raised run gets a
  ``opik_rigor:raised`` tag, an ``outcome`` of ``"raised"``, and Opik's own
  ``error_info``, so it is filterable, readable, and rendered as an error.
* **Everything sent is JSON-safe.** These payloads cross a network boundary. A
  ``numpy.float64`` from a statistical report, or a live exception object, would
  either be silently stringified by somebody else's encoder or drop the record on
  the floor. :func:`_jsonable` converts up front so the failure, if any, is here.

No object from Opik is ever introspected (``inspect.signature``,
``typing.get_type_hints``): ``inspect.signature(Trace.span)`` raises on Python
3.14 in opik 2.2.28. See COMPATIBILITY.md. Calling the methods is fine.
"""

from __future__ import annotations

import math
import traceback as traceback_module
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import RigorError
from ..sampling import Run, SampleResult

__all__ = [
    "MAX_REASON_CHARS",
    "SCORE_PREFIX",
    "STATISTIC_KEYS",
    "OpikIntegrationError",
    "log_assertion_to_opik",
    "log_sample_to_opik",
]

#: Every feedback score this module writes starts with this. Opik's feedback
#: scores share one flat namespace per trace, so an unprefixed ``passed`` or
#: ``mean`` would silently overwrite -- or be overwritten by -- a score the user
#: logged themselves from their own evaluator. The full shape is
#: ``rigor.<gate>.<key>``: ``rigor.pass_rate.passed``,
#: ``rigor.pass_rate.lower_bound``, ``rigor.no_regression.p_value``,
#: ``rigor.score_distribution.mean``. Keeping the gate name in the middle means
#: two different gates logged against the same trace do not collide either.
SCORE_PREFIX = "rigor."

#: Tag applied to every trace and span written here, so a workspace shared with
#: other tooling can be filtered down to opik_rigor's own records in one click.
TRACE_TAG = "opik_rigor"

#: Reserved key under which opik_rigor's own metadata is nested, on both traces and
#: spans. Caller-supplied ``metadata`` is merged at the top level *around* it, so
#: the two cannot overwrite each other -- except by the caller deliberately using
#: this key, in which case opik_rigor's block wins and the caller's is dropped.
METADATA_KEY = "opik_rigor"

#: Report keys logged as their own feedback score, in this order, when present
#: and numeric. Deliberately a whitelist rather than "every number in the
#: report": ``min_rate``, ``alpha``, ``confidence`` and ``n`` are *configuration
#: and sample size*, not measurements, and logging a threshold as a score would
#: put a constant into the dashboard's aggregates and make the average of
#: "min_rate" look like a finding. ``u_statistic`` is excluded for a different
#: reason -- it is unbounded and scales with n, so charting it next to a
#: probability is meaningless.
STATISTIC_KEYS = ("lower_bound", "p_value", "mean", "p10", "stddev", "pass_rate")

#: Feedback-score reasons are stored in a database column, not a blob. opik_rigor's
#: gate messages are deliberately verbose (they are the statistical report), so
#: they are truncated here rather than risking a rejected write that would lose
#: the score *and* the reason.
MAX_REASON_CHARS = 1000

#: Depth at which :func:`_jsonable` stops recursing and falls back to ``repr``.
#: A guard against a self-referential value in a judge's payload turning a
#: logging call into a RecursionError.
_MAX_DEPTH = 6

_INSTALL_HINT = 'pip install "opik-rigor[opik]"'


class OpikIntegrationError(RigorError):
    """Raised when the Opik integration cannot do its job.

    A subclass rather than a bare :class:`~opik_rigor.errors.RigorError` because of
    how this integration is actually used: telemetry is wrapped in a ``try`` so
    that a dashboard being unavailable does not fail a test run. Catching
    ``RigorError`` there would also swallow
    :class:`~opik_rigor.errors.RubricDriftError` and
    :class:`~opik_rigor.errors.JudgeOutputError` -- errors that say the *measurement*
    is invalid and must never be quietly ignored. ``except
    OpikIntegrationError`` means exactly one thing: the logging failed, the
    evaluation did not.
    """


def _import_opik() -> Any:
    """Import Opik on first use and name the extra if it is not installed.

    Deferred, not lazy-cached: ``sys.modules`` already caches it, and a module
    level cache here would only add a way for the two to disagree.
    """
    try:
        import opik
    except ImportError as exc:
        raise OpikIntegrationError(
            f"the Opik integration requires the optional [opik] extra, which is not "
            f"installed. Install it with: {_INSTALL_HINT}  "
            f"(opik_rigor's core never imports Opik, so the assertions, the judge and the "
            f"evidence log keep working without it -- you lose a dashboard, not a "
            f"test suite.)"
        ) from exc
    return opik


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """Convert ``value`` into something a JSON encoder cannot choke on.

    Applied to everything before it is handed to Opik. The alternative -- trusting
    the SDK's encoder -- fails in the two ways that matter here: a
    ``numpy.float64`` out of :mod:`opik_rigor.distribution` is not a plain float and a
    live exception object is not data, and a rejected payload is a *dropped*
    record rather than a raised one, so the loss would be invisible.

    Non-finite floats become their string spelling rather than ``null``: ``nan``
    is a real answer from a degenerate statistic, and turning it into "no value"
    would hide a result that a reader needs to see.
    """
    if value is None or isinstance(value, (str, bool)):
        return value

    # numpy scalars and 0-d arrays. Checked before int/float because np.int64 is
    # not an int (and is unencodable), while np.float64 *is* a float and would
    # otherwise pass straight through as a numpy object.
    if hasattr(value, "dtype") and hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError, TypeError):
            return repr(value)
        if value is None or isinstance(value, (str, bool)):
            return value

    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if _depth >= _MAX_DEPTH:
        return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item, _depth + 1) for item in value]
    return repr(value)


def _truncate(text: str) -> str:
    """Cap a reason at :data:`MAX_REASON_CHARS`, marking that it was cut."""
    if len(text) <= MAX_REASON_CHARS:
        return text
    return text[: MAX_REASON_CHARS - 3] + "..."


# --------------------------------------------------------------------------- #
# sample -> trace
# --------------------------------------------------------------------------- #


def _outcome_of(run: Run) -> str:
    """Name what happened in one run, keeping "raised" its own answer.

    Four values, not two. ``raised`` is checked first because a run that raised
    has no outcome to report, and the whole point of
    :attr:`opik_rigor.sampling.SampleResult.exceptions` is that it is not the failure
    bucket. ``unrecorded`` exists for a hand-built :class:`~opik_rigor.sampling.Run`
    with ``outcome=None`` and no error -- :func:`opik_rigor.sampling.sample` never
    produces one, and calling it a failure would invent a measurement.
    """
    if run.raised:
        return "raised"
    if run.outcome is True:
        return "passed"
    if run.outcome is False:
        return "failed"
    return "unrecorded"


def _error_info(run: Run) -> dict[str, str] | None:
    """Opik's ``ErrorInfoDict`` for a run that raised, else ``None``.

    ``exception_type`` and ``traceback`` are required by the TypedDict;
    ``message`` is optional and always supplied, because the type name alone
    ("TimeoutError") does not tell you which call timed out.
    """
    error = run.error
    if error is None:
        return None
    try:
        formatted = "".join(
            traceback_module.format_exception(type(error), error, error.__traceback__)
        )
    except Exception:  # noqa: BLE001 - a broken __traceback__ must not lose the run
        formatted = f"{type(error).__name__}: {error}"
    return {
        "exception_type": type(error).__name__,
        "message": str(error),
        "traceback": formatted,
    }


def _span_metadata(run: Run, outcome: str) -> dict[str, Any]:
    """What the run itself recorded, nested under :data:`METADATA_KEY`."""
    error = run.error
    return {
        METADATA_KEY: {
            "run_index": run.index,
            "outcome": outcome,
            "raised": run.raised,
            "duration_seconds": float(run.duration),
            "error_type": None if error is None else type(error).__name__,
            "error_message": None if error is None else str(error),
        }
    }


def log_sample_to_opik(
    result: SampleResult,
    *,
    name: str,
    client: Any = None,
    project_name: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    judge_name: str | None = None,
) -> str:
    """Map a :class:`~opik_rigor.sampling.SampleResult` to one Opik trace.

    One trace, one span per run. The trace carries
    :meth:`~opik_rigor.sampling.SampleResult.summary` as both its ``output`` and its
    metadata -- as output because that is what the sample produced, as metadata
    because that is where it is filterable.

    Each span is ``type="llm"`` and is tagged ``opik_rigor:passed``,
    ``opik_rigor:failed`` or ``opik_rigor:raised``. A run that raised additionally gets
    Opik's ``error_info``, which is what makes it render as an error rather than
    as an unremarkable span with a falsey output. Keeping those two apart is not
    cosmetic: a sample where five runs raised and none failed is a broken
    provider, and a sample where five failed and none raised is a broken model,
    and a dashboard that shows them identically will send you to debug the wrong
    system.

    Args:
        result: The sample to log. Every run in ``result.runs`` gets a span,
            including the ones that raised.
        name: Trace name, also the stem of each span name (``name[3]``). Required
            and non-empty -- ``client.trace()`` accepts ``name=None`` and gives
            you an unnamed trace with no error, which is unfindable later.
        client: An ``opik.Opik``. Constructed lazily here if omitted; never at
            import time, and never as a module-level singleton, because that
            would open a connection as a side effect of an import. A client
            created here is flushed before returning, since the caller has no
            handle to flush it with; a client you passed in is left alone, so
            your batching stays yours.
        project_name: Opik project. ``None`` uses the client's default.
        tags: Extra trace tags, on top of ``"opik_rigor"``. Order is preserved and
            duplicates are dropped.
        metadata: Extra trace metadata, merged at the top level around opik_rigor's
            own block. See :data:`METADATA_KEY`.
        judge_name: Recorded on the trace so a sample can be traced back to the
            judge that produced its verdicts. Optional because ``sample()`` does
            not require a judge at all.

    Returns:
        The trace id, which is what :func:`log_assertion_to_opik` needs.

    Raises:
        OpikIntegrationError: If the ``[opik]`` extra is not installed.
        ValueError: If ``name`` is empty.

    Note:
        No ``start_time``/``end_time`` is set on the spans. A
        :class:`~opik_rigor.sampling.SampleResult` records per-run *durations*, not
        wall-clock positions, so any timestamps here would be invented -- and a
        concurrent sample would be drawn as a neat sequence of runs that in fact
        overlapped. The measured duration is on each span's metadata as
        ``duration_seconds`` instead, which is the number that was actually
        observed.
    """
    opik = _import_opik()
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"name must be a non-empty string, got {name!r}")

    owns_client = client is None
    if owns_client:
        client = opik.Opik(project_name=project_name)

    summary = _jsonable(result.summary())
    trace_metadata: dict[str, Any] = dict(_jsonable(dict(metadata)) if metadata else {})
    trace_metadata[METADATA_KEY] = {
        "sample": summary,
        "judge_name": judge_name,
        "outcomes": {
            "passed": result.successes,
            "failed": result.failures,
            "raised": len(result.exceptions),
        },
    }
    trace_tags = list(dict.fromkeys([TRACE_TAG, *(tags or ())]))

    trace = client.trace(
        name=name,
        input={
            "runs": len(result.runs),
            "concurrency": result.concurrency,
            "errors_as_failures": result.errors_as_failures,
            "judge_name": judge_name,
        },
        output=summary,
        metadata=trace_metadata,
        tags=trace_tags,
        project_name=project_name,
    )

    for run in result.runs:
        outcome = _outcome_of(run)
        error_info = _error_info(run)
        span = trace.span(
            name=f"{name}[{run.index}]",
            type="llm",
            input={"run_index": run.index},
            output={
                "outcome": outcome,
                "value": _jsonable(run.value),
                "error": None if error_info is None else _jsonable(error_info),
            },
            metadata=_span_metadata(run, outcome),
            tags=[TRACE_TAG, f"{TRACE_TAG}:{outcome}"],
            error_info=error_info,
        )
        span.end()

    trace.end()
    if owns_client:
        client.flush()
    return str(trace.id)


# --------------------------------------------------------------------------- #
# assertion report -> feedback scores
# --------------------------------------------------------------------------- #


def _numeric(value: Any) -> float | None:
    """``value`` as a plain finite float, or ``None`` if it is not a number.

    ``bool`` is rejected on purpose: it is an ``int`` subclass, so a report key
    that happens to hold ``True`` would otherwise be logged as the statistic
    1.0. Non-finite floats are rejected too -- Opik's score is a float column,
    and a NaN there is a value nobody can chart or compare. The fact does not go
    missing: it stays in the reason.
    """
    if isinstance(value, bool) or value is None:
        return None
    converted = _jsonable(value)
    if isinstance(converted, bool) or not isinstance(converted, (int, float)):
        return None
    number = float(converted)
    return number if math.isfinite(number) else None


def _comparison(report: Mapping[str, Any]) -> str:
    """One line naming the number that decided the gate, and what it was against.

    Gate-specific because the deciding number is gate-specific: the pass-rate
    gate is decided by a lower bound and not by the observed rate, and a summary
    that printed the observed rate would misrepresent what the gate did.
    """
    gate = report.get("gate")
    if gate == "pass_rate":
        return (
            f"lower_bound {report.get('lower_bound')} vs min_rate {report.get('min_rate')} "
            f"({report.get('successes')}/{report.get('n')} passed)"
        )
    if gate == "no_regression":
        return (
            f"p_value {report.get('p_value')} vs alpha {report.get('alpha')} "
            f"(median {report.get('median_current')} vs baseline "
            f"{report.get('median_baseline')})"
        )
    if gate == "score_distribution":
        return (
            f"mean {report.get('mean')}, p10 {report.get('p10')}, "
            f"stddev {report.get('stddev')} over n={report.get('n')}"
        )
    known = [f"{key} {report[key]}" for key in STATISTIC_KEYS if key in report]
    return ", ".join(known) if known else f"n={report.get('n')}"


def _explanation(report: Mapping[str, Any]) -> str:
    """The text attached to every score as its ``reason``.

    An explicit ``report["message"]`` wins. That is the documented seam for the
    common case::

        try:
            assert_pass_rate(result, 0.9)
        except PassRateError as exc:
            log_assertion_to_opik({**exc.stats, "message": str(exc)}, trace_id=tid)

    -- the exception's message *is* the statistical report, and it is strictly
    better than anything reconstructed from the dict. Without it, one is
    reconstructed, because a score of 0.0 with no reason tells a reader that
    something failed and nothing about what.
    """
    message = report.get("message")
    if isinstance(message, str) and message.strip():
        return _truncate(message.strip())

    gate = str(report.get("gate") or "assertion")
    label = report.get("label")
    subject = f"{gate} gate" + (f" {label!r}" if label else "")
    verdict = "passed" if report.get("passed") else "failed"
    violations = report.get("violations") or ()
    detail = "; ".join(str(item) for item in violations) if violations else _comparison(report)
    return _truncate(f"{subject} {verdict}: {detail}")


def log_assertion_to_opik(
    report: Mapping[str, Any],
    *,
    trace_id: str,
    client: Any = None,
    project_name: str | None = None,
) -> None:
    """Attach a gate's report to an existing trace as feedback scores.

    Feedback scores, not an Opik experiment. ``create_experiment`` /
    ``Experiment.insert`` require an experiment item to reference a **dataset
    item that already exists in Opik**, which would mean opik_rigor creating datasets
    in your workspace as a side effect of logging a test result. That is a
    decision about your data, and it is yours; see COMPATIBILITY.md. If you want
    these verdicts inside an experiment, build the dataset yourself and pass the
    trace ids -- that is a seam, not a gap.

    Always at least two scores: ``<prefix><gate>.passed``, 1.0 or 0.0, and the
    gate's deciding statistic under its own name. Both carry the explanation as
    their ``reason``, because a bare 0.0 on a dashboard is an alarm without a
    cause, and the reader is usually not the person who ran the gate.

    Args:
        report: A gate report -- the dict returned by
            :func:`opik_rigor.distribution.assert_pass_rate` and friends, or the
            ``.stats`` of the exception they raise. Must carry ``passed``. May
            carry ``message``; see :func:`_explanation`.
        trace_id: The trace to attach to, i.e. the return value of
            :func:`log_sample_to_opik`. Opik requires this per score and will
            drop a batch without it.
        client: An ``opik.Opik``, constructed lazily here if omitted, on the same
            ownership rule as :func:`log_sample_to_opik`.
        project_name: Opik project. ``None`` uses the client's default. It must
            resolve to the project the trace is in; a score aimed at a project
            the trace is not in is silently dropped by Opik, not rejected.

    Raises:
        OpikIntegrationError: If the ``[opik]`` extra is not installed.
        ValueError: If ``trace_id`` is empty, or ``report`` has no ``passed``
            key -- a report without a verdict is not a gate report, and guessing
            one from the numbers would be inventing the result.
    """
    opik = _import_opik()
    if not isinstance(trace_id, str) or not trace_id.strip():
        raise ValueError(f"trace_id must be a non-empty string, got {trace_id!r}")
    if "passed" not in report:
        raise ValueError(
            f"report has no 'passed' key, so it is not a opik_rigor gate report: "
            f"got keys {sorted(map(str, report))}"
        )

    gate = str(report.get("gate") or "assertion")
    reason = _explanation(report)
    scores: list[dict[str, Any]] = [
        {
            "id": trace_id,
            "name": f"{SCORE_PREFIX}{gate}.passed",
            "value": 1.0 if report["passed"] else 0.0,
            "reason": reason,
        }
    ]
    for key in STATISTIC_KEYS:
        if key not in report:
            continue
        number = _numeric(report[key])
        if number is None:
            continue
        scores.append(
            {
                "id": trace_id,
                "name": f"{SCORE_PREFIX}{gate}.{key}",
                "value": number,
                "reason": reason,
            }
        )

    owns_client = client is None
    if owns_client:
        client = opik.Opik(project_name=project_name)
    client.log_traces_feedback_scores(scores, project_name=project_name)
    if owns_client:
        client.flush()
