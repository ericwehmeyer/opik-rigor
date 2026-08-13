"""The Opik integration, run against the real client with no server.

Two halves, and both of them run in every environment.

The first half is about the *extra being missing*, and it is the half that runs
on a machine where nobody has installed Opik. It asserts what a caller actually
depends on: that importing ``rigor.integrations.opik`` works anyway, that
calling into it says which extra to install rather than raising an ImportError
from three frames down, and that merely importing the module does not drag Opik
in. Absence is simulated with ``sys.modules["opik"] = None`` rather than assumed
from the environment, so these three run identically in the venv that has Opik
and the one that does not -- which is the point, because they are the tests most
likely to rot when someone adds a top-level import.

The second half exercises the real ``opik.Opik`` client through
``opik.record_traces_locally()``, which records what the client actually
produced without a backend. That is deliberately not a mock: a mock asserts that
we called the functions we meant to call, which is the one thing that never
breaks. These assert on the recorded trace. Opik's own docstring warns the
emulator is connection-scoped, so every lookup here is **by trace id**.

Nothing here touches the network. A loopback HTTP sink stands in for the Opik
backend and rejects everything with a 401 -- the same reply the public endpoint
gives an unconfigured client, and a path Opik handles cleanly. Without it the
SDK's online processor spends tens of seconds retrying a connection it will
never get, and the emulator, which sits behind it in the same processing chain,
does not see the messages until it gives up.

Determinism: every draw comes from an explicit ``random.Random(seed)``. The
mixed sample below has exactly four passes, three failures and two exceptions by
construction, in a shuffled but reproducible order, so "all three are
distinguishable" is asserted against known counts rather than against whatever
the seed happened to produce.
"""

from __future__ import annotations

import importlib
import json
import logging
import random
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from rigor import (
    EvidenceLog,
    FakeAdapter,
    PassRateError,
    PinnedJudge,
    SampleResult,
    assert_pass_rate,
    sample,
)
from rigor.integrations.opik import (
    SCORE_PREFIX,
    OpikIntegrationError,
    log_assertion_to_opik,
    log_sample_to_opik,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBRIC = REPO_ROOT / "rubrics" / "example-rubric.md"

#: Fixed so a failure here is reproducible by re-running, not by re-rolling.
SHUFFLE_SEED = 20260813
JUDGE_SEED = 424242

OUTAGE_MESSAGE = "upstream returned 503 while sampling"


class ProviderOutage(RuntimeError):
    """Stands in for the provider falling over mid-sample.

    A named type rather than a bare ``RuntimeError`` because the assertion under
    test is that *this name* survives the mapping into Opik: "the run raised" is
    not useful during an incident, "the run raised ProviderOutage" is.
    """


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


class _RejectingHandler(BaseHTTPRequestHandler):
    """Answers every request 401, immediately, and logs nothing."""

    protocol_version = "HTTP/1.1"

    def _reject(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = json.dumps({"code": 401, "message": "API key should be provided"}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # BaseHTTPRequestHandler dispatches on these exact names.
    do_GET = _reject
    do_POST = _reject
    do_PUT = _reject
    do_PATCH = _reject
    do_DELETE = _reject

    def log_message(self, *args: Any) -> None:
        return


@pytest.fixture(scope="module")
def opik_module(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The ``opik`` package, pointed at a loopback sink instead of the internet.

    Skips the whole second half of this file when the extra is absent, which is
    what makes it pass unchanged in the venv without Opik.

    ``OPIK_CONFIG_PATH`` is aimed at a path that does not exist so a developer's
    own ``~/.opik.config`` cannot leak a workspace, an API key, or a real host
    into these tests -- the failure mode being a "local" test that quietly logs
    a fixture's traces to somebody's production project.
    """
    opik = pytest.importorskip("opik")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _RejectingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    logger = logging.getLogger("opik")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OPIK_CONFIG_PATH", str(tmp_path_factory.mktemp("opik") / "absent.config"))
        patch.setenv("OPIK_URL_OVERRIDE", f"http://127.0.0.1:{port}/api")
        patch.setenv("OPIK_API_KEY", "rigor-offline-tests")
        patch.setenv("OPIK_WORKSPACE", "default")
        patch.setenv("OPIK_TRACK_DISABLE", "false")
        patch.setenv("OPIK_SENTRY_ENABLE", "false")
        patch.setenv("OPIK_CONSOLE_LOGGING_LEVEL", "CRITICAL")
        try:
            yield opik
        finally:
            logger.setLevel(previous_level)
            server.shutdown()
            server.server_close()


@pytest.fixture
def client(opik_module: Any) -> Any:
    return opik_module.Opik(project_name="rigor-integration-tests")


def trace_by_id(storage: Any, trace_id: str) -> Any:
    """The one recorded trace with this id.

    A lookup rather than ``storage.trace_trees[0]``: the local emulator is
    connection-scoped, so anything else logging on the same connection lands in
    the same handle, and indexing into it would make this file's assertions
    depend on what the rest of the suite happened to do.
    """
    found = [trace for trace in storage.trace_trees if trace.id == trace_id]
    assert len(found) == 1, f"expected exactly one trace with id {trace_id}, got {len(found)}"
    return found[0]


# --------------------------------------------------------------------------- #
# sample builders
# --------------------------------------------------------------------------- #


def mixed_sample(seed: int) -> SampleResult:
    """Four passes, three failures, two exceptions, in a reproducible order."""
    script = ["pass"] * 4 + ["fail"] * 3 + ["raise"] * 2
    random.Random(seed).shuffle(script)
    steps = iter(script)

    def once() -> bool:
        step = next(steps)
        if step == "raise":
            raise ProviderOutage(OUTAGE_MESSAGE)
        return step == "pass"

    return sample(once, len(script))


def all_passing_sample(n: int = 30) -> SampleResult:
    return sample(lambda: True, n)


def span_outcomes(trace: Any) -> dict[str, int]:
    """How many spans reported each outcome, read back off the metadata."""
    counts: dict[str, int] = {}
    for span in trace.spans:
        outcome = span.metadata["rigor"]["outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def scores_by_name(trace: Any) -> dict[str, Any]:
    return {score.name: score for score in trace.feedback_scores}


# --------------------------------------------------------------------------- #
# with the extra missing
# --------------------------------------------------------------------------- #


def test_the_module_imports_with_the_opik_extra_absent() -> None:
    # This file's own top-level import already proves it in the venv without
    # Opik; re-importing by name states the guarantee explicitly, so that a
    # future top-level `import opik` fails here rather than at collection time
    # for the whole suite.
    module = importlib.import_module("rigor.integrations.opik")

    assert callable(module.log_sample_to_opik)
    assert callable(module.log_assertion_to_opik)
    assert issubclass(module.OpikIntegrationError, Exception)


@pytest.mark.parametrize(
    ("call", "description"),
    [
        (lambda: log_sample_to_opik(all_passing_sample(2), name="s"), "log_sample_to_opik"),
        (lambda: log_assertion_to_opik({"passed": True}, trace_id="t"), "log_assertion_to_opik"),
    ],
)
def test_calling_either_function_without_opik_names_the_extra_and_the_pip_command(
    monkeypatch: pytest.MonkeyPatch, call: Any, description: str
) -> None:
    # Absence is simulated rather than assumed: `import opik` with None parked in
    # sys.modules raises ImportError, which is exactly what an uninstalled extra
    # raises, so this runs identically in both venvs.
    monkeypatch.setitem(sys.modules, "opik", None)

    with pytest.raises(OpikIntegrationError) as excinfo:
        call()

    message = str(excinfo.value)
    assert "[opik]" in message, f"{description} did not name the extra: {message}"
    assert 'pip install "opik-rigor[opik]"' in message


def test_importing_the_module_does_not_import_opik() -> None:
    # In a subprocess because sys.modules in this one is already contaminated by
    # whatever else the session imported -- in the venv that has Opik, its own
    # pytest plugin imports it before collection even starts. A fresh
    # interpreter is the only place the claim can actually be checked.
    code = (
        "import sys; import rigor.integrations.opik; "
        "print('IMPORTED' if 'opik' in sys.modules else 'ABSENT')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "ABSENT", completed.stdout + completed.stderr


# --------------------------------------------------------------------------- #
# with opik installed
# --------------------------------------------------------------------------- #


def test_a_sample_becomes_one_trace_with_one_span_per_run(opik_module: Any, client: Any) -> None:
    result = mixed_sample(SHUFFLE_SEED)

    with opik_module.record_traces_locally(client) as storage:
        trace_id = log_sample_to_opik(
            result, name="one-span-per-run", client=client, tags=["nightly"]
        )
        trace = trace_by_id(storage, trace_id)

    assert isinstance(trace_id, str) and trace_id
    assert len(trace.spans) == len(result.runs)
    assert {span.type for span in trace.spans} == {"llm"}
    assert sorted(span.metadata["rigor"]["run_index"] for span in trace.spans) == list(
        range(len(result.runs))
    )
    # The summary rides on the trace twice on purpose: as output because it is
    # what the sample produced, as metadata because that is where it is
    # filterable.
    assert trace.output == result.summary()
    assert trace.metadata["rigor"]["sample"] == result.summary()
    assert trace.tags == ["rigor", "nightly"]


def test_a_run_that_raised_is_distinguishable_from_a_run_that_failed(
    opik_module: Any, client: Any
) -> None:
    result = mixed_sample(SHUFFLE_SEED)
    assert (result.successes, result.failures, len(result.exceptions)) == (4, 3, 2)

    with opik_module.record_traces_locally(client) as storage:
        trace_id = log_sample_to_opik(result, name="three-outcomes", client=client)
        trace = trace_by_id(storage, trace_id)

    assert span_outcomes(trace) == {"passed": 4, "failed": 3, "raised": 2}

    raised = [span for span in trace.spans if span.metadata["rigor"]["outcome"] == "raised"]
    failed = [span for span in trace.spans if span.metadata["rigor"]["outcome"] == "failed"]
    assert len(raised) == 2

    for span in raised:
        # The exception's identity, not just the fact of one. During an outage
        # "ProviderOutage: upstream returned 503" is the whole diagnosis, and
        # "this run did not pass" is worthless.
        assert span.error_info["exception_type"] == "ProviderOutage"
        assert OUTAGE_MESSAGE in span.error_info["message"]
        assert "ProviderOutage" in span.error_info["traceback"]
        assert span.metadata["rigor"]["error_type"] == "ProviderOutage"
        assert span.metadata["rigor"]["error_message"] == OUTAGE_MESSAGE
        assert span.metadata["rigor"]["raised"] is True
        assert "rigor:raised" in span.tags
        # And it must not read as an ordinary failure anywhere a filter looks.
        assert "rigor:failed" not in span.tags
        assert span.output["outcome"] == "raised"

    for span in failed:
        # The converse: a genuine failure carries no error, or an error-rate
        # dashboard would count model quality as provider breakage.
        assert span.error_info is None
        assert span.metadata["rigor"]["raised"] is False
        assert span.metadata["rigor"]["error_type"] is None
        assert "rigor:failed" in span.tags
        assert "rigor:raised" not in span.tags

    assert trace.metadata["rigor"]["outcomes"] == {"passed": 4, "failed": 3, "raised": 2}


def test_a_passing_gate_attaches_feedback_scores_naming_the_gate(
    opik_module: Any, client: Any
) -> None:
    result = all_passing_sample(40)
    report = assert_pass_rate(result, 0.5, label="nightly")
    assert report["passed"] is True

    with opik_module.record_traces_locally(client) as storage:
        trace_id = log_sample_to_opik(result, name="passing-gate", client=client)
        log_assertion_to_opik(report, trace_id=trace_id, client=client)
        trace = trace_by_id(storage, trace_id)

    scores = scores_by_name(trace)
    assert f"{SCORE_PREFIX}pass_rate.passed" in scores
    assert f"{SCORE_PREFIX}pass_rate.lower_bound" in scores
    assert scores[f"{SCORE_PREFIX}pass_rate.passed"].value == 1.0
    assert scores[f"{SCORE_PREFIX}pass_rate.lower_bound"].value == pytest.approx(
        report["lower_bound"]
    )
    assert all(score.id == trace_id for score in trace.feedback_scores)
    assert "pass_rate gate 'nightly' passed" in scores[f"{SCORE_PREFIX}pass_rate.passed"].reason


def test_a_failing_gate_scores_zero_and_carries_the_explanation_as_the_reason(
    opik_module: Any, client: Any
) -> None:
    # Ten runs, all passing, gated at 0.9: the observed rate clears the bar and
    # the lower bound does not. That is the library's whole argument, and it is
    # the kind of failure whose explanation a dashboard has to carry -- "0.0"
    # alone would read as a broken model rather than an underpowered sample.
    result = all_passing_sample(10)
    with pytest.raises(PassRateError) as excinfo:
        assert_pass_rate(result, 0.9, label="underpowered")
    report = {**excinfo.value.stats, "message": str(excinfo.value)}

    with opik_module.record_traces_locally(client) as storage:
        trace_id = log_sample_to_opik(result, name="failing-gate", client=client)
        log_assertion_to_opik(report, trace_id=trace_id, client=client)
        trace = trace_by_id(storage, trace_id)

    scores = scores_by_name(trace)
    passed_score = scores[f"{SCORE_PREFIX}pass_rate.passed"]
    assert passed_score.value == 0.0
    assert scores[f"{SCORE_PREFIX}pass_rate.lower_bound"].value == pytest.approx(
        report["lower_bound"]
    )

    reason = passed_score.reason
    assert "underpowered" in reason
    assert "min_rate" in reason
    assert reason == scores[f"{SCORE_PREFIX}pass_rate.lower_bound"].reason
    assert len(reason) <= 1000


def test_a_report_without_a_verdict_is_refused_rather_than_guessed(
    opik_module: Any, client: Any
) -> None:
    # A report with numbers but no `passed` is not a gate report. Inferring the
    # verdict from the numbers would put a result on the dashboard that no gate
    # ever produced.
    with pytest.raises(ValueError, match="passed"):
        log_assertion_to_opik(
            {"gate": "pass_rate", "lower_bound": 0.4}, trace_id="t", client=client
        )


def test_numpy_scalars_are_converted_before_they_cross_the_wire(
    opik_module: Any, client: Any
) -> None:
    # numpy leaks in from rigor.distribution's reports and from any caller who
    # computes their own metadata. np.int64 is not a Python int and a JSON
    # encoder refuses it -- and a refused payload is a *dropped* record, not a
    # raised one, so the loss would be invisible on the dashboard.
    numpy = pytest.importorskip("numpy")
    result = all_passing_sample(3)

    with opik_module.record_traces_locally(client) as storage:
        trace_id = log_sample_to_opik(
            result,
            name="json-safety",
            client=client,
            metadata={"draws": numpy.int64(7), "mean": numpy.float32(0.5)},
        )
        trace = trace_by_id(storage, trace_id)

    assert type(trace.metadata["draws"]) is int
    assert type(trace.metadata["mean"]) is float
    # The real assertion: the whole payload survives an encoder.
    json.dumps({"metadata": trace.metadata, "output": trace.output})


def test_a_real_judged_sample_round_trips_into_opik(
    opik_module: Any, client: Any, tmp_path: Path
) -> None:
    # The end-to-end shape: a genuinely stochastic judge over a real rubric,
    # sampled, then logged. One of the scripted responses is unparseable, so the
    # judge raises JudgeOutputError and the sample carries all three outcomes
    # without any of them being staged.
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    script = [
        '{"pass": true, "score": 5, "reason": "faithful and complete"}',
        '{"pass": true, "score": 4, "reason": "minor secondary point missing"}',
        '{"pass": false, "score": 2, "reason": "omits the stated caveat"}',
        "I would rather not answer in JSON.",
    ]
    judge = PinnedJudge(FakeAdapter(responses=script, seed=JUDGE_SEED), RUBRIC, log, name="round")

    result = sample(
        lambda: judge.evaluate("Summarise the memo.", "A summary."),
        60,
        evidence=log,
        label="round-trip",
    )
    # The fixture has to actually produce all three, or this asserts nothing.
    assert result.successes > 0
    assert result.failures > 0
    assert len(result.exceptions) > 0

    with opik_module.record_traces_locally(client) as storage:
        trace_id = log_sample_to_opik(
            result,
            name="round-trip",
            client=client,
            judge_name=judge.name,
            metadata={"rubric_hash": judge.rubric_hash},
        )
        trace = trace_by_id(storage, trace_id)

    assert len(trace.spans) == len(result.runs) == 60
    assert span_outcomes(trace) == {
        "passed": result.successes,
        "failed": result.failures,
        "raised": len(result.exceptions),
    }
    assert trace.metadata["rigor"]["judge_name"] == "round"
    assert trace.metadata["rubric_hash"] == judge.rubric_hash
    assert trace.output["pass_rate"] == pytest.approx(result.pass_rate)

    # Durations survive as data even though no timestamps are invented for them.
    durations = [span.metadata["rigor"]["duration_seconds"] for span in trace.spans]
    assert len(durations) == 60
    assert all(isinstance(value, float) for value in durations)
