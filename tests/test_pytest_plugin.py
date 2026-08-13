"""The pytest plugin, tested by running pytest inside pytest.

Every assertion here is about what a *real* pytest run does with the plugin
loaded, not about the hook functions in isolation. That is the only level at which
the interesting questions can be asked -- does the marker survive
``--strict-markers``, does the failure message reach the terminal, does Opik's
plugin sit quietly next to ours -- and the built-in ``pytester`` fixture makes it
cheap enough to ask them all.

The plugin is loaded by its ``pytest11`` entry point, exactly as a user's install
would load it -- see :data:`PLUGIN_ARGS`. Everything here keeps working unchanged after
that, since an entry-point plugin and a ``-p``-loaded one are the same object.

Every random draw comes from an explicit ``random.Random(seed)``. The expected
counts below are recomputed from that same seed rather than pasted in, so a
changed seed changes the expectation instead of turning the test red -- a suite
about flakiness cannot afford to be flaky, and it cannot afford magic numbers
either.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from opik_rigor import EvidenceLog, wilson_lower_bound
from opik_rigor.evidence import EVENT_ASSERTION, EVENT_SAMPLE_COMPLETED

pytest_plugins = ["pytester"]

#: How the inner runs load the plugin under test -- nothing, now that
#: ``pyproject.toml`` declares the ``pytest11`` entry point and pytest loads it
#: automatically. Passing ``-p opik_rigor.integrations.pytest_plugin`` as well is not
#: merely redundant, it is fatal: pluggy registers the module under the dotted
#: name, then entry-point loading tries to register the *same module object*
#: under the entry-point name ``rigor`` and raises "Plugin already registered under a
#: different name". Kept as a named empty tuple so the call sites still read as
#: "however the plugin gets loaded", and so re-adding an argument is one edit.
PLUGIN_ARGS: tuple[str, ...] = ()

#: Every generated module is named ``test_inner_*``. An in-process pytester run
#: shares ``sys.modules`` with this suite, so a generated ``test_evidence.py``
#: collides with the real ``tests/test_evidence.py`` -- which shows up only when
#: the whole suite runs, not when this file runs alone.

#: Sample size for the repeated bodies. Small enough to keep the inner runs fast,
#: large enough that the Wilson bound is not degenerate.
N = 20

#: Seed for the flaky body. Named so a failure is reproducible by hand.
FLAKY_SEED = 20260813
FLAKY_RATE = 0.3

#: A bar the flaky body cannot clear, so the gate's failure output is on screen.
HIGH_BAR = 0.9

RUBRIC_TEXT = """
# Faithfulness

Score 5 if the response is faithful to the input, 1 if it invents facts.
Answer with a single JSON object and nothing else.
"""

FLAKY_SOURCE = """
import random
import pytest

# A private Random, seeded: never bare random.*, which another test could reseed.
RNG = random.Random({seed})


@pytest.mark.rigor_repeat(n={n}, min_rate={min_rate})
def test_flaky_body():
    assert RNG.random() < {rate}
"""

EXPLODING_SOURCE = """
import pytest


@pytest.mark.rigor_repeat(n={n}, min_rate={min_rate})
def test_harness_breaks():
    raise RuntimeError("the provider never answered")
"""

SUMMARY_PATTERN = re.compile(
    r"\[rigor_repeat\] (?P<nodeid>\S+) runs=(?P<runs>\d+) successes=(?P<successes>\d+) "
    r"failures=(?P<failures>\d+) exceptions=(?P<exceptions>\d+)"
)


def expected_successes(seed: int = FLAKY_SEED, n: int = N, rate: float = FLAKY_RATE) -> int:
    """Replay the flaky body's RNG to learn how many of its n runs pass.

    Recomputed rather than hard-coded: the number is a consequence of the seed,
    and a test that asserts a pasted constant is asserting that nobody has
    touched the seed, which is a different (and less useful) claim.
    """
    rng = random.Random(seed)
    return sum(1 for _ in range(n) if rng.random() < rate)


def flaky_source(min_rate: float = HIGH_BAR) -> str:
    """The seeded flaky body, gated at ``min_rate``."""
    return FLAKY_SOURCE.format(seed=FLAKY_SEED, n=N, min_rate=min_rate, rate=FLAKY_RATE)


def summaries(stdout: str) -> dict[str, dict[str, int]]:
    """The plugin's per-test breakdown lines, keyed by test name."""
    keys = ("runs", "successes", "failures", "exceptions")
    found = {}
    for match in SUMMARY_PATTERN.finditer(stdout):
        name = match.group("nodeid").rsplit("::", 1)[-1]
        found[name] = {key: int(match.group(key)) for key in keys}
    return found


# --------------------------------------------------------------------------- #
# the marker
# --------------------------------------------------------------------------- #


def test_a_marked_test_runs_n_times_and_reports_one_result(pytester: pytest.Pytester) -> None:
    # The contract in one test: n executions of the body, one line in the report.
    # A repeat plugin that reported n results would make the suite's own pass
    # count a function of the sample size.
    pytester.makepyfile(
        test_inner_deterministic="""
        import pathlib
        import pytest

        LEDGER = pathlib.Path(__file__).with_name("ledger.txt")


        @pytest.mark.rigor_repeat(n=20, min_rate=0.5)
        def test_always_passes():
            with LEDGER.open("a", encoding="utf-8") as handle:
                handle.write("ran\\n")
        """
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(passed=1)
    ledger = (pytester.path / "ledger.txt").read_text(encoding="utf-8").split()
    assert len(ledger) == 20


def test_fixtures_still_reach_the_body_on_every_repeat(pytester: pytest.Pytester) -> None:
    # Repeating the call must not repeat fixture setup: a function-scoped fixture
    # is set up once per test, and the body sees the same object each time.
    pytester.makepyfile(
        test_inner_fixtures="""
        import pathlib

        import pytest

        SETUPS = pathlib.Path(__file__).with_name("setups.txt")
        CALLS = pathlib.Path(__file__).with_name("calls.txt")


        def note(path):
            with path.open("a", encoding="utf-8") as handle:
                handle.write("x\\n")


        @pytest.fixture
        def gadget(tmp_path):
            note(SETUPS)
            return {"scratch": tmp_path}


        @pytest.mark.rigor_repeat(n=12, min_rate=0.5)
        def test_uses_a_fixture(gadget):
            note(CALLS)
            assert gadget["scratch"].exists()
        """
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(passed=1)
    assert len((pytester.path / "setups.txt").read_text(encoding="utf-8").split()) == 1
    assert len((pytester.path / "calls.txt").read_text(encoding="utf-8").split()) == 12


def test_a_flaky_body_fails_the_gate_with_the_statistical_report(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(test_inner_flaky=flaky_source())

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(failed=1)
    successes = expected_successes()
    assert 0 < successes < N  # otherwise this test is measuring a constant
    lower = wilson_lower_bound(successes, N)
    stdout = result.stdout.str()

    # The failure message is the statistical report: the counts, the bound that
    # was gated on, and the bar it missed -- not "assert 0.3 >= 0.9".
    assert f"{successes}/{N} passed" in stdout
    assert f"Wilson lower bound {lower:.4f}" in stdout
    assert f"min_rate {HIGH_BAR:.4f}" in stdout
    assert "more runs will not fix it" in stdout


def test_an_exception_is_counted_apart_from_a_failure(pytester: pytest.Pytester) -> None:
    # sampling.py's central thesis, enforced at the plugin boundary: a body that
    # asserts its way to a false verdict is a *failure* of the system under test,
    # while a body that raises anything else is an *exception* -- the harness
    # broke and produced no observation. pytest calls both "a failed test", so
    # this is exactly where the distinction would get flattened.
    pytester.makepyfile(
        test_inner_flaky=flaky_source(),
        test_inner_exploding=EXPLODING_SOURCE.format(n=N, min_rate=HIGH_BAR),
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(failed=2)
    reported = summaries(result.stdout.str())
    failing = reported["test_flaky_body"]
    exploding = reported["test_harness_breaks"]

    assert failing == {
        "runs": N,
        "successes": expected_successes(),
        "failures": N - expected_successes(),
        "exceptions": 0,
    }
    assert exploding == {"runs": N, "successes": 0, "failures": 0, "exceptions": N}
    assert exploding["failures"] != failing["failures"]
    assert exploding["exceptions"] != failing["exceptions"]
    # And the exceptions are visible as exceptions, with the original traceback.
    assert "RuntimeError" in result.stdout.str()


def test_control_flow_out_of_the_body_is_not_an_observation(pytester: pytest.Pytester) -> None:
    # A skip means "this test does not apply", which is a statement about all n
    # runs at once. Counting it -- in either bucket -- would invent measurements.
    pytester.makeini(
        """
        [pytest]
        rigor_evidence_path = evidence/skipped.jsonl
        """
    )
    pytester.makepyfile(
        test_inner_skipping="""
        import pytest


        @pytest.mark.rigor_repeat(n=50, min_rate=0.99)
        def test_skips_immediately(rigor_evidence):
            pytest.skip("no provider configured")
        """
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(skipped=1)
    # And nothing was recorded: a sample.completed record claiming 50 runs would
    # be a fabricated measurement of a body that ran once and bowed out.
    assert not (pytester.path / "evidence" / "skipped.jsonl").exists()


def test_the_marker_is_registered_for_strict_markers(pytester: pytest.Pytester) -> None:
    # This repo runs --strict-markers, so an unregistered marker is a hard error
    # rather than a warning. The plugin must register its own.
    pytester.makepyfile(
        test_inner_strict="""
        import pytest


        # Positional spelling, which the marker accepts as a prefix of
        # (n, min_rate, confidence, errors_as_failures).
        @pytest.mark.rigor_repeat(5, 0.5)
        def test_marked():
            pass
        """
    )

    result = pytester.runpytest("--strict-markers", *PLUGIN_ARGS)

    result.assert_outcomes(passed=1)
    assert "not registered" not in result.stdout.str()

    listing = pytester.runpytest("--markers", *PLUGIN_ARGS)
    assert "rigor_repeat" in listing.stdout.str()


def test_a_misspelled_marker_argument_is_refused(pytester: pytest.Pytester) -> None:
    # A dropped min_rate would leave the test passing on a gate nobody set, which
    # is the failure mode the whole library exists to prevent.
    pytester.makepyfile(
        test_inner_typo="""
        import pytest


        @pytest.mark.rigor_repeat(n=5, minimum_rate=0.9)
        def test_typo():
            pass
        """
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(failed=1)
    assert "unexpected argument(s): minimum_rate" in result.stdout.str()


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def test_the_evidence_fixture_writes_under_tmp_path(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner_evidence="""
        import pathlib


        def test_records_survive_to_the_file(rigor_evidence, tmp_path):
            assert rigor_evidence.path.parent == tmp_path

            rigor_evidence.append("demo.event", {"answer": 42})
            records = rigor_evidence.read()

            assert [record.event_type for record in records] == ["demo.event"]
            assert records[0].payload == {"answer": 42}
            pathlib.Path(__file__).with_name("where.txt").write_text(
                str(rigor_evidence.path), encoding="utf-8"
            )
        """
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(passed=1)
    # Read the log from the outside: the records outlived the test that wrote them.
    written = Path((pytester.path / "where.txt").read_text(encoding="utf-8"))
    assert written.exists()
    assert [record.event_type for record in EvidenceLog(written).read()] == ["demo.event"]


def test_the_ini_option_redirects_the_log_and_appends(pytester: pytest.Pytester) -> None:
    # Append rather than truncate: the log is append-only by construction, and a
    # per-test truncation would reduce a suite-wide audit trail to whatever the
    # last test happened to write.
    pytester.makeini(
        """
        [pytest]
        rigor_evidence_path = evidence/suite.jsonl
        """
    )
    pytester.makepyfile(
        test_inner_shared="""
        def test_first(rigor_evidence):
            rigor_evidence.append("demo.event", {"who": "first"})


        def test_second(rigor_evidence):
            rigor_evidence.append("demo.event", {"who": "second"})
        """
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(passed=2)
    log = pytester.path / "evidence" / "suite.jsonl"
    assert log.exists()
    assert [record.payload["who"] for record in EvidenceLog(log).read()] == ["first", "second"]


def test_the_judge_fixture_builds_a_judge_that_records_its_verdict(
    pytester: pytest.Pytester,
) -> None:
    pytester.makefile(".md", rubric=RUBRIC_TEXT)
    pytester.makepyfile(
        test_inner_judge="""
        import pathlib

        from opik_rigor import FakeAdapter

        RUBRIC = pathlib.Path(__file__).with_name("rubric.md")
        VERDICT = '{"pass": true, "score": 5, "reason": "faithful to the source"}'


        def test_a_judged_response(rigor_judge, rigor_evidence):
            judge = rigor_judge(
                FakeAdapter(responses=[VERDICT], cycle=True),
                RUBRIC,
                name="demo",
            )

            verdict = judge.evaluate("Summarise the memo.", "A summary.")

            assert verdict.passed is True
            assert verdict.score == 5.0
            assert verdict.rubric_hash == judge.rubric_hash
            recorded = rigor_evidence.last("judge.verdict", judge="demo")
            assert recorded is not None
            assert recorded.payload["passed"] is True
        """
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(passed=1)


def test_a_repeated_test_records_its_sample_and_gate_in_the_evidence_log(
    pytester: pytest.Pytester,
) -> None:
    # A test that asks for the evidence log gets the repeat itself recorded in it,
    # so the log reconstructs the run rather than the judge calls alone.
    pytester.makeini(
        """
        [pytest]
        rigor_evidence_path = evidence/repeat.jsonl
        """
    )
    pytester.makepyfile(
        test_inner_recorded="""
        import pytest


        @pytest.mark.rigor_repeat(n=8, min_rate=0.5)
        def test_recorded(rigor_evidence):
            pass
        """
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(passed=1)
    records = EvidenceLog(pytester.path / "evidence" / "repeat.jsonl").read()
    events = [record.event_type for record in records]
    assert events.count(EVENT_SAMPLE_COMPLETED) == 1
    assert events.count(EVENT_ASSERTION) == 1
    sampled = next(r for r in records if r.event_type == EVENT_SAMPLE_COMPLETED)
    assert sampled.payload["runs"] == 8
    assert sampled.payload["label"].endswith("::test_recorded")


# --------------------------------------------------------------------------- #
# co-installation with Opik's own pytest plugin
# --------------------------------------------------------------------------- #


@pytest.mark.requires_opik
def test_both_plugins_load_and_collect_together(pytester: pytest.Pytester) -> None:
    # The check the build plan calls for. Opik registers a pytest11 entry point
    # named `opik`; ours is named `opik_rigor`. A subprocess run is used so both are
    # loaded the way a user's environment loads them -- from entry points -- and
    # not because this process happens to have imported them already.
    pytest.importorskip("opik")
    pytester.makefile(".md", rubric=RUBRIC_TEXT)
    pytester.makepyfile(
        test_inner_together="""
        import pathlib

        import pytest
        from opik_rigor import FakeAdapter

        RUBRIC = pathlib.Path(__file__).with_name("rubric.md")
        VERDICT = '{"pass": true, "score": 4, "reason": "close enough"}'


        def test_opik_plugin_is_actually_loaded(pytestconfig):
            assert pytestconfig.pluginmanager.hasplugin("opik")


        @pytest.mark.rigor_repeat(n=10, min_rate=0.5)
        def test_rigor_marker_still_works(rigor_judge):
            judge = rigor_judge(FakeAdapter(responses=[VERDICT], cycle=True), RUBRIC)
            assert judge.evaluate("in", "out").passed
        """
    )

    result = pytester.runpytest_subprocess(*PLUGIN_ARGS)

    assert result.ret == 0
    result.assert_outcomes(passed=2)
    assert "error" not in result.stdout.str().lower()


@pytest.mark.requires_opik
def test_opik_stays_inert_when_no_llm_unit_tests_are_collected(
    pytester: pytest.Pytester,
) -> None:
    # opik_rigor must never switch Opik on: no --opik, no opik_pytest_enabled. With
    # neither set and no llm_unit-decorated test collected, Opik's plugin does
    # nothing at all, so installing opik_rigor changes nothing about an Opik user's
    # suite (and vice versa).
    pytest.importorskip("opik")
    pytester.makepyfile(
        test_inner_inert="""
        import pytest


        def test_opik_was_not_switched_on(pytestconfig):
            assert pytestconfig.getoption("opik") is False
            assert pytestconfig.getini("opik_pytest_enabled") is False


        @pytest.mark.rigor_repeat(n=10, min_rate=0.5)
        def test_rigor_runs_unaffected():
            pass
        """
    )

    assert "--opik" not in PLUGIN_ARGS

    result = pytester.runpytest_subprocess(*PLUGIN_ARGS)

    assert result.ret == 0
    result.assert_outcomes(passed=2)
    # Opik prints this panel only when its plugin is active.
    assert "Opik: LLM Test Results" not in result.stdout.str()


@pytest.mark.requires_opik
def test_the_plugin_never_imports_opik(pytester: pytest.Pytester) -> None:
    # The rule that keeps opik_rigor's core independent of a vendor SDK, checked where
    # it is easiest to break: an entry-point plugin is imported in every pytest
    # process on the machine, so an import of opik here would be a hard dependency
    # in disguise.
    pytest.importorskip("opik")
    pytester.makepyfile(
        test_inner_no_opik="""
        import sys


        def test_opik_is_not_imported_by_rigors_plugin():
            for name in list(sys.modules):
                if name == "opik" or name.startswith("opik."):
                    del sys.modules[name]

            import opik_rigor.integrations.pytest_plugin  # noqa: F401

            assert not [name for name in sys.modules if name.split(".")[0] == "opik"]
        """
    )

    # -p no:opik keeps Opik's own plugin from importing opik on our behalf; the
    # question here is only what opik_rigor's module pulls in.
    result = pytester.runpytest_subprocess("-p", "no:opik", *PLUGIN_ARGS)

    result.assert_outcomes(passed=1)


def test_the_evidence_and_judge_fixtures_exist_without_opik_installed(
    pytester: pytest.Pytester,
) -> None:
    # The unmarked counterpart of the tests above: this one runs in the main venv
    # too, where opik is not installed at all.
    pytester.makepyfile(
        test_inner_fixture_listing="""
        def test_fixtures_are_available(request):
            for name in ("rigor_evidence", "rigor_judge"):
                assert name in request.fixturenames or request.getfixturevalue(name)
        """
    )

    result = pytester.runpytest(*PLUGIN_ARGS)

    result.assert_outcomes(passed=1)
