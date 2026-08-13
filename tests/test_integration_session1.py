"""End-to-end wiring of the Session 1 modules.

The unit suites deliberately test each module against a local double: the judge
tests script their own adapter, and the adapter tests never construct a judge.
That keeps them independent, but it leaves one thing unproven -- that the real
:class:`~rigor.FakeAdapter`, the real :class:`~rigor.PinnedJudge`, and the real
:class:`~rigor.EvidenceLog` actually fit together. These tests cover that seam,
and the package-level invariants that no single module owns.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import rigor
from rigor import (
    EvidenceLog,
    FakeAdapter,
    JudgeOutputError,
    ModelPinError,
    PinnedJudge,
    RubricDriftError,
)
from rigor.evidence import (
    EVENT_JUDGE_INIT,
    EVENT_JUDGE_PARSE_FAILURE,
    EVENT_JUDGE_VERDICT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_RUBRIC = REPO_ROOT / "rubrics" / "example-rubric.md"

PASS_RESPONSE = '{"pass": true, "score": 5, "reason": "faithful and complete"}'
FAIL_RESPONSE = '{"pass": false, "score": 2, "reason": "omits the stated caveat"}'


def judge_on(
    tmp_path: Path,
    responses: list[str],
    *,
    rubric: Path | None = None,
    **kwargs: object,
) -> tuple[PinnedJudge, EvidenceLog, FakeAdapter]:
    """A judge wired to a real FakeAdapter and a real log, as a caller would."""
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    adapter = FakeAdapter(responses=responses)
    judge = PinnedJudge(adapter, rubric or SHIPPED_RUBRIC, log, **kwargs)  # type: ignore[arg-type]
    return judge, log, adapter


# --------------------------------------------------------------------------- #
# the seam
# --------------------------------------------------------------------------- #


def test_the_shipped_rubric_and_the_real_fake_adapter_produce_a_recorded_verdict(
    tmp_path: Path,
) -> None:
    judge, log, adapter = judge_on(tmp_path, [PASS_RESPONSE])

    verdict = judge.evaluate("Summarise the memo.", "The memo says X, with caveat Y.")

    assert verdict.passed is True
    assert verdict.score == 5.0
    assert verdict.model_id == adapter.model_id

    # The whole audit trail for this run: one init, one verdict, nothing else.
    records = log.read()
    assert [record.event_type for record in records] == [EVENT_JUDGE_INIT, EVENT_JUDGE_VERDICT]
    assert records[1].payload["rubric_hash"] == judge.rubric_hash


def test_the_prompt_the_adapter_actually_received_contains_the_shipped_rubric(
    tmp_path: Path,
) -> None:
    judge, _log, adapter = judge_on(tmp_path, [PASS_RESPONSE])

    judge.evaluate("Summarise the memo.", "A summary.")

    (prompt,) = adapter.calls
    assert "faithful summarisation" in prompt.lower()
    assert "Summarise the memo." in prompt
    assert '"pass"' in prompt  # the output-format contract reached the model


def test_a_default_fake_adapter_is_pinned_enough_for_a_judge(tmp_path: Path) -> None:
    # FakeAdapter's default model id has to satisfy the judge's pin rule, or every
    # example in the docs would need a special case.
    judge, _log, _adapter = judge_on(tmp_path, [PASS_RESPONSE])
    assert rigor.is_pinned(judge.model_id)


def test_an_unpinned_adapter_is_refused_however_it_is_wired(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    adapter = FakeAdapter(model_id="claude-3-5-sonnet-latest", responses=[PASS_RESPONSE])

    with pytest.raises(ModelPinError):
        PinnedJudge(adapter, SHIPPED_RUBRIC, log)

    assert log.read() == []  # a refused judge records nothing


# --------------------------------------------------------------------------- #
# drift, against the real rubric file
# --------------------------------------------------------------------------- #


def test_editing_the_rubric_between_runs_stops_the_second_run(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.md"
    rubric.write_text(SHIPPED_RUBRIC.read_text(encoding="utf-8"), encoding="utf-8")
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    PinnedJudge(FakeAdapter(responses=[PASS_RESPONSE]), rubric, log, name="summariser")

    rubric.write_text("# A different rubric\n\nPass if it is short.\n", encoding="utf-8")

    with pytest.raises(RubricDriftError):
        PinnedJudge(FakeAdapter(responses=[PASS_RESPONSE]), rubric, log, name="summariser")


def test_the_evidence_log_survives_a_rubric_change_that_was_declared(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.md"
    rubric.write_text("# v1\n\nPass if faithful.\n", encoding="utf-8")
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    first = PinnedJudge(FakeAdapter(responses=[PASS_RESPONSE]), rubric, log, name="s")
    first.evaluate("in", "out")

    rubric.write_text("# v2\n\nPass if faithful and complete.\n", encoding="utf-8")
    second = PinnedJudge(
        FakeAdapter(responses=[FAIL_RESPONSE]), rubric, log, name="s", accept_rubric_change=True
    )
    second.evaluate("in", "out")

    # A reader of the log can tell which verdicts were graded on which rubric,
    # which is the only reason accepting a change is allowed at all.
    verdicts = [r for r in log.read() if r.event_type == EVENT_JUDGE_VERDICT]
    assert [v.payload["rubric_hash"] for v in verdicts] == [first.rubric_hash, second.rubric_hash]
    assert first.rubric_hash != second.rubric_hash


# --------------------------------------------------------------------------- #
# failure paths across the seam
# --------------------------------------------------------------------------- #


def test_a_garbled_provider_response_is_recorded_and_not_scored(tmp_path: Path) -> None:
    judge, log, _adapter = judge_on(tmp_path, ["I think it's probably fine, honestly."])

    with pytest.raises(JudgeOutputError):
        judge.evaluate("in", "out")

    events = [record.event_type for record in log.read()]
    assert EVENT_JUDGE_PARSE_FAILURE in events
    # The critical invariant: missing data never becomes a failing verdict.
    assert EVENT_JUDGE_VERDICT not in events


def test_a_provider_outage_propagates_rather_than_scoring_zero(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    adapter = FakeAdapter(
        responses=[PASS_RESPONSE, PASS_RESPONSE],
        fail_with=rigor.AdapterError("provider is down"),
        fail_after=1,
    )
    judge = PinnedJudge(adapter, SHIPPED_RUBRIC, log)
    judge.evaluate("in", "out")

    with pytest.raises(rigor.AdapterError):
        judge.evaluate("in", "out")

    # Session 2 note: the sampler owns provider failures, so the judge writes no
    # evidence for one. Exactly one verdict was recorded, from the call that worked.
    verdicts = [r for r in log.read() if r.event_type == EVENT_JUDGE_VERDICT]
    assert len(verdicts) == 1


# --------------------------------------------------------------------------- #
# concurrency across the seam
# --------------------------------------------------------------------------- #


def test_concurrent_evaluations_record_one_verdict_each(tmp_path: Path) -> None:
    # Session 2's sampler will drive exactly this shape. If either the adapter's
    # cursor or the log's append were unguarded, the count would come up short.
    n = 60
    log = EvidenceLog(tmp_path / "evidence.jsonl")
    adapter = FakeAdapter(responses=[PASS_RESPONSE, FAIL_RESPONSE], cycle=True)
    judge = PinnedJudge(adapter, SHIPPED_RUBRIC, log)

    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = list(pool.map(lambda i: judge.evaluate(f"in {i}", f"out {i}"), range(n)))

    assert len(verdicts) == n
    assert adapter.call_count == n
    assert len([r for r in log.read() if r.event_type == EVENT_JUDGE_VERDICT]) == n
    assert all(json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines())


def test_a_seeded_adapter_makes_a_judge_run_reproducible(tmp_path: Path) -> None:
    # The property Session 2's statistical gates depend on: a stochastic judge
    # whose randomness is fixed, so a failing gate is a bug and not a coin flip.
    def run(directory: Path) -> list[bool]:
        log = EvidenceLog(directory / "evidence.jsonl")
        adapter = FakeAdapter(responses=[PASS_RESPONSE, FAIL_RESPONSE], seed=1234)
        judge = PinnedJudge(adapter, SHIPPED_RUBRIC, log)
        return [judge.evaluate("in", "out").passed for _ in range(40)]

    first = run(tmp_path / "a")
    second = run(tmp_path / "b")

    assert first == second
    assert len(set(first)) == 2  # and it genuinely varied, or the test proves nothing


# --------------------------------------------------------------------------- #
# package-level invariants
# --------------------------------------------------------------------------- #


def test_constructing_a_real_adapter_needs_no_provider_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Invariant 3 in PROGRESS.md, stated as a property of rigor rather than of the
    # environment: an adapter can be *constructed* without its SDK, because the
    # import is deferred to complete(). The earlier version asserted the SDKs were
    # not installed, which broke as soon as a venv installed opik (which depends on
    # openai) -- it was measuring the venv, not the library.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")

    adapter = rigor.AnthropicAdapter("claude-sonnet-4-5-20250929")

    assert adapter.model_id == "claude-sonnet-4-5-20250929"


def test_importing_rigor_in_a_clean_interpreter_pulls_in_no_integrations() -> None:
    # Invariant 1: core never imports integrations, and no provider SDK is loaded
    # as a side effect of importing the package. A subprocess is the only honest
    # way to check this -- this test session has already imported half the world.
    code = (
        "import sys, rigor; "
        "leaked = sorted(m for m in sys.modules "
        "if m.startswith(('rigor.integrations', 'anthropic', 'openai', 'opik'))); "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert result.stdout.strip() == ""


def test_every_public_name_is_importable_from_the_package_root() -> None:
    missing = [name for name in rigor.__all__ if not hasattr(rigor, name)]
    assert missing == []
    assert rigor.__version__ == importlib.metadata.version("opik-rigor")
