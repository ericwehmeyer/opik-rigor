"""The example has to keep working, so it is run rather than imported.

Importing ``summarise_eval`` and calling its functions would test the pieces and
miss the thing that matters: that ``python examples/summarise_eval.py`` finishes,
offline, on a machine with no Opik and no credentials, and prints a walkthrough
that still says what the README claims it says. So every test here starts a
subprocess and reads its stdout, exactly as a reader would.

Fast by construction: the sampled "model" is a plain function and the judge is a
scripted fake, so a 40-run walkthrough costs about a second, nearly all of it
interpreter startup.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent / "summarise_eval.py"

#: Enough runs for the headline gate to clear its bar. Below roughly 30 the
#: Wilson lower bound sits under 0.60 and the example stops early *by design* --
#: which is the library's own lesson, and a bad exit criterion for this test.
N = 40
SEED = 7


def run_example(
    *args: str, out: Path, seed: int = SEED
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(EXAMPLE),
        "--seed",
        str(seed),
        "--n",
        str(N),
        "--out",
        str(out),
        *args,
    ]
    return subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)


def normalised(completed: subprocess.CompletedProcess[str], out: Path) -> str:
    """stdout with the run directory masked.

    ``--out`` differs between two runs of the same seed, and it is printed --
    deliberately, since a reader has to be able to go and look at the evidence
    log. Masking it is what makes "identical byte for byte" a claim about the
    measurement rather than about the filesystem.
    """
    return completed.stdout.replace(str(out), "<out>")


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory: pytest.TempPathFactory) -> subprocess.CompletedProcess[str]:
    """One offline run, shared by the assertions about its output."""
    return run_example(out=tmp_path_factory.mktemp("run"))


def test_the_offline_example_runs_to_completion(
    offline_run: subprocess.CompletedProcess[str],
) -> None:
    assert offline_run.returncode == 0, offline_run.stdout + offline_run.stderr


@pytest.mark.parametrize(
    "phrase",
    [
        # the pinned judge refused an alias, and pinned its rubric
        "an alias is refused, at construction time",
        "rubric sha256",
        # it sampled rather than measured once
        "observed pass rate",
        "a point estimate; never gate on it",
        # both statistical gates ran and reported
        "Wilson lower bound",
        "score distribution gate",
        # both ways a pass-rate gate can fail, distinguishable on screen
        "underpowered sample, not a demonstrated failure",
        "the system missed the bar, and more runs will not fix it",
        # the baseline and the regression it caught
        "baseline written",
        "regression gate 'summariser-nightly' failed",
        "Mann-Whitney U",
        # the audit trail
        "evidence.jsonl",
        "judge.verdict",
        "assertion.evaluated",
    ],
)
def test_the_walkthrough_says_what_it_is_supposed_to_say(
    offline_run: subprocess.CompletedProcess[str], phrase: str
) -> None:
    # Whitespace is collapsed first: the example wraps its prose to 96 columns,
    # so a phrase that is present on screen can still straddle a newline. The
    # test is about what the reader is told, not about where the lines break.
    assert phrase in " ".join(offline_run.stdout.split())


def test_the_offline_run_needs_no_opik(offline_run: subprocess.CompletedProcess[str]) -> None:
    # Not "opik is not installed" -- the example must not touch the integration
    # without --opik even on a machine that has it.
    assert "mirroring the run into Opik" not in offline_run.stdout


def test_the_same_seed_reproduces_the_output_byte_for_byte(tmp_path: Path) -> None:
    # If this fails, every number the example prints is a coin flip and the
    # walkthrough in examples/README.md is fiction.
    first = run_example(out=tmp_path / "first")
    second = run_example(out=tmp_path / "second")

    assert first.returncode == second.returncode == 0
    assert normalised(first, tmp_path / "first") == normalised(second, tmp_path / "second")


def test_a_different_seed_gives_different_numbers(tmp_path: Path) -> None:
    # The complement of the test above: identical output from every seed would
    # also be "deterministic", and would mean the fixture had stopped varying.
    seeded = run_example(out=tmp_path / "a")
    other = run_example(out=tmp_path / "b", seed=1234)

    assert other.returncode == 0, other.stdout + other.stderr
    assert normalised(seeded, tmp_path / "a") != normalised(other, tmp_path / "b")


def test_the_opik_leg_never_fails_the_script(tmp_path: Path) -> None:
    # The contract for the optional leg: with no opik installed, or opik
    # installed and nothing listening, the run still exits 0 and the evaluation
    # above it is unaffected.
    completed = run_example("--opik", out=tmp_path / "opik")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "mirroring the run into Opik" in completed.stdout
    assert "-- done " in completed.stdout
