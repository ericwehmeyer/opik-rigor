"""What the *built wheel* contains, as opposed to what the source tree contains.

Every other test in this suite imports ``opik_rigor`` from an editable install,
which resolves into ``src/``. That is exactly the wrong instrument for a
packaging question: ``src/opik_rigor/py.typed`` existing proves nothing about
whether ``hatchling`` copied it into the artifact a consumer installs, and a test
that reads the source tree while claiming to check the wheel is worse than no
test, because it reports success for the case it cannot see.

So these tests build a wheel into a temporary directory and read the zip. They
also import out of an *extracted* wheel with that directory first on ``sys.path``
and assert the module actually loaded from there -- a check that has to be made
explicitly, because a developer's own ``src/`` will otherwise fill in silently
whatever the wheel omitted, and the test would pass on a wheel that ships
nothing.

If ``build`` is unavailable the tests skip with a reason naming what went
unverified. A skip is not a pass, and the release procedure in PROGRESS.md
re-checks these same two paths against the artifact that is actually uploaded.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Paths inside the wheel that are data rather than code, and are therefore the
#: ones a build backend can silently drop without breaking an import.
DATA_FILES_THE_WHEEL_MUST_CARRY = (
    "opik_rigor/py.typed",
    "opik_rigor/rubrics/example-rubric.md",
)

#: Modules the README hands a reader as a runnable address. Code rather than data,
#: so no build backend is going to drop them by accident -- but the *address* is the
#: fragile part, and it has been wrong once already: the quickstart's last line said
#: ``python examples/summarise_eval.py``, and ``examples/`` is a directory in the git
#: tree that appears in no wheel. Naming the member here means a move back out of the
#: package fails a test instead of failing a stranger.
MODULES_THE_WHEEL_MUST_CARRY = (
    "opik_rigor/examples/__init__.py",
    "opik_rigor/examples/summarise_eval.py",
)

#: Names 0.1.1 promises at the package root. Checked against the wheel rather
#: than against ``opik_rigor.__all__`` so that a name which is exported but not
#: shipped cannot pass.
NAMES_THE_PACKAGE_ROOT_MUST_EXPORT = (
    "SCORE_MIN",
    "SCORE_MAX",
    "hash_rubric_file",
    "hash_rubric_text",
    "example_rubric_path",
)


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a wheel from this checkout and return the path to it.

    Session-scoped: a wheel build is seconds, and every test here wants the same
    one. ``--no-isolation`` when the backend is already importable, because that
    is offline and deterministic; otherwise the isolated build, which fetches
    ``hatchling`` and therefore needs a network or a populated pip cache.
    """
    if not PYPROJECT.is_file():
        pytest.skip(
            f"not a source checkout ({PYPROJECT} does not exist), so the wheel "
            f"contents of {', '.join(DATA_FILES_THE_WHEEL_MUST_CARRY)} are UNVERIFIED"
        )
    if importlib.util.find_spec("build") is None:
        pytest.skip(
            "the `build` package is not installed, so the wheel contents of "
            f"{', '.join(DATA_FILES_THE_WHEEL_MUST_CARRY)} are UNVERIFIED -- "
            "install the dev extra (`pip install -e '.[dev]'`) to run this check"
        )

    outdir = tmp_path_factory.mktemp("wheel")
    command = [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)]
    if importlib.util.find_spec("hatchling") is not None:
        command.append("--no-isolation")

    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"`{' '.join(command)}` failed with exit {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    wheels = sorted(outdir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel in {outdir}, got {wheels}"
    return wheels[0]


@pytest.fixture(scope="session")
def extracted_wheel(built_wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The built wheel unpacked, as an installer would lay it down on disk."""
    target = tmp_path_factory.mktemp("unpacked")
    with zipfile.ZipFile(built_wheel) as archive:
        archive.extractall(target)
    return target


@pytest.mark.parametrize("member", DATA_FILES_THE_WHEEL_MUST_CARRY)
def test_the_built_wheel_carries_the_data_files_the_package_promises(
    built_wheel: Path, member: str
) -> None:
    # py.typed is the PEP 561 marker: without it inside the *wheel*, a type
    # checker discards every annotation in an installed copy no matter how many
    # of them the source tree has. The rubric is what `example_rubric_path()`
    # returns; 0.1.0's wheel shipped neither, which is the whole reason both are
    # named here rather than assumed.
    with zipfile.ZipFile(built_wheel) as archive:
        members = archive.namelist()

    assert member in members, (
        f"{member} is missing from {built_wheel.name}; it exists in the source "
        f"tree, so this is a build-configuration fault, not a missing file. "
        f"Wheel contains: {sorted(members)}"
    )


@pytest.mark.parametrize("member", MODULES_THE_WHEEL_MUST_CARRY)
def test_the_built_wheel_carries_the_worked_example_the_readme_tells_people_to_run(
    built_wheel: Path, member: str
) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        members = archive.namelist()

    assert member in members, (
        f"{member} is missing from {built_wheel.name}. The README's quickstart ends "
        f"on `python -m opik_rigor.examples.summarise_eval`, which is only runnable "
        f"by someone who ran `pip install opik-rigor` if the module is in the wheel. "
        f"Wheel contains: {sorted(members)}"
    )


def test_the_worked_example_runs_out_of_the_extracted_wheel(extracted_wheel: Path) -> None:
    """`-m` against the wheel and nothing else, which is what a reader has.

    Not a duplicate of ``examples/test_example_runs.py``: that runs the example
    through whatever ``opik_rigor`` the developer's environment resolves, which in a
    checkout is ``src/``. This one puts the extracted wheel on the path and refuses
    any other answer, so it fails on the day the example stops being packaged even
    though the source tree still has it.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opik_rigor.examples.summarise_eval",
            "--seed",
            "7",
            # The walkthrough's own default. Below roughly 30 the headline gate stops
            # the example early by design, which would make this test's exit code a
            # statement about the sample size rather than about the packaging.
            "--n",
            "40",
        ],
        capture_output=True,
        text=True,
        cwd=extracted_wheel,
        env={**_clean_env(), "PYTHONPATH": str(extracted_wheel)},
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "rubric sha256" in result.stdout


def test_the_public_names_import_out_of_the_wheel_and_not_out_of_src(
    extracted_wheel: Path,
) -> None:
    # The point of the subprocess is the assertion about __file__. Putting the
    # extracted wheel first on sys.path is not by itself proof that the wheel is
    # what answered: a developer running this has an editable opik_rigor on the
    # same path, and a test that only checks `hasattr` would pass identically if
    # the wheel were empty. So the child reports where it loaded from, and the
    # parent refuses any answer that points outside the extraction directory.
    code = (
        "import json, pathlib, opik_rigor;"
        "print(json.dumps({"
        "'file': str(pathlib.Path(opik_rigor.__file__).resolve()),"
        "'rubric': str(opik_rigor.example_rubric_path().resolve()),"
        "'missing': [n for n in " + repr(list(NAMES_THE_PACKAGE_ROOT_MUST_EXPORT)) + " "
        "if not hasattr(opik_rigor, n)],"
        "'version': opik_rigor.__version__,"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT.parent,
        env={**_clean_env(), "PYTHONPATH": str(extracted_wheel)},
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    report = json.loads(result.stdout)
    loaded_from = Path(report["file"])
    rubric = Path(report["rubric"])

    assert loaded_from.is_relative_to(extracted_wheel), (
        f"opik_rigor was imported from {loaded_from}, which is not inside the "
        f"extracted wheel at {extracted_wheel} -- the source tree answered instead, "
        f"so this test proves nothing about the wheel"
    )
    assert rubric.is_relative_to(extracted_wheel), (
        f"example_rubric_path() returned {rubric}, outside the extracted wheel"
    )
    assert rubric.is_file()
    assert report["missing"] == []


def _clean_env() -> dict[str, str]:
    """The parent environment minus anything that would preload a second copy."""
    import os

    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


# ----------------------------------------------------------------------------------
# The sdist: what must NOT be in it
# ----------------------------------------------------------------------------------

#: Path fragments that must never appear in a source distribution. Each is a
#: directory some tool creates inside the checkout and nobody intends to publish.
#: `.claude/worktrees` holds a full second copy of the tree; `.remember` holds
#: session memory, which is conversational content rather than source.
FRAGMENTS_THE_SDIST_MUST_NOT_CARRY = (".claude", ".remember", ".venv", "node_modules")


@pytest.fixture(scope="session")
def built_sdist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build an sdist from this checkout and return the path to it."""
    if not PYPROJECT.is_file():
        pytest.skip(f"not a source checkout ({PYPROJECT} does not exist)")
    if importlib.util.find_spec("build") is None:
        pytest.skip(
            "the `build` package is not installed, so the sdist contents are "
            "UNVERIFIED -- install the dev extra (`pip install -e '.[dev]'`)"
        )

    outdir = tmp_path_factory.mktemp("sdist")
    command = [sys.executable, "-m", "build", "--sdist", "--outdir", str(outdir)]
    if importlib.util.find_spec("hatchling") is not None:
        command.append("--no-isolation")

    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"`{' '.join(command)}` failed with exit {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    archives = sorted(outdir.glob("*.tar.gz"))
    assert len(archives) == 1, f"expected exactly one sdist in {outdir}, got {archives}"
    return archives[0]


@pytest.mark.parametrize("fragment", FRAGMENTS_THE_SDIST_MUST_NOT_CARRY)
def test_the_sdist_carries_nothing_from_the_working_directory(built_sdist: Path, fragment: str):
    """The sdist must not sweep in directories that only exist while working here.

    Every other packaging check in this file asks whether something the package
    promised is *present*. Nothing asked whether something nobody promised had
    been included, and that asymmetry shipped a real defect: with no
    ``[tool.hatch.build.targets.sdist]`` declared, hatchling fell back to
    "everything ``.gitignore`` does not exclude", and these directories are
    excluded in ``.git/info/exclude``, which hatchling does not read. A local
    ``python -m build`` produced a 122-member sdist of which 71 members were a
    second full copy of the tree under ``.claude/worktrees/`` plus 25
    ``.remember/`` files.

    The published 0.1.1 sdist is clean, verified by downloading it: CI builds from
    a fresh checkout where none of these directories exist. That is the point of
    this test. The artifact was safe by accident of the build environment, and an
    accident is not a guarantee -- the first release built on a developer's
    machine would have published the lot.
    """
    import tarfile

    with tarfile.open(built_sdist) as archive:
        offenders = [name for name in archive.getnames() if fragment in name]

    assert offenders == [], (
        f"the sdist contains {len(offenders)} member(s) under {fragment!r}, which is "
        f"working-directory content and not source. First few: {offenders[:5]}"
    )
