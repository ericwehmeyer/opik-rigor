"""The release gate needs tools installed, and only the workflow can install them.

`scripts/verify_release.py` is the gate that decides whether an artifact ships.
Several of its checks need a tool that is not a runtime dependency of this
package -- `wheel-annotations` runs `mypy --strict` against the extracted wheel to
establish that the `py.typed` marker is telling a downstream checker the truth.
When mypy is absent that row does not fail, it **skips**, and the script exits 2
rather than 0 precisely so a gate cannot read a skip as green.

That is the right design and it has one consequence worth a test: the gate's
correctness now depends on something outside the gate, in a file the gate never
reads. On 2026-08-14 `publish.yml` ran the gate with a plain `pip install -e .`
while `ci.yml` installed `.[typecheck]`. CI was green across eight matrix cells,
the release checks reported *16 passed, 0 failed*, and 0.2.0 stopped one step
before the upload. `ci.yml` had carried a comment predicting exactly that failure
since the extra was introduced; nothing propagated it to the workflow where it
decides whether a release ships.

**What these tests do not do.** There is no YAML parser in this project's
dependencies and one is not worth adding for this, so the check is file-scoped,
not job-scoped: it asserts that a workflow *file* which runs the gate also
installs the typechecker somewhere. A workflow that installed the extra in one
job and ran the gate in a different one would pass this test and fail in CI. That
is a real gap and it is stated here rather than papered over -- the regression
this exists to catch was a whole file missing the install, which is the shape it
does catch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The extra that carries the tools the gate's checks need. Named here rather than
#: spelled inline at each use so the test and the workflows have one shared name to
#: drift from, instead of three.
TYPECHECK_EXTRA = "typecheck"

#: The gate. A workflow that names this script is making a release decision with
#: it, whatever the job is called.
GATE_SCRIPT = "scripts/verify_release.py"

#: `pip install` with the extra attached, in any of the spellings the workflows
#: use: `-e ".[typecheck]"`, `-e .[typecheck]`, `.[typecheck]`. The quoting varies
#: because some shells eat the brackets and some do not.
INSTALLS_EXTRA = re.compile(
    r"pip\s+install\b[^\n]*\.\[[^\]]*\b" + re.escape(TYPECHECK_EXTRA) + r"\b[^\]]*\]"
)


def _workflow_files() -> list[Path]:
    if not WORKFLOW_DIR.is_dir():
        return []
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def _workflows_running_the_gate() -> list[Path]:
    return [p for p in _workflow_files() if GATE_SCRIPT in p.read_text(encoding="utf-8")]


def test_the_workflow_directory_is_where_this_test_thinks_it_is() -> None:
    """Guard the guard: if the path is wrong every other test here passes vacuously.

    A file-scanning test that finds no files reports success, which is the failure
    mode this whole module exists to argue against.
    """
    if not WORKFLOW_DIR.is_dir():
        pytest.skip(
            f"no {WORKFLOW_DIR.relative_to(REPO_ROOT)} in this tree -- workflow "
            "invariants are UNVERIFIED (expected when running from an install "
            "rather than a checkout)"
        )
    assert _workflow_files(), f"{WORKFLOW_DIR} exists but holds no workflow files"


def test_every_workflow_that_runs_the_release_gate_installs_the_typechecker() -> None:
    """The regression of 2026-08-14, pinned.

    Both `ci.yml` and `publish.yml` run `verify_release.py`; only one of them
    installed mypy, and the one that did not was the one that gates the upload.
    """
    if not WORKFLOW_DIR.is_dir():
        pytest.skip("no .github/workflows in this tree -- UNVERIFIED")

    gating = _workflows_running_the_gate()
    assert gating, (
        f"no workflow runs {GATE_SCRIPT}. Either the gate stopped running in CI, "
        "or it was renamed and this test is now checking nothing."
    )

    missing = [p.name for p in gating if not INSTALLS_EXTRA.search(p.read_text(encoding="utf-8"))]
    assert not missing, (
        f"{', '.join(missing)} runs {GATE_SCRIPT} but never installs "
        f"`.[{TYPECHECK_EXTRA}]`. Without it the `wheel-annotations` check SKIPS, "
        "the script exits 2, and the job fails with 0 failures reported -- which "
        "reads as a broken release rather than an unverified one. This is the "
        "defect that stopped the 0.2.0 publish."
    )


def test_the_typecheck_extra_still_provides_mypy() -> None:
    """The other half of the same invariant.

    The test above proves the workflows ask for the extra. It cannot prove the
    extra still delivers the tool: emptying or renaming it would leave every
    workflow installing something that no longer supplies mypy, and the gate would
    go back to skipping with nothing to point at.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        pytest.skip("tomllib needs Python 3.11+, so the extra's contents are UNVERIFIED")

    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    assert TYPECHECK_EXTRA in extras, (
        f"pyproject.toml has no `{TYPECHECK_EXTRA}` extra, but the workflows "
        "install it. `pip install .[missing-extra]` is a warning, not an error, so "
        "this drift would surface as a skipped check rather than a failed install."
    )
    requirements = extras[TYPECHECK_EXTRA]
    names = [req.split(">=")[0].split("[")[0].strip() for req in requirements]
    assert "mypy" in names, (
        f"the `{TYPECHECK_EXTRA}` extra no longer carries mypy: {requirements}. "
        "The `wheel-annotations` check needs it and skips without it."
    )
