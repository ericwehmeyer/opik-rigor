"""What ``import opik_rigor`` is allowed to drag in, and what it is not.

Importing this package used to cost 1018.6 ms of warm import against a ~40 ms
interpreter floor, because :mod:`opik_rigor.distribution` imported
``scipy.stats`` at module scope -- which pulls ``scipy.optimize``,
``scipy.spatial``, ``scipy.sparse`` and ``scipy.linalg`` behind it. Every suite
paid that, including the overwhelming majority that only ever call
:func:`~opik_rigor.assert_pass_rate`.

That cost is now paid by exactly one caller, :func:`~opik_rigor.assert_no_regression`,
which is the only function in the package that genuinely needs SciPy. The tests
below are the thing that keeps it that way: an import-cost fix is a single line
away from being silently undone by the next person who adds
``from scipy.stats import something`` to the top of a module, and nothing else in
this suite would notice, because the package works perfectly well either way.

They are written as subprocess checks against ``sys.modules`` rather than as
timings. A wall-clock assertion would be flaky on a loaded CI box and would tell
you the symptom rather than the cause; ``"scipy" in sys.modules`` is the cause,
and it is a fact rather than a measurement.

The second group checks the arithmetic consequence of the same change: the Wilson
z now comes from :class:`statistics.NormalDist`, so scipy's ``norm.ppf`` -- which
is no longer used in ``src/`` at all -- becomes available here as an *independent*
oracle for it, which is a better position than the module was in before.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scipy.stats import norm

import opik_rigor
from opik_rigor.distribution import _runs_needed, _wilson, _z, wilson_interval, wilson_lower_bound

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Every scipy subpackage the old module-scope ``from scipy.stats import ...``
#: pulled in. Naming them individually means a failure says which one came back.
SCIPY_SUBPACKAGES = ("scipy", "scipy.stats", "scipy.optimize", "scipy.sparse", "scipy.linalg")


def _modules_after(statement: str) -> set[str]:
    """Run ``statement`` in a fresh interpreter, return its ``sys.modules`` keys.

    A subprocess is not an optimisation here, it is the whole method: this test
    module has already imported scipy for its own oracle, so asking the current
    process whether scipy is loaded can only ever answer "yes".
    """
    code = (
        "import sys, json\n"
        f"{statement}\n"
        "print(json.dumps(sorted(sys.modules)))"
    )
    # Point the child at the *same* opik_rigor the parent imported. Without this
    # the child resolves through whatever editable install happens to be on the
    # machine, and the test would silently report on a different working tree.
    env = dict(os.environ)
    package_parent = str(Path(opik_rigor.__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{package_parent}{os.pathsep}{existing}" if existing else package_parent
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = set(json.loads(completed.stdout.splitlines()[-1]))
    assert "opik_rigor" in loaded
    return loaded


# --------------------------------------------------------------------------- #
# what the import path is allowed to contain
# --------------------------------------------------------------------------- #


def test_importing_the_package_does_not_import_scipy() -> None:
    loaded = _modules_after("import opik_rigor")
    still_there = sorted(name for name in SCIPY_SUBPACKAGES if name in loaded)
    assert not still_there, (
        f"import opik_rigor pulled in {still_there}. The whole point of the deferred "
        f"import in distribution._mannwhitneyu is that this list stays empty; some "
        f"module has grown a module-scope scipy import again."
    )


@pytest.mark.parametrize(
    ("gate", "statement"),
    [
        (
            "assert_pass_rate",
            "import opik_rigor; opik_rigor.assert_pass_rate((19, 20), min_rate=0.5)",
        ),
        (
            "assert_score_distribution",
            "import opik_rigor; opik_rigor.assert_score_distribution("
            "[4.0, 5.0, 4.0, 3.0, 5.0], min_mean=1.0)",
        ),
        (
            "wilson_lower_bound",
            "import opik_rigor; opik_rigor.wilson_lower_bound(18, 20)",
        ),
    ],
)
def test_the_gates_that_do_not_need_scipy_never_load_it(gate: str, statement: str) -> None:
    # These three are the ones on the hot path, and this is the assertion that
    # says the fix went all the way. Deferring the scipy import while leaving
    # norm.ppf on the pass-rate path was measured at 1096.7 ms for
    # `import + assert_pass_rate` against the 1070.6 ms it was meant to improve:
    # the module avoided the import and the first gate call performed it anyway.
    loaded = _modules_after(statement)
    assert "scipy" not in loaded, f"{gate} loaded scipy"


def test_the_regression_gate_does_load_scipy() -> None:
    # The other half of the claim, and the one that would otherwise rot silently:
    # if someone "fixed" the deferred import by reimplementing Mann-Whitney, this
    # test fails and says so. scipy's implementation carries the tie-corrected
    # ranks and the exact null distribution, and this suite has no independent
    # oracle for the p-value -- so a reimplementation must not slip in quietly.
    loaded = _modules_after(
        "import opik_rigor; opik_rigor.assert_no_regression([4.0, 5.0, 4.0], [4.0, 5.0, 4.0])"
    )
    assert "scipy.stats" in loaded


def test_numpy_is_a_declared_dependency_and_not_an_accident() -> None:
    # distribution.py imports numpy directly and pins mean/p10/stddev to numpy's
    # exact semantics in a public docstring. Before this change it was undeclared
    # and worked only because scipy happened to require it -- an edge that
    # disappears the day scipy moves behind an extra.
    if not PYPROJECT.is_file():
        pytest.skip(
            f"not a source checkout ({PYPROJECT} does not exist), so the declared "
            f"dependencies are UNVERIFIED"
        )
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10, which this package still supports
        pytest.skip("tomllib needs Python 3.11+, so the declared dependencies are UNVERIFIED")
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    assert any(spec.startswith("numpy") for spec in dependencies), dependencies
    # scipy stays a *hard* dependency: the regression gate imports it lazily but
    # still requires it, so `pip install opik-rigor` must keep bringing it.
    assert any(spec.startswith("scipy") for spec in dependencies), dependencies
    extras = metadata["project"].get("optional-dependencies", {})
    for name, specs in extras.items():
        assert not any(spec.startswith("scipy") for spec in specs), (
            f"scipy appears in the {name!r} extra; moving it out of the base install "
            f"is a runtime break for assert_no_regression and waits for 0.2."
        )


# --------------------------------------------------------------------------- #
# the arithmetic is unchanged: stdlib z against scipy's, as an oracle
# --------------------------------------------------------------------------- #

#: Enough of the open interval to be a real check rather than a lookup table:
#: both tails, the values a caller actually types, and the far reaches where a
#: series approximation would fall apart.
PROBABILITIES = [
    1e-12, 1e-8, 1e-6, 1e-4, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999,
    1 - 1e-6, 1 - 1e-8, 1 - 1e-12,
]


@pytest.mark.parametrize("p", PROBABILITIES)
def test_the_stdlib_quantile_agrees_with_scipys_to_the_last_few_ulp(p: float) -> None:
    # statistics.NormalDist.inv_cdf is CPython's Wichura AS241 and scipy's ndtri
    # is the Cephes routine; they are two implementations of the same function,
    # not the same code, so this is a genuine cross-check. Swept over 6.4M points
    # of (0, 1) the worst disagreement was 1.22e-15 relative, 8 ULP.
    assert _z(p) == pytest.approx(float(norm.ppf(p)), rel=1e-14, abs=1e-13)


# 0.51 rather than 0.5 at the low end: the one-sided bound refuses a confidence at
# or below a half, where its z is zero or negative and the bound stops being a
# floor. What this parameter is for is the small-z end of the quantile, and 0.51
# reaches it.
@pytest.mark.parametrize("confidence", [0.51, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999])
@pytest.mark.parametrize(
    ("successes", "n"), [(0, 1), (1, 2), (18, 20), (5, 6), (1, 120), (900, 1000)]
)
def test_the_reported_bound_is_unchanged_by_the_new_quantile(
    confidence: float, successes: int, n: int
) -> None:
    # Propagated through _wilson the quantile difference shrinks rather than
    # grows: the worst case over 87,486 bounds was 3.3e-16 absolute.
    from_scipy = _wilson(successes, n, float(norm.ppf(confidence)))
    from_stdlib = _wilson(successes, n, _z(confidence))
    assert from_stdlib[0] == pytest.approx(from_scipy[0], abs=1e-14)
    assert from_stdlib[1] == pytest.approx(from_scipy[1], abs=1e-14)
    assert wilson_lower_bound(successes, n, confidence) == pytest.approx(
        from_scipy[0], abs=1e-14
    )
    # wilson_interval is two-sided, so its z is the other one -- a detail worth
    # asserting separately rather than assuming, since mixing the two is exactly
    # the mistake the one-sided/two-sided docstrings exist to prevent.
    two_sided_from_scipy = _wilson(successes, n, float(norm.ppf(1.0 - (1.0 - confidence) / 2.0)))
    assert wilson_interval(successes, n, confidence) == pytest.approx(
        two_sided_from_scipy, abs=1e-14
    )


@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99])
@pytest.mark.parametrize("p", [0.91, 0.95, 0.99, 0.995])
@pytest.mark.parametrize("min_rate", [0.5, 0.8, 0.9])
def test_runs_needed_returns_the_same_integer(p: float, min_rate: float, confidence: float) -> None:
    # _runs_needed answers in whole runs, so a 1e-16 wobble in the bound has to
    # move the answer by a whole run to matter at all. Over 72,814 cases with a
    # defined answer, none did.
    if not p > min_rate:
        pytest.skip("no answer is defined when the observed rate is at or below the bar")
    answer = _runs_needed(p, min_rate, confidence)
    assert answer is not None
    # Recompute the same search with scipy's z injected, so the only variable is
    # the quantile source. This mirrors _runs_needed's body: binary search on the
    # bound at the least favourable rounding of p*n -- the one predicate here that
    # really is monotone in n -- then walk down to the last n that still failed.
    # Mirrored rather than shared on purpose: if the search itself changes, this
    # has to be changed too and read while it is, because the question it asks is
    # "did the quantile move the answer", not "does _runs_needed agree with
    # itself".
    z_scipy = float(norm.ppf(confidence))

    def guaranteed(size: int) -> bool:
        return _wilson(min(max(0, math.floor(p * size - 0.5)), size), size, z_scipy)[0] >= min_rate

    def clears(size: int) -> bool:
        return _wilson(min(round(p * size), size), size, z_scipy)[0] >= min_rate

    low, high = 1, 10_000_000
    assert guaranteed(high)
    while low < high:
        mid = (low + high) // 2
        if guaranteed(mid):
            high = mid
        else:
            low = mid + 1
    while low > 1 and clears(low - 1):
        low -= 1
    assert answer == low


def test_the_number_the_readme_prints_is_byte_identical() -> None:
    # The one honest residual of this change: wilson_lower_bound(18, 20) moves in
    # its last ULP, 0.7383369536731331 -> 0.7383369536731332. Every number this
    # package *prints* is formatted to four decimals, so the README's pasted
    # failure message -- and every failure message a user has ever read -- is
    # unchanged. This test pins the formatted form, which is the contract.
    assert format(wilson_lower_bound(18, 20), ".4f") == "0.7383"
    lower, upper = wilson_interval(18, 20)
    assert (format(lower, ".4f"), format(upper, ".4f")) == ("0.6990", "0.9721")


# --------------------------------------------------------------------------- #
# the error a deferred import makes newly visible
# --------------------------------------------------------------------------- #


class _BlockImport:
    """A meta-path finder that makes one package un-importable.

    Raising from ``find_spec`` rather than returning ``None`` is deliberate: a
    ``None`` would simply let the next finder on ``sys.meta_path`` succeed, and
    scipy is genuinely installed in this environment.
    """

    def __init__(self, blocked: str) -> None:
        self.blocked = blocked

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname == self.blocked or fullname.startswith(f"{self.blocked}."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def test_a_missing_scipy_names_itself_instead_of_leaking_a_bare_import_error() -> None:
    # Before this change a scipy-less environment failed at `import opik_rigor`,
    # where the traceback at least pointed at an import line. Deferring it moves
    # the failure into the middle of a gate, where a bare
    # "ModuleNotFoundError: No module named 'scipy'" reads like a bug in this
    # package rather than a broken install -- so the message has to say what is
    # missing, which gate wanted it, and that the other gates are fine.
    from opik_rigor import assert_no_regression

    saved = {name: mod for name, mod in sys.modules.items() if name.split(".")[0] == "scipy"}
    blocker = _BlockImport("scipy")
    sys.meta_path.insert(0, blocker)
    try:
        for name in saved:
            del sys.modules[name]
        with pytest.raises(ModuleNotFoundError) as caught:
            assert_no_regression([4.0, 5.0, 4.0], [4.0, 5.0, 4.0])
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)

    message = str(caught.value)
    assert "SciPy" in message
    assert "assert_no_regression" in message
    assert "pip install scipy" in message
    assert "assert_pass_rate" in message
    # A ModuleNotFoundError, so `except ImportError` around the call still catches
    # it -- the improvement is the text, not a new exception type to handle.
    assert isinstance(caught.value, ImportError)
    assert isinstance(caught.value.__cause__, ImportError)


def test_the_scipy_free_gates_still_work_with_scipy_blocked() -> None:
    # The claim that matters to a user who has not installed scipy yet: the
    # pass-rate and distribution gates are genuinely off it, not merely lazy
    # about it.
    from opik_rigor import assert_pass_rate, assert_score_distribution

    saved = {name: mod for name, mod in sys.modules.items() if name.split(".")[0] == "scipy"}
    blocker = _BlockImport("scipy")
    sys.meta_path.insert(0, blocker)
    try:
        for name in saved:
            del sys.modules[name]
        assert assert_pass_rate((19, 20), min_rate=0.5)["passed"] is True
        assert assert_score_distribution([4.0, 5.0, 4.0, 3.0, 5.0], min_mean=1.0)["passed"] is True
        assert wilson_lower_bound(18, 20) == pytest.approx(0.73833695367313, abs=1e-13)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
