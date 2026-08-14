"""Tests for scripts/verify_release.py.

The release script decides whether a release is allowed to happen, so its own
logic needs checking by something other than itself.

Two kinds of test live here.

* **Pure helpers.** Every expectation is derived by reading the rule, not by
  running the code. The README-scanning rules are the sibling project's
  `docs/readme-scan-contract.md`, frozen 2026-08-13; its hand-derived tables are
  reproduced below with this project's names substituted, and the new
  `readme_package_symbols` rule has its own table derived the same way.
* **Wheel-shaped checks**, driven by synthetic zip files rather than by invoking
  hatchling. The interesting cases are the *broken* ones -- a wheel with no
  `py.typed`, a wheel whose entry point targets a module it does not contain --
  and this repository's pyproject.toml cannot produce those on purpose.

Nothing here runs the isolated subprocess probe: that is verified by running the
script itself against a deliberately damaged wheel, which is a different kind of
evidence and belongs in the release record rather than in a unit test.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_release.py"
_README = Path(__file__).resolve().parent.parent / "README.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_release", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vr = _load_module()


# ----------------------------------------------------------------------------------
# Name and requirement parsing
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("opik-rigor", "opik-rigor"),
        ("opik_rigor", "opik-rigor"),
        ("Opik.Rigor", "opik-rigor"),
        ("OPIK--RIGOR", "opik-rigor"),
        ("  opik-rigor  ", "opik-rigor"),
    ],
)
def test_pep503_normalisation_collapses_the_spellings(raw, expected):
    """The distribution is `opik-rigor` and the import is `opik_rigor`; PEP 503 says
    the first two are one name, which is what lets the README be checked at all."""
    assert vr.normalize_project_name(raw) == expected


def test_opik_is_a_different_project_from_opik_rigor():
    """Do not let PEP 503's convenience become an assumption: `opik` is somebody
    else's distribution, and this library declares it as an optional dependency."""
    assert vr.normalize_project_name("opik") != vr.normalize_project_name("opik-rigor")


@pytest.mark.parametrize(
    ("requirement", "name"),
    [
        ("scipy>=1.10", "scipy"),
        ("opik>=2.0,<3", "opik"),
        ("opik-rigor[opik]", "opik-rigor"),
        ("pytest-cov>=4.0; extra == 'dev'", "pytest-cov"),
        ("ruff", "ruff"),
    ],
)
def test_requirement_name(requirement, name):
    assert vr.requirement_name(requirement) == name


@pytest.mark.parametrize(
    ("requirement", "extras"),
    [
        ("opik-rigor", []),
        ("opik-rigor[opik]", ["opik"]),
        ("opik-rigor[opik,pytest]", ["opik", "pytest"]),
        ("opik-rigor[opik] ; python_version < '3.11'", ["opik"]),
        # The bracket belongs to the marker, not to the requirement, so nothing is
        # claimed about extras here.
        ("scipy>=1.10", []),
    ],
)
def test_requirement_extras(requirement, extras):
    """A README that advertises `pip install "opik-rigor[opik]"` is making a claim
    about pyproject's optional-dependencies. An undeclared extra installs the bare
    package and gives the reader none of the integration they came for."""
    assert vr.requirement_extras(requirement) == extras


def test_requirement_marker_is_empty_when_unconditional():
    assert vr.requirement_marker("scipy>=1.10") == ""
    assert vr.requirement_marker("tomli>=2.0; python_version < '3.11'") == "python_version < '3.11'"


def test_markers_compare_across_quote_styles():
    """hatchling normalises quotes into METADATA and pyproject.toml does not, so a
    byte comparison would report a difference that does not exist."""
    assert vr.markers_equivalent("python_version < '3.11'", 'python_version < "3.11"')
    assert vr.markers_equivalent("python_version<'3.11'", "python_version < '3.11'")
    assert not vr.markers_equivalent("python_version < '3.11'", "python_version < '3.12'")
    assert not vr.markers_equivalent("", "python_version < '3.11'")


def test_extras_are_not_runtime_requirements():
    """`opik` and `pytest` are extras of this project. Counting them as runtime
    dependencies would report the wheel installing Opik for everybody, which is the
    exact thing COMPATIBILITY.md says must not happen."""
    runtime, extras = vr.split_requires_dist(
        [
            "scipy>=1.10",
            "opik<3,>=2.0; extra == 'opik'",
            "pytest>=7.0; extra == 'pytest'",
            "ruff>=0.6; extra == \"dev\"",
        ]
    )
    assert [vr.requirement_name(r) for r in runtime] == ["scipy"]
    assert [vr.requirement_name(r) for r in extras] == ["opik", "pytest", "ruff"]


# ----------------------------------------------------------------------------------
# Version parsing
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "is_dev"),
    [("0.1.0", False), ("0.1.1", False), ("0.1.0.dev0", True), ("0.2.0dev", True)],
)
def test_dev_versions_are_recognised(version, is_dev):
    assert vr.version_is_dev(version) is is_dev


def test_versions_are_read_out_of_both_filenames():
    """PEP 427 puts the version second in a wheel filename; an sdist puts it last."""
    assert vr.wheel_version_from_filename("opik_rigor-0.1.1-py3-none-any.whl") == "0.1.1"
    assert vr.sdist_version_from_filename("opik_rigor-0.1.1.tar.gz") == "0.1.1"


def test_dunder_version_is_read_as_text_not_imported(tmp_path):
    """Importing would need scipy and would answer with the *installed* version --
    which in this repository's own venv is a different tree entirely."""
    init = tmp_path / "__init__.py"
    init.write_text('x = 1\n__version__ = "0.1.1"\n', encoding="utf-8")
    assert vr.read_dunder_version(init) == "0.1.1"
    assert vr.read_dunder_version(tmp_path / "absent.py") is None


def test_dunder_all_is_read_as_text(tmp_path):
    init = tmp_path / "__init__.py"
    init.write_text('__all__ = [\n    "sample",\n    "SCORE_MIN",\n]\n', encoding="utf-8")
    assert vr._source_dunder_all(init) == ["sample", "SCORE_MIN"]


# ----------------------------------------------------------------------------------
# Licence identification
# ----------------------------------------------------------------------------------

MIT_TEXT = """MIT License

Copyright (c) 2026 Somebody

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
"""

APACHE_HEAD = """
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
"""


def test_mit_text_is_identified():
    assert vr.spdx_from_license_text(MIT_TEXT) == "MIT"


def test_apache_text_is_identified_so_a_swap_would_be_caught():
    """pyproject declares `license = "MIT"`. If the shipped LICENSE were Apache the
    declaration and the bytes would disagree, and that is the failure to detect."""
    assert vr.spdx_from_license_text(APACHE_HEAD) == "Apache-2.0"


def test_unknown_licence_text_yields_none_rather_than_a_guess():
    """A guess here would manufacture agreement between the declared identifier and
    the shipped bytes, which is the one thing this check exists to detect."""
    assert vr.spdx_from_license_text("Do what you like. Seriously, anything.") is None


# ----------------------------------------------------------------------------------
# Isolation: which sys.path entries a probe may keep
# ----------------------------------------------------------------------------------


# The entries below are built with the *native* separator rather than written as
# literals, because that is the only form the function will ever be handed: its
# input is the running interpreter's `sys.path`, and no interpreter puts a
# foreign-separator path there. Windows-literal paths were what this file used to
# assert on, and they passed on Windows and failed on all four Ubuntu cells --
# `Path(r"C:\x\site-packages").name` on POSIX is the whole string, because POSIX
# has no backslash separator, so the entry was dropped rather than kept. The
# property under test ("kept by directory name, not by location") is
# platform-independent; only the spelling of a path is not.
def _entry(*parts: str) -> str:
    return str(Path(*parts))


def test_only_site_packages_survives_as_a_dependency_path():
    """`src` is the entry an editable install of this repo adds, and it is the one
    that would answer every question about the wheel with the source tree. It is
    excluded by name, not by location, so this holds for any checkout."""
    site = _entry("repos", "opik-rigor", ".venv", "Lib", "site-packages")
    dist = _entry("usr", "lib", "python3", "dist-packages")
    kept = vr.dependency_paths(
        [
            "",
            _entry("repos", "opik-rigor", "src"),
            site,
            _entry("Python", "Lib"),
            dist,
        ]
    )
    assert kept == [site, dist]


def test_dependency_paths_are_deduplicated():
    entry = _entry("repos", "opik-rigor", ".venv", "Lib", "site-packages")
    assert vr.dependency_paths([entry, entry]) == [entry]


# ----------------------------------------------------------------------------------
# Reading another tool's human-facing output
# ----------------------------------------------------------------------------------


def test_a_colourised_twine_pass_is_still_counted():
    """The exact bytes GitHub Actions produced, which the check read as zero passes.

    `check_twine` counts lines ending in `PASSED`. `twine check` prints a bare
    `PASSED` on a Windows dev shell and a colour-wrapped one on CI, so the line
    ends in `\\x1b[0m` there and the count came out 0 -- a FAIL on a build that was
    fine. It passed on every machine it was written on and failed the first time it
    ran where it counts. Note the wrapped first line: twine breaks the path across
    lines too, so the word is not on the line that starts with `Checking`.
    """
    captured = (
        "Checking \n"
        "/home/runner/work/opik-rigor/opik-rigor/dist/"
        "opik_rigor-0.1.1-py3-none-any.whl: \x1b[32mPASSED\x1b[0m\n"
        "Checking /home/runner/work/opik-rigor/opik-rigor/dist/"
        "opik_rigor-0.1.1.tar.gz: \x1b[32mPASSED\x1b[0m\n"
    )
    lines = vr.plain_lines(captured)
    assert sum(1 for line in lines if line.strip().endswith("PASSED")) == 2
    assert "\x1b" not in "".join(lines)


def test_a_plain_twine_pass_is_unchanged():
    """The uncoloured form still has to count, or the strip fixed CI and broke every
    developer machine instead."""
    captured = "Checking dist/opik_rigor-0.1.1-py3-none-any.whl: PASSED\n"
    assert vr.plain_lines(captured) == ["Checking dist/opik_rigor-0.1.1-py3-none-any.whl: PASSED"]


def test_a_failure_is_not_turned_into_a_pass_by_stripping():
    """`FAILED` must survive the same treatment. A strip that ate the distinction
    would make this check report success on a broken artifact, which is worse than
    the defect it was written to fix."""
    captured = "Checking dist/x.whl: \x1b[31mFAILED\x1b[0m\n  `long_description` is missing\n"
    lines = vr.plain_lines(captured)
    assert sum(1 for line in lines if line.strip().endswith("PASSED")) == 0
    assert lines[0].endswith("FAILED")


def _probe(tmp_path):
    return vr.Probe(extract=tmp_path / "wheel-extract", workdir=tmp_path, deps=[])


def _elsewhere(*parts: str) -> str:
    """An absolute path that is definitely not inside the extraction directory.

    Absolute on whichever platform is running, which the Windows literals this
    replaced were not: on POSIX, `C:\\repos\\opik-rigor\\src` is a single *relative*
    filename with no separators, so those cases still refused the answer but for
    the wrong reason -- they exercised "a relative path" rather than "an absolute
    path somewhere else on the developer's machine", which is the case the probe
    exists to catch. `abspath` of a rooted path supplies the current drive on
    Windows and changes nothing on POSIX.
    """
    return os.path.abspath(os.sep + os.path.join(*parts))


def test_an_answer_from_the_extracted_wheel_is_accepted(tmp_path):
    probe = _probe(tmp_path)
    package = probe.extract / "opik_rigor"
    assert (
        probe.isolation_broken(
            {
                "package_path": [str(package)],
                "package_file": str(package / "__init__.py"),
            }
        )
        is None
    )


@pytest.mark.parametrize(
    ("answer", "needle"),
    [
        ({"package_path": [], "package_file": None}, "no __path__"),
        (
            {"package_path": [_elsewhere("repos", "opik-rigor", "src", "opik_rigor")],
             "package_file": None},
            "outside the extracted wheel",
        ),
        # A namespace package multiplexes: the developer's src/ silently supplies
        # whatever the wheel omitted. This is the failure mode the sibling project
        # was actually bitten by, and an answer produced this way proves nothing.
        (
            {"package_path": ["EXTRACT", _elsewhere("repos", "opik-rigor", "src", "opik_rigor")]},
            "outside the extracted wheel",
        ),
    ],
)
def test_an_answer_from_anywhere_else_is_refused(tmp_path, answer, needle):
    probe = _probe(tmp_path)
    package = probe.extract / "opik_rigor"
    answer = dict(answer)
    answer["package_path"] = [
        str(package) if p == "EXTRACT" else p for p in answer["package_path"]
    ]
    answer.setdefault("package_file", str(package / "__init__.py"))
    broken = probe.isolation_broken(answer)
    assert broken is not None
    assert needle in broken


def test_a_file_outside_the_extraction_is_refused_even_when_the_path_looks_right(tmp_path):
    """`__path__` can be right while `__file__` came from elsewhere; both are checked
    because either one alone can be satisfied by a tree that is not the wheel."""
    probe = _probe(tmp_path)
    broken = probe.isolation_broken(
        {
            "package_path": [str(probe.extract / "opik_rigor")],
            "package_file": _elsewhere("repos", "opik-rigor", "src", "opik_rigor", "__init__.py"),
        }
    )
    assert broken is not None
    assert "__file__" in broken


# ----------------------------------------------------------------------------------
# README: fenced code blocks (frozen contract, table F)
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("F1", "a\n```\nb\n```\nc", ["b\n"]),
        ("F2", "a\n~~~\nb\n~~~\nc", ["b\n"]),
        ("F3", "```bash\nb\n```", ["b\n"]),
        ("F4", "```\na\n~~~\nb\n```", ["a\n~~~\nb\n"]),
        ("F5", "```\nx\n", ["x\n"]),
        ("F6", "no fences here", []),
        ("F7", "```\na\n```\nprose\n```\nb\n```", ["a\n", "b\n"]),
        ("F8", "````\na\n```\nb\n````", ["a\n```\nb\n"]),
        ("F9", "    indented code", []),
        # Not in the frozen table: this repository is developed on Windows and its
        # history already contains one CRLF defect.
        ("F10", "a\r\n```\r\nb\r\n```\r\n", ["b\n"]),
    ],
)
def test_fenced_code_blocks(label, text, expected):
    assert vr.fenced_code_blocks(text) == expected, label


# ----------------------------------------------------------------------------------
# README: pip install lines (frozen contract, table P)
# ----------------------------------------------------------------------------------


def _fenced(*lines: str) -> str:
    return "```bash\n" + "\n".join(lines) + "\n```\n"


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("P1", "So `pip install opik-rigor` does not work today.", []),
        ("P2", _fenced("pip install opik-rigor"), ["opik-rigor"]),
        ("P3", _fenced(".venv/bin/python -m pip install ."), []),
        ("P4", _fenced('python -m pip install -e ".[dev]"'), []),
        ("P5", _fenced("# Windows: python -m pip install scipy"), ["scipy"]),
        ("P6", _fenced('echo "pip install nonsense"'), []),
        ("P7", _fenced("pip install scipy && pip install numpy"), ["scipy", "numpy"]),
        ("P8", _fenced("$ pip install opik-rigor>=0.1"), ["opik-rigor>=0.1"]),
        ("P9", _fenced("pip install ./dist/x.whl"), []),
        ("P10", _fenced("PS> py -m pip install scipy"), ["scipy"]),
        # New here, and the reason this returns requirements rather than bare names:
        # the extras are a separate claim and they are checked separately.
        ("P11", _fenced('pip install "opik-rigor[opik]"'), ["opik-rigor[opik]"]),
    ],
)
def test_readme_pip_install_requirements(label, text, expected):
    assert vr.readme_pip_install_requirements(text) == expected, label


def test_the_real_readme_installs_this_project_and_only_real_extras():
    """An oracle against the document itself, kept deliberately loose so that adding
    a paragraph does not break it: what must hold is that the README's install lines
    are about *this* distribution and no invented extra."""
    requirements = vr.readme_pip_install_requirements(_README.read_text(encoding="utf-8"))
    assert requirements, "the README used to tell people how to install this; it should still"
    for requirement in requirements:
        assert vr.normalize_project_name(vr.requirement_name(requirement)) == "opik-rigor"
        assert set(vr.requirement_extras(requirement)) <= {"opik", "pytest", "dev"}


# ----------------------------------------------------------------------------------
# README: symbols (new rule, table S, hand-derived from the docstring of
# readme_package_symbols)
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("S1", "opik_rigor.example_rubric_path()", ["example_rubric_path"]),
        # The trap this rule exists for: a pasted traceback is output, not
        # instruction. A colon is not a parenthesis. This is the sibling's `migkit:`
        # log-prefix problem wearing Python clothing, and this README has one.
        ("S2", "opik_rigor.distribution.PassRateError: pass rate gate failed: 18/20", []),
        (
            "S3",
            "from opik_rigor import EvidenceLog, FakeAdapter",
            ["EvidenceLog", "FakeAdapter"],
        ),
        ("S4", "from opik_rigor.judge import hash_rubric_text", ["judge.hash_rubric_text"]),
        ("S5", 'opik_rigor.judge.hash_rubric_text(b"x")', ["judge.hash_rubric_text"]),
        # A module named in prose is a mention. Only a call is a claim that the name
        # exists and is callable, which is why prose is scanned at all.
        ("S6", "reaching into `opik_rigor.judge` was the only honest option", []),
        (
            "S7",
            "python -c \"import opik_rigor, shutil; shutil.copy("
            "opik_rigor.example_rubric_path(), 'rubric.md')\"",
            ["example_rubric_path"],
        ),
        ("S8", "from opik_rigor import sample as take", ["sample"]),
        ("S9", "from opik_rigor import (\n    Baseline,\n    Verdict,\n)", ["Baseline", "Verdict"]),
        # A line a reader is being shown *not* to run is not a claim.
        ("S10", "# from opik_rigor import Nope", []),
        ("S11", ">>> from opik_rigor import sample", ["sample"]),
        ("S12", "import opik_rigor", []),
        # Duplicates collapse and the result is sorted, so the evidence lines are
        # stable between runs.
        (
            "S13",
            "opik_rigor.sample()\nopik_rigor.sample()\nopik_rigor.Baseline()",
            ["Baseline", "sample"],
        ),
    ],
)
def test_readme_package_symbols(label, text, expected):
    assert vr.readme_package_symbols(text) == expected, label


def test_the_real_readme_names_the_accessor_that_0_1_0_did_not_ship():
    """The defect this check was built for. The published 0.1.0 wheel contains no
    `example_rubric_path` anywhere in it, and the README's quickstart calls it."""
    symbols = vr.readme_package_symbols(_README.read_text(encoding="utf-8"))
    assert "example_rubric_path" in symbols


def test_the_real_readmes_pasted_traceback_is_not_read_as_a_claim():
    """`opik_rigor.distribution.PassRateError` appears in the README's pasted failure
    output. If the scanner read it as a call, the check would assert a name that is
    reached through a submodule the README never tells anyone to import."""
    symbols = vr.readme_package_symbols(_README.read_text(encoding="utf-8"))
    assert not [s for s in symbols if s.startswith("distribution.")]


# ----------------------------------------------------------------------------------
# entry_points.txt
# ----------------------------------------------------------------------------------


def test_entry_points_are_parsed_by_group():
    text = (
        "[pytest11]\nrigor = opik_rigor.integrations.pytest_plugin\n"
        "\n[console_scripts]\n# a comment\nx = m:f\n"
    )
    assert vr._parse_entry_points(text) == {
        "pytest11": [("rigor", "opik_rigor.integrations.pytest_plugin")],
        "console_scripts": [("x", "m:f")],
    }


def test_an_empty_entry_points_file_is_no_groups():
    assert vr._parse_entry_points("") == {}


# ----------------------------------------------------------------------------------
# Wheel-shaped checks, against synthetic zips
# ----------------------------------------------------------------------------------

_JUDGE_SOURCE = 'EXAMPLE_RUBRIC_NAME = "example-rubric.md"\n'


_WHEEL_NAME = "opik_rigor-0.1.1-py3-none-any.whl"


def _wheel(tmp_path: Path, members: dict[str, bytes], name: str = _WHEEL_NAME) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for member, payload in members.items():
            zf.writestr(member, payload)
    return path


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    package = repo / "src" / "opik_rigor"
    (package / "rubrics").mkdir(parents=True)
    (package / "py.typed").write_bytes(b"")
    # Bytes, not text: `write_text` translates "\n" to "\r\n" on Windows, and this
    # check compares the tree with the wheel byte for byte.
    (package / "rubrics" / "example-rubric.md").write_bytes(b"# rubric\n")
    return repo


def _record(members: list[str]) -> bytes:
    return "\n".join(f"{m},," for m in members).encode()


def test_py_typed_in_the_tree_does_not_make_it_present_in_the_wheel(tmp_path):
    """The whole reason this check reads the zip. The tree below carries `py.typed`
    and the wheel does not -- which is precisely what the published 0.1.0 wheel
    shipped, and what an assertion written as
    `Path(opik_rigor.__file__).parent / "py.typed"` cannot see, because in a
    checkout that path *is* the source tree."""
    repo = _repo(tmp_path)
    wheel = _wheel(
        tmp_path,
        {
            "opik_rigor/__init__.py": b"",
            "opik_rigor-0.1.1.dist-info/RECORD": _record(["opik_rigor/__init__.py"]),
        },
    )
    result = vr.check_wheel_py_typed(wheel, repo)
    assert result.status == vr.FAIL
    assert "not in the wheel" in result.summary
    assert any("exists=True" in line for line in result.evidence)


def test_py_typed_present_in_the_wheel_passes_even_though_it_is_empty(tmp_path):
    """PEP 561's marker is meant to be empty; an emptiness check would be wrong."""
    repo = _repo(tmp_path)
    members = ["opik_rigor/__init__.py", "opik_rigor/py.typed"]
    wheel = _wheel(
        tmp_path,
        {
            "opik_rigor/__init__.py": b"",
            "opik_rigor/py.typed": b"",
            "opik_rigor-0.1.1.dist-info/RECORD": _record(members),
        },
    )
    assert vr.check_wheel_py_typed(wheel, repo).status == vr.PASS


def test_py_typed_in_the_zip_but_not_in_record_is_a_failure(tmp_path):
    """RECORD is what an installer copies from. A file in the zip and absent from
    RECORD is in the artifact and not in the install."""
    repo = _repo(tmp_path)
    wheel = _wheel(
        tmp_path,
        {
            "opik_rigor/__init__.py": b"",
            "opik_rigor/py.typed": b"",
            "opik_rigor-0.1.1.dist-info/RECORD": _record(["opik_rigor/__init__.py"]),
        },
    )
    assert vr.check_wheel_py_typed(wheel, repo).status == vr.FAIL


def test_a_wheel_without_the_rubric_fails_and_says_what_the_user_would_get(tmp_path):
    repo = _repo(tmp_path)
    wheel = _wheel(tmp_path, {"opik_rigor/__init__.py": b"", "opik_rigor/judge.py": b""})
    result = vr.check_wheel_example_rubric(wheel, repo)
    assert result.status == vr.FAIL
    assert any("nothing to point it at" in line for line in result.evidence)


def test_a_rubric_that_differs_from_the_tree_is_a_stale_build(tmp_path):
    repo = _repo(tmp_path)
    wheel = _wheel(
        tmp_path,
        {
            "opik_rigor/judge.py": _JUDGE_SOURCE.encode(),
            "opik_rigor/rubrics/example-rubric.md": b"# something else\n",
        },
    )
    result = vr.check_wheel_example_rubric(wheel, repo)
    assert result.status == vr.FAIL
    assert any("differs from" in line for line in result.evidence)


def test_a_renamed_rubric_that_the_code_no_longer_looks_for_is_caught(tmp_path):
    """The file is in the wheel and the constant names something else. Nothing fails
    until a user calls the accessor, and then it raises FileNotFoundError on their
    machine and on nobody's CI."""
    repo = _repo(tmp_path)
    wheel = _wheel(
        tmp_path,
        {
            "opik_rigor/judge.py": b'EXAMPLE_RUBRIC_NAME = "rubric.md"\n',
            "opik_rigor/rubrics/example-rubric.md": b"# rubric\n",
        },
    )
    result = vr.check_wheel_example_rubric(wheel, repo)
    assert result.status == vr.FAIL
    assert any("the code looks for" in line for line in result.evidence)


def test_a_matching_rubric_passes(tmp_path):
    repo = _repo(tmp_path)
    wheel = _wheel(
        tmp_path,
        {
            "opik_rigor/judge.py": _JUDGE_SOURCE.encode(),
            "opik_rigor/rubrics/example-rubric.md": b"# rubric\n",
        },
    )
    assert vr.check_wheel_example_rubric(wheel, repo).status == vr.PASS


_PYPROJECT_WITH_PLUGIN = """
[project]
name = "opik-rigor"
version = "0.1.1"

[project.entry-points.pytest11]
rigor = "opik_rigor.integrations.pytest_plugin"
"""


def test_a_pytest_plugin_pointing_at_a_module_the_wheel_omits_is_caught(tmp_path):
    """This one does not break rigor's tests -- it breaks the *user's* entire suite,
    at collection time, before any of their tests run."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(_PYPROJECT_WITH_PLUGIN, encoding="utf-8")
    wheel = _wheel(
        tmp_path,
        {
            "opik_rigor/__init__.py": b"",
            "opik_rigor-0.1.1.dist-info/entry_points.txt": (
                b"[pytest11]\nrigor = opik_rigor.integrations.pytest_plugin\n"
            ),
        },
    )
    result = vr.check_entry_points(wheel, repo)
    assert result.status == vr.FAIL
    assert "points at nothing" in result.summary


def test_an_entry_point_declared_in_pyproject_and_missing_from_the_wheel_is_caught(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(_PYPROJECT_WITH_PLUGIN, encoding="utf-8")
    wheel = _wheel(tmp_path, {"opik_rigor/__init__.py": b""})
    result = vr.check_entry_points(wheel, repo)
    assert result.status == vr.FAIL
    assert "declares none" in result.summary


def test_an_entry_point_whose_target_is_shipped_passes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(_PYPROJECT_WITH_PLUGIN, encoding="utf-8")
    wheel = _wheel(
        tmp_path,
        {
            "opik_rigor/__init__.py": b"",
            "opik_rigor/integrations/pytest_plugin.py": b"",
            "opik_rigor-0.1.1.dist-info/entry_points.txt": (
                b"[pytest11]\nrigor = opik_rigor.integrations.pytest_plugin\n"
            ),
        },
    )
    assert vr.check_entry_points(wheel, repo).status == vr.PASS


# ----------------------------------------------------------------------------------
# Result rendering: design rule 2 says every check prints what it checked
# ----------------------------------------------------------------------------------


def test_every_evidence_line_survives_rendering():
    rendered = vr.bad("some-check", "it did not hold", ["first", "second"]).render()
    assert "[FAIL   ] some-check: it did not hold" in rendered
    assert "first" in rendered and "second" in rendered


def test_a_skip_is_reported_as_its_own_status_and_not_as_a_pass():
    """Design rule 1. The whole exit-code-2 arrangement rests on these not being
    conflated."""
    assert vr.skipped("x", "twine is absent").status == vr.SKIP
    assert vr.skipped("x", "twine is absent").status != vr.PASS


# ----------------------------------------------------------------------------------
# wheel-annotations
# ----------------------------------------------------------------------------------


def test_the_wheel_typecheck_ignores_exactly_what_pyproject_ignores():
    """The two lists must agree, and nothing but this test makes them.

    `check_wheel_annotations` runs mypy against an extracted wheel, which does not
    contain pyproject.toml -- so the set of unresolvable imports is spelled out a
    second time in the script. Two copies of a list drift, and the drift is silent
    in the direction that matters: an entry added to pyproject.toml and forgotten
    here turns the release gate red for a reason that has nothing to do with the
    release, and an entry removed from pyproject.toml and left here leaves the gate
    quietly ignoring a module it should be checking.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - 3.10 only
        pytest.skip("tomllib needs Python 3.11+, so the two lists are UNVERIFIED")

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    overrides = metadata["tool"]["mypy"]["overrides"]

    declared: set[str] = set()
    for override in overrides:
        if override.get("ignore_missing_imports"):
            module = override["module"]
            declared.update([module] if isinstance(module, str) else module)

    assert declared == set(vr.UNTYPED_OR_OPTIONAL_IMPORTS)


def test_the_wheel_typecheck_skips_rather_than_passes_when_mypy_is_absent(monkeypatch):
    """A checker that is not installed has verified nothing.

    Design rule 1 again, and it matters most here: `py.typed` is a promise to a
    downstream user, and a green row for a check that never ran would be the
    second false promise stacked on the first.
    """
    monkeypatch.setattr(vr, "_module_available", lambda module: False)
    probe = vr.Probe(extract=Path("nowhere"), workdir=Path("nowhere"), deps=[])

    result = vr.check_wheel_annotations(probe, Path("some.whl"))

    assert result.status == vr.SKIP
    assert "mypy" in result.summary
    assert any("UNVERIFIED" in line for line in result.evidence)
