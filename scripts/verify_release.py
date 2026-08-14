#!/usr/bin/env python
"""Verify that the artifact opik-rigor would publish is the library it claims to be.

    .\\.venv\\Scripts\\python.exe scripts\\verify_release.py

CI runs `python -m build` and `twine check`. Neither of them opens the wheel. Every
defect the first external consumer reported against the published 0.1.0 wheel --
a missing PEP 561 marker, a rubric that lived only in the repository, names that
were public in spirit and unreachable in an install -- was invisible in the source
tree and obvious in the zip. This script reads the zip.

Ported from the sibling project's `scripts/verify_release.py`, which was written
after a night in which three variants of one bug all hid in the gap between the
source tree and the wheel. Its three design rules are carried over unchanged,
because they are the whole point:

1. **A skipped check is never a passing check.** Anything that cannot run --
   because `twine` is not installed, because no wheel was produced -- prints
   SKIPPED with the reason and pushes the process exit code to 2. A verification
   script that quietly skips manufactures confidence, which is worse than no
   script at all.
2. **Every check prints what it checked**, not just a verdict. The evidence lines
   under each result are the thing you paste into a release record.
3. **The wheel is the subject.** The source tree, an editable install and this
   repo's own `sys.path` all lie in the same direction: they make a wheel that is
   missing `py.typed` or the example rubric look fine. Wheel-derived checks
   therefore run in a subprocess with `-E -S`, with the *extracted wheel* first on
   `sys.path`, and assert that `opik_rigor.__path__` and `opik_rigor.__file__`
   point inside the extraction and nowhere else. A check that can be satisfied by
   the source tree is not a check.

On rule 3, one adaptation from the sibling, stated plainly because it is the only
place the isolation is weaker than the original. The sibling's package had no
`__init__.py`, so it was a *namespace* package and `importlib.resources`
multiplexed it: the developer's `src/` silently supplied whatever the wheel
omitted, and only a bare `-S` interpreter could stop it. `opik_rigor` is a regular
package -- a single `__path__` entry, first hit on `sys.path` wins -- but it
imports scipy and numpy at module scope, so a bare `-S` interpreter cannot import
it at all. The probe therefore puts the extracted wheel *first* on `sys.path` and
appends only directories literally named `site-packages`/`dist-packages` at the
*end*, for the third-party dependencies. `-E -S` still suppresses `PYTHONPATH` and
every `.pth` file, which is what an editable install of this repo uses to put
`src/` on the path -- and that editable install is present in this project's own
`.venv`, so without `-S` every check below would be a claim about `src/`. The
probe reports which other `sys.path` entries also contain an `opik_rigor`, so the
shadowing is visible rather than assumed, and fails if the winner was not the
wheel.

Exit codes:

    0   every check ran and passed
    1   at least one check FAILED, or raised a contract FLAG
    2   nothing failed, but at least one check could not run (SKIPPED)

Only stdlib is used, plus `build` and `twine`. Both are optional here and their
absence is reported, never silently tolerated.
"""

from __future__ import annotations

import argparse
import email.message
import email.parser
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib  # type: ignore[no-redef]

# --------------------------------------------------------------------------------------
# What this project is. Changing any of these is a release-contract change, not a tweak.
# --------------------------------------------------------------------------------------

DIST_NAME = "opik-rigor"
IMPORT_NAME = "opik_rigor"

# PEP 561. One empty file, and without it a type checker must discard every
# annotation in an installed copy -- which is what the published 0.1.0 wheel did.
PY_TYPED = "py.typed"

# The worked example rubric. `pip install opik-rigor` at 0.1.0 gave you a
# PinnedJudge and nothing to point it at; the README linked a repository path.
RUBRIC_SUBDIR = "rubrics"
RUBRIC_NAME = "example-rubric.md"
RUBRIC_ACCESSOR = "example_rubric_path"

# MIT, and no NOTICE: unlike Apache-2.0 §4(d), MIT makes no third file load-bearing.
LICENSE_FILES = ("LICENSE",)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIPPED"
FLAG = "FLAG"

# --------------------------------------------------------------------------------------
# Result plumbing
# --------------------------------------------------------------------------------------


@dataclass
class Result:
    """One checklist row: a verdict, and the evidence that produced it."""

    name: str
    status: str
    summary: str
    evidence: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = f"[{self.status:7}] {self.name}: {self.summary}"
        body = "".join(f"\n            {line}" for line in self.evidence)
        return head + body


def ok(name: str, summary: str, evidence: list[str] | None = None) -> Result:
    return Result(name, PASS, summary, evidence or [])


def bad(name: str, summary: str, evidence: list[str] | None = None) -> Result:
    return Result(name, FAIL, summary, evidence or [])


def skipped(name: str, reason: str, evidence: list[str] | None = None) -> Result:
    return Result(name, SKIP, reason, evidence or [])


def flagged(name: str, summary: str, evidence: list[str] | None = None) -> Result:
    return Result(name, FLAG, summary, evidence or [])


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_release_checks.py)
# --------------------------------------------------------------------------------------


def normalize_project_name(name: str) -> str:
    """PEP 503 normalisation. `opik-rigor`, `opik_rigor` and `Opik.Rigor` are one project."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def requirement_name(requirement: str) -> str:
    """Extract the distribution name from a PEP 508 requirement string.

    >>> requirement_name("opik>=2.0,<3")
    'opik'
    >>> requirement_name("opik-rigor[opik]")
    'opik-rigor'
    """
    head = requirement.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", head)
    return match.group(1) if match else head


def requirement_extras(requirement: str) -> list[str]:
    """The extras named in a PEP 508 requirement: `opik-rigor[opik,pytest]` -> both.

    An extra the README advertises but pyproject does not declare installs the bare
    distribution and silently gives the reader none of the integration they came for.
    """
    match = re.search(r"\[([^\]]*)\]", requirement.split(";", 1)[0])
    if not match:
        return []
    return [part.strip() for part in match.group(1).split(",") if part.strip()]


def requirement_marker(requirement: str) -> str:
    """The environment marker of a PEP 508 requirement, or '' when there is none."""
    if ";" not in requirement:
        return ""
    return requirement.split(";", 1)[1].strip()


def markers_equivalent(left: str, right: str) -> bool:
    """Compare two environment markers ignoring quote style and whitespace.

    `python_version < "3.11"` and `python_version < '3.11'` are the same marker;
    hatchling normalises quotes on the way into METADATA and pyproject.toml does
    not, so a byte comparison would report a difference that does not exist.
    """

    def canon(text: str) -> str:
        return re.sub(r"\s+", "", text).replace('"', "'")

    return canon(left) == canon(right)


def split_requires_dist(values: list[str]) -> tuple[list[str], list[str]]:
    """Partition Requires-Dist into (runtime, extras-only).

    A requirement guarded by `extra == '...'` is installed only for that extra and
    is not part of the runtime dependency claim pyproject makes.
    """
    runtime: list[str] = []
    extras: list[str] = []
    for value in values:
        if re.search(r"\bextra\s*==", requirement_marker(value)):
            extras.append(value)
        else:
            runtime.append(value)
    return runtime, extras


LICENSE_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Apache-2.0", ("apache license", "version 2.0, january 2004")),
    ("MIT", ("mit license", "permission is hereby granted, free of charge")),
    ("BSD-3-Clause", ("redistribution and use in source and binary forms", "neither the name")),
    ("GPL-3.0-only", ("gnu general public license", "version 3, 29 june 2007")),
)


def spdx_from_license_text(text: str) -> str | None:
    """Identify the licence from its shipped text, or None if unrecognised.

    Deliberately conservative: an unrecognised licence yields None so the caller
    reports SKIPPED rather than inventing agreement between text and identifier.
    """
    lowered = " ".join(text.lower().split())
    for spdx, needles in LICENSE_SIGNATURES:
        if all(needle in lowered for needle in needles):
            return spdx
    return None


def version_is_dev(version: str) -> bool:
    """True for a development version. `.dev` must not survive into a release."""
    return bool(re.search(r"\.?dev\d*", version))


def parse_metadata(raw: str) -> email.message.Message:
    return email.parser.Parser().parsestr(raw)


def wheel_version_from_filename(filename: str) -> str:
    """`opik_rigor-0.1.0-py3-none-any.whl` -> `0.1.0` (PEP 427 field order)."""
    return Path(filename).name.split("-")[1]


def sdist_version_from_filename(filename: str) -> str:
    """`opik_rigor-0.1.0.tar.gz` -> `0.1.0`."""
    return Path(filename).name.rsplit(".tar.gz", 1)[0].split("-")[-1]


def read_dunder_version(init_file: Path) -> str | None:
    """`__version__` read as text, never by importing -- an import would need the
    package's dependencies and would give the *installed* answer, not this tree's."""
    if not init_file.is_file():
        return None
    match = re.search(
        r"^__version__\s*[:=]\s*['\"]([^'\"]+)['\"]", init_file.read_text(encoding="utf-8"), re.M
    )
    return match.group(1) if match else None


def dependency_paths(entries: list[str]) -> list[str]:
    """The `sys.path` entries an isolated probe may keep, for third-party imports only.

    `opik_rigor` imports numpy and scipy at module scope, so the probe cannot run on
    a bare `-S` path. Only directories literally named `site-packages` or
    `dist-packages` survive, which is precisely what excludes this repo's own `src/`
    -- the entry an editable install adds, and the one that would answer every
    question below with the source tree instead of the wheel. They are appended
    *after* the extracted wheel, never before it.
    """
    keep: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry:
            continue
        path = Path(entry)
        if path.name not in ("site-packages", "dist-packages"):
            continue
        marker = str(path).lower()
        if marker in seen:
            continue
        seen.add(marker)
        # The entry is passed through verbatim rather than round-tripped through
        # Path: it goes back onto a child interpreter's sys.path, and rewriting a
        # POSIX separator into a backslash on the way would be a silent corruption.
        keep.append(entry)
    return keep


# --------------------------------------------------------------------------------------
# Reading the README.
#
# The fenced-block and command-position rules below are the sibling's
# `docs/readme-scan-contract.md`, frozen 2026-08-13, ported unchanged. That document
# is the specification for everything down to `readme_pip_install_requirements`.
# `readme_package_symbols` is new and its own rule is stated at its docstring.
# --------------------------------------------------------------------------------------

_FENCE = re.compile(r"^(`{3,}|~{3,})(.*)$")


def fenced_code_blocks(text: str) -> list[str]:
    """The body of every fenced code block in `text`, the fence lines excluded.

    Contract rule 1. Prose is not shell. An inline code span is a *mention*, and a
    mention is frequently a claim that the command does *not* work -- the sibling's
    README said exactly that about its own `pip install` line, and scanning flat
    text turned the sentence into the package names `does`, `not`, `work`, `today.`.

    A fence closes only on a run of the same character, at least as long as the one
    that opened it, carrying no info string -- so a tilde fence inside a backtick
    block is body text. An unterminated block yields its body to the end of the
    input: the README would be malformed, but silently dropping the tail would hide
    commands instead of reporting them. Four-space indented blocks are deliberately
    not recognised, because reading an indented paragraph as shell is the very
    mistake this rule exists to stop.
    """
    blocks: list[str] = []
    body: list[str] | None = None
    char, length = "", 0
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        fence = _FENCE.match(line.strip())
        if body is None:
            if fence:
                char, length = fence.group(1)[0], len(fence.group(1))
                body = []
            continue
        closes = fence and fence.group(1)[0] == char and len(fence.group(1)) >= length
        if closes and not fence.group(2).strip():
            blocks.append("".join(f"{entry}\n" for entry in body))
            body = None
            continue
        body.append(line)
    if body is not None:
        blocks.append("".join(f"{entry}\n" for entry in body))
    return blocks


# `PS C:\...>` as well as `$` and `>`, because README transcripts get pasted from both
# shells.
_PROMPT = re.compile(r"^\s*(?:PS[^>]*>|[$>])\s")

# `&&`, `||`, `|`, `;`, `&` end a command; `#` begins a comment, which is where a
# second platform's command lives, so it begins a segment rather than ending the line.
_SEPARATOR = re.compile(r"&&|\|\||[|;&]|(?=#)")

# `Windows:` in `# Windows: python -m pip install .` -- a label naming the platform,
# not the program being run. Capped at 30 characters so a comment written as a
# sentence cannot swallow its own verb.
_COMMENT_LABEL = re.compile(r"^\s*[^\s:][^:]{0,28}:\s")

_PATH_PREFIX = re.compile(r"^\S*[/\\]")


def command_segments(line: str) -> list[str]:
    """Every point on `line` where a command could begin, each cut back to that point.

    Contract rule 2. Restricting the scan to code blocks is not enough on its own:
    the sibling's CI example contained `*) echo "migkit failed" ; exit 1 ;;` and a
    match anywhere in the line turned that string into a subcommand that does not
    exist. Only the head of a segment is a command; a name inside an argument to
    `echo` is data.

    Separators inside quotes are not tracked. That is a deliberate limit rather than
    an oversight: a false split can only lose a match, never invent one, and the
    alternative is a shell parser.
    """
    segments: list[str] = []
    for raw in _SEPARATOR.split(_PROMPT.sub("", line, count=1)):
        segment = _COMMENT_LABEL.sub("", raw[1:], count=1) if raw.startswith("#") else raw
        # Both ends: leading whitespace hides the head of the command, and trailing
        # whitespace is only an artifact of where the separator happened to fall.
        segment = segment.strip()
        if segment[:1] in ('"', "'"):
            segment = segment[1:]
        segment = _PATH_PREFIX.sub("", segment, count=1)
        if segment:
            segments.append(segment)
    return segments


_PIP_FLAGS_WITH_VALUE = {
    "-r",
    "-c",
    "-i",
    "--index-url",
    "--extra-index-url",
    "--requirement",
    "--constraint",
    "--find-links",
    "--target",
    "--python-version",
}

# `[<python> -m ] pip[3] install <args...>`, anchored, so it only fires at the head of
# a segment. `python.exe` and `py` are here because Windows lines use them.
_PIP_INSTALL = re.compile(r"^(?:(?:python3?|py)(?:\.exe)?\s+-m\s+)?pip3?\s+install\s+(.*)$")


def _pip_argument_requirements(tail: str) -> list[str]:
    """The requirement strings among the arguments of one `pip install`.

    Local paths, wheels, `-e .` and flag values are not names a user can get wrong,
    so they are dropped. What survives is a claim about what this project is called
    and which extras it offers, so extras and version specifiers are *kept* here and
    stripped by the caller that does not want them.
    """
    requirements: list[str] = []
    skip_next = False
    for token in tail.replace("`", " ").split():
        if skip_next:
            skip_next = False
            continue
        if token in _PIP_FLAGS_WITH_VALUE:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        candidate = token.strip("\"'")
        if not candidate or candidate.startswith("."):
            continue
        if "/" in candidate or "\\" in candidate:
            continue
        if candidate.endswith((".whl", ".tar.gz", ".zip", ".txt")):
            continue
        if candidate:
            requirements.append(candidate)
    return requirements


def readme_pip_install_requirements(text: str) -> list[str]:
    """Every requirement a `pip install` line in the README would resolve.

    Only fenced code blocks are read, and only at command position, so a prose
    sentence about `pip install opik-rigor` stays prose. Returned with extras and
    version specifiers intact: `opik-rigor[opik]` is two separate claims.
    """
    found: list[str] = []
    for block in fenced_code_blocks(text):
        for line in block.splitlines():
            for segment in command_segments(line):
                match = _PIP_INSTALL.match(segment)
                if match:
                    found.extend(_pip_argument_requirements(match.group(1)))
    return found


# A *call*: `opik_rigor.example_rubric_path(` -- an attribute chain followed by an
# opening parenthesis.
def _symbol_call_pattern(package: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(package)}((?:\.[A-Za-z_]\w*)+)\s*\(")


# `from opik_rigor import a, b` and `from opik_rigor.judge import c`, allowing a
# leading backtick (an inline span in prose), a `$`/`>` prompt or a `>>>` doctest
# marker. The parenthesised alternative is tried first so a multi-line import list
# is captured whole rather than truncated at the newline.
def _from_import_pattern(package: str) -> re.Pattern[str]:
    return re.compile(
        rf"^[\s`>$]*from\s+({re.escape(package)}(?:\.[A-Za-z_]\w*)*)\s+import\s+"
        r"(\([^)]*\)|[^`\n]+)",
        re.M,
    )


def readme_package_symbols(text: str, package: str = IMPORT_NAME) -> list[str]:
    """Every name the README claims `opik_rigor` provides, dotted, relative to the package.

    The sibling scans the README for *subcommands*; this library has no CLI, so the
    equivalent claim is a symbol. The rule that separates a claim from a mention is
    different here, and deliberately so.

    For shell, only a fenced block is an instruction, because an inline span is
    usually a remark about a command. For Python, **call syntax is self-identifying**:
    `opik_rigor.judge` in a sentence is a mention of a module, whereas
    `opik_rigor.example_rubric_path()` anywhere -- prose, fence, or the middle of a
    `python -c` string -- is a claim that the name exists and is callable. So this
    scans the whole document and requires either a call or an import statement.

    That distinction is what keeps this README's own pasted traceback out of the
    results: `opik_rigor.distribution.PassRateError: pass rate gate failed` is
    followed by a colon, not a parenthesis, and is output rather than instruction.
    The same trap the sibling hit with a log prefix (`migkit:`), in Python clothing.

    A commented-out import (`# from opik_rigor import Nope`) is not matched: a line
    a reader is being shown *not* to run is not a claim. Over-reporting is otherwise
    preferred to under-reporting, exactly as the frozen contract's "accepted
    over-reports" section argues -- a loud false failure naming a symbol is a better
    outcome than a narrower rule that hides a name the wheel does not have.
    """
    prefix = f"{package}."
    found: list[str] = []

    for match in _symbol_call_pattern(package).finditer(text):
        found.append(match.group(1).lstrip("."))

    for match in _from_import_pattern(package).finditer(text):
        module, raw = match.group(1), match.group(2)
        stem = "" if module == package else module[len(prefix) :] + "."
        for token in raw.strip().strip("()").split(","):
            name = token.split(" as ")[0].strip().strip("`").strip()
            if not name or name == "*" or not re.fullmatch(r"[A-Za-z_]\w*", name):
                continue
            found.append(stem + name)

    return sorted(set(found))


# --------------------------------------------------------------------------------------
# README: addresses.
#
# `readme_package_symbols` above asks whether a *name* the README uses exists in the
# wheel. These three ask the same question of every *address* it gives: a link a
# reader can click, a file a command tells them to run, a module a `-m` names. The
# defect is one defect, and this project has now shipped it three times -- a claim
# that is true of the source tree and false of the artifact somebody installs:
#
#   0.1.0  the README linked `rubrics/example-rubric.md`, a repository path, and the
#          wheel carried no rubric at all.
#   0.1.0  the README linked `LICENSE` and `COMPATIBILITY.md` relatively, and PyPI
#          renders a long description with no repository behind it, so they 404ed.
#   0.1.1  the quickstart ended on `python examples/summarise_eval.py`, and the whole
#          distribution is `opik_rigor/` plus `opik_rigor-0.1.1.dist-info/`.
#
# The first was caught by `check_wheel_example_rubric`, after it shipped. The other
# two were caught by a stranger installing from PyPI and following the README
# literally, which is not a release process. Hence these rules.
# --------------------------------------------------------------------------------------

# `[text](target)`, `![alt](src)`, and a trailing title -- `[x](LICENSE "the licence")`.
#
# The link text is allowed to contain one level of brackets, which is not pedantry:
# a shields.io badge is written `[![License: MIT](https://img.shields.io/...)](LICENSE)`
# and a flat `[^\]]*` stops at the *inner* `]`, matches the badge image's absolute
# https target, and reports nothing. That is the shape of one of the four dead links
# this rule exists to catch, so the flat form would have missed a quarter of the
# defect while looking like it worked. The two alternatives start with disjoint
# characters, so there is no backtracking to worry about.
#
# Angle-bracket targets containing spaces are deliberately not supported: they do not
# appear here, and a rule nobody exercises is a rule nobody can trust.
_MD_INLINE_LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)<>\s]+)>?[^)]*\)")

# A reference definition: `[label]: target`, up to three leading spaces per CommonMark.
_MD_REFERENCE_DEF = re.compile(r"^ {0,3}\[[^\]]+\]:\s*<?(\S+?)>?\s*$", re.M)

# An address that already resolves from anywhere: any scheme (`https:`, `mailto:`) or
# a protocol-relative `//host/path`.
_ABSOLUTE_TARGET = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*:|//)")


def readme_relative_links(text: str) -> list[str]:
    """Every markdown link target in `text` that resolves only inside a checkout.

    Rule: a link is *relative* when it carries no scheme, is not protocol-relative,
    and is not a bare `#fragment` pointing at this same document. Those are the ones
    that 404 from the project page, because PyPI renders the long description
    standing on nothing -- there is no repository, no branch and no directory behind
    it. `LICENSE`, `examples/` and `COMPATIBILITY.md` were all live on this README
    and all four of those links were dead on the index.

    A trailing `#fragment` is stripped: `docs/x.md#heading` is an address to
    `docs/x.md`, and the heading is not a file that can be missing.

    The whole document is scanned, fences included, and that is a deliberate
    over-report in the spirit of the frozen contract's "accepted over-reports": a
    markdown link inside a code fence does not render as a link, but writing one is
    already a mistake, and a loud false failure naming a target beats a quiet rule
    that lets a real one through.
    """
    found: list[str] = []
    for pattern in (_MD_INLINE_LINK, _MD_REFERENCE_DEF):
        for match in pattern.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith("#"):
                continue
            if _ABSOLUTE_TARGET.match(target):
                continue
            target = target.split("#", 1)[0]
            if target:
                found.append(target)
    return sorted(set(found))


# A file extension at the end of a token: `.py`, `.jsonl`, `.md`. Bounded so that a
# version specifier or a sentence's full stop cannot masquerade as one.
_FILE_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,8}$")

# A URL, as distinct from a path. `git clone https://github.com/...` names no file
# in this artifact.
_URLISH = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def readme_command_paths(text: str) -> list[str]:
    """Every path-shaped *argument* to a command in the README's fenced code blocks.

    Rule, in three parts, each of which excludes a real thing that would otherwise
    be reported wrongly:

    * **Fenced blocks only**, per the frozen contract's rule 1. A path named in
      prose is usually a remark about the repository, not an instruction.
    * **Arguments only, never the head of a command segment.** The head is the
      program. `.venv/bin/python -m pytest` names the reader's own virtualenv, which
      is not a claim about this artifact at all -- and it is the one path-shaped
      token in this README that must *not* be looked for in the wheel.
    * **A separator and an extension.** `examples/summarise_eval.py` qualifies;
      `dist/` and `".[dev]"` do not. A directory argument is an address too, but it
      is nearly always a link rather than a command argument, and
      :func:`readme_relative_links` already has it.

    Backslashes are folded to forward slashes so that a Windows spelling and a POSIX
    one are one address rather than two.
    """
    found: list[str] = []
    for block in fenced_code_blocks(text):
        for line in block.splitlines():
            for segment in command_segments(line):
                for token in segment.split()[1:]:
                    candidate = token.strip("\"'`")
                    if not candidate or candidate.startswith("-"):
                        continue
                    if _URLISH.match(candidate):
                        continue
                    if "/" not in candidate and "\\" not in candidate:
                        continue
                    if not _FILE_SUFFIX.search(candidate):
                        continue
                    found.append(candidate.replace("\\", "/"))
    return sorted(set(found))


# `[<path/to/>]python[.exe] -m <module>`. The path prefix is already stripped by
# `command_segments`, so only the `.exe` suffix has to be tolerated here.
_PYTHON_DASH_M = re.compile(r"^(?:python3?|py)(?:\.exe)?\s+-m\s+([A-Za-z_][\w.]*)")


def readme_module_targets(text: str, package: str = IMPORT_NAME) -> list[str]:
    """Every `python -m <module>` in the README that names a module of this package.

    The replacement address for the defect above: `python examples/summarise_eval.py`
    became `python -m opik_rigor.examples.summarise_eval`, which is a claim about
    what the *wheel* contains rather than about what a git checkout contains. It is
    checked as such.

    `python -m pytest`, `-m build` and `-m venv` are other people's modules and are
    not this project's to guarantee, so only targets equal to the package or beneath
    it are returned. `opik_rigorous.thing` is a different distribution and is not
    beneath `opik_rigor` -- the dot is required, not merely a prefix match.
    """
    prefix = f"{package}."
    found: list[str] = []
    for block in fenced_code_blocks(text):
        for line in block.splitlines():
            for segment in command_segments(line):
                match = _PYTHON_DASH_M.match(segment)
                if match and (match.group(1) == package or match.group(1).startswith(prefix)):
                    found.append(match.group(1))
    return sorted(set(found))


# --------------------------------------------------------------------------------------
# Small process/archive utilities
# --------------------------------------------------------------------------------------


def run(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


#: A CSI escape sequence. Stripped from any subprocess output this script parses.
#: Not cosmetic: `twine check` prints a bare `PASSED` on a Windows dev shell and a
#: colour-wrapped `\x1b[32mPASSED\x1b[0m` on GitHub Actions, so a check that looked
#: for the word at end of line passed everywhere it was written and failed the
#: first time it ran where it counts.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def plain_lines(text: str) -> list[str]:
    """Non-empty lines with any terminal colouring removed."""
    return [_ANSI_RE.sub("", line) for line in text.splitlines() if line.strip()]


def tail(text: str, limit: int = 12) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def dist_info_member(zf: zipfile.ZipFile, suffix: str) -> str | None:
    for name in zf.namelist():
        if ".dist-info/" in name and name.endswith(suffix):
            return name
    return None


def wheel_metadata(wheel: Path) -> email.message.Message | None:
    with zipfile.ZipFile(wheel) as zf:
        member = dist_info_member(zf, "METADATA")
        if member is None:
            return None
        return parse_metadata(zf.read(member).decode("utf-8"))


def _module_available(module: str) -> bool:
    return run([sys.executable, "-c", f"import {module}"]).returncode == 0


# --------------------------------------------------------------------------------------
# The isolated probe. Rule 3 lives here; every wheel-derived check goes through it,
# so there is exactly one isolation preamble to audit rather than one per check.
# --------------------------------------------------------------------------------------

ISOLATION_PROBE = r'''
import json, os, sys

request = json.loads(open(sys.argv[1], encoding="utf-8").read())
package = request["package"]
extract = request["extract"]

# The extracted wheel goes FIRST -- before the probe's own directory and before the
# dependency directories appended below -- so that it, and nothing else, answers
# `import opik_rigor`. The parent asserts that it did.
sys.path.insert(0, extract)
sys.path.extend(request["deps"])

out = {"sys_path": list(sys.path), "executable": sys.executable}
try:
    import importlib
    import importlib.resources as ir

    pkg = importlib.import_module(package)
    out["package_file"] = getattr(pkg, "__file__", None)
    out["package_path"] = [str(entry) for entry in list(getattr(pkg, "__path__", []))]
    out["all"] = list(getattr(pkg, "__all__", []))

    # Which other path entries hold a package of the same name. Reported rather than
    # assumed absent: this venv has an editable install of this very project.
    shadow = []
    for entry in sys.path[1:]:
        if os.path.isdir(os.path.join(entry, package)) or os.path.isfile(
            os.path.join(entry, package + ".py")
        ):
            shadow.append(entry)
    out["shadowing"] = shadow

    resources = {}
    for anchor, name in request.get("resources", []):
        try:
            entry = ir.files(anchor) / name
            resources[anchor + "/" + name] = {
                "size": len(entry.read_bytes()) if entry.is_file() else -1,
                "anchor": str(ir.files(anchor)),
            }
        except Exception as exc:
            resources[anchor + "/" + name] = {
                "size": -1,
                "anchor": "%s: %s" % (type(exc).__name__, exc),
            }
    out["resources"] = resources

    symbols = {}
    for dotted in request.get("symbols", []):
        obj, trail, found = pkg, package, True
        for part in dotted.split("."):
            trail += "." + part
            if hasattr(obj, part):
                obj = getattr(obj, part)
                continue
            try:
                obj = importlib.import_module(trail)
            except Exception:
                found = False
                break
        symbols[dotted] = {"found": found, "kind": type(obj).__name__ if found else None}
    out["symbols"] = symbols

    calls = {}
    for name in request.get("call", []):
        try:
            calls[name] = {"value": str(getattr(pkg, name)()), "error": None}
        except Exception as exc:
            calls[name] = {"value": None, "error": "%s: %s" % (type(exc).__name__, exc)}
    out["calls"] = calls
except Exception as exc:  # noqa: BLE001 - reported verbatim to the parent
    out["error"] = "%s: %s" % (type(exc).__name__, exc)

print(json.dumps(out))
'''


@dataclass
class Probe:
    """The extracted wheel, and a way to ask questions of it in isolation."""

    extract: Path
    workdir: Path
    deps: list[str]

    def ask(self, **request: object) -> tuple[dict, list[str]]:
        """Run the probe. Returns (parsed answer, evidence lines). `error` key on failure."""
        payload = {
            "package": IMPORT_NAME,
            "extract": str(self.extract),
            "deps": self.deps,
            **request,
        }
        request_file = self.workdir / "_probe_request.json"
        request_file.write_text(json.dumps(payload), encoding="utf-8")
        script = self.workdir / "_isolation_probe.py"
        script.write_text(ISOLATION_PROBE, encoding="utf-8")
        proc = run([sys.executable, "-E", "-S", str(script), str(request_file)], cwd=self.workdir)
        evidence = [
            f"probe: {Path(sys.executable).name} -E -S, cwd={self.workdir}",
            f"sys.path[0]: {self.extract}  (the extracted wheel, ahead of everything)",
            f"dependency entries appended: {self.deps}",
        ]
        if proc.returncode != 0:
            return {"error": f"probe exited {proc.returncode}"}, evidence + tail(proc.stderr, 6)
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"error": "the probe produced no JSON"}, evidence + tail(
                proc.stdout + proc.stderr, 6
            )
        return data, evidence

    def isolation_evidence(self, data: dict) -> list[str]:
        return [
            f"{IMPORT_NAME}.__file__ = {data.get('package_file')}",
            f"{IMPORT_NAME}.__path__ = {data.get('package_path')}",
            f"other sys.path entries holding an {IMPORT_NAME}: "
            f"{data.get('shadowing') or 'none'}",
        ]

    def isolation_broken(self, data: dict) -> str | None:
        """The reason this answer proves nothing, or None when the wheel really answered."""
        expected = (self.extract / IMPORT_NAME).resolve()
        paths = [Path(p).resolve() for p in data.get("package_path", [])]
        if not paths:
            return f"{IMPORT_NAME} has no __path__, so it is not the package we extracted"
        leaked = [str(p) for p in paths if p != expected]
        if leaked:
            return f"__path__ reaches outside the extracted wheel: {leaked}"
        if len(paths) > 1:
            return f"__path__ has {len(paths)} entries; a multiplexed package proves nothing"
        file_ = data.get("package_file")
        if not file_ or Path(file_).resolve().parent != expected:
            return f"__file__ is {file_}, which is not inside the extracted wheel"
        return None


def _probe_failed(name: str, data: dict, evidence: list[str]) -> Result:
    """A probe that could not run is a FAIL, not a SKIP: the wheel is right there and
    the reason it could not be imported is itself the answer to the question."""
    return bad(name, "the isolated probe could not import the wheel", evidence + [data["error"]])


def extract_wheel(wheel: Path, workdir: Path) -> Path:
    extract = workdir / "wheel-extract"
    if extract.exists():
        shutil.rmtree(extract)
    extract.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(extract)
    return extract


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def check_build(
    repo: Path, dist_dir: Path, do_build: bool
) -> tuple[Result, Path | None, Path | None]:
    """Build sdist + wheel, or adopt what is already in dist/ under --no-build."""
    name = "build"
    if do_build:
        if not _module_available("build"):
            return (
                skipped(
                    name,
                    "`python -m build` is unavailable in this interpreter",
                    [
                        f"interpreter: {sys.executable}",
                        "fix: .\\.venv\\Scripts\\python.exe -m pip install --upgrade build twine",
                        "every wheel-derived check below is skipped as a consequence",
                    ],
                ),
                None,
                None,
            )
        dist_dir.mkdir(parents=True, exist_ok=True)
        stale = sorted(glob.glob(str(dist_dir / "*.whl")) + glob.glob(str(dist_dir / "*.tar.gz")))
        for path in stale:
            os.remove(path)
        proc = run([sys.executable, "-m", "build", "--outdir", str(dist_dir)], cwd=repo)
        if proc.returncode != 0:
            return (
                bad(
                    name,
                    f"`python -m build` exited {proc.returncode}",
                    [
                        f"removed {len(stale)} stale artifact(s) first",
                        *tail(proc.stdout + proc.stderr),
                    ],
                ),
                None,
                None,
            )

    wheels = sorted(glob.glob(str(dist_dir / "*.whl")))
    sdists = sorted(glob.glob(str(dist_dir / "*.tar.gz")))
    if not wheels or not sdists:
        return (
            bad(
                name,
                "dist/ does not hold both a wheel and an sdist",
                [f"dist dir: {dist_dir}", f"wheels: {wheels}", f"sdists: {sdists}"],
            ),
            None,
            None,
        )
    if len(wheels) > 1 or len(sdists) > 1:
        return (
            bad(
                name,
                "more than one artifact in dist/ -- which one would be published?",
                [
                    f"wheels: {[Path(w).name for w in wheels]}",
                    f"sdists: {[Path(s).name for s in sdists]}",
                ],
            ),
            None,
            None,
        )
    wheel, sdist = Path(wheels[0]), Path(sdists[0])
    verb = "built" if do_build else "adopted (--no-build)"
    return (
        ok(
            name,
            f"{verb} one sdist and one wheel",
            [
                f"wheel: {wheel.name} ({wheel.stat().st_size:,} bytes)",
                f"sdist: {sdist.name} ({sdist.stat().st_size:,} bytes)",
                f"source tree: {repo}",
            ],
        ),
        sdist,
        wheel,
    )


def check_wheel_py_typed(wheel: Path, repo: Path) -> Result:
    """PEP 561's marker must be in the wheel, not merely in the tree.

    The published 0.1.0 wheel had none, so every annotation in this library was
    discarded by a type checker in an installed copy. The tree can carry the file
    and the wheel still drop it -- `packages = ["src/opik_rigor"]` is what decides,
    and a `.gitignore` rule or a build backend change can silently split the two.
    A test that reads `Path(opik_rigor.__file__).parent / "py.typed"` cannot tell
    the difference, because in a dev checkout that path *is* the source tree.
    """
    name = "wheel-py-typed"
    member = f"{IMPORT_NAME}/{PY_TYPED}"
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        record_member = dist_info_member(zf, "RECORD")
        record = zf.read(record_member).decode("utf-8") if record_member else ""
        size = zf.getinfo(member).file_size if member in names else None

    on_disk = repo / "src" / IMPORT_NAME / PY_TYPED
    evidence = [
        f"looked for: {member}",
        f"source tree: {on_disk} exists={on_disk.is_file()}",
        f"listed in RECORD: {member in record}",
    ]
    if size is None:
        return bad(
            name,
            f"{member} is not in the wheel; the library's annotations are dead on arrival",
            evidence
            + [
                "PEP 561: a type checker must ignore annotations in an installed package",
                "with no marker. This is what the published 0.1.0 wheel shipped.",
                "An empty file is the whole fix -- but it has to be in `packages`.",
            ],
        )
    evidence.append(f"in wheel: {member} ({size:,} bytes; empty is correct)")
    if member not in record:
        return bad(name, f"{member} is in the zip but absent from RECORD", evidence)
    return ok(name, f"{member} is inside {wheel.name}", evidence)


def check_wheel_example_rubric(wheel: Path, repo: Path) -> Result:
    """The worked example rubric must be in the wheel and match the tree byte for byte.

    `pip install opik-rigor` at 0.1.0 gave you a PinnedJudge and nothing to point it
    at: the rubric existed only in the repository and the README linked a relative
    path that resolves to nothing on PyPI. A packaged `.md` is exactly the kind of
    file a wheel drops without anyone noticing, because no import fails.
    """
    name = "wheel-example-rubric"
    member = f"{IMPORT_NAME}/{RUBRIC_SUBDIR}/{RUBRIC_NAME}"
    on_disk = repo / "src" / IMPORT_NAME / RUBRIC_SUBDIR / RUBRIC_NAME
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        if member not in names:
            shipped_rubrics = sorted(n for n in names if f"/{RUBRIC_SUBDIR}/" in n)
            return bad(
                name,
                f"{member} is not in the wheel",
                [
                    f"looked for: {member}",
                    f"{RUBRIC_SUBDIR}/ entries actually in the wheel: "
                    f"{shipped_rubrics or 'none'}",
                    f"source tree: {on_disk} exists={on_disk.is_file()}",
                    "an install would give a PinnedJudge and nothing to point it at",
                ],
            )
        payload = zf.read(member)
        judge_source = zf.read(f"{IMPORT_NAME}/judge.py").decode("utf-8")

    evidence = [f"in wheel: {member} ({len(payload):,} bytes)"]
    problems: list[str] = []
    if not payload.strip():
        problems.append(f"{member} is empty")
    if not on_disk.is_file():
        problems.append(f"{member} is in the wheel but not at {on_disk}")
    elif on_disk.read_bytes() != payload:
        problems.append(f"{member} differs from {on_disk} (stale build?)")
    else:
        evidence.append(f"byte-identical to {on_disk.relative_to(repo).as_posix()}")

    # The name the code looks for must be the name the wheel ships. A rename that
    # touches one and not the other raises FileNotFoundError on a user's machine only.
    declared = re.search(r"^EXAMPLE_RUBRIC_NAME\s*=\s*['\"]([^'\"]+)['\"]", judge_source, re.M)
    if declared is None:
        problems.append("judge.py in the wheel declares no EXAMPLE_RUBRIC_NAME to cross-check")
    else:
        evidence.append(
            f"judge.py in the wheel declares EXAMPLE_RUBRIC_NAME = {declared.group(1)!r}"
        )
        if declared.group(1) != RUBRIC_NAME:
            problems.append(
                f"the code looks for {declared.group(1)!r} and the wheel ships {RUBRIC_NAME!r}"
            )

    if problems:
        return bad(name, "the packaged rubric is not coherent", evidence + problems)
    return ok(name, f"the example rubric is inside {wheel.name}", evidence)


def check_resources_isolated(probe: Probe, wheel: Path) -> Result:
    """Reach the rubric the way a user does -- `importlib.resources` -- with only the wheel.

    Without the isolation this check is worthless: this repo's `.venv` carries an
    editable install pointing at `src/`, so an unisolated interpreter answers every
    question about the wheel with the source tree, in the direction that hides the
    defect. The probe therefore runs with `-E -S` (no PYTHONPATH, no `.pth` files,
    so no editable finder) and asserts the package resolved inside the extraction.
    """
    name = "wheel-rubric-importable"
    anchor = f"{IMPORT_NAME}.{RUBRIC_SUBDIR}"
    data, evidence = probe.ask(
        resources=[[anchor, RUBRIC_NAME]],
        call=[RUBRIC_ACCESSOR],
    )
    if "error" in data:
        return _probe_failed(name, data, evidence)

    evidence += probe.isolation_evidence(data)
    broken = probe.isolation_broken(data)
    if broken:
        return bad(name, "the import did not resolve to the wheel, so this proves nothing",
                   evidence + [broken])

    key = f"{anchor}/{RUBRIC_NAME}"
    entry = data.get("resources", {}).get(key, {})
    evidence.append(f"anchor {anchor} -> {entry.get('anchor')}")
    evidence.append(f"{key}: {entry.get('size')} bytes via importlib.resources")

    call = data.get("calls", {}).get(RUBRIC_ACCESSOR, {})
    problems: list[str] = []
    if int(entry.get("size", -1)) <= 0:
        problems.append(f"{key} is unreachable or empty through importlib.resources")
    if call.get("error"):
        problems.append(f"{IMPORT_NAME}.{RUBRIC_ACCESSOR}() raised: {call['error']}")
    else:
        value = str(call.get("value"))
        evidence.append(f"{IMPORT_NAME}.{RUBRIC_ACCESSOR}() -> {value}")
        resolved = Path(value).resolve()
        if probe.extract.resolve() not in resolved.parents:
            problems.append(
                f"{RUBRIC_ACCESSOR}() returned {value}, which is outside the extracted wheel"
            )
        elif not resolved.is_file():
            problems.append(f"{RUBRIC_ACCESSOR}() returned a path that is not a file: {value}")

    if problems:
        return bad(
            name, "the packaged rubric is not reachable from an install", evidence + problems
        )
    return ok(
        name,
        f"importlib.resources and {RUBRIC_ACCESSOR}() both reach the rubric in {wheel.name}",
        evidence,
    )


def check_exports_importable(probe: Probe, wheel: Path, repo: Path) -> Result:
    """Every name in `__all__` must be importable from the package root of the wheel.

    This is the shape the sibling's console-script check takes in a library: there
    is no `migkit` command to point at a module that does not exist, but there is an
    `__all__` -- the list this package tells the world is its public surface -- and a
    name in it that an installed copy cannot supply is the same defect. It is checked
    against the *wheel*, in isolation, because `from opik_rigor import X` in this
    repo's own interpreter is answered by `src/` and would pass regardless.

    The tree's `__all__` is compared with the wheel's as well, so a stale artifact is
    reported as a stale artifact rather than as a missing name.
    """
    name = "wheel-exports-importable"
    data, evidence = probe.ask()
    if "error" in data:
        return _probe_failed(name, data, evidence)

    evidence += probe.isolation_evidence(data)
    broken = probe.isolation_broken(data)
    if broken:
        return bad(name, "the import did not resolve to the wheel, so this proves nothing",
                   evidence + [broken])

    exported = list(data.get("all", []))
    if not exported:
        return bad(
            name,
            f"the wheel's {IMPORT_NAME} declares no __all__",
            evidence + ["a package with no declared surface cannot have that surface verified"],
        )

    tree_all = _source_dunder_all(repo / "src" / IMPORT_NAME / "__init__.py")
    if tree_all is not None and sorted(tree_all) != sorted(exported):
        return bad(
            name,
            "the wheel's __all__ differs from the source tree's -- the artifact is stale",
            evidence
            + [
                f"only in the wheel: {sorted(set(exported) - set(tree_all))}",
                f"only in the tree:  {sorted(set(tree_all) - set(exported))}",
            ],
        )

    resolved, resolve_evidence = probe.ask(symbols=exported)
    if "error" in resolved:
        return bad(name, "the isolated probe failed while resolving __all__",
                   resolve_evidence + [resolved["error"]])
    answers = resolved.get("symbols", {})
    missing = sorted(n for n in exported if not answers.get(n, {}).get("found"))
    kinds: dict[str, int] = {}
    for entry in answers.values():
        if entry.get("found"):
            kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1

    evidence += [
        f"__all__ in the wheel ({len(exported)} names): {exported}",
        f"resolved from the wheel by kind: {dict(sorted(kinds.items()))}",
    ]
    if missing:
        return bad(
            name,
            f"{len(missing)} name(s) in __all__ cannot be imported from the wheel: {missing}",
            evidence
            + [
                "`from opik_rigor import <name>` fails for these in a real install, while",
                "passing in this checkout, because the checkout has src/ on sys.path.",
            ],
        )
    return ok(name, f"all {len(exported)} names in __all__ import from {wheel.name}", evidence)


def _source_dunder_all(init_file: Path) -> list[str] | None:
    """`__all__` read as text from the tree, for a stale-artifact comparison only."""
    if not init_file.is_file():
        return None
    text = init_file.read_text(encoding="utf-8")
    match = re.search(r"^__all__\s*=\s*\[(.*?)\]", text, re.M | re.S)
    if not match:
        return None
    return re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))


def check_entry_points(wheel: Path, repo: Path) -> Result:
    """Every entry point the wheel declares must target a module the wheel ships.

    rigor registers a `pytest11` plugin, which pytest imports at *collection* time in
    any environment where this package is installed. A plugin pointing at a module
    the wheel omitted does not fail this library's tests -- it fails the user's
    entire suite, before any of their tests run.
    """
    name = "wheel-entry-points"
    with zipfile.ZipFile(wheel) as zf:
        member = dist_info_member(zf, "entry_points.txt")
        names = set(zf.namelist())
        text = zf.read(member).decode("utf-8") if member else ""

    declared = {
        group: dict(targets)
        for group, targets in _parse_entry_points(text).items()
    }
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    expected = pyproject["project"].get("entry-points", {})
    for key in ("scripts", "gui-scripts"):
        if pyproject["project"].get(key):
            expected[key] = pyproject["project"][key]

    evidence = [f"entry_points.txt in the wheel: {declared or 'none'}"]
    problems: list[str] = []
    if expected and not declared:
        return bad(
            name,
            "pyproject declares entry points and the wheel declares none",
            evidence + [f"pyproject: {expected}"],
        )

    for group, targets in expected.items():
        for key, value in targets.items():
            built = declared.get(group, {}).get(key)
            evidence.append(f"pyproject [{group}] {key} = {value!r}; wheel says {built!r}")
            if built is None:
                problems.append(f"[{group}] {key} is in pyproject but not in the wheel")

    for group, targets in declared.items():
        for key, value in targets.items():
            module = value.split(":", 1)[0].strip()
            candidates = [
                module.replace(".", "/") + ".py",
                module.replace(".", "/") + "/__init__.py",
            ]
            present = [c for c in candidates if c in names]
            evidence.append(f"[{group}] {key} -> {module}: {present[0] if present else 'ABSENT'}")
            if not present:
                problems.append(
                    f"[{group}] {key} targets {module}, which the wheel does not contain"
                )

    if problems:
        return bad(name, "an entry point points at nothing", evidence + problems)
    total = sum(len(t) for t in declared.values())
    return ok(name, f"all {total} entry point(s) target modules the wheel ships", evidence)


def _parse_entry_points(text: str) -> dict[str, list[tuple[str, str]]]:
    """Minimal INI read of entry_points.txt: {group: [(name, target), ...]}."""
    groups: dict[str, list[tuple[str, str]]] = {}
    current = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            groups.setdefault(current, [])
            continue
        if "=" in stripped and current:
            key, value = stripped.split("=", 1)
            groups[current].append((key.strip(), value.strip()))
    return groups


SDIST_REQUIRED = ("LICENSE", "README.md", "CHANGELOG.md", "pyproject.toml")


def check_sdist_contents(sdist: Path) -> Result:
    """The sdist is what a distro packager and a `pip install --no-binary` build use."""
    name = "sdist-contents"
    with tarfile.open(sdist) as tf:
        members = tf.getnames()
    root = Path(members[0]).parts[0] if members else ""
    relative = {str(Path(m).relative_to(root)).replace("\\", "/") for m in members if m != root}

    missing = [f for f in SDIST_REQUIRED if f not in relative]
    has_src = any(r.startswith("src/") for r in relative)
    has_tests = any(r.startswith("tests/") for r in relative)
    package_data = [
        f"src/{IMPORT_NAME}/{PY_TYPED}",
        f"src/{IMPORT_NAME}/{RUBRIC_SUBDIR}/{RUBRIC_NAME}",
    ]
    missing_data = [m for m in package_data if m not in relative]

    evidence = [
        f"sdist root: {root}/ ({len(relative)} entries)",
        f"required files present: {[f for f in SDIST_REQUIRED if f in relative]}",
        f"src/ present: {has_src}; tests/ present: {has_tests}",
        f"package data in sdist: {[m for m in package_data if m in relative]}",
    ]
    problems = []
    if missing:
        problems.append(f"missing: {missing}")
    if not has_src:
        problems.append("no src/ tree")
    if not has_tests:
        problems.append("no tests/ tree -- a packager cannot verify what they rebuild")
    if missing_data:
        problems.append(f"package data missing from sdist: {missing_data}")
    if problems:
        return bad(name, "the sdist is missing required content", evidence + problems)
    return ok(
        name, "sdist carries licence, readme, changelog, pyproject, src/ and tests/", evidence
    )


def check_license_metadata(wheel: Path, repo: Path) -> Result:
    """PEP 639 coherence, read off the built metadata rather than pyproject.

    pyproject's own comment records the trap: PyPI rejects an upload that carries
    both the SPDX `license` field and a deprecated `License ::` classifier. That is
    a claim about the built METADATA, so it is checked there.
    """
    name = "license-metadata"
    evidence: list[str] = []
    problems: list[str] = []
    notes: list[str] = []

    with zipfile.ZipFile(wheel) as zf:
        metadata_member = dist_info_member(zf, "METADATA")
        if metadata_member is None:
            return bad(name, "the wheel has no METADATA", [])
        msg = parse_metadata(zf.read(metadata_member).decode("utf-8"))
        names = zf.namelist()

        expression = (msg.get("License-Expression") or "").strip()
        evidence.append(f"License-Expression: {expression or '(absent)'}")
        if not expression:
            problems.append("no License-Expression: the PEP 639 SPDX field is missing")

        legacy = (msg.get("License") or "").strip()
        if legacy:
            evidence.append(f"License: {legacy[:60]!r}... ({len(legacy)} chars)")
            if len(legacy) > 200 or "\n" in legacy:
                problems.append(
                    "the legacy License: field holds the licence body -- "
                    "`license = { file = ... }` has crept back"
                )
        else:
            evidence.append("License: (absent, correct under PEP 639)")

        classifiers = [c for c in msg.get_all("Classifier") or [] if c.startswith("License ::")]
        evidence.append(f"deprecated 'License ::' classifiers: {classifiers or 'none'}")
        if classifiers:
            problems.append(f"PyPI rejects SPDX + classifier together: {classifiers}")

        declared_files = [v.strip() for v in msg.get_all("License-File") or []]
        evidence.append(f"License-File: {declared_files or 'none'}")
        for required in LICENSE_FILES:
            if required not in declared_files:
                problems.append(f"License-File: {required} is not declared")
            member = next(
                (n for n in names if ".dist-info/licenses/" in n and Path(n).name == required), None
            )
            if member is None:
                problems.append(f"{required} is not in .dist-info/licenses/ of the wheel")
                continue
            shipped = zf.read(member)
            evidence.append(f"in wheel: {member} ({len(shipped):,} bytes)")
            on_disk = repo / required
            if on_disk.is_file() and on_disk.read_bytes() != shipped:
                problems.append(f"{member} differs from the repo's {required}")

        license_member = next(
            (n for n in names if ".dist-info/licenses/" in n and Path(n).name == "LICENSE"), None
        )
        if license_member and expression:
            text = zf.read(license_member).decode("utf-8", "replace")
            detected = spdx_from_license_text(text)
            first = " / ".join(line.strip() for line in text.splitlines()[:2] if line.strip())
            evidence.append(f"shipped LICENSE begins: {first}")
            if detected is None:
                notes.append(
                    "the shipped licence text matches no known signature, so text-vs-SPDX "
                    "agreement is UNVERIFIED -- check it by hand"
                )
            elif detected not in expression:
                problems.append(
                    f"the shipped text is {detected} but the metadata declares "
                    f"'{expression}' -- declaration and bytes disagree"
                )
            else:
                evidence.append(
                    f"shipped text identified as {detected}, consistent with '{expression}'"
                )

    if problems:
        return bad(name, "licence metadata is not coherent", evidence + problems)
    if notes:
        return skipped(name, "licence text could not be identified mechanically", evidence + notes)
    return ok(name, "SPDX expression, licence file and classifiers are coherent", evidence)


def check_dependencies(wheel: Path, repo: Path) -> Result:
    """Every Requires-Dist is accounted for by pyproject, and vice versa."""
    name = "dependencies-declared"
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(pyproject["project"].get("dependencies", []))
    declared_python = str(pyproject["project"].get("requires-python", "")).strip()
    declared_extras = dict(pyproject["project"].get("optional-dependencies", {}))

    msg = wheel_metadata(wheel)
    if msg is None:
        return bad(name, "the wheel has no METADATA", [])

    built_runtime, built_extras = split_requires_dist(list(msg.get_all("Requires-Dist") or []))
    built_python = (msg.get("Requires-Python") or "").strip()
    provides = sorted(v.strip() for v in msg.get_all("Provides-Extra") or [])

    built_names = {normalize_project_name(requirement_name(r)) for r in built_runtime}
    declared_names = {normalize_project_name(requirement_name(r)) for r in declared}

    evidence = [
        f"pyproject dependencies ({len(declared)}): {declared}",
        f"metadata Requires-Dist, runtime ({len(built_runtime)}): {built_runtime}",
        f"metadata Requires-Dist, extras ({len(built_extras)}): {built_extras}",
        f"Requires-Python: metadata {built_python!r} vs pyproject {declared_python!r}",
        f"Provides-Extra: metadata {provides} vs pyproject {sorted(declared_extras)}",
    ]
    problems = []
    unexplained = sorted(built_names - declared_names)
    unbuilt = sorted(declared_names - built_names)
    if unexplained:
        problems.append(f"in the wheel but not in pyproject: {unexplained}")
    if unbuilt:
        problems.append(f"in pyproject but not in the wheel: {unbuilt}")
    if built_python != declared_python:
        problems.append("Requires-Python disagrees between metadata and pyproject")
    if provides != sorted(declared_extras):
        problems.append("Provides-Extra disagrees between metadata and pyproject")

    for requirement in declared:
        marker = requirement_marker(requirement)
        built = next(
            (
                r
                for r in built_runtime
                if normalize_project_name(requirement_name(r))
                == normalize_project_name(requirement_name(requirement))
            ),
            None,
        )
        if built is None:
            continue
        if not markers_equivalent(marker, requirement_marker(built)):
            problems.append(
                f"{requirement_name(requirement)}: marker {requirement_marker(built)!r} in the "
                f"wheel, {marker!r} in pyproject"
            )

    if problems:
        return bad(
            name, "declared dependencies do not match the built metadata", evidence + problems
        )
    return ok(name, f"all {len(built_runtime)} runtime requirements accounted for", evidence)


def check_version_coherence(wheel: Path, sdist: Path, repo: Path) -> list[Result]:
    """Every place the version is written agrees, and none of them says `.dev`."""
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    sources: dict[str, str] = {}
    notes: list[str] = []

    if "version" in project:
        sources["pyproject [project].version"] = str(project["version"])
    elif "version" in project.get("dynamic", []):
        notes.append("pyproject declares a dynamic version (hatch reads __init__.py)")

    msg = wheel_metadata(wheel)
    if msg is not None:
        sources["wheel METADATA Version"] = (msg.get("Version") or "").strip()
    sources["wheel filename"] = wheel_version_from_filename(wheel.name)
    sources["sdist filename"] = sdist_version_from_filename(sdist.name)

    init_file = repo / "src" / IMPORT_NAME / "__init__.py"
    dunder = read_dunder_version(init_file)
    results: list[Result] = []

    if dunder is None:
        results.append(
            skipped(
                "version-dunder",
                f"{init_file.relative_to(repo).as_posix()} has no __version__",
                [
                    "__version__ is in this package's __all__, so its absence is itself a",
                    "defect; until it exists the coherence check has one fewer source",
                ],
            )
        )
    else:
        sources["src/opik_rigor/__init__.py __version__"] = dunder

    distinct = sorted(set(sources.values()))
    evidence = [f"{label}: {value}" for label, value in sources.items()] + notes
    if len(distinct) == 1:
        results.append(
            ok(
                "version-coherence",
                f"all {len(sources)} version sources say {distinct[0]}",
                evidence,
            )
        )
    else:
        results.append(
            bad(
                "version-coherence",
                f"version sources disagree: {distinct}",
                evidence
                + ["a user filing a bug quotes __version__; the index serves the metadata"],
            )
        )

    dev = sorted({v for v in sources.values() if version_is_dev(v)})
    if dev:
        results.append(
            bad(
                "version-not-dev",
                f"a development version would be published: {dev}",
                [
                    "PyPI would accept this and every consumer's resolver would then see a",
                    "pre-release. Pass --allow-dev-version to acknowledge it mid-build.",
                ],
            )
        )
    else:
        results.append(ok("version-not-dev", f"no .dev suffix in {distinct[0]}", []))

    if dunder is not None:
        results.append(_check_installed_version(repo, dunder))
    return results


def _check_installed_version(repo: Path, dunder: str) -> Result:
    """`importlib.metadata.version(...)` in this interpreter vs `__version__`.

    Guarded: if the interpreter's install of the distribution is some *other* tree
    -- an editable install pointing at a different checkout, which is exactly the
    situation in a worktree -- the comparison would be about that tree, not this
    one, so it is reported SKIPPED rather than dressed up as agreement.
    """
    name = "version-matches-installed"
    code = (
        "import json, importlib.metadata as md\n"
        f"d = md.distribution({DIST_NAME!r})\n"
        "print(json.dumps({'version': d.version, 'location': str(getattr(d, '_path', ''))}))"
    )
    proc = run([sys.executable, "-c", code])
    if proc.returncode != 0:
        return skipped(
            name,
            f"{DIST_NAME} is not installed in {Path(sys.executable).name}",
            tail(proc.stderr, 3),
        )
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return skipped(name, "importlib.metadata produced no parsable answer", tail(proc.stdout))
    evidence = [
        f"interpreter: {sys.executable}",
        f"importlib.metadata.version({DIST_NAME!r}) = {data['version']}",
        f"distribution location: {data['location']}",
        f"__version__ in the tree under verification = {dunder}",
    ]
    location = Path(data["location"]).resolve() if data["location"] else None
    elsewhere = location is not None and repo.resolve() not in [location, *location.parents]
    if elsewhere and data["version"] != dunder:
        return skipped(
            name,
            "the installed distribution is a different tree, so a mismatch here proves nothing",
            evidence + [f"tree under verification: {repo}"],
        )
    if data["version"] != dunder:
        return bad(name, "__version__ and the installed metadata disagree", evidence)
    return ok(name, f"__version__ and installed metadata agree on {dunder}", evidence)


def check_twine(sdist: Path, wheel: Path, repo: Path) -> Result:
    name = "twine-check"
    if not _module_available("twine"):
        return skipped(
            name,
            "twine is not installed in this interpreter",
            [
                f"interpreter: {sys.executable}",
                "fix: .\\.venv\\Scripts\\python.exe -m pip install --upgrade build twine",
                "PyPI's own rendering check is therefore UNVERIFIED, not passed",
            ],
        )
    # NO_COLOR so twine emits the plain word, and the ANSI strip below in case a
    # future twine ignores it. Belt and braces on purpose: this check reads
    # another tool's human-facing output, which is not an interface anybody
    # promised to keep stable, and its whole value is that it fails loudly rather
    # than counting wrong.
    proc = run(
        [sys.executable, "-m", "twine", "check", str(sdist), str(wheel)],
        cwd=repo,
        env={**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"},
    )
    lines = plain_lines(proc.stdout + proc.stderr)
    passed = sum(1 for line in lines if line.strip().endswith("PASSED"))
    evidence = [f"checked: {sdist.name}, {wheel.name}", *lines]
    if proc.returncode != 0:
        return bad(name, f"twine check exited {proc.returncode}", evidence)
    if passed < 2:
        return bad(name, f"expected PASSED twice, saw it {passed} time(s)", evidence)
    return ok(name, "twine check PASSED on both sdist and wheel", evidence)


def check_readme_pip_install(repo: Path) -> Result:
    """Any `pip install` line in the README must name the real distribution and real extras."""
    name = "readme-pip-install"
    readme = repo / "README.md"
    if not readme.is_file():
        return bad(name, "README.md does not exist", [])
    text = readme.read_text(encoding="utf-8")
    requirements = readme_pip_install_requirements(text)
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    extras = set(pyproject["project"].get("optional-dependencies", {}))
    allowed = {normalize_project_name(DIST_NAME)}
    for requirement in pyproject["project"].get("dependencies", []):
        allowed.add(normalize_project_name(requirement_name(requirement)))
    for group in pyproject["project"].get("optional-dependencies", {}).values():
        for requirement in group:
            allowed.add(normalize_project_name(requirement_name(requirement)))

    if not requirements:
        return ok(
            name,
            "no `pip install <name>` line in README.md to get wrong",
            [f"scanned {len(text.splitlines())} lines of {readme.name}"],
        )
    evidence = []
    problems = []
    for requirement in requirements:
        normalized = normalize_project_name(requirement_name(requirement))
        named = requirement_extras(requirement)
        unknown = [e for e in named if e not in extras]
        verdict = "ok" if normalized in allowed and not unknown else "WRONG"
        evidence.append(
            f"pip install {requirement}  ->  name {normalized}, extras {named or '[]'}  ->  "
            f"{verdict}"
        )
        if normalized not in allowed:
            problems.append(f"{requirement!r} is not {DIST_NAME} nor a declared dependency")
        if unknown:
            problems.append(
                f"{requirement!r} asks for extras {unknown}, which pyproject does not "
                f"declare (declared: {sorted(extras)}) -- pip installs the bare package"
            )
    if problems:
        return bad(
            name, "the README's install lines do not match this project", evidence + problems
        )
    return ok(
        name,
        f"all {len(requirements)} pip-install target(s) name a real distribution and real extras",
        evidence,
    )


def check_readme_symbols(probe: Probe, repo: Path, wheel: Path) -> Result:
    """Every symbol the README shows being called must exist in the built wheel.

    This library's equivalent of the sibling's README command scanner, and it exists
    because of a defect that actually shipped: the README tells the reader to call
    `opik_rigor.example_rubric_path()`, and the published 0.1.0 wheel has no such
    name anywhere in it. A README that names a symbol its own release does not
    export is a broken quickstart on the project's front page, and neither
    `python -m build` nor `twine check` can see it -- nor can any test in this repo,
    because in a checkout `src/` answers the import.
    """
    name = "readme-symbols"
    readme = repo / "README.md"
    if not readme.is_file():
        return bad(name, "README.md does not exist", [])
    symbols = readme_package_symbols(readme.read_text(encoding="utf-8"))
    if not symbols:
        return ok(
            name,
            f"README.md shows no {IMPORT_NAME}.<name> call or import to get wrong",
            [f"scanned {readme}"],
        )

    data, evidence = probe.ask(symbols=symbols)
    if "error" in data:
        return _probe_failed(name, data, evidence)
    evidence += probe.isolation_evidence(data)
    broken = probe.isolation_broken(data)
    if broken:
        return bad(name, "the import did not resolve to the wheel, so this proves nothing",
                   evidence + [broken])

    answers = data.get("symbols", {})
    missing = []
    for symbol in symbols:
        entry = answers.get(symbol, {})
        found = bool(entry.get("found"))
        evidence.append(
            f"README shows {IMPORT_NAME}.{symbol}  ->  "
            f"{'in the wheel as ' + str(entry.get('kind')) if found else 'NOT IN THE WHEEL'}"
        )
        if not found:
            missing.append(symbol)
    if missing:
        return bad(
            name,
            f"the README names {len(missing)} symbol(s) the wheel does not export: {missing}",
            evidence
            + [
                "This is the 0.1.0 defect exactly: the README's quickstart called",
                f"{IMPORT_NAME}.{RUBRIC_ACCESSOR}() and the published wheel had no such name.",
                "Fix the wheel or fix the README -- but do not delete this check.",
            ],
        )
    return ok(
        name,
        f"all {len(symbols)} symbol(s) the README names exist in {wheel.name}",
        evidence,
    )


def _module_members(module: str) -> tuple[str, str]:
    """The two wheel members either of which would make `module` importable."""
    stem = module.replace(".", "/")
    return f"{stem}.py", f"{stem}/__init__.py"


def check_readme_paths(wheel: Path, repo: Path) -> Result:
    """Every address the README hands a reader must resolve from where they stand.

    A reader stands on the project page and then on an install. Neither is a
    checkout, and this is the check that says so:

    * a **relative link** is dead on PyPI, which renders the long description with
      no repository, no branch and no directory behind it;
    * a **path argument** in a code block must be inside the wheel, or the command
      cannot be run by anyone who followed the install line above it;
    * a **`python -m` target** under this package must be a module the wheel ships.

    Each finding prints whether the address exists *in the source tree*, because
    that is the shape of every instance of this fault: true of the tree, false of
    the artifact. An address that is wrong in both places is a different and much
    louder bug; an address that is right in the tree and absent from the wheel is
    this one, and it is invisible to `python -m build`, to `twine check`, and to
    every test in this repository that imports through `src/`.

    One thing this check deliberately does *not* do is accept a relative link on the
    grounds that the file is in the wheel. `LICENSE` really is in the wheel, under
    `.dist-info/licenses/`, and the link still 404s for the reader who clicks it.
    Reachable and addressable are different claims.
    """
    name = "readme-paths"
    readme = repo / "README.md"
    if not readme.is_file():
        return bad(name, "README.md does not exist", [])
    text = readme.read_text(encoding="utf-8")

    with zipfile.ZipFile(wheel) as zf:
        members = set(zf.namelist())
    roots = sorted({member.split("/", 1)[0] for member in members})

    evidence = [
        f"scanned: {readme}",
        f"top-level entries in {wheel.name}: {roots}",
    ]
    problems: list[str] = []

    links = readme_relative_links(text)
    evidence.append(f"repo-relative markdown links ({len(links)}): {links or 'none'}")
    for target in links:
        in_tree = (repo / target).exists()
        evidence.append(
            f"link -> {target}: in the source tree={in_tree}, "
            f"resolvable from the PyPI project page=False"
        )
        problems.append(
            f"[...]({target}) is repo-relative and 404s from the project page; "
            f"use the full https://github.com/... URL"
        )

    paths = readme_command_paths(text)
    evidence.append(f"path arguments in code blocks ({len(paths)}): {paths or 'none'}")
    for path in paths:
        in_wheel = path in members
        in_tree = (repo / path).exists()
        evidence.append(f"command path -> {path}: in tree={in_tree}, in wheel={in_wheel}")
        if not in_wheel:
            problems.append(
                f"the README tells the reader to run or open {path!r}, which is not in "
                f"{wheel.name}"
                + (
                    " -- it exists only in a checkout, which is the whole defect"
                    if in_tree
                    else " and does not exist in this tree either"
                )
            )

    modules = readme_module_targets(text)
    evidence.append(f"`python -m` targets under {IMPORT_NAME} ({len(modules)}): "
                    f"{modules or 'none'}")
    for module in modules:
        candidates = _module_members(module)
        present = [c for c in candidates if c in members]
        in_tree = any((repo / "src" / c).exists() for c in candidates)
        evidence.append(
            f"module -> {module}: in tree={in_tree}, in wheel={present[0] if present else 'ABSENT'}"
        )
        if not present:
            problems.append(
                f"the README says `python -m {module}` and the wheel ships neither "
                f"{candidates[0]} nor {candidates[1]}"
            )

    if problems:
        return bad(
            name,
            f"the README gives {len(problems)} address(es) a reader cannot reach",
            evidence
            + problems
            + [
                "This is the third instance of one fault: a claim true of the source",
                "tree and false of the artifact. Fix the address or ship the file --",
                "and note that a README correction reaches PyPI only on the next",
                "upload, because a long description is frozen at upload time.",
            ],
        )
    total = len(links) + len(paths) + len(modules)
    return ok(
        name,
        f"every address the README gives resolves ({total} checked, 0 repo-relative links)",
        evidence,
    )


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

WHEEL_DERIVED = (
    "wheel-py-typed",
    "wheel-example-rubric",
    "wheel-rubric-importable",
    "wheel-exports-importable",
    "wheel-entry-points",
    "sdist-contents",
    "license-metadata",
    "dependencies-declared",
    "version-coherence",
    "version-not-dev",
    "twine-check",
    "readme-symbols",
    "readme-paths",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify_release.py",
        description="Verify that the artifact opik-rigor would publish is what it claims to be.",
    )
    default_repo = Path(__file__).resolve().parent.parent
    parser.add_argument("--repo", type=Path, default=default_repo, help="tree to verify")
    parser.add_argument(
        "--dist-dir", type=Path, default=None, help="where artifacts go (default <repo>/dist)"
    )
    parser.add_argument(
        "--no-build", action="store_true", help="use the artifacts already in dist/"
    )
    parser.add_argument(
        "--allow-dev-version",
        action="store_true",
        help="downgrade the .dev version check to a skip, for mid-build runs",
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="do not delete the scratch directory"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo: Path = args.repo.resolve()
    dist_dir: Path = (args.dist_dir or repo / "dist").resolve()
    deps = dependency_paths(list(sys.path))

    print("=" * 100)
    # Printed output stays ASCII-only: this runs on Windows consoles whose code page
    # is whatever the machine's locale says, and a UnicodeEncodeError in the banner
    # would take down the check that was about to report a real problem.
    print("opik-rigor release verification")
    print("the wheel is the subject; the source tree is not evidence")
    print("=" * 100)
    print(f"repo        : {repo}")
    print(f"dist dir    : {dist_dir}")
    print(f"interpreter : {sys.executable}")
    print(f"platform    : {sys.platform}")
    print(f"probe deps  : {deps}")
    print()

    results: list[Result] = []
    workdir = Path(tempfile.mkdtemp(prefix="rigor-verify-"))

    def emit(result: Result) -> None:
        results.append(result)
        print(result.render())

    try:
        build_result, sdist, wheel = check_build(repo, dist_dir, do_build=not args.no_build)
        emit(build_result)

        if wheel is None or sdist is None:
            reason = "no wheel was produced, so this could not be checked"
            for pending in WHEEL_DERIVED:
                emit(skipped(pending, reason, [f"see the `{build_result.name}` row above"]))
        else:
            probe = Probe(extract=extract_wheel(wheel, workdir), workdir=workdir, deps=deps)
            emit(check_wheel_py_typed(wheel, repo))
            emit(check_wheel_example_rubric(wheel, repo))
            emit(check_resources_isolated(probe, wheel))
            emit(check_exports_importable(probe, wheel, repo))
            emit(check_entry_points(wheel, repo))
            emit(check_sdist_contents(sdist))
            emit(check_license_metadata(wheel, repo))
            emit(check_dependencies(wheel, repo))
            for result in check_version_coherence(wheel, sdist, repo):
                is_dev_fail = result.name == "version-not-dev" and result.status == FAIL
                if is_dev_fail and args.allow_dev_version:
                    result = skipped(
                        result.name,
                        "a .dev version is present and --allow-dev-version was passed",
                        [result.summary, "this must be a PASS before a tag is cut"],
                    )
                emit(result)
            emit(check_twine(sdist, wheel, repo))
            emit(check_readme_symbols(probe, repo, wheel))
            emit(check_readme_paths(wheel, repo))

        emit(check_readme_pip_install(repo))
    finally:
        if args.keep_temp:
            print(f"\nscratch kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    failures = [r for r in results if r.status == FAIL]
    flags = [r for r in results if r.status == FLAG]
    skips = [r for r in results if r.status == SKIP]
    passes = [r for r in results if r.status == PASS]

    print()
    print("=" * 100)
    print(
        f"{len(passes)} passed, {len(failures)} failed, {len(flags)} flagged, "
        f"{len(skips)} skipped, {len(results)} checks total"
    )
    for group, label in ((failures, "FAILED"), (flags, "FLAGGED"), (skips, "SKIPPED")):
        for result in group:
            print(f"  {label:8} {result.name}: {result.summary}")
    print("=" * 100)

    if failures or flags:
        print("Release is blocked. Every line above is reproducible; fix the cause, not the check.")
        return 1
    if skips:
        print(
            "Nothing failed, but a check could not run. A skip is not a pass -- exit code 2 so a\n"
            "release gate cannot mistake this for green."
        )
        return 2
    print("Every check ran and passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
