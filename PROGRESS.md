# PROGRESS

Running state of the v0.1 build. Updated at the end of every session. If you are
picking this up cold, read this file, then [README.md](README.md) for what the
library does and [COMPATIBILITY.md](COMPATIBILITY.md) for what the Opik
integration was verified against — nothing important lives only in a chat
transcript.

[docs/build-plan.md](docs/build-plan.md) is the plan this build followed, committed
**verbatim and unedited**, including the parts it got wrong. It was written and
approved before any code existed: module contracts, session boundaries sized to fit
in one context window, a test inventory used as the acceptance contract, and
pre-decided answers to the risks. It is here as evidence rather than as
documentation — a plan edited after the fact to match what happened is not a plan,
it is a reconstruction.

## Where the build stands

| Session | Scope | Status |
|---|---|---|
| 1 | Core, no network: scaffold, evidence, adapters, judge | **complete** — 218 passed, 1 skipped |
| 2 | Statistics: sampling, distribution, Baseline | **complete** — 478 passed, 1 skipped |
| 3 | Integrations: Opik, pytest plugin, example | **complete** — 514 passed offline / 523 with opik |
| 4 | Ship: README, rubric, tag v0.1.0 | **complete** — tagged v0.1.0 |
| — | Publish | **v0.1.0 published to PyPI 2026-08-13** — `pip install opik-rigor` |
| 5 | Phase 3: close the consumer-reported API gaps, additively | **complete** — items 10–15 and item 8's message closed; 534 passed offline / 543 with opik |
| — | Release 0.1.1 | **prepared, not released** — branch `release/0.1.1`, 537 passed offline; see [Releasing 0.1.1](#releasing-011--what-is-prepared-and-what-remains) |

## Session 1 — module status

| Module | State | Tests |
|---|---|---|
| `src/opik_rigor/errors.py` | done | exercised via the modules that raise |
| `src/opik_rigor/evidence.py` | done | `tests/test_evidence.py` — 72 |
| `src/opik_rigor/pinning.py` | done | `tests/test_pinning.py` — 21 |
| `src/opik_rigor/adapters/base.py` | done — Protocol, env constants, credential guards | via adapter tests |
| `src/opik_rigor/adapters/{fake,anthropic,openai_compat}.py` | done | `tests/test_adapters.py` — 66 (+1 network-skipped) |
| `src/opik_rigor/judge.py` | done | `tests/test_judge.py` — 46 |
| `src/opik_rigor/rubrics/example-rubric.md` | done — moved inside the package in Phase 3 | asserted to state the output format exactly **once** in the rendered prompt |
| `src/opik_rigor/__init__.py` | done — re-exports the public surface | `tests/test_integration_session1.py` — 13 |

## Session 2 — module status

| Module | State | Tests |
|---|---|---|
| `src/opik_rigor/sampling.py` | done | `tests/test_sampling.py` — 64 |
| `src/opik_rigor/distribution.py` | done | `tests/test_distribution.py` — 118 |
| `src/opik_rigor/baseline.py` | done | `tests/test_baseline.py` — 64 |
| dogfooding suite | done | `tests/test_integration_session2.py` — 9 |

**Implementation and tests were written by different authors, deliberately.** The
agent that wrote `distribution.py` was told it would not write the tests, and the
test author was given expected values derived independently — by root-finding the
score-test inequality that *defines* the Wilson interval — before the
implementation existed. `tests/test_distribution.py` embeds that bisection oracle,
so the independent check ships with the library rather than living in a scratchpad.
Mann-Whitney gets the same treatment: `U` is hand-counted from its definition,
because checking scipy against scipy would prove nothing when scipy *is* the
implementation.

The structure paid for itself — three real bugs, each found by an author with no
stake in the implementation:

1. **`TimeoutError` shadowing in `sampling.py`.** Since Python 3.11
   `concurrent.futures.TimeoutError` *is* `builtins.TimeoutError`, so catching it
   in the collector rewrote a provider's own socket timeout as rigor's budget
   expiring — the module violating its own thesis that a failure and an exception
   are facts about different systems.
2. **The per-run timeout was not per-run.** The collector waited on futures in
   submission order, so a run queued behind others was granted several budgets'
   worth of wall clock. Both fixed by running the same `_run_once` on both paths
   and timing from inside the worker.
3. **`wilson_lower_bound(0, n)` returned a positive number.** At p̂=0 `centre` and
   `half` are the same expression analytically, but evaluated in different orders,
   and `_clamp` squeezed only the negative side — so ~15% of `(n, confidence)`
   pairs returned a lower confidence bound sitting *above* its own point estimate.

## Session 3 — module status

| Module | State | Tests |
|---|---|---|
| `COMPATIBILITY.md` | written **before** any integration code | — |
| `src/opik_rigor/integrations/opik.py` | done | `tests/test_integration_opik.py` — 11 |
| `src/opik_rigor/integrations/pytest_plugin.py` | done | `tests/test_pytest_plugin.py` — 15 |
| `examples/summarise_eval.py` + README | done | `examples/test_example_runs.py` — 19 |

Two environments are maintained: `.venv` (no Opik — the suite must be green
without it) and `.venv-opik` (Opik 2.2.28 installed, for the integration and
co-installation tests). Both must pass before a Session 3 change is done.

**An invariant was stated wrongly and had to be re-framed.** Three tests asserted
`importlib.util.find_spec("openai") is None` — which passed only because no
environment had the SDK, and broke the moment a venv installed Opik, which depends
on `openai`. They were testing the environment, not the library. The real invariant
is that *rigor never imports a provider SDK*, which is now checked in a subprocess
and holds whether or not one is installed. The one test that genuinely needs the
SDK absent (the missing-SDK error message) skips when it is present, rather than
monkeypatching the import machinery and testing the mock.

## Caller friction, found by writing the example

The agent that wrote `examples/` was asked to report where the library was awkward
to *use*, and did. These are v0.1 roadmap items, not Session 3 bugs — recorded here
so Session 4's roadmap section is written from evidence rather than imagination.

1. **`FakeAdapter(seed=...)` is unusable for the fake worth building.** A judge
   drawing uniformly from a global script is noise; a demo needs verdicts that
   *correlate* with what the system under test did, which needs
   `responses=<callable>` — and `seed=` is then rejected outright. The docstring
   sells `seed` as the mechanism for reproducible stochasticity, but the only shape
   of fake that can react to its input is the one shape that cannot take a seed.
2. **`sample(fn, n)` hands `fn` nothing** — no index, no item. Every real eval
   iterates a dataset, so every caller writes the same `itertools.cycle` closure.
   `sample_of` reads like it might be the dataset helper and is not.
3. **The report/exception split forces two code paths for one piece of
   information.** Success returns a dict; failure carries the same numbers on
   `exc.stats` — and `underpowered`/`runs_needed` exist *only* on failure. Printing
   "what did the gate conclude" uniformly means `try/except` around every gate. A
   non-raising `check_pass_rate(...) -> report` beside the asserting one is the
   shape a dashboard or report actually wants.
4. **Report dicts are `dict[str, Any]`** — no autocomplete, no typo protection, and
   the key names are not guessable (`lower_bound` vs `interval_lower` vs
   `min_rate`; `n_current` vs `n`). A frozen dataclass with `.to_dict()` would
   serialise identically and read far better.
5. **`SampleResult.scores()` is a method while `.outcomes`/`.values`/`.durations`
   are properties** — and `Baseline.scores` is a plain tuple, so
   `assert_no_regression(after.scores(), recorded.scores)` looks like a typo and
   is not.
6. **Reproducible output and the evidence log are in tension.** `EvidenceRecord`
   has no rendering helper and always carries a real timestamp, so an example that
   must print byte-identical output cannot show a record as it exists on disk.
7. **`assert_score_distribution`'s first parameter is named `scores`** but accepts
   a `SampleResult`.

Counterweight, equally worth recording: the failure messages needed no framing at
all. The two best screens in the example are copy-pasted library prose.

## Caller friction, found by the first external consumer

`migration-kit` (`C:\Users\ewehm\repos\migration-kit`) consumes the **published**
0.1.0 wheel from PyPI, not this working copy, which is the only way it counts as a
consumer. Its Session 1 runner turned up two more roadmap items. Both are recorded
here rather than worked around in the caller, per the dependency-direction rule.

8. **`sample` records a classifier failure in the same field as a call failure,
   and the default classifier fails on the most common return type there is.**
   `default_outcome` raises `TypeError` on a plain `str` — reasonably, since it
   cannot know whether `"Paris"` is a pass — but that exception lands on
   `Run.error`, exactly where an exception raised by `fn` lands. A caller who only
   wants the text back, and has no pass/fail question yet, therefore gets `n` runs
   that each carry `value="Paris"` *and* an error. migration-kit's first end-to-end
   run reported 6 completions and 6 provider failures against a `FakeAdapter` that
   answered every prompt correctly. The fix in the caller is one explicit
   `outcome=`, so this is friction rather than a defect — but the failure mode is
   silent in the direction that matters: it makes a model look like it answered
   nothing. Options: let `outcome=None` mean "do not classify" and leave
   `Run.outcome` as `None` without an error, or keep classifier errors in a
   separate field from call errors so the two are never confused.
9. **The `Adapter` seam exposes no usage data.** `complete(prompt) -> str` is the
   whole protocol, so a caller that wants token counts — for a cost gate, or for
   the "what did this verdict cost" line in a report — cannot get them without
   reaching past the seam into a provider SDK. migration-kit's `Completion` carries
   `tokens_in`/`tokens_out` fields that it must leave `None` for every adapter rigor
   ships. An optional second method (`complete_with_usage`, or a `last_usage`
   property) would keep the one-method protocol intact for adapters that cannot
   report it.
10. **Two names every judging consumer needs are not public.** `SCORE_MIN` /
    `SCORE_MAX` and `hash_rubric_file` live in `opik_rigor.judge` and appear in no
    `__all__` and no document, so migration-kit reaches into a submodule to get
    them — a violation of its own "public API only" invariant, committed by the
    same author who wrote the invariant, and found by a mechanical grep rather
    than by review. The need is unavoidable rather than incidental: a consumer
    that must impute a score for an ungradeable response has to know what the
    bottom of the scale *is*, and one that hashes a judge config has to hash the
    rubric the same way rigor does or the two disagree about whether the
    instrument changed. Re-deriving either in the consumer is worse than the
    import: a hard-coded `1.0` silently becomes wrong the day the scale changes,
    which is exactly the class of drift this library exists to catch. Fix in 0.2
    by exporting all three from the package root. Note the sharper reason to fix
    it: if a future release renames `hash_rubric_file`, the consumer's pinned CI
    stays green and its users discover the break on upgrade. Nothing in either
    repo would catch that except the drift canary.
    **Closed (Phase 3, additively).** All four — `SCORE_MIN`, `SCORE_MAX`,
    `hash_rubric_file` and `hash_rubric_text` — are re-exported from
    `opik_rigor/__init__.py` and are in `__all__`, with the reasoning in the module
    docstring so a later reader does not delete them as clutter.
    `tests/test_integration_session1.py::test_the_names_a_judging_consumer_needs_are_public_at_the_package_root`
    asserts each is in `__all__` **and** is the same object as
    `opik_rigor.judge.<name>`, so the two spellings cannot drift apart. The
    submodule path still works, so migration-kit's existing imports are untouched;
    it can move to the package root at its leisure.
11. **No `py.typed`, so none of the annotations reach a consumer.** The library is
    thoroughly annotated and ships no marker file, which under PEP 561 means a
    type checker ignores every one of them. This is arguably a larger typing gap
    than item 4, which only concerns the untyped report dicts — and it is one
    empty file to fix.
    **Closed (Phase 3).** `src/opik_rigor/py.typed` exists and, more to the point,
    is *in the wheel* — `opik_rigor/py.typed` appears in
    `zipfile.ZipFile(wheel).namelist()`, and a clean-venv install of that wheel
    reports `(Path(opik_rigor.__file__).parent / "py.typed").exists() == True`.
    No packaging change was needed: hatchling's `packages = ["src/opik_rigor"]`
    already ships every non-Python file under the package. That was checked by
    building rather than assumed, because "the marker exists in the tree" and "the
    marker reaches a consumer" are different claims and only the second one is the
    bug.
12. **`SampleResult.exceptions` returns `Run` objects, not exceptions.** The
    obvious line `[str(e) for e in result.exceptions]` yields run reprs, silently.
    `errored_runs` would name the thing it returns.
    **Closed (Phase 3, additively).** `SampleResult.errored_runs` is the named
    accessor; `.exceptions` returns the same tuple and is documented as deprecated.
    **No `DeprecationWarning` is emitted**, deliberately: `.exceptions` is read
    inside loops and inside every consumer assertion, so a warning there produces a
    wall of test output rather than a migration, and the two names are the same
    object so nothing is at risk while a caller moves. A test pins the silence
    (`warnings.simplefilter("error")` around a read of `.exceptions`), because
    "we chose not to warn" is only a decision if it is enforced.
13. **`hash_rubric_text(data: bytes)` takes bytes despite the name**, and passing a
    `str` fails inside the library on its own `b"\r\n"` literal with
    `TypeError: replace() argument 1 must be str, not bytes` — a message that
    describes the inverse of the caller's actual mistake.
    **Closed (Phase 3, additively).** The parameter is now `bytes | str`; a `str`
    is encoded UTF-8 and hashes identically, so
    `hash_rubric_text(path.read_text(encoding="utf-8")) == hash_rubric_file(path)`
    for LF and CRLF files. Anything that is neither text nor bytes-like is refused
    at the boundary with a message naming the type it got and pointing at
    `hash_rubric_file` for the argument callers actually reach for. The tests
    check against the **published FIPS 180-4 SHA-256 vectors** for `"abc"` and the
    empty input rather than against a second call to `hashlib`, so they would catch
    the normalisation silently changing.
14. **`assert_no_regression` on text reports "current has no scores"** — "you have
    no data" when the truth is "your data is the wrong shape". Same confusion as
    item 8, and a caller who is holding 200 completions will not read that message
    as being about types.
    **Closed (Phase 3).** The opening sentence is unchanged, so anything matching
    on it still matches; what follows now names the type (`SampleResult`), the run
    count, and the first offending value (`run 0 returned str 'Paris'`) — or, when
    every run raised, the first error, because "wrong shape" and "the provider was
    down" call for opposite responses. A genuinely empty input still gets the plain
    sentence and nothing more; the diagnosis appears only when there is a diagnosis
    to give. What the gate accepts is unchanged.
15. **The wheel ships no rubric.** `rubrics/example-rubric.md` is repo-only, so
    `pip install opik-rigor` gives you a `PinnedJudge` and nothing to point it at,
    while `README.md` line 117 points at the unpackaged file. Related, and
    verified: that example ends with `OUTPUT_FORMAT_INSTRUCTION` while
    `PROMPT_TEMPLATE` already appends it, so a rubric copied from it carries the
    format block twice.
    **Closed (Phase 3).** The rubric moved to
    `src/opik_rigor/rubrics/example-rubric.md` — inside the package, so it is in
    the wheel (verified by listing `namelist()` and by copying it out of a
    clean-venv install) — and `opik_rigor.example_rubric_path()` returns its path.
    The README now tells you to copy it out of the install rather than linking a
    repository file, and the example and three test modules read it through the
    same public function, so nothing in this repo points at a path an installed
    user does not have. The double-format bug is fixed in the shipped file: the
    "Output format" section now says why it is empty. The old assertion (rubric
    *ends with* `OUTPUT_FORMAT_INSTRUCTION`) was inverted rather than deleted —
    `test_the_shipped_rubric_states_the_output_format_exactly_once` asserts the
    instruction is absent from the file and appears exactly once in the rendered
    prompt, which is the invariant the original test was reaching for.

Item 8 has a sharper form than the one recorded above, found by the same audit and
worth stating because it changes how the bug presents: `SampleResult.completed`
filters on `run.raised`, so when the default classifier raises, `.values` and
`.outcomes` come back **empty**. The caller does not merely see an extra error —
the accessor that means "give me the text back" returns nothing at all, and
`pass_rate=0.0` beside `failures=0` reads as "the system never responded."

**Partly closed in Phase 3, and the rest deliberately left.** Judged against the
sharper form, the old message did *not* already say what to do: it named the type
and offered `outcome=`, but said nothing about the run being dropped, so a caller
staring at `pass_rate=0.0`, `failures=0` and empty `.values` had no route from
what they were looking at to the sentence that explains it. `default_outcome` now
names the offending value (clipped, so a 4kB completion cannot become the
traceback), states that the run is dropped from `.values`, `.outcomes`,
`.successes` and `.completed`, spells out that this reads as `pass_rate=0.0`
beside `failures=0`, and gives the one-liner for the "I only want the values back"
case (`outcome=lambda value: True`).

The **behaviour** is unchanged and is now pinned by a test that asserts the bad
reading — `pass_rate == 0.0`, `failures == 0`, `values == ()` — precisely so that
fixing it later is a visible, deliberate act. It is a major-version change, not an
additive one: every candidate fix (a fourth outcome state, a separate field for
classifier errors, `outcome=None` meaning "do not classify") changes what a
recorded sample *means*, and therefore what `pass_rate` and `n` are, and therefore
the verdict a consumer reads off `PassRateError.stats`. migration-kit's
`COMPATIBILITY.md` §2.1–2.3 sits directly on those. It waits for 0.2.

Item 9 (no token usage on the `Adapter` seam) is likewise still open, for the same
reason: adding `complete_with_usage` to the protocol is additive for adapters but
not for code that type-checks against `Adapter`.

### 16. `is_pinned` rejects every current frontier Anthropic model id — **RESOLVED**

Found on 2026-08-13, hours after 0.1.1 shipped, by an agent designing a *third*
consumer — not by reading this repository. Verified against the published 0.1.1
wheel in a clean venv:

```
claude-opus-5                      is_pinned=False
claude-sonnet-5                    is_pinned=False
claude-opus-4-8                    is_pinned=False
claude-haiku-4-5-20251001          is_pinned=True
gpt-4o                             is_pinned=False
gpt-4o-2024-08-06                  is_pinned=True
claude-3-7-sonnet-20250219         is_pinned=True
```

The rule requires a date suffix. Anthropic's current ids do not carry one, so the
only Anthropic models `require_pinned` accepts are the ones that still use the old
dated spelling — and `claude-3-7-sonnet-20250219`, which it happily accepts, was
**retired on 2026-02-19**. The check therefore admits a model that no longer
exists and refuses every model a caller would actually reach for. `require_pinned`
is called at config load in the consumers that use it, so this is not a warning
they can route around; it refuses to start.

This is worse than an inconvenience, because the library's argument is that an
unpinned alias silently invalidates a calibration. That argument is right. The
implementation encodes a *proxy* for it — "has a date in the name" — and the proxy
has stopped tracking the thing it stood for. A vendor changed its naming
convention and rigor's gate started measuring spelling instead of stability.

The fix is not simply to widen the regex, and that is why this was not a one-liner:
whether an id re-points is a provider *policy*, and `claude-opus-5` and `gpt-5` are
the same shape with opposite answers, so no vendor-neutral rule can separate them.

**Resolved.** The rule now checks immutability directly instead of checking for a
date, in three clauses. (1) No alias token — `latest`, `newest`, `current`,
`stable`, `default`; this half was always the load-bearing one and is unchanged.
(2) The id must end in a *release designator*: splitting on `-`, `_`, `@`, `:`, the
last component must be nothing but a version (`5`, `8`, `20251001`, `06`, `v1`,
`2.1.0`). A last component that is a *word* — `sonnet`, `mini`, `4o`, `large`,
`instruct` — names a kind of model, and a kind is exactly what a provider
re-points at new weights. (3) One documented table, `_MOVING_FAMILIES`, for
providers that publish `<family>-<number>` as a moving pointer; today that is
OpenAI only, and it is the reason `gpt-5` and `gpt-4.1` are refused while
`claude-opus-5` is accepted.

The relaxation was checked in both directions, because a fix that makes everything
pass is the removal of the check rather than a repair of it. Every current
Anthropic id is accepted; `claude-3-5-sonnet-latest`, bare `gpt-4o`, `claude`,
`my-finetune`, `mistral-large`, the empty string and non-strings are still refused;
and `gpt-4.1`, which **0.1.1 wrongly accepted** as pinned (its `4.1` satisfied the
old dotted-version branch), is now correctly refused. That last one was found by
running the hand-derived verdict table against the shipped predicate and is a
second, previously unrecorded defect in the same function.

Two things the new rule deliberately does not do, both written into the module
docstring so the next person meets them before a consumer does. It **accepts** a
moving `<family>-<number>` pointer from a provider not yet in the table — the same
class of failure as the defect it replaces, now costed at one line in a table
rather than a rewrite. And it **refuses** a self-hosted id ending in a word even
when its weights never move; the fix is a `-v1` suffix, and the asymmetry is
chosen, because a false refusal is loud and a false acceptance silently
invalidates every score recorded after it.

Two things it does not *mean*, also stated in the docstring: pinned is not
*available* (`claude-3-7-sonnet-20250219` is retired and still names one immutable
version, which is what a score recorded against it in 2025 needs), and pinned is
not *correct* (nothing here checks that a model exists).

This is **additive under the compatibility rule**: no signature changed, no name
was renamed, and no id that 0.1.1 accepted is now refused *except* `gpt-4.1`, which
0.1.1 should not have accepted. Ids that were refused and are now accepted cannot
break a caller, because the only thing `require_pinned` did with them was raise.
The rejection *message* changed, which is the same kind of change 0.1.1 shipped
additively as the *message* half of item 8.

## Phase 3 — closing the recorded gaps

Items 10, 11, 12, 13, 14 and 15 are closed, plus the *message* half of item 8.
Items 8 (behaviour) and 9 are deliberately left; the reasoning is under each.
Nothing was renamed, no signature rejects a call that used to work, and no gate's
verdict moved, because 0.1.0 is on PyPI with a consumer pinned to
`>=0.1.0,<0.2`.

**Backwards compatibility was checked mechanically, against the built wheel rather
than against this working copy.** A script transcribes the 0.1.0 `__all__`, the
thirteen signatures, and the attribute-level dependencies out of
`migration-kit/COMPATIBILITY.md` §1 — a record written from *outside*, by
introspecting the published artifact — then asserts every one of them still holds
in a clean venv holding only the new wheel. It also re-checks the numbers that
decide a verdict: `assert_pass_rate`'s success-dict key set, `underpowered` and
`runs_needed` on the failure path, `lower_bound == 0.8596681784340271` for 38/40,
and the 16-key regression report. It reports no problems.

**Test counts, and why the headline number depends on the command.**

| command | result |
|---|---|
| `.venv\Scripts\python.exe -m pytest` | 515 passed, 11 skipped (was 495 / 11) |
| `.venv\Scripts\python.exe -m pytest tests examples` | 534 passed, 11 skipped (was 514 / 11) |
| `.venv-opik\Scripts\python.exe -m pytest tests examples` | 543 passed, 2 skipped (was 523 / 2) |

Twenty tests added, no test deleted. The "523 tests" figure this project quotes is
the **third** row: `pyproject.toml` sets `testpaths = ["tests"]`, so a bare
`pytest` does not collect `examples/test_example_runs.py` at all and reports 19
fewer. That is worth knowing before someone reads a bare `pytest` run as a
regression against a number that was never measured that way. One test was
rewritten rather than added — `test_shipped_rubric_ends_with_the_output_format_the_judge_parses`
became `test_the_shipped_rubric_states_the_output_format_exactly_once`, asserting
the inverse, because the thing it pinned turned out to be the bug.

## Releasing 0.1.1 — what is prepared, and what remains

Prepared on branch `release/0.1.1`, up to but not including the tag. Nothing has
been tagged, pushed, released, or uploaded.

What is done: the version is `0.1.1` in `pyproject.toml` and in
`src/opik_rigor/__init__.py`, which must always move together because
`test_every_public_name_is_importable_from_the_package_root` compares
`__version__` against install-time metadata. `CHANGELOG.md` closes the section as
`## [0.1.1] - 2026-08-13`, with `[Unreleased]` and `[0.1.1]` link definitions that
previously did not exist — a bracketed reference with no definition renders as
literal text on GitHub and on PyPI, which is what `[Unreleased]` had been doing.
`tests/test_packaging.py` is new: it builds a wheel into a temporary directory and
reads the zip, because the earlier py.typed assertion read
`Path(opik_rigor.__file__).parent`, which under an editable install is `src/` —
the source tree, i.e. exactly the thing the changelog sentence claimed it was not.

Measured on a throwaway venv holding this branch (Python 3.14.4):
`pytest tests examples` → **537 passed, 11 skipped**; `ruff check src tests
examples` clean; `python -m build` and `twine check dist/*` clean; and the built
wheel verified by installing it alone into a second empty virtualenv and importing
from there, which is the only arrangement in which the developer's own `src/`
cannot answer on the wheel's behalf.

**0.1.1 is new public API in a PATCH release, deliberately.** The reasoning is in
the changelog under the version heading; the short form is that this file reserved
0.2 for items 8 and 9 in writing, and the only known consumer pinned
`>=0.1.0,<0.2` on that reading.

**Ship it rather than sit on it.** `main` currently documents
`opik_rigor.example_rubric_path()` in the README quickstart, and the only
installable version is 0.1.0, which has no such function. That window opened when
`11da812` merged and closes when 0.1.1 is on the index. (The *published* 0.1.0
README does not mention it — that page has its own, older instance of the same
fault, recorded in the changelog.)

### The trusted-publisher registrations already exist. Do not redo them.

Both were created for 0.1.0 and are bound to owner, repository, workflow filename
and environment name — none of which change for a second release from the same
repository: `ericwehmeyer/opik-rigor`, `publish.yml`, and the environments `pypi`
and `testpypi`, which already exist too. Commit `2c7cd46` records that trusted
publishing took three attempts, all for one cause: PyPI and TestPyPI are separate
sites needing separate registrations with a different environment name each. That
cause is spent. A *pending* publisher is consumed by the first successful upload
and correctly leaves nothing to re-create, so its absence from the PyPI UI is not
evidence that anything is missing — the publisher is now attached to the project.
Re-registering "to be safe" is superstition. If an upload ever fails at the auth
step again, the fault is a name that *changed*, not one that is missing.

### Steps remaining, in order

1. **Review and merge `release/0.1.1` into `main`.** Terminal, or a PR in a
   browser if you want CI to run on the merge candidate.
2. **Confirm CI is green on `main`** across the 3.10–3.13 × Ubuntu/Windows matrix.
   *Browser* (GitHub Actions), or `gh run list`.
3. **Tag the merge commit `v0.1.1` and push the tag.** Terminal. The tag alone
   publishes nothing — `publish.yml` triggers on a *published release*, not on a
   tag push. Do not tag anything but the exact commit CI went green on.
4. **Optional dry run: dispatch `Publish` manually to reach TestPyPI.** *Browser*
   (Actions → Publish → Run workflow), or `gh workflow run publish.yml`. The two
   upload jobs are gated on the event, so a dispatch physically cannot reach the
   real index. This rehearsal is what caught the environment-name mistake before
   0.1.0's irreversible run; it is cheap and it is the reason 0.1.0 was still
   unclaimed when the real upload first failed.
5. **Create and publish the GitHub Release for `v0.1.1`.** *Browser* (Releases →
   Draft a new release). Body: the `## [0.1.1]` section of `CHANGELOG.md`.
   **Publishing the release is the irreversible act** — it is what starts the
   upload. A version can be yanked but its number can never be reused, so a wrong
   0.1.1 ships as 0.1.2.
6. **Watch the `publish to PyPI` job.** *Browser*. Its first step refuses to
   upload if the release tag and the built wheel's version disagree; `v0.1.1`
   against `version = "0.1.1"` satisfies it, and that step failing means the tag
   is wrong, not the guard.
7. **Verify from the real index, not from this tree.** In a clean virtualenv:
   `pip install opik-rigor==0.1.1`, then check that `example_rubric_path()`
   returns a file inside site-packages, that `py.typed` sits beside `__init__.py`,
   and that `SCORE_MIN` and `hash_rubric_file` import from the package root.
8. **Retire the sibling's violation.** `model-migration-kit` reaches into
   `opik_rigor.judge` for `SCORE_MIN` and `hash_rubric_file` because 0.1.0 offered
   no other route. Repoint it at the package root and move its pin to
   `>=0.1.1,<0.2`. This release exists largely for that, so it is not done until
   this is done.
9. **Record the publish here**, as the `v0.1.0` row above records its own, and
   move `[Unreleased]`'s compare link forward if anything lands after the tag.

## Decisions made, and why

**The pin rule lives in one module.** `pinning.py` is the single definition of
"reproducible model id", imported by both the adapters and the judge. It was
written before either of them precisely so the two could not drift apart on what
the word means. An id must not contain `latest`, `newest`, `current`, `stable`, or
`default`, and must end in a release designator — a release number
(`claude-opus-4-8`), a date stamp (`-20250514`, `-2024-08-06`), or an explicit
version (`-v1`, `-2.1.0`). It must also not be a member of one small, documented
table of providers that publish `<family>-<number>` as a moving pointer.

**The vendor-specific part of the pin rule is one table, on purpose.** Item 16
below is what happens when a rule written against one vendor's naming convention
outlives the convention. The replacement separates the half that is a property of
naming in general — an alias has to be *named* `latest`, and a name ending in a
word names a kind of model rather than a release of one — from the half that is
irreducibly a provider policy, which no string can reveal. The second half is
`_MOVING_FAMILIES` in `pinning.py`: one tuple, one provider today, with a comment
saying it will need updating and a docstring section saying what the rule cannot
catch. The point is that the next convention change costs a line in a table rather
than a rewrite of the predicate — and that the gap is visible in advance rather
than discovered by a consumer.

**The evidence log has no delete API.** Not an oversight — `tests/test_evidence.py`
asserts that no public name on `EvidenceLog` matches delete/remove/clear/truncate/
rotate/purge/update/pop, so adding one breaks the suite. Discarding evidence has
to be a deliberate act outside the library, which leaves a visible hole in the
timeline rather than a silent edit.

**A torn final line is tolerated on read; a malformed line anywhere else is an
error.** A process killed mid-append leaves a fragment with no trailing newline.
Dropping it costs one record; refusing to read the file costs the entire history.
A bad line in the middle is a different thing — that is corruption, and it raises.

**Line endings are normalised to LF via `.gitattributes`.** The judge hashes its
rubric file. Without this, the same rubric checked out on Windows and on a Linux CI
runner would hash differently, and the drift check would fire on every
cross-platform run — correctly, but uselessly.

**MIT license, `opik-rigor` distribution name, `opik_rigor` import name.** The
distribution name was decided at the v0.1.0 tag as the build plan required, and
is free on both PyPI and TestPyPI (checked 2026-08-13).

**The import name was originally `rigor`, and that was wrong.** The warning
previously recorded here — check the names before publishing — was acted on after
the tag, and the check found that `rigor` is already taken on PyPI by an unrelated
HTTP-API-testing DSL whose wheel installs a top-level `rigor/`. Any environment
holding both would have had one silently shadowing the other depending on path
order. For a library whose entire argument is that nothing should change
underneath you unannounced, that was not a name to keep, so the package was
renamed to `opik_rigor` and v0.1.0 retagged.

Deliberately still called `rigor`, because they live in namespaces that cannot
collide with a distribution: the `rigor_repeat` marker, the `rigor_evidence` and
`rigor_judge` fixtures, the `rigor_evidence_path` ini option, the pytest11
entry-point name, and the Opik feedback-score prefix.

The lesson generalises past this repo: the check was cheap, it was written down as
a known gap rather than acted on, and it sat unactioned right through the tag. A
gap you have recorded is not a gap you have closed.

**Session 4 README quickstart was executed, not written.** Every code block was
run in a clean virtualenv against the built wheel, and the output pasted verbatim.
The headline example deliberately *fails* — 18/20 with `min_rate=0.9` — because
that failure message is the single clearest statement of what the library is for,
and a quickstart that only shows success would be selling the wrong thing.

## Environment

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

Local venv is Python 3.14.4 with scipy 1.18.0 and pytest 9.1.1. CI runs Ubuntu and
Windows across Python 3.10–3.13.

Note that `pytest` alone honours `testpaths = ["tests"]` and therefore skips
`examples/`. Run `pytest tests examples` to get the number this file quotes.

## Invariants that must survive every later session

1. **Core never imports integrations.** `integrations/` may import core; nothing in
   `src/opik_rigor/*.py` may import `opik_rigor.integrations` or any provider SDK at module
   scope.
2. **The suite is green with no credentials present.** CI blanks `ANTHROPIC_API_KEY`
   and `OPENAI_API_KEY`. Anything needing a live endpoint is marked
   `requires_network` or `requires_opik` and deselected.
3. **`import opik_rigor.adapters` must not require a provider SDK.** The `anthropic` and
   `openai` imports are lazy, inside `complete()`.
4. **A parse failure is never a fail-verdict.** Unparseable judge output is missing
   data; scoring it as a failure biases every statistic downstream of it.
5. **Every `evaluate()` writes exactly one verdict record** — never zero, never two.

**Unit suites test modules in isolation; one integration suite tests the seam.**
`tests/test_judge.py` scripts its own adapter rather than importing `FakeAdapter`,
and `tests/test_adapters.py` never builds a judge — which keeps them independent
but proves nothing about the two fitting together.
`tests/test_integration_session1.py` covers exactly that, plus the package-level
invariants no single module owns (notably a subprocess check that importing
`opik_rigor` loads no provider SDK and no integration module).

**Credential guards live in `adapters/base.py`, not `fake.py`.** They started in
`fake.py` because it was the only provider-free module at the time; that left the
real adapters importing from the test double, which is backwards. `base.py` is
the seam module and has no provider dependency either.

**The pass-rate gate uses a ONE-SIDED bound.** `wilson_lower_bound` takes
`z = norm.ppf(confidence)`, not `norm.ppf(1 - (1-c)/2)`. A gate only ever asks "is
the true rate at least X?" — there is no upper bar to defend, so the whole error
budget goes to the floor. Using the two-sided z would silently make a gate labelled
95% behave as 97.5%. The difference is not cosmetic: for 14/20 the one-sided bound
is 0.5162 and the two-sided is 0.4810. Anyone comparing rigor's numbers against a
textbook table needs to know which convention is in force, so it is stated in the
docstring and pinned by a test that would fail under the other z.

## Known gaps entering Session 2

- **A provider exception during `evaluate()` writes no evidence.** The judge lets
  `AdapterError` propagate, so a provider outage leaves no trace in the audit
  trail. This is deliberate for now — `sampling.py` owns failure accounting and is
  where an outage should be recorded — but decide in Session 2 whether the log
  should carry a `judge.call_failure` event too. `test_a_provider_outage_
  propagates_rather_than_scoring_zero` pins the current behaviour.
- **The judge's score range is hardcoded to 1–5** (`SCORE_MIN`/`SCORE_MAX` in
  `judge.py`, interpolated into the prompt so rubric, prompt, and validator cannot
  disagree). Fine for v0.1; a configurable range is a roadmap item, not a v0.1 gap.
- Coverage is measured but no threshold is enforced in CI. Decide a floor in
  Session 2, once the statistics modules exist and the number means something.
- ~~CI has never actually run — the workflow is written but there is no remote yet.~~
  Resolved: pushed to `github.com/ericwehmeyer/opik-rigor`, and the matrix went
  green on the first run across py3.10–3.13 on Ubuntu and Windows.
