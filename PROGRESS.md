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
| — | Release 0.1.1 | **v0.1.1 published to PyPI 2026-08-13** — `pip install opik-rigor==0.1.1` |
| 6 | Documentation defects found by a cold-start stranger installing from PyPI | **complete** — see [Documentation defects](#documentation-defects-found-from-outside-2026-08-14) |
| 7 | Adversarial review of the published 0.1.1: statistics, pinning, adapters, packaging, typing | **complete, unreleased** — 1039 passed, `verify_release.py` 17/17 |

### Unreleased on `main`, 2026-08-14 — read this before quoting the source tree

**PyPI serves 0.1.1. `main` is well ahead of it, and one change is not additive.**
Anyone reasoning about what a user gets must read the published wheel, not `src/`.
The sibling project depends on this library and has been bitten by exactly that
confusion.

**Needs a version decision before release.** `wilson_lower_bound` and
`assert_pass_rate` now refuse `confidence <= 0.5`, which they used to accept and
answer. That is the only signature narrowing in the release — everything else is
additive — but under semver it is the difference between 0.1.2 and 0.2.0. It is a
defect fix: below 0.5 the z is negative, so the "lower bound" rose *above* the
observed rate, more data made it worse, and `assert_pass_rate((20, 20), 1.0,
confidence=0.5)` **passed** — "twenty runs prove perfection", the exact claim the
module's opening paragraph exists to refuse.

What else is on `main` and not on PyPI:

- **`is_pinned` was rewritten.** 0.1.1 refuses `claude-opus-5`, `claude-sonnet-5`
  and `claude-opus-4-8`, accepts `claude-3-7-sonnet-20250219` (retired February
  2026), and wrongly *accepts* `gpt-4.1`, an OpenAI alias that re-points.
- **`AnthropicAdapter` sent `temperature` unconditionally**, which every current
  Anthropic model rejects with a 400 (roadmap item 19). With the pin rule, these
  were sequential blockers: the gate was the front door locked, the 400 was there
  being no room behind it. **Both are needed before this library can judge with a
  current model, and both are unreleased.**
- **A score-distribution gate returned green on infinite input** — `inf - inf` is
  `nan`, and `nan > 0.001` is False, so a 0.001 spread gate passed on a sample
  containing infinity.
- **`py.typed` was a false claim.** `AnthropicAdapter(..., api_key="sk-…")`
  type-checked clean under mypy *and* pyright and raised `TypeError` at runtime,
  because `**forbidden: object` accepts everything. Now `NoReturn`, and
  `verify_release.py` runs `mypy --strict` against the **extracted wheel**.
- **`import opik_rigor` went from ~1019 ms to ~303 ms** — scipy is imported lazily
  and `assert_pass_rate` no longer pulls it at all.
- **The worked example ships inside the wheel** as `opik_rigor.examples`. The
  README's headline command previously named `examples/summarise_eval.py`, a path
  that exists only in a git checkout, and the README also printed a line built
  from a key (`observed`) that the returned dict does not have.
- **The sdist was sweeping in the working directory** — a local build produced 122
  members, 71 of them a second copy of the tree under `.claude/worktrees/` plus 25
  `.remember/` files. The published 0.1.1 sdist is clean, verified by downloading
  it; CI builds from a fresh checkout where those directories do not exist. That
  was protection by accident of environment, and it is now an allowlist plus a
  test.

#### What to do next, in order

1. **Decide the version number.** `confidence <= 0.5` is the only non-additive
   change. 0.1.2 says "bug fix"; 0.2.0 says "we removed accepted inputs". The
   consumer (`model-migration-kit`) pins `>=0.1.1,<0.2`, so **0.2.0 would not
   reach it without a bound change there** — which is an argument for 0.1.2 and
   also exactly the kind of argument that should be made deliberately rather than
   by default.
2. **Ship it.** The two Anthropic blockers are both fixed on `main` and neither is
   published, so today `pip install opik-rigor` still cannot judge with a current
   model. That is the strongest reason to cut a release soon.
3. **Note the README problem.** PyPI freezes the long description at upload, so
   0.1.1's project page keeps its old README — including the dead
   `examples/summarise_eval.py` command — until a new version goes up. It does not
   self-heal.
4. Items 20 and 21 below are open and neither blocks a release.

#### Three lessons this session, all the same shape

A check that passes where it was written and fails only where it matters:

- `twine check`'s output is **colourised under GitHub Actions**, so a gate
  counting lines ending in `PASSED` counted zero on a healthy build.
- Two tests hardcoded a Windows path; on POSIX `Path(r"C:\x\site-packages").name`
  is the whole string, so four Ubuntu cells failed while Windows passed. The
  sibling had the mirror-image failure the same night.
- `wheel-annotations` ran `mypy --strict` over a wheel containing
  `import pytest`, in a build job that did not install pytest.

Each was invisible locally. The general form is worth stating: **an expectation
can quietly encode the environment of whoever wrote it**, and the only instrument
that finds that is running it somewhere else.

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
| `src/opik_rigor/examples/summarise_eval.py` + `examples/README.md` | done — **moved inside the package** on 2026-08-14, so `python -m opik_rigor.examples.summarise_eval` works from a bare install | `examples/test_example_runs.py` — 19, plus two wheel checks in `tests/test_packaging.py` |

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

### 17. `import opik_rigor` costs a second, for a gate most suites never call

Found on 2026-08-14 by measuring the published 0.1.1 install rather than by reading
the tree — the friction is invisible in source, because the import that causes it is
one perfectly ordinary line. `src/opik_rigor/distribution.py:33` imported
`from scipy.stats import mannwhitneyu, norm` at module scope, and `__init__.py`
imports from that module, so `import opik_rigor` pulled in all of `scipy.stats` —
which drags `scipy.optimize`, `scipy.spatial`, `scipy.sparse` and `scipy.linalg`
behind it.

The cost is paid by every suite that imports the package, including the
overwhelming majority that only ever call `assert_pass_rate`. It is worse than a
one-off: this library registers a `pytest11` entry point, so the import happens at
*collection*, before a single test runs, on every pytest invocation in a project
that has it installed — including invocations of test files that never touch it.
A consumer running the gate in a pre-commit hook pays it on every commit.

**What the fix is not, and this is the whole point.** `mannwhitneyu` has exactly
one call site, so deferring the scipy import into `assert_no_regression` looks like
the whole answer. It is not. `norm.ppf` has three call sites — `wilson_lower_bound`,
`wilson_interval` and `_runs_needed` — and the first of those sits directly on the
pass-rate path, the most-used gate in the package. A lazy import that the first gate
call immediately triggers is not a lazy import.

That is measurable rather than arguable. Three trees were timed **interleaved**, so
background load hit all three equally — `BEFORE` (main @ `df93f43`), `LAZY-ONLY`
(both scipy names deferred into functions, but the Wilson z still `norm.ppf`), and
`AFTER` (this branch). Warm, minimum of 20 runs, milliseconds:

```
scenario                              BEFORE   LAZY-ONLY      AFTER
interpreter floor                       36.6        39.3       45.3
import opik_rigor                     1018.6       213.4      247.2
import + assert_pass_rate             1070.6      1096.7      251.0
import + assert_score_distribution    1320.5       248.6      247.9
import + assert_no_regression         1143.8      1048.1     1023.2
```

Read the `LAZY-ONLY` column downward. Its bare import is fast — 213.4 ms, the fix
apparently working — and then `assert_pass_rate` costs **1096.7 ms**, no better than
the 1070.6 ms it was trying to improve on, because the gate imports the scipy the
module just finished not importing. `assert_score_distribution`, which never touches
`norm.ppf`, *is* genuinely fixed at 248.6 ms. So the half-fix does not merely
under-deliver; it moves the cost from a place you would measure to a place you would
not, and leaves the single most-called gate exactly where it was.

**The fix, and it is additive.** `statistics.NormalDist().inv_cdf` is CPython's
implementation of Wichura's AS241, in the standard library since 3.8; this package
already requires >= 3.10. Replacing `norm.ppf` with it *and* moving `mannwhitneyu`
inside `assert_no_regression` takes warm import from 1018.6 ms to 247.2 ms against a
~40 ms interpreter floor, and — the part that matters — takes `assert_pass_rate`
from 1070.6 ms to 251.0 ms. The absolute figures move with machine load; the columns
above were measured against each other in one interleaved run for exactly that
reason. `assert_no_regression` is unchanged (1143.8 → 1023.2 ms, a difference inside
the run-to-run spread): it still pays the full SciPy import on first call, which is
the one caller that should.
Nothing is renamed, no signature rejects a call it used to accept, and no gate's
verdict moves — the equivalence evidence is in CHANGELOG.md and in
`tests/test_import_cost.py`. **So this does not wait for 0.2, and it shipped
additively.**

**Two adjacent changes do wait for 0.2, and the ordering matters.** The obvious
next moves are an `opik-rigor[regression]` extra that takes SciPy out of the base
install, and dropping NumPy so the base install is stdlib-only. Both are runtime
breaks: `pip install opik-rigor` followed by `assert_no_regression` would raise
where it used to work, and this project's convention — written down before 0.1.1
shipped, and relied on by a consumer pinned `>=0.1.0,<0.2` — is that `0.MINOR`
means breaking. The ordering matters in the other direction too: item 17 had to
land *first*, and additively, because it is what makes the extra worth having.
Without it, `opik-rigor[regression]` would remove a dependency the base install
still imports at module scope, which is not a smaller install but a broken one.
Deferring the import is the prerequisite; moving the dependency is the follow-on.
Do them in that order, one release apart, and no consumer is ever caught between.

### 18. NumPy is imported and never declared

Found the same day, by reading `pyproject.toml` next to the file that imports.
`dependencies` listed only `scipy>=1.10`, while `distribution.py:32` imported
`numpy as np` at module scope and used it in `_coerce_pass_data`, `_coerce_scores`
and every score summary. The public docstring around `distribution.py:604-608`
goes further and pins the reported statistics to NumPy's exact semantics: **mean**
is `numpy.mean`, **p10** is `numpy.percentile` with NumPy's default *linear*
interpolation, **stddev** is `numpy.std(ddof=1)`. That is a documented promise
about a library the package never said it needed.

It worked only because SciPy requires NumPy transitively. That is an accident, not
a contract — and it is exactly the accident item 17's follow-on destroys: the day
SciPy moves behind an extra, a base install has no NumPy, and `import opik_rigor`
fails on a line nobody changed. **Declaring it is additive and shipped with item
17** (`numpy>=1.21`, where the current scalar type names and `numpy.typing`
surface settled; `scipy>=1.10` already requires `>=1.19.5`, so no existing
environment is excluded). `tests/test_import_cost.py` asserts the declaration is
there, and asserts SciPy is absent from every extra, so the 0.2 change has to be
deliberate.

**Removing NumPy is the part that waits for 0.2**, and it waits on test debt
rather than on the calendar. A prototype that replaced it with `statistics` and
`math` passed the entire suite and still silently broke `np.bool_`, `np.integer`
and `np.float32` acceptance in `_coerce_pass_data` and `_coerce_scores` — inputs
that arrive routinely from array code and CSV round-trips, and that **no test
covers**. Until those coercion paths have tests that would notice, dropping NumPy
is a change whose regression nothing in this repository can catch, which is the
one kind of change this project does not make.

### 19. `AnthropicAdapter` cannot call any current frontier Anthropic model — **RESOLVED 2026-08-14**

**Resolved by the first option below, with a piece of the second.** `temperature`
now defaults to `None`, meaning the key is *absent from the request* rather than
set to something chosen for you, and it is omitted unconditionally — on every
model, current or older. An explicitly-passed value is still sent to a model that
accepts one; against a model that does not, it is refused at **construction**,
with a message naming the model, rather than becoming a 400 partway through a run
that has already spent calls.

Omitting unconditionally is the part worth recording. A model-conditional default
would put the vendor table on the happy path, so the day Anthropic ships a model
the table has not heard of, the zero-argument constructor starts returning 400s
again — the same defect, waiting on a release date. Omitting always is correct on
every model Anthropic serves and cannot rot. The table (`_SAMPLING_REMOVED`) is
therefore consulted *only* to reject an explicit value, which means its staleness
costs exactly one thing: an explicit `temperature` against a model released after
the table was written produces the vendor's 400 instead of our `ValueError` —
which is the behaviour that shipped before the table existed. It can never break a
call that works.

What it costs: an older model that still accepts sampling parameters now gets the
API default rather than `0.0` unless asked. On a current model that lever no
longer exists at all, which is a fact about the provider and not a choice this
library gets to make.

Verified against the merged tree: the zero-argument constructor yields
`temperature=None` for `claude-opus-5`, `claude-sonnet-5`, `claude-opus-4-6` and
`claude-haiku-4-5-20251001`; `AnthropicAdapter("claude-opus-5", temperature=0.0)`
raises `ValueError`; `AnthropicAdapter("claude-opus-4-6", temperature=0.0)` keeps
`0.0`; the gateway spelling `anthropic.claude-opus-5` matches the table and the
finetune name `my-claude-opus-5-tuned` does not. Tests assert the *absence* of the
key in the built payload, not merely that a call succeeds.

**With items 16 and 19 both closed, rigor can judge with a current Anthropic
model.** Those were the two blockers and they were sequential: 16 was the front
door locked, 19 was there being no room behind it.

The original record follows, because the reasoning that chose between the options
is the part worth keeping.

---

Found on 2026-08-14 by a coordinating agent reading Anthropic's own migration
reference against `src/opik_rigor/adapters/anthropic.py`, and confirmed here by
reading the call shape rather than by spending a credential. The adapter passed
`temperature` on every request:

```python
# src/opik_rigor/adapters/anthropic.py:96-101
message = client.messages.create(
    model=self._model_id,
    max_tokens=self._max_tokens,
    temperature=self._temperature,          # constructor default 0.0, line 50
    messages=[{"role": "user", "content": prompt}],
)
```

`temperature`, `top_p` and `top_k` were **removed** on Claude Opus 5, Opus 4.8,
Opus 4.7, Sonnet 5 and Fable 5: sending any of them returns a **400**. On Sonnet 5
the rule is narrower — a *non-default* value returns 400 — and the API default is
`1.0`, so this adapter's `0.0` is non-default and 400s there too. There is no
configuration a caller can supply that avoids it: `temperature` has no sentinel
meaning "omit", the constructor validates `0.0 <= temperature <= 1.0` and so
rejects `None`, and the parameter is passed unconditionally. **Every call this
adapter makes to a current Anthropic model fails at the API boundary.** It fails
loudly at least — `complete()` wraps provider exceptions, so the caller sees
`AdapterError: anthropic call failed for model 'claude-opus-5': BadRequestError:
...` rather than a wrong answer — but it fails on every call.

**This is worse than item 16 was, and until item 16 was fixed the two compounded.**
Item 16 was a gate that *refused to start* with a current model id; a consumer
could route around it by not calling `require_pinned`, which was a bad workaround
but was a workaround. This one is at the API call itself, so routing around the
gate buys a 400 instead of a judgement. Item 16 was the front door being locked;
item 19 is there being no room behind it. **Item 16 is now fixed and this one is
not**, which means the door opens onto the 400: `PinnedJudge(AnthropicAdapter(
"claude-opus-5"), ...)` now constructs and then fails on its first `complete()`.
That is a better failure than the previous one — it is at the call, with a message
naming the model and the provider's own error — but rigor still cannot judge with
anything Anthropic currently serves until this item lands.

**Why this is recorded rather than implemented.** The obvious fix — omit
`temperature` when the model id looks current — is a runtime behaviour change on a
published adapter, and it is not purely internal: the constructor validates the
range and the class exposes a `temperature` property, both of which are public
surface a consumer may read. The honest options:

* **Omit when unset, honour when set** — give `temperature` a `None` default
  meaning "do not send", keep the validation for explicitly-passed values, and
  keep the property (returning `None`). This is *additive*: no existing call is
  rejected, and a caller who passed `0.0` explicitly against an older model keeps
  getting `0.0`. It changes the default's behaviour, which is the part that needs
  argument — but the current default's behaviour is a 400, so nothing that works
  today stops working.
* **Per-model parameter rules** — a table of which models accept which sampling
  parameters, so the adapter sends what the target actually supports. Correct, and
  it acquires the same maintenance problem item 16 had: a vendor changes its
  surface and the table silently stops tracking it. Item 16's fix is the precedent
  for how to hold that honestly rather than avoid it — `_MOVING_FAMILIES` in
  `pinning.py` is one table, commented as needing updates, with the gap it leaves
  written into the module docstring instead of discovered by a consumer. A table
  whose staleness is documented and localised is a different thing from a rule
  that has silently stopped tracking reality, which is what item 16 was.
* **Drop the parameter entirely** — smallest code, and it removes a public
  property and a constructor keyword. That is breaking, and waits for 0.2.

The first option looks additive enough to ship before 0.2, and is the recommended
one; it needs a test that asserts the key is *absent* from the request payload,
not merely that the call succeeds. Deciding between it and the per-model table is
the open question, and it is deliberately left open here rather than settled in a
branch whose subject is import cost.

**How it was settled:** the first option, plus the second's table used *only* on
the explicit-value path so it can never sit between a default-constructed adapter
and a working call. See the resolution note at the top of this item.

### 20. The pytest plugin costs every pytest process on the machine 88.7 ms

Measured 2026-08-14: `pytest --collect-only` costs **+88.7 ms** with the plugin
enabled versus `-p no:rigor` — min of 15 interleaved runs in one environment — and
it is paid by every suite on the machine, including suites that never use the
marker. A pytest11 entry point is loaded at interpreter start whether or not
anything asks for it, which is the price of the zero-configuration discovery
`COMPATIBILITY.md` argues for, and the argument is still right; the cost was
simply never stated.

The cause is not the plugin. Importing it runs `opik_rigor/__init__.py`, which
re-exports `.distribution`, which imports numpy. **Deferring the plugin's own
imports was tried and measured to change nothing**, and reverted rather than
shipped as churn with a rationale it had not earned — which is the right instinct
and worth keeping as the record of what does *not* work.

The real fix is PEP 562 lazy re-exports on the package's front door, and it is not
obviously worth it: `__getattr__` moves when an `AttributeError` surfaces, and
`verify_release.py`'s `wheel-exports-importable` check probes exactly that
surface. So it trades a measured 88.7 ms against a weakened release gate. Open
deliberately.

### 21. `sample_of()` cannot do the thing its own docstring says it is for

Its docstring says "notably feeding a stored baseline into the regression gate".
It cannot:

```python
>>> sample_of([4.0] * 8).scores()
()
```

`scores()` harvests `getattr(run.value, "score", None)`, and a float has no
`.score`, so every value is dropped. With the default outcome all eight runs also
land in the *errored* bucket. Reproduced against the merged tree on 2026-08-14.

The workaround is trivial — pass the list straight to the gate, since `ScoreData`
accepts a `Sequence[float]` — so nothing is blocked. Left open because the fix is
genuinely ambiguous and picking one silently would be the wrong move: either the
docstring is wrong and should stop advertising the path, or `scores()` should
accept plain numbers, which widens a public method's contract. That is a decision,
not a typo.

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

**Test counts as of Phase 3, and why the headline number depends on the command.**
These are the numbers Phase 3 measured, kept as its record. The current ones are
under [Documentation defects](#documentation-defects-found-from-outside-2026-08-14)
below, and they are larger because `main` has taken property-based tests since.

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

## Releasing 0.1.1 — what shipped, and what remains

**0.1.1 was published to PyPI on 2026-08-13.** `pip install opik-rigor` now
resolves to it, and a clean-venv install was re-checked on 2026-08-14 while fixing
the documentation defects below:

```
python -m venv venv-bare
venv-bare\Scripts\python.exe -m pip install --no-cache-dir opik-rigor
venv-bare\Scripts\python.exe -c "import opik_rigor; print(opik_rigor.__version__)"
```

→ `0.1.1`, from `venv-bare\Lib\site-packages\opik_rigor\__init__.py`. The
paragraphs below were written before the upload and are kept as the record of what
was prepared; the wording has been corrected where it said "not released", which is
the same defect this file exists to catch — a document that is true of the moment
it was written and false of the world it describes.

What was done before the tag: the version is `0.1.1` in `pyproject.toml` and in
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

**Ship it rather than sit on it.** ~~`main` currently documents
`opik_rigor.example_rubric_path()` in the README quickstart, and the only
installable version is 0.1.0, which has no such function.~~ That window opened when
`11da812` merged and **closed on 2026-08-13 when 0.1.1 reached the index**. (The
*published* 0.1.0 README does not mention it — that page has its own, older
instance of the same fault, recorded in the changelog.)

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

### Steps, in order — 1 to 7 are done

Steps 1–7 were carried out on 2026-08-13 and the upload succeeded; they are kept
because the next release repeats them. **Steps 8 and 9 are the ones still open.**

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

## Documentation defects found from outside (2026-08-14)

Ten defects, reported by a cold-start stranger who installed `opik-rigor` 0.1.1
from PyPI and followed `README.md` literally. Every one of them is the same fault
this file has already recorded twice under other names: **a claim that is true of
the source tree and false of the artifact a user installs.** The full list and the
per-defect reasoning is in `CHANGELOG.md` under `[Unreleased]`; what belongs here
is the state it leaves the build in.

**The worked example moved into the package.** `examples/summarise_eval.py` is now
`src/opik_rigor/examples/summarise_eval.py`, and the README's last quickstart line
is `python -m opik_rigor.examples.summarise_eval --seed 7 --n 40`. The old address
was a directory in the git tree; the wheel is `opik_rigor/` plus
`opik_rigor-0.1.1.dist-info/` and nothing else, so the command the quickstart ended
on could not be run by anyone who had followed the install line above it. This is
the same call that was already made for the rubric in Phase 3, item 15, and it was
made the same way: ship the asset, do not delete the claim. `examples/` keeps its
walkthrough and its subprocess test; the test now invokes `-m`, never a file path.

**One new release check, `readme-paths`.** It reads every address the README hands
a reader — markdown link targets, path arguments inside fenced code blocks, and
`python -m` targets under `opik_rigor` — and asserts that each one resolves for
somebody standing on PyPI or on an install rather than in a checkout. Wired into
`scripts/verify_release.py` beside `readme-symbols` and unit-tested in
`tests/test_release_checks.py` against hand-derived tables (L, C and M) and
synthetic wheels, in the style of the rest of that file. Run against the README as
it stood before this session it reports four unreachable addresses and blocks the
release; run against the current one it passes.

The check earns its place twice over: it also caught a sentence *added during this
session*. A first draft of the extras paragraph wrote
`from opik_rigor import log_sample_to_opik` in prose to say that the import does
not work, and `readme-symbols` — correctly — read it as a claim that it does.

**Measured on this branch** (Windows, Python 3.14.4, `PYTHONPATH` pointed at this
worktree's `src/` so that the shared `.venv`'s editable install of the main
checkout cannot answer instead):

| command | result |
|---|---|
| `.venv\Scripts\python.exe -m pytest` | **959 passed, 11 skipped, 1 xfailed** |
| `.venv\Scripts\python.exe -m pytest tests examples` | **978 passed, 11 skipped, 1 xfailed** |
| `.venv\Scripts\python.exe -m ruff check src tests scripts examples` | clean |
| `.venv\Scripts\python.exe scripts\verify_release.py` | **16 passed, 0 failed, 0 skipped** |

Measured after merging `main`, which landed the pinning rewrite, the import-cost
work and the property-based suites in the same window; before that merge the same
two commands on this branch reported 691 and 710. Both numbers are recorded
because a reader comparing against either has to know which tree it was.

`pytest` alone still honours `testpaths = ["tests"]` and therefore collects 19
fewer than `pytest tests examples`; both numbers are above so neither can be read
as a regression against the other.

**The `.venv-opik` row is deliberately absent, and that is itself a finding.**
`.venv-opik\Scripts\python.exe -m pytest tests examples` reports 976 passed, 12
skipped, 1 xfailed and **1 failed** —
`test_every_public_name_is_importable_from_the_package_root`, on its last line,
which compares `opik_rigor.__version__` against `importlib.metadata.version`. That
venv holds an installed `opik_rigor-0.1.0.dist-info` while the tree says `0.1.1`,
so the failure is a stale environment and not a defect in this branch. It is
recorded rather than quietly omitted: reinstalling that venv is a one-line fix
(`.venv-opik\Scripts\python.exe -m pip install -e ".[dev]"`) and a number nobody
can reproduce is worse than no number.

**Install cost, measured because the README now states it.** Each into its own
empty virtualenv, cold (`--no-cache-dir`), Python 3.14.4 on Windows:

| requirement | time | venv size | packages |
|---|---|---|---|
| `opik-rigor` | 54.0 s | 11.5 MiB → 192.8 MiB | 4 |
| `opik-rigor[opik]` | 287.5 s | 11.5 MiB → 414.9 MiB | 74 |

scipy is 109.6 MiB and numpy 31.2 MiB, plus 19.3 and 20.1 MiB of bundled shared
libraries; rigor's own code is 0.4 MiB. The extra adds litellm (101.9 MiB), openai,
tokenizers, huggingface-hub, hf-xet, tiktoken, sentry-sdk, three `tree-sitter`
grammars — and pytest, which Opik pulls in for its own plugin.

**A README fix does not reach an existing release.** A project's long description
is frozen at upload, so the page rendered on PyPI for 0.1.1 is still the old README
— dead links, elided hashes, unrunnable example and all — and stays that way until
a new version is uploaded. Anyone reading this should treat that as the reason to
cut 0.1.2, not as something that self-heals.

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
.\.venv\Scripts\python.exe -m pytest tests examples
.\.venv\Scripts\python.exe -m ruff check src tests scripts examples
.\.venv\Scripts\python.exe scripts\verify_release.py
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
