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

## Session 1 — module status

| Module | State | Tests |
|---|---|---|
| `src/opik_rigor/errors.py` | done | exercised via the modules that raise |
| `src/opik_rigor/evidence.py` | done | `tests/test_evidence.py` — 72 |
| `src/opik_rigor/pinning.py` | done | `tests/test_pinning.py` — 21 |
| `src/opik_rigor/adapters/base.py` | done — Protocol, env constants, credential guards | via adapter tests |
| `src/opik_rigor/adapters/{fake,anthropic,openai_compat}.py` | done | `tests/test_adapters.py` — 66 (+1 network-skipped) |
| `src/opik_rigor/judge.py` | done | `tests/test_judge.py` — 46 |
| `rubrics/example-rubric.md` | done | asserted to end with `OUTPUT_FORMAT_INSTRUCTION` |
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

## Decisions made, and why

**The pin rule lives in one module.** `pinning.py` is the single definition of
"reproducible model id", imported by both the adapters and the judge. It was
written before either of them precisely so the two could not drift apart on what
the word means. An id must end in a concrete version marker (`-20250514`,
`-2024-08-06`, `-v1`, `-2.1.0`) and must not contain `latest`, `newest`,
`current`, `stable`, or `default`.

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
