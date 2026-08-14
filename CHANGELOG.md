# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Import cost, and the dependency that was never declared. Both changes are
**additive**: nothing is renamed, no signature rejects a call it used to accept,
and no gate's verdict moves. The items under *Not fixed, and why* below remain
queued for 0.2, which is where this project puts changes that alter what a
recorded sample means.

### Changed

- **`import opik_rigor` no longer imports SciPy** — 1018.6 ms of warm import
  becomes 247.2 ms, and `assert_pass_rate` 1070.6 ms becomes 251.0 ms, against a
  ~40 ms interpreter floor. `distribution.py` imported
  `from scipy.stats import mannwhitneyu, norm` at module scope, and `__init__.py`
  imports from that module, so every consumer paid for all of `scipy.stats` — and
  with it `scipy.optimize`, `scipy.spatial`, `scipy.sparse` and `scipy.linalg`.
  This package registers a `pytest11` entry point, so the cost landed at
  *collection*, on every pytest run in a project that has it installed.

  Two changes, and both were needed. `mannwhitneyu` moved inside
  `assert_no_regression`, its only caller. The Wilson `z` now comes from
  `statistics.NormalDist().inv_cdf` — CPython's Wichura AS241, stdlib since 3.8,
  and this package already requires >= 3.10 — rather than `scipy.stats.norm.ppf`.
  Deferring the SciPy import *alone* does almost nothing, and the measurement is
  worth stating because the failure is invisible from the obvious benchmark. Timed
  interleaved against a tree with both scipy names deferred but the Wilson z still
  on `norm.ppf` (warm, min of 20, ms):

  ```
  scenario                              BEFORE   LAZY-ONLY      AFTER
  interpreter floor                       36.6        39.3       45.3
  import opik_rigor                     1018.6       213.4      247.2
  import + assert_pass_rate             1070.6      1096.7      251.0
  import + assert_score_distribution    1320.5       248.6      247.9
  import + assert_no_regression         1143.8      1048.1     1023.2
  ```

  The lazy-only tree's *import* looks fixed at 213.4 ms, and then its
  `assert_pass_rate` costs 1096.7 ms — no better than the 1070.6 ms it replaced,
  because `norm.ppf` sits on the pass-rate path and the first gate call triggers
  the import the module just deferred. The pass-rate and score-distribution gates
  are now off SciPy entirely; the regression gate still pays the full import on
  first call, which is the one caller that needs it, and is unchanged
  (1143.8 → 1023.2 ms, inside the run-to-run spread). Absolute figures move with
  machine load, which is why all three arms were measured in one interleaved run.

  **SciPy remains a hard dependency**, so `pip install opik-rigor` continues to
  give a working `assert_no_regression`. Mann-Whitney U is *not* reimplemented:
  SciPy's carries the tie-averaged ranks and the cached exact null distribution
  that make the p-value trustworthy, and this suite has no independent oracle for
  a p-value — `tests/test_distribution.py` hand-counts the U statistic but asserts
  only `p_value < 1e-4`, which it describes as generous. A reimplementation would
  ship with nothing able to catch it being wrong.

  **No gate's verdict changes, and that was proved rather than assumed.**
  `NormalDist().inv_cdf` was swept against `norm.ppf` over 6,401,023 points of the
  open interval (0, 1) — a dense uniform grid, log-spaced approaches to both
  endpoints down to `5e-324` and `nextafter(1.0, 0.0)`, uniform random draws,
  random raw bit patterns across every exponent band, and every confidence a
  caller would plausibly type. Maximum absolute deviation 2.84e-14, maximum
  relative deviation 1.22e-15, maximum 8 ULP. Propagated through `_wilson` over
  87,486 one- and two-sided bounds the worst deviation *shrinks* to 3.33e-16.
  `_runs_needed`, which answers in whole runs, changed **0 of 72,814** answers.
  Across **1,443,519** combinations of confidence x successes x n x `min_rate`,
  **0** verdicts flip.

  **The one real residual, stated rather than hidden.** The last ULP of a
  full-precision float in an evidence log can move:
  `wilson_lower_bound(18, 20)` was `0.7383369536731331` and is now
  `0.7383369536731332`. Every number this package *prints* is formatted to four
  decimals, so every failure message is byte-identical — the README's pasted
  `PassRateError` example reproduces character for character, and a test pins the
  formatted form. A consumer diffing raw `lower_bound` floats between 0.1.1 and
  this release at full `repr` precision may see a final-digit difference; one
  comparing them at any tolerance, or reading the printed report, will not.

### Added

- **`numpy>=1.21` is now a declared dependency.** It always was one in fact:
  `distribution.py` imports NumPy directly, and `assert_score_distribution`'s
  public docstring pins `mean`, `p10` and `stddev` to NumPy's exact semantics
  (`numpy.mean`, `numpy.percentile` with linear interpolation,
  `numpy.std(ddof=1)`). It worked only because SciPy pulls NumPy transitively —
  an edge that disappears the moment SciPy ever moves behind an extra. No
  environment is newly excluded: `scipy>=1.10` already requires `numpy>=1.19.5`.

- **`tests/test_import_cost.py`** — asserts, in fresh subprocesses, that
  `import opik_rigor` and the pass-rate and distribution gates load no `scipy`
  module, that `assert_no_regression` does, that NumPy is declared and SciPy is in
  no extra, that the stdlib quantile matches `norm.ppf` (which is now available as
  an *independent* oracle, since `src/` no longer uses it), and that the formatted
  bound the README prints is unchanged.

- **A named error when SciPy is missing.** Deferring the import makes a
  scipy-free environment fail inside `assert_no_regression` rather than at
  `import opik_rigor`, where a bare `ModuleNotFoundError: No module named 'scipy'`
  would read as a bug in this package. It now names what is missing, which gate
  wanted it, why the test is not reimplemented, how to install it, and that the
  other two gates keep working without it. It is still a `ModuleNotFoundError`
  chained to the original, so `except ImportError` around the call still catches
  it — the improvement is the text, not a new exception type to handle.

## [0.1.1] - 2026-08-13

Seven gaps closed, all of them reported by the first external consumer against
the published 0.1.0 wheel rather than found by reading this repository. Every
change is **additive**: nothing is renamed, no signature rejects a call it used
to accept, and no gate's verdict moves. A consumer pinned to `>=0.1.0,<0.2` can
take this release without reading it. One more defect was found while preparing
the release, by reading the artifact on PyPI instead of the tree, and is the first
entry under **Fixed**.

**New public API in a PATCH release is a deliberate deviation from strict
SemVer.** Strict SemVer makes any addition to the public API a MINOR bump, which
would put this at 0.2.0. This project instead adopted `0.MINOR = breaking` in
writing before this release existed: `PROGRESS.md` records items 8 and 9 as
waiting for 0.2 precisely because they change what a recorded sample means, and
the only known consumer pinned itself `>=0.1.0,<0.2` on that reading. Publishing
purely additive work as 0.2.0 would lock that consumer out of a release that
cannot break it, in order to honour a rule about a number. So the additions ship
as 0.1.1, and the reservation of 0.2 stands.

### Added

- **`py.typed`.** The library is annotated throughout and shipped no PEP 561
  marker, which means a type checker had to discard every one of those
  annotations in an installed copy. One empty file — and `tests/test_packaging.py`
  builds a wheel into a temporary directory and reads the zip's namelist for it,
  rather than reading the source tree, because a marker sitting in the tree is
  exactly the state 0.1.0 shipped from while its wheel carried nothing. The same
  file checks `opik_rigor/rubrics/example-rubric.md`, and imports the public names
  out of the *extracted* wheel while asserting that `opik_rigor.__file__` really
  resolved there — a developer's own `src/` answers otherwise, and the check would
  pass against an empty wheel. `build` and `hatchling` join the `dev` extra so it
  runs rather than skips; where they are absent it skips with a message naming what
  went unverified.
- **`SCORE_MIN`, `SCORE_MAX`, `hash_rubric_file` and `hash_rubric_text` are
  exported from the package root** and are in `__all__`. They were public in
  spirit and unreachable in practice: a consumer that imputes a score for an
  ungradeable response has to know where the bottom of the scale is, and one that
  hashes a judge config has to hash the rubric exactly as rigor does or the two
  disagree about whether the instrument changed. Re-deriving either — a
  hard-coded `1.0`, a hand-rolled sha256 — is the drift this library exists to
  catch, so reaching into `opik_rigor.judge` was the only honest option and it
  should not have been necessary.
- **`SampleResult.errored_runs`**, the correctly named accessor for the runs that
  raised. `SampleResult.exceptions` returns the same tuple and keeps working
  exactly as before; it is documented as deprecated and emits **no**
  `DeprecationWarning`, because that attribute is read inside loops and inside
  every downstream assertion, and a warning there buys a wall of test output
  rather than a migration.
- **The example rubric ships inside the package**, at
  `opik_rigor/rubrics/example-rubric.md`, reachable as
  `opik_rigor.example_rubric_path()`. `pip install opik-rigor` previously gave you
  a `PinnedJudge` and nothing to point it at — and this is worth stating as the
  admission it is rather than as an addition. **0.1.0's own published README told
  an installed user to "save a rubric as `rubric.md` (the one in
  `rubrics/example-rubric.md`)", and the 0.1.0 wheel does not contain that file.**
  Verified against the artifact rather than the tree: the description baked into
  `opik_rigor-0.1.0-py3-none-any.whl` on PyPI is exactly that sentence, and the
  wheel's twenty-one entries include no rubric and no `examples/`. A published
  document instructing a reader to use something their installed copy does not
  have is the precise defect class `COMPATIBILITY.md` exists to record in other
  people's libraries; the first one this project shipped was its own.
- **`opik_rigor.judge.EXAMPLE_RUBRIC_NAME`**, the filename of that packaged rubric
  (`"example-rubric.md"`), documented so it can be found in a wheel listing without
  running Python. It is deliberately **not** re-exported at the package root and
  **not** in `__all__`. `example_rubric_path()` is the supported way to reach the
  file; a root-level export would promise the *filename* as API and invite callers
  to rebuild the path out of it, which is the re-derivation this library exists to
  catch. `SCORE_MIN` and `hash_rubric_file` were lifted to the root because a
  consumer had no other way to get at what they mean — this constant has one, so
  the argument that moved those does not carry it. Named here rather than left
  undocumented because it is public by position whether or not it is advertised.

### Changed

- **`hash_rubric_text` accepts `str` as well as bytes.** The name says "text", so
  refusing text was a trap. A `str` is encoded as UTF-8 and hashed identically to
  the same bytes, so the two spellings cannot disagree about whether a rubric
  changed; anything that is neither text nor bytes is now refused at the boundary,
  by name, with a pointer to `hash_rubric_file` for the argument people actually
  reach for. What the old call did instead is under **Fixed**.
- **`assert_no_regression` says which shape the data is, not just that there is
  none.** "current has no scores" reads as *you have no data* to a caller holding
  two hundred completions, when the truth is that `SampleResult.scores()` harvests
  `getattr(run.value, "score", None)` and a sample of plain strings harvests
  nothing. The message now names the type it was given, how many runs it holds,
  and the first offending value — or, when every run raised, the first error. What
  the gate *accepts* is unchanged, and a genuinely empty input still gets the
  original sentence.
- **The default classifier's refusal explains where the runs went.**
  `default_outcome` correctly refuses to guess whether `"Paris"` is a pass, but
  `sample` files that refusal on `Run.error` — the field a provider outage lands
  in — and `SampleResult.completed` filters those runs out. An adapter that
  answered every prompt correctly therefore reported `pass_rate=0.0` beside
  `failures=0` with empty `.values`, which reads as a total outage. The message
  now names the value, states that the run has been dropped from `.values`,
  `.outcomes`, `.successes` and `.completed`, and gives the one-line fix
  (`outcome=lambda value: True` if you only want the values back). **The
  behaviour itself is unchanged and pinned by a test** — letting `outcome=None`
  mean "do not classify", or splitting classifier errors out of `Run.error`,
  changes what a recorded sample means and is a major-version change, not this
  one.

### Fixed

- **The PyPI project page told readers to install opik-rigor from a git clone.**
  The 0.1.0 wheel was built at the `v0.1.0` tag, and the README's install block was
  rewritten from `git clone … && pip install .` to `pip install opik-rigor` in the
  commit *after* the upload — so the description baked into that wheel's METADATA,
  which is what pypi.org renders for that release and will render for it forever,
  still tells you to build from source and to write `pip install ".[opik]"`.
  Nothing in the repository was ever wrong; the artifact was cut before the
  repository caught up, and an artifact does not update when the tree does. 0.1.1's
  METADATA carries the current README, checked by reading it back out of the built
  wheel rather than by looking at `README.md`.
- **`hash_rubric_text` reported a `str` argument as the opposite mistake.** It
  validated nothing on entry, so a `str` got several lines in and died on the
  function's own `b"\r\n"` literal with `TypeError: replace() argument 1 must be
  str, not bytes` — which reads as *you passed bytes where text was wanted*, the
  exact inverse of what happened, and sends the reader off to inspect a string
  that was fine. `str` is now accepted (see **Changed**), and an argument that is
  genuinely neither text nor bytes is named and refused at the boundary.
- **The example rubric restated the response format, so anyone who copied it sent
  that block to the model twice.** It used to end with `OUTPUT_FORMAT_INSTRUCTION`
  verbatim, on the reasoning that a rubric should read as a whole prompt on its
  own — but `PROMPT_TEMPLATE` already appends it. The instruction belongs to the
  library and the criteria belong to the rubric; a test now pins that the rendered
  prompt contains it exactly once.

  Fixing it changed the file's bytes, and the file also moved from
  `rubrics/example-rubric.md` in the repository root to
  `src/opik_rigor/rubrics/example-rubric.md` inside the package. Its sha256 went
  from `e62bdbb21a0ecbd6f66d4761f8bf8dc48c61dfe62527dd902561181066d69cf4` to
  `556c1383350d73d71235e40c719cdf816bec8a5693cc4750ad01ae421128dc5d`. Who a
  recorded hash of the old file can actually reach:

  - **Not an installed consumer.** 0.1.0's wheel carried no rubric at all, so no
    installed copy of rigor can have recorded the old hash from one.
  - **A repository checkout that graded against the repository file** raises
    `RubricDriftError` on its next run, which is the mechanism working. Acknowledge
    it with `accept_rubric_change=True`, or keep your own copy of the old file.
  - **An unpacked 0.1.0 sdist**, which is the one case the "additive" claim above
    does not cover, so it is stated rather than left implied. The 0.1.0 sdist did
    carry the old file, at `rubrics/example-rubric.md` in its unpacked root — the
    wheel's omission was not the sdist's. Nothing on that disk moves when 0.1.1 is
    published, so a hash recorded against it keeps agreeing and no drift fires by
    itself. What changed is that 0.1.1's sdist has no `rubrics/` directory at its
    root at all: someone who unpacks the newer sdist over that path gets
    `FileNotFoundError` from their own configured rubric path, which is louder than
    a drift error and much louder than silence. An sdist is neither installable as
    a rubric source nor importable, so this is a footnote and not a caveat on the
    upgrade.

### Not fixed, and why

- **`SampleResult.completed` still filters out a run whose classifier raised**, so
  the "total outage" reading above is still reachable by anyone who does not read
  the message. Every fix — a fourth outcome state, a separate field for classifier
  errors, or `outcome=None` meaning "do not classify" — changes the meaning of a
  recorded sample and of `pass_rate`, and a consumer reading `.stats` off a raised
  gate would silently get different verdicts. It waits for 0.2.
- **The `Adapter` protocol still exposes no token usage**, so no cost gate is
  possible without reaching past the seam into a provider SDK. Adding
  `complete_with_usage` to the protocol is additive for adapters but not for
  anything that type-checks against `Adapter`; it is a 0.2 item alongside the
  typed report objects.

## [0.1.0] - 2026-08-13

First release. Two primitives — statistical gates and a pinned judge — plus the
append-only evidence log they both write to. There is no earlier version.

### Added

- `assert_pass_rate(result, min_rate=...)` gates on the one-sided Wilson lower
  confidence bound, never on the observed rate.
- The pass-rate failure message is the statistical report, and distinguishes
  missing the bar from not having sampled enough to tell.
- `assert_score_distribution(scores, min_mean=..., min_p10=..., max_stddev=...)`
  applies each threshold independently and reports every violation at once.
- `assert_no_regression(current, baseline)` uses the Mann-Whitney U test,
  nonparametric because judge scores are ordinal and routinely multi-modal.
- `wilson_lower_bound` and `wilson_interval` are public, for callers who want
  the number without the assertion.
- `PinnedJudge` refuses an aliased model id at construction — an id must end in
  a concrete version marker and must not contain `latest`, `newest`, `current`,
  `stable`, or `default`.
- `PinnedJudge` hashes its rubric file (sha256) and raises `RubricDriftError`
  when the rubric changes underneath a recorded history;
  `accept_rubric_change=True` acknowledges it and records both hashes.
- Judge responses are parsed strictly: an unparseable response raises
  `JudgeOutputError` and is never coerced into a fail-verdict, because missing
  data is not evidence of failure.
- `EvidenceLog` is an append-only JSONL log with a fixed envelope and no delete,
  clear, truncate, rotate, or purge API; a test asserts no such name exists.
- The evidence log tolerates a torn final line on read (a process killed
  mid-append) but raises on a malformed line anywhere else.
- `sample(fn, n, concurrency=..., timeout=...)` returns a `SampleResult` that
  keeps failures and exceptions in separate buckets and records per-run
  durations.
- `Baseline` stores a versioned JSON file carrying a sha256 of its own contents
  and verifies that hash on load.
- Adapters `FakeAdapter`, `AnthropicAdapter`, and `OpenAICompatAdapter` behind a
  one-method protocol; credentials come from environment variables only, and the
  provider SDKs are imported lazily inside `complete()`.
- `pinning.is_pinned` / `pinning.require_pinned` are the single definition of a
  reproducible model id, shared by the adapters and the judge.
- Optional `[opik]` extra: `log_sample_to_opik` maps a sample to a trace with
  one span per run, and `log_assertion_to_opik` maps a gate's verdict to
  feedback scores. Verified against opik 2.2.28 — see
  [COMPATIBILITY.md](COMPATIBILITY.md).
- Optional `[pytest]` extra: `@pytest.mark.rigor_repeat(n=..., min_rate=...)`
  runs a test n times and applies the gate, plus a `rigor_judge` fixture.
  Registered as `rigor`, and tested to co-exist with Opik's own pytest plugin.
- `examples/summarise_eval.py` and `rubrics/example-rubric.md`: a worked eval
  that runs offline with no credentials.
- Python 3.10 through 3.13, on Linux and Windows. scipy is the only required
  dependency.

### Fixed

Four defects were found and fixed during the build. The mechanism that caught
each one is worth recording, because it is why they were found at all.

Three were caught by authorship separation — the agent that wrote a module did
not write its tests, and the test author derived expected values independently
of the implementation:

- `sampling.py` caught `concurrent.futures.TimeoutError`, which since Python
  3.11 *is* `builtins.TimeoutError`. A provider's own socket timeout was
  therefore rewritten as rigor's budget expiring — the module violating its own
  thesis that a failure and an exception are facts about different systems.
- The per-run timeout budget was not per-run. The collector waited on futures in
  submission order, so a run queued behind others was granted several budgets'
  worth of wall clock. Timing now happens inside the worker.
- `wilson_lower_bound(0, n)` returned roughly 1e-17 rather than 0 — a lower
  confidence bound sitting above its own point estimate — on about 15% of
  `(n, confidence)` pairs. At p̂=0 the centre and half-width are the same
  expression analytically but were evaluated in different orders, and the clamp
  squeezed only the negative side.

A fourth defect was **not** caught by authorship separation. Three tests
asserted that `importlib.util.find_spec("openai") is None` — testing the
environment rather than the library, and passing only because no environment had
the SDK installed. It surfaced when a second virtualenv installed the Opik SDK,
which depends on `openai`. The invariant was re-stated as the one that was
actually meant — rigor never imports a provider SDK — and is now checked in a
subprocess, so it holds whether or not a provider SDK is present.

Also fixed after the v0.1.0 tag:

- The import package was renamed `rigor` to `opik_rigor`. `rigor` is already
  taken on PyPI by an unrelated project whose wheel installs a top-level
  `rigor/`, so the original name would have shadowed it in any environment
  holding both. The distribution name `opik-rigor` is unchanged.

### Known limitations

- Published to PyPI on 2026-08-13 as `opik-rigor` (import `opik_rigor`).
- The judge's score range is fixed at 1 to 5. `SCORE_MIN` and `SCORE_MAX` are
  interpolated into the prompt so that rubric, prompt, and validator cannot
  disagree; a configurable range is a roadmap item.
- The pass-rate gate is one-sided by design. `wilson_lower_bound` takes
  `z = norm.ppf(confidence)`, not the two-sided `norm.ppf(1 - (1-c)/2)`, so the
  numbers will not match a two-sided textbook table: for 14/20 the one-sided
  bound is 0.5162 where the two-sided is 0.4810.
- Report objects are `dict[str, Any]`. No autocomplete, no protection against a
  mistyped key, and the key names are not guessable.
- The `underpowered` and `runs_needed` fields exist only on the failure path,
  carried on the raised exception's `stats`. Reporting what a gate concluded,
  pass or fail, means wrapping every gate in `try`/`except`; a non-raising
  `check_*` beside each `assert_*` is a roadmap item.
- `sample(fn, n)` passes `fn` no item and no index, so a caller iterating a
  dataset writes the same cycling closure every time.
- `FakeAdapter` rejects `seed=` in exactly the `responses=<callable>` mode that
  a fake reacting to its input requires — the one shape that can vary with the
  system under test is the one that cannot take a seed.
- The Opik integration was verified against opik 2.2.28 only; the extra declares
  `opik>=2.0,<3` on the reasoning recorded in
  [COMPATIBILITY.md](COMPATIBILITY.md), which also records an upstream
  introspection bug in 2.2.28, one stale documentation URL, and a retraction of
  this project's own earlier claim that Opik's SDK reference rendered parameter
  names wrongly. It did not; the fault was in the HTML-to-markdown converter used
  to read it.

[Unreleased]: https://github.com/ericwehmeyer/opik-rigor/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ericwehmeyer/opik-rigor/releases/tag/v0.1.1
[0.1.0]: https://github.com/ericwehmeyer/opik-rigor/releases/tag/v0.1.0
