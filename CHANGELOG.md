# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Seven gaps closed, all of them reported by the first external consumer against
the published 0.1.0 wheel rather than found by reading this repository. Every
change is **additive**: nothing is renamed, no signature rejects a call it used
to accept, and no gate's verdict moves. A consumer pinned to `>=0.1.0,<0.2` can
take this release without reading it.

### Added

- **`py.typed`.** The library is annotated throughout and shipped no PEP 561
  marker, which means a type checker had to discard every one of those
  annotations in an installed copy. One empty file, and a test asserts it is
  inside the built wheel rather than merely inside the tree.
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
  a `PinnedJudge` and nothing to point it at, while the README linked a file that
  only existed in the repository.

### Changed

- **`hash_rubric_text` accepts `str` as well as bytes.** The name says "text", so
  refusing text was a trap — and passing a `str` failed several lines in on the
  function's own `b"\r\n"` literal with `TypeError: replace() argument 1 must be
  str, not bytes`, a message describing the exact inverse of the caller's
  mistake. A `str` is encoded as UTF-8 and hashed identically; anything that is
  neither text nor bytes is now refused at the boundary, by name, with a pointer
  to `hash_rubric_file` for the argument people actually reach for.
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
- **The example rubric no longer restates the response format.** It used to end
  with `OUTPUT_FORMAT_INSTRUCTION` verbatim, on the reasoning that a rubric should
  read as a whole prompt on its own — but `PROMPT_TEMPLATE` already appends that
  block, so anyone starting from the example shipped the format instructions to
  the model twice. The instruction belongs to the library and the criteria belong
  to the rubric; a test now pins that the rendered prompt contains it exactly once.
  The file moved from `rubrics/example-rubric.md` to inside the package, so its
  sha256 changed. That cannot reach an installed consumer — 0.1.0's wheel carried
  no rubric, so no installed copy of rigor can have recorded the old hash — but a
  checkout that graded against the repository file will raise `RubricDriftError`
  on its next run, which is the mechanism working. Acknowledge it with
  `accept_rubric_change=True`, or keep your own copy of the old file.

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

[0.1.0]: https://github.com/ericwehmeyer/opik-rigor/releases/tag/v0.1.0
