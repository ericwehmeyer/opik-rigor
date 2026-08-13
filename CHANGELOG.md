# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- Not published to PyPI or any other package index. Install from source.
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
  introspection bug in 2.2.28 and four points where the published Opik docs are
  wrong.

[0.1.0]: https://github.com/ericwehmeyer/opik-rigor/releases/tag/v0.1.0
