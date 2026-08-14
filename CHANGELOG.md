# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet. Both items under *Not fixed, and why* in 0.1.1 below are still open:
`SampleResult.completed` still filters out a run whose classifier raised, and the
`Adapter` protocol still exposes no token usage. Each changes what a recorded
sample means, so each waits for the minor after this one.

## [0.2.0] - 2026-08-14

**This release refuses input that 0.1.1 accepted and answered, and that is the
entire reason the number is 0.2.0 rather than 0.1.2.** `wilson_lower_bound` and
`assert_pass_rate` now raise `ValueError` on a `confidence` at or below 0.5. **A
confidence at or below 0.5 produces an interval that means nothing, and the
library will no longer manufacture false evidence out of one.**

Below 0.5 the one-sided `z = ppf(c)` is negative, so the "lower bound" comes out
*above* the observed rate and gets **worse** as the sample grows:
`wilson_lower_bound(89, 100, 0.0001)` is 0.9615, and the same rate over 1000 runs
gives 0.9216. At exactly 0.5 the z is zero and the bound *is* `successes / n`, so
`assert_pass_rate((20, 20), 1.0, confidence=0.5)` passed — twenty runs proving
perfection, which is the claim the module's opening paragraph exists to refuse.
Every one of those numbers was arithmetically correct at the level asked for,
which is why this is a narrowed domain and not a corrected formula: a gate
written `confidence=0.3` reads in a test file as an act of caution and was looser
than comparing the raw rate. A one-sided bound is a floor you are willing to
defend, and there is no level of belief below a coin flip that anyone defends.

**Migration is one line: raise the confidence above 0.5, or drop the argument and
take the 0.95 default.** If what you actually want is the full two-sided range,
`wilson_interval` is the escape hatch and is unaffected — its z is
`ppf((1 + c) / 2)`, non-negative across the whole open interval, so it never
inverts, and it is deliberately not routed through the new check.

**A third surface reaches the refusal, and it is not in `__all__`.** The pytest
marker `@pytest.mark.rigor_repeat(n=..., min_rate=..., confidence=...)` hands its
`confidence` to `assert_pass_rate` unvalidated — the plugin fills in the default
and type-checks only `errors_as_failures` — so a suite carrying `confidence=0.3`
on a marker goes from running to erroring, and the failure now arrives *after*
the runs have been spent rather than before. That is the zero-configuration
surface this project argues for, and because the marker is not a name in
`__all__`, a consumer auditing `__all__` for breakage does not see it. Grep your
suite for `rigor_repeat` as well as for the two function names.

**`model-migration-kit` pins `opik-rigor>=0.1.1,<0.2`**, so 0.2.0 does not reach
it until that bound moves. 0.1.1 published its additions as a PATCH to keep that
consumer's upgrade path open and recorded that the reservation of 0.2 stood for
changes that alter what a recorded sample means. This is the release that spends
it, and it spends it on exactly one change: everything else here is additive —
nothing renamed, no other signature narrowed, and no gate's verdict moved for
input either function still accepts.

The rest of the release is four independent reviews of the published 0.1.1
wheel, landing together.

The fourth is the odd one out and is listed first because it is the one a new
reader meets first: a cold-start stranger who installed `opik-rigor` 0.1.1 from
PyPI and followed `README.md` literally. They did not get as far as the numbers.
Ten defects, and they are not ten faults — they are **one fault, ten times: a
claim that is true of the source tree and false of the artifact a user
installs**, which is the sentence this changelog already used once, for 0.1.0's
rubric. The worked example the quickstart ends on is the first entry under
**Added**, the release check that would have caught it is the second, and the
remaining nine defects are the last nine entries under **Fixed**.

**None of the documentation work reaches an existing release.** A project's long
description is frozen at upload time, so the page PyPI renders for 0.1.1 is the
old README — dead links, elided hashes, an unrunnable example — and stays that way
until a new version is uploaded. Those fixes reach readers with 0.2.0, and not
one of them reaches a reader of 0.1.1's page.

The first of the numerical reviews was adversarial, working by derivation rather
than by reading this code: bisection on the score-test inequality, exact `Fraction`
arithmetic, and full brute-force scans. It confirmed that the Wilson bounds are
exact to 2.2e-16 across 105 grid points, that nothing anywhere conflates
one-sided with two-sided, that the pass-rate gate never gates on the point
estimate, that Mann-Whitney's direction and statistic are right, and that the
realised type-I error is calibrated at 0.048. It found seven defects: the
confidence refusal above, which is the first entry under **Changed**, and six
under **Fixed** — the infinity gate, both `_runs_needed` defects, the two
refusals that named a type instead of a value, the numpy count pair, and the
wrong worked example in the `_wilson` docstring. The next measured what
`import opik_rigor` costs, and produced the SciPy entry under **Changed**. The
last came from outside the numbers
entirely — an agent designing a third consumer, which could not construct a judge
at all — and produced the two `is_pinned` entries in the middle of **Fixed**, the
defect `PROGRESS.md` recorded as item 16.

### Fixed

- **`AnthropicAdapter` could not call any current Anthropic model.** It sent
  `temperature` on every request, with a constructor default of `0.0`, and
  `temperature`/`top_p`/`top_k` were removed from the Messages API on the current
  generation — Opus 5, Opus 4.8, Opus 4.7, Fable 5 and Mythos 5 return a **400**
  for any of them, and Sonnet 5 returns one for any *non-default* value, which
  comes to the same thing here because the only reason to pass the parameter is
  to set a value that is not the default. There was no way to construct an
  adapter that avoided it:
  the parameter had no omit-sentinel, the constructor rejected `None`, and the
  value was passed unconditionally. A judge built the way the README builds one
  could not complete a single call.

  `temperature` now defaults to `None`, meaning the key is **absent** from the
  request rather than set to something chosen for you, and it is omitted on every
  model. An explicit value is still sent to a model that accepts one; against a
  model that does not, it is refused at *construction*, naming the model, rather
  than becoming a 400 partway through a run that has already spent calls.

  Omitting unconditionally is deliberate. A model-conditional default would put
  the vendor table on the happy path, so the day a model ships that the table has
  not heard of, the zero-argument constructor would start returning 400s again —
  the same defect, waiting on a release date. The table is consulted only to
  reject an explicit value, so its going stale costs one thing: an explicit
  `temperature` against a newer model gets the vendor's 400 instead of our
  `ValueError`. It can never break a call that works.

  Together with the pin-rule fix below, this closes the second of two blockers:
  `is_pinned` refusing every current model id was the front door locked, and this
  was there being no room behind it. Cost worth naming: an older model that still
  accepts sampling parameters now gets the API default rather than `0.0` unless
  asked, and on a current model that lever no longer exists at all.

- **A score-distribution gate returned green on infinite input.** This is the
  worst thing in the release and it is the exact failure the library exists to
  prevent. `_coerce_scores` refused NaN and never checked for infinity, so
  `assert_score_distribution([1.0, 1.0, float("inf")], max_stddev=0.001)` computed
  a standard deviation of `nan` (because `inf - inf` is `nan`), found that `nan >
  0.001` is False, recorded no violation, and **passed** — a 0.001 spread gate,
  cleared by a sample containing infinity. `min_p10=4.9` against `-inf` passed the
  same way, and `assert_no_regression` reported `mean_current=inf` and passed. The
  only outward sign was a bare `RuntimeWarning: invalid value encountered in
  subtract` from numpy on stderr, which CI discards. Infinity is now refused where
  NaN is, with a message that names the value and says what it would have done.
  The function's own docstring had already made this argument for the `n < 2`
  case: "reporting it as 0.0 would pass the strictest possible stddev gate on no
  evidence at all."

  The property suite asserts that no gate ever reports a NaN or an infinity in a
  numeric field, and that property **held** while this defect was live — its score
  generator draws from a 1-5 scale, a 0-1 scale and bounded uniforms, every one of
  them finite, so no case it produced could reach the gate with a non-finite
  score. The property was sound and its input domain stopped short of the defect,
  which is the more useful half of the finding. Non-finite scores now sit in that
  file's refusal table instead, where a refused input belongs.
- **`_runs_needed` reported a number that was not the one it promised.** Its
  docstring claimed the smallest n at which the observed rate clears the bar, and
  justified a binary search with "the bound is monotone in n for fixed p". That is
  false: `successes = round(p * n)` makes the predicate *oscillate*. At `p=0.95,
  min_rate=0.90, confidence=0.95` it holds at n = 86-90, fails at 91-99, holds at
  100-109, fails at 110-112 and holds from 113 on, so a binary search returns
  whichever clearing n it happens to land on. 28 of 45 grid cases disagreed with a
  brute-force scan. What is reported now is the point past which the answer stops
  depending on where the rounding lands — 113 in that case — found by binary
  search on a genuinely monotone predicate (the bound at the least favourable
  rounding, `floor(p*n - 0.5)`) followed by a short walk down to the last n that
  still failed. It is **not** the smallest n that clears, and deliberately so: 86
  clears only because `round(0.95 * 86) = 82` rounds the rate up to 0.9535, and a
  reader told "86 runs" who runs 91 for luck fails.

  1,399 lines of property tests sat on this function and asserted only that the
  answer was *sufficient* — that the bound really does clear at the number
  returned — which 113 satisfies exactly as well as 86 does. The obvious repair,
  "and `runs_needed - 1` must not clear", turns out to catch **0 of 45** grid
  cases, because a binary search converges to a point where `low` clears and
  `low - 1` fails whatever the predicate does in between; it is asserted now
  because it is true, not because it is load-bearing. The assertion that bites is
  the other one: every n *above* the answer must clear too. That is what the
  oscillation breaks, and it is now checked over a window in the property suite
  and against a full brute-force scan over a 45-point grid in the example suite.
- **The runs-needed cap was 8,388,608, not the documented 10,000,000.** The search
  approached the cap by doubling from 1 under `while high <= cap`, so 16,777,216
  overshot and fell to the `else`: every answer in the 8.4M-10M band came back as
  `None`, meaning "no sample size can do this", for margins where a finite in-cap
  answer exists. `_runs_needed(0.9001645, 0.9, 0.95)` returned `None` against a
  true answer of 9,004,248. The cap is now searched directly.
- **Two refusals named the offending type and not the value**, unlike every other
  refusal in the module: `min_mean must be a number or None, got str` and `scores
  must be a sequence of numbers, got str`. Both now quote the value, which is what
  a caller whose threshold arrived from a config file needs to see.
- **A `(numpy.int64, numpy.int64)` count pair was misdiagnosed as outcomes.**
  `_looks_like_counts` tested `isinstance(item, int)`, which numpy integers fail,
  so `assert_pass_rate((np.int64(3), np.int64(5)), 0.5)` was refused with
  `result[0] must be a bool (or 0/1), got int64` — pointing the reader at their
  data when the answer was their dtype. `_as_count` had accepted numpy integers
  all along, for the stated reason that they arrive routinely from array code.
- **The `_wilson` docstring's worked example was wrong.** It said 20/20 gives
  "roughly `[0.86, 1.0]`". The two-sided 95% interval is `[0.8389, 1.0]` and the
  one-sided 95% lower bound is `0.8808`; 0.86 is neither. A worked example in a
  statistics library is a claim, and this one is now asserted against an oracle in
  the test suite rather than trusted.
- **`is_pinned` rejected every current frontier Anthropic model id**, and
  `require_pinned` is on the path a new user hits in their first five minutes.
  `claude-opus-5`, `claude-sonnet-5`, `claude-opus-4-8` and every other current id
  were refused, because the rule required a trailing date and Anthropic's ids no
  longer carry one — while `claude-3-7-sonnet-20250219`, retired on 2026-02-19,
  was accepted. The rule had been checking spelling in place of the property it
  stood for. It now checks the property: an id is pinned when it contains no alias
  token (`latest`, `newest`, `current`, `stable`, `default`) **and** ends in a
  release designator — a release number (`claude-opus-4-8`), a date stamp
  (`-20251001`, `-2024-08-06`), or an explicit version (`-v1`, `-2.1.0`). A last
  component that is a *word* (`gpt-4o`, `mistral-large`, `…-instruct`) names a kind
  of model rather than one release of it, and a kind is what providers re-point.
  `_`, `@` and `:` now count as component separators, so Vertex
  (`claude-opus-4-5@20251101`) and Bedrock (`…-20241022-v2:0`) snapshot spellings
  are recognised too. Whether an id re-points is ultimately a provider *policy* and
  no string carries it, so that residue is isolated in one documented table in
  `pinning.py` rather than spread through the predicate, and the module docstring
  states what the rule still cannot catch.
- **`gpt-4.1` was accepted as pinned**, and it is an OpenAI alias that re-points to
  the newest dated snapshot — its `4.1` satisfied the old dotted-version branch.
  Found while checking the fix above in *both* directions rather than only the one
  that was reported. The same table refuses `gpt-5` and `gpt-4`;
  `gpt-4o-2024-08-06` and `gpt-4.1-2025-04-14` are unaffected.

- **`py.typed` was a false claim.** The marker tells a downstream checker to trust
  this package's annotations, and the annotations could not catch the error they
  existed to catch: `AnthropicAdapter(..., api_key="sk-…")` type-checked clean
  under mypy *and* pyright and raised `TypeError` at runtime, because the
  `**forbidden: object` that guarded the credential keywords accepts every
  keyword. It is `NoReturn` now, so passing one is a type error where the caller
  is looking at it. The gate moved with it: `verify_release.py` runs
  `mypy --strict` against the **extracted wheel** rather than the source tree,
  because the claim `py.typed` makes is about the artifact.

- **The sdist swept in the working directory.** A local build produced 122
  members — 71 of them a second copy of the tree under `.claude/worktrees/`, plus
  25 `.remember/` files. **The published 0.1.1 sdist is clean**, verified by
  downloading it: CI builds from a fresh checkout where those directories do not
  exist. That is the point worth stating rather than the leak — the artifact was
  protected by an accident of where it happened to be built, which is not a
  property anyone can rely on. It is an explicit allowlist plus a test now.

- **Four repo-relative links in `README.md` 404 from the PyPI project page**
  (`LICENSE` twice, `examples/`, `COMPATIBILITY.md`). PyPI renders a long
  description with no repository, no branch and no directory behind it. All are now
  absolute `https://github.com/...` URLs.

- **The `[opik]` extra's two functions were unimportable from where the README
  implied.** `log_sample_to_opik` and `log_assertion_to_opik` were named in prose
  with no import path; they live in `opik_rigor.integrations.opik`, and reaching
  for them at the package root raises `ImportError`. The README now gives the
  import line.

  The other direction — re-export both at the root — was considered and rejected,
  because it is the one change that would break the property the same section
  advertises: `opik_rigor/__init__.py` imports no integration at module scope, a
  subprocess test asserts it, and that is what keeps `import opik_rigor` from ever
  dragging in a vendor SDK. Here the docs were wrong and the code was right.

- **The one *passing* output in the quickstart could not be produced from the code
  the README gave.** It said "change `20` to `200` and `min_rate` to `0.8`" and
  then showed `passed=True observed=0.9150 ...`. Doing exactly that prints nothing:
  on success `assert_pass_rate` returns a report dict and is silent. The README now
  shows the assignment and the `print`, and lists the full key set.

  Worse than a missing `print`, and now stated in the README rather than left to be
  discovered: the line reads `observed=` and the key is **`pass_rate`**. The
  roadmap's own complaint that "the key names are not guessable" was being
  illustrated, unwittingly, by the roadmap's own document.

- **A block labelled "the output is pasted verbatim, nothing here is illustrative"
  was neither.** The rubric-drift example showed a judge named `'j'` that appears
  nowhere in the quickstart and elided its hashes with `...`, which the real message
  never does. It is now a real message from a real run: judge `'summariser'`, both
  sha256 hashes in full, and the exact edit that produces it. The paragraph making
  the claim now states precisely what was done to the output — hard-wrapped, and
  nothing else.

- **`PROGRESS.md` said 0.1.1 was unreleased.** It has been on PyPI since
  2026-08-13. Its test counts were stale as well and have been re-derived by
  running, with the command beside each number.

- **`examples/README.md` told the reader the judge was backed by
  `rigor.FakeAdapter`.** The import package is `opik_rigor`; `rigor` on PyPI is an
  unrelated HTTP-API-testing DSL, so that line pointed at somebody else's package —
  the exact collision this project renamed itself to avoid, reintroduced in prose.

- **`sample_of` was in `__all__` and in no document**, next to a `sample_over` that
  the roadmap lists as not yet built. Two names one letter apart, one shipped and
  undocumented, one documented and absent. `sample_of` is now described where the
  gates are, and the roadmap entry for `sample_over` names the collision.

- **The `## Development` block gave only `.venv/bin/python`.** The primary
  development machine is Windows, where the interpreter is
  `.\.venv\Scripts\python.exe`. Both forms are given, and both now run
  `pytest tests examples` and `ruff check src tests scripts`.

- **Install cost was stated nowhere.** Measured, into empty virtualenvs, cold:
  `pip install opik-rigor` takes 54 s and grows a virtualenv from 11.5 MiB to
  192.8 MiB (4 packages); `pip install "opik-rigor[opik]"` takes 4 min 48 s and
  reaches 414.9 MiB across 74 packages, pulling litellm, openai, tokenizers,
  huggingface-hub, tiktoken, sentry-sdk and three `tree-sitter` grammars. A reader
  of "two functions, not a framework" does not expect the second number, so the
  README now gives both.

### Changed

- **A one-sided `confidence` at or below 0.5 is refused instead of inverting the
  gate.** The release's only non-additive change, argued at the top of this
  section. `wilson_lower_bound` and `assert_pass_rate` are the whole of the
  affected surface: they are the only two callers of the validator, and
  `wilson_interval` is deliberately not one of them, so the two-sided interval
  still spans the full open unit interval of confidences. The refusal is
  hard-wrapped here and otherwise verbatim, with `got 0.5` being the value as the
  caller passed it:

  ```
  confidence must be greater than 0.5 for a one-sided bound, got 0.5. Below
  0.5 the z is negative, so the 'lower bound' sits above the observed rate
  and falls as n grows -- more evidence, worse bound -- and a gate set there
  is looser than comparing the raw rate. At exactly 0.5 the bound is the raw
  rate. Use wilson_interval if you want the full range two-sided
  ```

  It is listed under **Changed** rather than **Fixed** because every number the
  old code returned was arithmetically correct; what changed is the domain the
  functions accept, not the formula they evaluate on it.

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

- **The underpowered pass-rate message no longer reads as a power calculation,
  because it never was one.** It said "At this observed rate roughly N runs would
  clear the bar", which is true only if the next N runs reproduce the observed
  rate exactly — and they land above or below it at random. The exact binomial
  probability that a fresh sample of the recommended size clears the gate is
  **0.66** at the 113 recommended above, 0.54 at 613 and 0.59 at 42: a coin flip
  presented as a budget, and the same defect class a sibling project found by
  simulation when it certified n=25 adequate at a real power of 33.9%. The message
  now states the recommendation, says in terms that it is arithmetic on one rate
  rather than a power calculation, and quotes the power alongside it.

- **The `ModelPinError` message now names the clause that refused the id**, and
  quotes worked examples that are themselves pinned. The old message instructed
  the reader to add a date suffix, which for a current Anthropic id produces an id
  that does not exist — advice that is confidently wrong. A test now asserts that
  every example the message quotes satisfies `is_pinned`, so that class of untrue
  advice fails the suite rather than relying on someone proof-reading it.

### Added

- **The worked example ships inside the wheel**, as
  `opik_rigor/examples/summarise_eval.py`, and the quickstart's last line is now
  `python -m opik_rigor.examples.summarise_eval --seed 7 --n 40`.

  It used to be `python examples/summarise_eval.py`. `examples/` is a directory in
  the git tree; the distribution is `opik_rigor/` plus
  `opik_rigor-0.1.1.dist-info/` and nothing else, so the one command the
  quickstart ended on — after four paragraphs selling it — could not be run by
  anybody who had followed the `pip install` line above it. The script itself was
  never the problem: fetched from the repository it runs clean against the
  published wheel, exit 0. Its *address* was the problem.

  Two honest fixes existed — ship the example, or stop advertising it — and this
  is the first, for the same reason 0.1.1 chose it for the rubric: a reader who
  installs a library and is told there is a worked example is better off with the
  example than with a shorter README. `--out` now defaults to `.rigor-run` under
  the caller's working directory rather than beside the script, which after the
  move would have been inside site-packages.

  `tests/test_packaging.py` asserts both modules are in the built zip and runs the
  example out of the *extracted* wheel with nothing else on the path, so a move
  back out of the package fails a test rather than a stranger.

- **`readme-paths`, a new check in `scripts/verify_release.py`.** It reads every
  address the README hands a reader — markdown link targets, path arguments inside
  fenced code blocks, and `python -m` targets under `opik_rigor` — and asserts each
  resolves for somebody standing on the project page or on an install, rather than
  in a checkout. It is the single check that catches both the entry above and the
  dead links under **Fixed**, and it generalises: this was the third time the
  project shipped an address that only a checkout could resolve. Unit-tested in
  `tests/test_release_checks.py`
  against hand-derived tables in the style of that file, plus synthetic wheels for
  the broken cases this repository can no longer produce on purpose.

  It refuses a relative link *even when the file is in the wheel*. `LICENSE` really
  is shipped, under `.dist-info/licenses/`, and the link still 404s for the reader
  who clicks it. Reachable and addressable are different claims, and conflating
  them is how four dead links survived two releases.

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

- **A genuinely powered recommendation beside the arithmetic one.** The report
  dict gains `power_at_runs_needed`, `target_power` (0.80) and
  `runs_for_target_power`, and the failure message offers the last of these as
  "the number to plan against" — 188 rather than 113 in the case above, 1.5-2.2x
  larger across the grid. It is derived, not asserted: the power is the exact
  binomial probability that `Binomial(n, observed)` reaches the smallest count
  clearing the Wilson bound, computed in the standard library from a log-gamma
  seed and a multiplicative recurrence over a 14-sigma window, and the tests
  recompute every reported figure in exact `Fraction` arithmetic over the whole
  tail. The recommendation is the point past which the power *stays* at or above
  the target, for the same lattice reason `_runs_needed` reports a stable point:
  164 is the first n to reach 80% and 165-187 fall back below it.

- **`pinning.PINNED_EXAMPLES`** — the worked examples the `ModelPinError` message
  quotes, exposed so that a caller, and the test suite, can check the advice
  against the predicate instead of against a proof-read.

**On the pin rule and compatibility.** No *pinning* signature changed and nothing
was renamed — the one narrowed signature in this release is the confidence check
at the top of this section. Every model id 0.1.1 accepted is still accepted **except `gpt-4.1`**,
which it should not have accepted. Ids that were refused and are now accepted
cannot break a caller, because the only thing `require_pinned` ever did with them
was raise.

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

[Unreleased]: https://github.com/ericwehmeyer/opik-rigor/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ericwehmeyer/opik-rigor/releases/tag/v0.2.0
[0.1.1]: https://github.com/ericwehmeyer/opik-rigor/releases/tag/v0.1.1
[0.1.0]: https://github.com/ericwehmeyer/opik-rigor/releases/tag/v0.1.0
