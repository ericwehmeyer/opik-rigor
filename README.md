# opik-rigor

[![PyPI](https://img.shields.io/pypi/v/opik-rigor)](https://pypi.org/project/opik-rigor/)
[![CI](https://github.com/ericwehmeyer/opik-rigor/actions/workflows/ci.yml/badge.svg)](https://github.com/ericwehmeyer/opik-rigor/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Statistical assertions and pinned-judge evaluation primitives for LLM test suites.

```python
assert_pass_rate(result, min_rate=0.9)   # not: assert pass_rate >= 0.9
```

Two primitives, done properly, with an audit trail. Optional Opik integration.

---

## The problem

You have an eval. It calls a model 20 times, 18 pass, and your test asserts
`pass_rate >= 0.9`. It goes green and you ship.

That test told you almost nothing, for three separate reasons.

**You measured a stochastic system once.** 18/20 is a sample, not a property. The
same system on the same inputs gives you 17/20 tomorrow and the suite goes red
with nothing having changed. So the team adds a retry, or drops the bar to 0.85,
and now the gate is measuring the team's patience rather than the model.

**Your judge moved.** The model id was `claude-3-5-sonnet-latest`. The provider
re-pointed the alias in March. Every score you recorded before March is not
comparable to every score after, and nothing anywhere says so. Or somebody
improved the wording of the rubric, which is the same problem wearing different
clothes.

**Your failures and your outages are in the same bucket.** The provider 500ed
four times, your harness counted those as failures, and now a quality gate is
reporting an infrastructure incident. Nobody notices, because the number moved in
a direction that looks like a real regression.

rigor fixes exactly these three things and nothing else.

---

## The primitives

### 1. Statistical gates

An assertion that accounts for having sampled a stochastic system *n* times rather
than measured it once. `assert_pass_rate` compares the **one-sided Wilson lower
confidence bound** against your bar, never the observed rate.

The practical consequence is worth internalising before you use it:

| observed | n | 95% lower bound | `min_rate=0.9` |
|---|---|---|---|
| 90% | 20 | 0.7383 | **fails** |
| 90% | 200 | 0.8596 | **fails** |
| 90% | 1000 | 0.8833 | **fails** |
| 95% | 200 | 0.9181 | passes |

**You cannot pass a 90% gate by scoring 90%.** The bound approaches the observed
rate from below and never reaches it, so you need real headroom above the bar —
and how much headroom is exactly what *n* buys you. That is the whole idea. A gate
that let 18/20 through would be telling you a story about 20 coin flips.

Three gates ship:

- `assert_pass_rate(result, min_rate=...)` — Wilson lower bound vs a floor.
- `assert_score_distribution(result, min_mean=..., min_p10=..., max_stddev=...)` —
  each threshold independent and optional, every violation reported at once. A
  mean gate alone passes a system that is excellent four times in five and
  unusable the fifth time, which is the failure users actually notice.
- `assert_no_regression(current, baseline)` — Mann-Whitney U against a recorded
  baseline. Nonparametric because judge scores are ordinal and routinely
  multi-modal; a t-test there is testing an assumption the data does not meet.

The failure message *is* the statistical report. It distinguishes the two failures
that matter, in as many words: **you missed the bar** versus **you did not sample
enough to tell**.

### 2. A pinned judge

`PinnedJudge` refuses to run against an aliased model id — at construction, not
after a week of wasted compute:

```
judge 'summariser' refuses unpinned model id 'claude-3-5-sonnet-latest'. It contains
the alias token 'latest', which names whatever the provider is serving today rather
than one fixed version. A pinned id names one immutable model version ... An alias
re-points over time, which silently invalidates every score recorded against it.
```

An id is pinned when it carries no alias token (`latest`, `newest`, `current`,
`stable`, `default`) and ends in a release designator — a release number
(`claude-opus-5`, `claude-opus-4-8`), a date stamp (`claude-haiku-4-5-20251001`,
`gpt-4o-2024-08-06`), or an explicit version (`-v1`, `-2.1.0`). The property being
checked is *immutability*, not spelling: a retired id that still names one fixed
set of weights is pinned, and an id ending in a *word* (`gpt-4o`, `mistral-large`)
is not, because a word names a kind of model and kinds get re-pointed. What no
string can tell you is a provider's policy, so the one place that needs vendor
knowledge — providers that publish `<family>-<number>` as a moving pointer — is a
single documented table in `pinning.py`, and its limits are written down there.

It hashes its rubric and raises when the rubric changes underneath a baseline
(`accept_rubric_change=True` acknowledges it and records both hashes). And it
parses the judge's response strictly: an unparseable response raises, and is
**never** converted into a failing verdict — missing data is not evidence of
failure, and folding it into the failure bucket biases your pass rate by exactly
the judge's own flakiness rate.

### 3. An evidence log you cannot edit

Everything above writes to an append-only JSONL log with a fixed envelope and
**no delete, truncate, or rotate method**. That absence is a feature, and a test
enforces it — an audit trail you can quietly edit is not an audit trail.

---

## Quickstart

Everything below was executed in a clean virtualenv against the built wheel, and
the output is pasted verbatim. Nothing here is illustrative.

```bash
pip install opik-rigor
```

**What that pulls in.** opik-rigor's own code is 0.33 MB. It requires NumPy and
SciPy, which come to **180.6 MB on disk** between them (numpy 2.5.2 and scipy
1.18.0 on CPython 3.14/Windows, counting the `numpy.libs` and `scipy.libs`
directories that carry the bundled BLAS/LAPACK). NumPy is used by every gate.
SciPy is used by exactly one function — `assert_no_regression`, for
`scipy.stats.mannwhitneyu` — and is imported on first call rather than at package
import, so a suite that never calls that gate never loads SciPy and does not pay
for it at import time. Both stay required, so `pip install opik-rigor` continues
to give you a working `assert_no_regression`.

A worked example rubric ships **inside the package**, so the install gives you
something to point the judge at:

```bash
python -c "import opik_rigor, shutil; shutil.copy(opik_rigor.example_rubric_path(), 'rubric.md')"
```

Read it, then edit it into your own — a rubric is the measuring instrument, and
one copied from a library measures the library's idea of quality rather than
yours. It deliberately says nothing about JSON: `PinnedJudge` appends the
response-format instruction to the prompt itself, so a rubric that restates it
sends the same block twice. Then:

```python
from opik_rigor import EvidenceLog, FakeAdapter, PinnedJudge, assert_pass_rate, sample

log = EvidenceLog("evidence.jsonl")
adapter = FakeAdapter(  # a real judge would be AnthropicAdapter("claude-...-20250929")
    responses=['{"pass": true, "score": 5}'] * 9 + ['{"pass": false, "score": 2}'],
    seed=1,
)
judge = PinnedJudge(adapter, "rubric.md", log, name="summariser")

result = sample(lambda: judge.evaluate("Summarise this.", "A summary."), 20, evidence=log)
assert_pass_rate(result, min_rate=0.9, evidence=log)
```

This **fails**, and the failure is the point:

```
opik_rigor.distribution.PassRateError: pass rate gate failed: 18/20 passed (observed
0.9000); one-sided 95% Wilson lower bound 0.7383 < min_rate 0.9000. Two-sided 95%
interval [0.6990, 0.9721]. The observed rate 0.9000 clears min_rate 0.9000 but the
lower bound does not: this is an underpowered sample, not a demonstrated failure.
20 runs cannot distinguish a system at 90.0% from one at 73.8%. The observed rate
sits exactly on min_rate, and the lower bound approaches the observed rate from
below without ever reaching it: no sample size clears this bar at exactly 0.9000.
The system needs real headroom above min_rate, or min_rate has to come down.
```

`assert pass_rate >= 0.9` would have gone green on that sample.

Same judge, same seed, sampled properly and gated at a bar it can actually
defend — change `20` to `200` and `min_rate` to `0.8`:

```
passed=True  observed=0.9150  lower_bound=0.8768  min_rate=0.8
```

And the other primitive — edit the rubric between two runs:

```
RubricDriftError: rubric drift for judge 'j': evidence log last recorded
a0a929f4c657...b6847, rubric file now hashes to 0c423008a579...bdb38. Scores
before and after this change are not comparable. Pass accept_rubric_change=True
to acknowledge and record the change.
```

A full worked example — corpus, judge, both gates, a baseline, a simulated
regression, and the audit trail — is in [`examples/`](examples/) and runs offline
with no credentials:

```bash
python examples/summarise_eval.py --seed 7 --n 40
```

---

## Optional extras

```bash
pip install "opik-rigor[opik]"     # log samples and verdicts to Opik
pip install "opik-rigor[pytest]"   # @pytest.mark.rigor_repeat, rigor_judge fixture
```

**Opik** — two functions, not a framework: `log_sample_to_opik` maps a sample to a
trace with one span per run (a run that *raised* is visibly distinct from one that
*failed*), and `log_assertion_to_opik` maps a gate's verdict to feedback scores.
The verified API surface, the version bounds, the reasoning behind them, and a
correction to a claim this project got wrong about Opik's own documentation are
all in [COMPATIBILITY.md](COMPATIBILITY.md).

**pytest** — `@pytest.mark.rigor_repeat(n=50, min_rate=0.9)` runs a test *n* times
and applies the gate to the outcomes. A body that returns passes; one that raises
`AssertionError` is a failure; anything else is an exception, counted separately.
Registered as `rigor`, and verified to co-exist with Opik's own pytest plugin.

**The core never imports either.** If a vendor SDK breaks, you lose a dashboard,
not a test suite.

---

## Designed to support SR 11-7 model validation

[SR 11-7](https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107.pdf)
(Federal Reserve / OCC 2011-12, April 2011) is the US supervisory guidance on model
risk management. It is not a checklist and this library does not make you compliant
with it — compliance is an institutional programme with independent review,
governance, and validators, and no Python package delivers that.

What rigor does is make three things it asks for cheap to produce as a by-product
of testing, rather than reconstructed from memory at review time:

**Conceptual soundness** is documented where the choices are made. Why Wilson over
Clopper-Pearson (coverage vs power at the small *n* an eval can afford), why
Mann-Whitney over a t-test (ordinal, non-normal, multi-modal scores), and why a
parse failure is never a fail-verdict — all in the docstrings of the functions
that implement them, not in a slide deck that drifts away from the code.

**Ongoing monitoring** is what the gates are. A recorded baseline carries a sha256
of its own contents and is verified on load, so a regression cannot be made to
disappear by editing the file it is compared against.

**Outcomes analysis** is what the evidence log holds: every verdict, every sample,
every gate decision, appended and never rewritten, each carrying the judge's pinned
model id and the sha256 of the exact rubric revision that produced it. The question
"what exactly was this number measured with, and has that changed since?" has a
file-backed answer.

**Effective challenge** is a property of your organisation, not your tooling. But
challenge needs something to bite on, and "the rubric hashed to `e62bdbb2…` and the
judge was pinned to `claude-sonnet-4-5-20250929`" is a materially better starting
point than "we ran the eval and it looked fine."

If you work in a regulated setting, treat this as plumbing that makes evidence
falsifiable and cheap — and treat the guidance as your compliance team's to
interpret.

---

## Roadmap

v0.1 is two primitives done rigorously. These are the good ideas that were
deliberately parked, most of them discovered by writing the example and finding
the library annoying to use:

- **A non-raising `check_*` beside each `assert_*`.** Today success returns a
  report dict and failure carries the same numbers on `exc.stats` — and
  `underpowered`/`runs_needed` exist only on the failure path. Printing "what did
  the gate conclude" means `try/except` around every gate.
- **Typed report objects.** The reports are `dict[str, Any]`: no autocomplete, no
  typo protection, and the key names are not guessable (`lower_bound` vs
  `interval_lower` vs `min_rate`).
- **`sample_over(items, fn)`.** `sample(fn, n)` hands `fn` nothing, so every caller
  writing a real eval reimplements the same dataset-cycling closure.
- **A seedable callable `FakeAdapter`.** `seed=` is rejected in exactly the
  `responses=<callable>` mode a realistic fake needs — the one shape that can react
  to its input is the one that cannot take a seed.
- Cost and latency gates. `SampleResult` already records per-run durations.
- Clopper-Pearson as an option, for settings that need guaranteed coverage.
- A configurable score range (currently fixed at 1–5).

Explicitly **not** planned: becoming an eval platform. Datasets, dashboards,
prompt management, and orchestration are what Opik is for, and rigor integrates
with it rather than competing.

---

## Development

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests
```

The suite is green with **no credentials and no network**. Anything needing a live
provider is marked `requires_network` or `requires_opik` and deselected in CI. The
Opik integration is tested against a real Opik client via
`opik.record_traces_locally()` pointed at a loopback server — not against mocks.

Two environments are maintained: one without Opik (the suite must pass without it)
and one with, for the integration and plugin co-installation tests.

## License

MIT — see [LICENSE](LICENSE).
