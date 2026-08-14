# Examples

One example, told as one story: a summariser that is quietly bad at the thing
that matters, and what rigor does about it.

**The example itself is not in this directory.** It ships *inside the package*, at
`src/opik_rigor/examples/summarise_eval.py`, so that the command below works from
a bare `pip install opik-rigor` with no checkout. This directory holds the
walkthrough you are reading and the test that keeps the example honest.

```
python -m opik_rigor.examples.summarise_eval --seed 7 --n 40
```

On Windows, from this repository's virtualenv:

```
.\.venv\Scripts\python.exe -m opik_rigor.examples.summarise_eval --seed 7 --n 40
```

That is the whole setup. **No network, no credentials, no API keys, no model
provider, no Opik.** The "model under test" is a plain Python function and the
judge is backed by `opik_rigor.FakeAdapter` — the import package is `opik_rigor`;
`rigor` on PyPI is an unrelated HTTP-API-testing DSL — so the run finishes in
about a second and prints the same bytes every time for a given `--seed`.

| Flag | Default | What it does |
|---|---|---|
| `--seed` | `7` | Seeds the summariser and the judge. Same seed, same output, byte for byte. |
| `--n` | `40` | Judge calls per run. Two runs are made — healthy and degraded. |
| `--out` | `.rigor-run` | Where the evidence log and the baseline are written, **relative to your current directory**. Cleared at the start of every run. |
| `--opik` | off | Additionally mirror the run into Opik. **This is the only flag that needs anything outside the process** — see [The Opik leg](#the-opik-leg). |

The test that keeps this honest:

```
.\.venv\Scripts\python.exe -m pytest examples/test_example_runs.py
```

It runs the example as a subprocess — `python -m opik_rigor.examples.summarise_eval`,
the exact address given above, not a file path — checks the exit code and the
phrases the walkthrough is supposed to contain, and runs it twice with the same
seed to assert the output is byte-identical.

---

## What it actually does

**A summariser worth evaluating.** The corpus is six short source documents, each
one carrying an outcome, the main stated reason for it, and a **caveat** — the
sentence a reader would be hurt by not knowing. The summariser is extractive: it
copies sentences rather than writing them, and its only defect is which ones it
keeps. Fifteen percent of the time it drops the caveat.

That number is the point of the whole example. A summariser that is always
perfect makes every gate below unfalsifiable, and a summariser that is randomly
bad makes them meaningless. This one has a specific, plausible, *invisible*
failure — the summary still reads well — which is precisely the kind an eval has
to catch.

**A judge that is pinned and imperfect.** `PinnedJudge` grades against the
example rubric that ships inside the package (`opik_rigor.example_rubric_path()`,
so this example reads exactly the file a `pip install` gives you), refuses an
aliased model id at construction time,
and records the sha256 of the rubric so a later edit cannot silently move the
scale. Its answers come from a seeded `FakeAdapter` whose scripted verdicts are
chosen *per tier of summary* — complete, reason missing, caveat missing — so the
judge's opinions track the summariser's actual behaviour instead of being noise.
Within each tier the judge still disagrees with itself, and in one case out of
five it fails to notice a missing caveat at all. That is deliberate: a judge is a
measuring instrument with its own error term, and the reason you sample one
rather than calling it once.

**Then the gates**, in the order a real suite would meet them: sample, pass-rate
gate, score-distribution gate, baseline, regression gate — with everything
written to an append-only evidence log along the way.

---

## Reading the output

Numbers below are from `--seed 7 --n 40`.

### Sections 0–1: the instrument, and the thing being measured

The header prints the judge's pinned model id and the rubric's sha256, then shows
an aliased model id being refused:

```
  an alias is refused, at construction time:
    judge 'aliased' refuses unpinned model id 'fake-judge-latest'. It contains the alias
    token 'latest', which names whatever the provider is serving today ...
```

Section 1 prints one source document and two summaries of it — one keeping every
load-bearing point, one with the caveat gone. Both read fine. That is the problem.

### Section 2: sampling, not measuring

```
  runs                  40
  passed                31
  failed                9
  exceptions            0  (a system that did not run, counted separately)
  observed pass rate    0.7750   <- a point estimate; never gate on it
```

`exceptions` is a separate line from `failed` because they are facts about
different systems — a failure is the model missing the bar, an exception is the
model never running — and a suite that adds them together turns a provider outage
into a quality regression.

### Section 3: the gate passes

```
  observed              31/40 = 0.7750
  bar (min_rate)        0.6000
  95% lower bound       0.6510   <- what the gate compares, not the observed rate
  95% interval          [0.6250, 0.8768]
  gate                  PASS
```

The gate compares `0.6510` — the one-sided Wilson lower bound — against the bar.
It never compares `0.7750`. Forty runs at 77.5% are consistent with a true rate
of about 65%, and the gate is only allowed to claim what forty runs buy.

### Section 4: the failure mode nobody expects

The same sample is then gated at `0.7500`, a bar that sits *below* the observed
`0.7750`. A plain `assert pass_rate >= 0.75` sails through. rigor does not:

```
  pass rate gate 'summariser-healthy-strict' failed: 31/40 passed (observed 0.7750); one-sided
  95% Wilson lower bound 0.6510 < min_rate 0.7500. ... The observed rate 0.7750 clears min_rate
  0.7500 but the lower bound does not: this is an underpowered sample, not a demonstrated
  failure. 40 runs cannot distinguish a system at 77.5% from one at 65.1%. At this observed rate
  roughly 818 runs would clear the bar.

  underpowered          True
  runs needed           818
```

**This is the most important screen in the example.** The failure message is the
statistical report, and it says which of the two things went wrong: you did not
sample enough to tell. The fix is 818 runs, not a code change. Compare section 6b
below, where the fix is a code change and no number of runs will help.

### Section 5: shape, not just average

```
  scores                40
  mean                  4.0500  (bar 3.5)
  p10                   2.0000  (bar 1.5)
  stddev (ddof=1)       1.0610  (bar 1.75)
  range                 [2.0, 5.0]
  gate                  PASS
```

A mean gate on its own passes a system that is excellent four times in five and
unusable the fifth — which is the failure users actually notice. `p10` bounds the
bad tail; `stddev` bounds the swing.

### Section 6: a baseline, and a regression against it

The healthy run is recorded as a `Baseline` carrying a sha256 of its own contents
plus the judge's model id and the rubric hash. Then the summariser's caveat-drop
rate goes from 15% to 75% — one constant, which is what most real regressions look
like from the outside — and the same judge grades the same documents in the same
order.

**6b, the other pass-rate failure mode:**

```
  ... The observed rate 0.4250 is itself below min_rate 0.6000: the system missed the bar, and
  more runs will not fix it.

  underpowered          False  <- contrast with section 4
```

Same gate, same bar, opposite diagnosis, opposite advice. Distinguishing these
two on screen is the reason the library exists.

**6d, the regression gate:**

```
  regression gate 'summariser-nightly' failed: current scores are significantly lower than
  baseline. Mann-Whitney U = 372.5, p = 8.23226e-06 < alpha 0.05 (one-sided, alternative='less':
  current stochastically smaller than baseline). current n=40, median=3.0000, mean=3.0250;
  baseline n=40, median=4.0000, mean=4.0500; median delta -1.0000. ...
```

Mann-Whitney rather than a t-test, because judge scores are ordinal and bounded
and a t-test would be testing an assumption the data does not meet. One-sided,
because getting better is not a regression.

### Section 7: the audit trail

```
  records               89
    assertion.evaluated 6
    judge.init          1
    judge.verdict       80
    sample.completed    2
```

Every verdict, every sample, and every gate — passes included — is one JSON line
in `--out/evidence.jsonl`. There is no delete API. The last few records are
printed with **timestamps elided**, because the walkthrough claims byte-identical
output for a given seed and a real timestamp would make that false; the file
itself has them.

Go and look at it:

```
type .rigor-run\evidence.jsonl
```

---

## The Opik leg

```
.\.venv\Scripts\python.exe -m opik_rigor.examples.summarise_eval --seed 7 --n 40 --opik
```

Everything above needs nothing outside the process. `--opik` is the exception: it
mirrors the sample into Opik as a trace, with one span per run, and attaches the
pass-rate gate's report as feedback scores on that trace.

It needs two things the script cannot conjure:

1. **the client** — `pip install "opik-rigor[opik]"`;
2. **somewhere to send it** — a local instance
   (`git clone https://github.com/comet-ml/opik`, then
   `cd opik/deployment/docker-compose && docker compose up -d`, then
   <http://localhost:5173>), or Opik Cloud via `opik configure`.

If either is missing, the script says so, explains how to fix it, and **exits 0**.
That is not politeness. rigor's core never imports an integration, so a dashboard
that is down costs you a dashboard and not a test suite, and the example is built
to demonstrate that rather than assert it.

`opik_rigor.integrations.opik` is imported *inside* the `--opik` branch. Without the
flag the module is never touched at all.

Two things to expect when you do have `opik` installed:

- **Opik's client does not raise when the destination is unreachable.** It batches
  in a background thread and writes `OPIK:` warnings to stderr. So the script says
  the trace was *handed to* Opik, and prints the trace id — that is proof rigor
  built the trace, not proof a server accepted it. Check your project.
- Those `OPIK:` lines on stderr are Opik's, not rigor's.

---

## Files

| File | What it is |
|---|---|
| `../src/opik_rigor/examples/summarise_eval.py` | The example. One script, one story, runnable — and **in the wheel**, which is why it is not in this directory. |
| `test_example_runs.py` | Runs the example as a subprocess (`-m opik_rigor.examples.summarise_eval`) and asserts it still works and is still deterministic. |
| `.rigor-run/` | Output: the evidence log and the baseline. Written under your current directory, cleared on every run. |
