# Announcement drafts

Drafts only. **Nothing here has been posted anywhere.** Edit freely before use.

## State, as of 2026-08-13

| Draft | State |
|---|---|
| 1. Opik post | **on hold — no venue.** Discussions disabled on `comet-ml/opik`; no community channel in their `CONTRIBUTING.md`. See the note on that section. |
| 2. LinkedIn post | **rewritten, ready.** Unposted. |
| 3. README badge row | **applied** to `README.md` at the 0.1.0 release. |

### Why the LinkedIn post was rewritten

The first version opened on the method — four sessions, plan-first, role
separation — and never said what the library actually does until its
second-to-last line. Every fact in it was correct; the ordering was wrong for a
feed, where a reader decides in two lines whether to keep going.

The rewrite leads with the concrete result instead: 18/20 and 900/1000 are both
90%, only one is evidence, and neither clears a 90% gate. That states what the
tool is for before asking anyone to care how it was built, and the method story
lands better as the second beat because the attention has been earned. Length and
register are unchanged — under 200 words, no hype adjectives, no emoji.

Facts re-verified at rewrite time: both Wilson bounds, the count of bugs found by
authorship separation (three, all in code the author had written or specified),
the test count, and the install command. One claim from the original was
deliberately dropped rather than carried over — that a specific provider alias
re-pointed on a specific date. Aliases re-pointing is true by definition and is
the argument; that particular event was illustrative and was never verified.

---

## 1. Opik post — ON HOLD, no venue

**Checked 2026-08-13: `comet-ml/opik` has GitHub Discussions disabled.** Its
`CONTRIBUTING.md` names no Slack, Discord, or forum, and says nothing about
sharing third-party tools built on Opik. Issues are open (196 of them, 21.4k
stars), but announcing your own project there is off-topic by default.

This draft was written for a venue that was never verified to exist — the same
mistake as the retracted documentation claim in `COMPATIBILITY.md`: asserting
something about another project's setup without checking it.

Held rather than deleted, because the text is still good and a venue may appear.
If one is wanted sooner, the honest route is a real contribution — an integration
docs PR — not an announcement. Meanwhile the package is discoverable on PyPI
under the `opik` keyword, and its README links back.

**Title (if a venue appears):** Statistical gates and judge pinning on top of Opik

I have been using Opik for tracing and evaluation, and built a small library for
two problems adjacent to it. A test asserting `pass_rate >= 0.9` over 20 runs
measures a stochastic system once; and `claude-3-5-sonnet-latest` re-points, so
scores from before it are not comparable to scores after.

opik-rigor's gates compare a one-sided Wilson lower bound against your bar, not
the observed rate: 18/20 gives 0.7383, 900/1000 gives 0.8833, so a 90% observed
rate never clears `min_rate=0.9` at any n. The judge refuses aliased model ids
at construction, hashes its rubric, and raises on drift.

A complement, not an alternative: Opik owns datasets, dashboards, tracing; this
owns the gate and the pin. The integration is two functions; the core never
imports Opik.

Verified against opik 2.2.28 on 2026-08-13. One field note: Opik's `pytest11`
entry point is named `opik`, so ours registers as `rigor`; both collect together,
and Opik's stays inert unless `llm_unit` tests are collected. `record_traces_locally`
made the integration testable without a server, which is why its tests drive the
real client rather than a mock.

pip install opik-rigor. MIT, Python 3.10+.

https://github.com/ericwehmeyer/opik-rigor

---

## 2. LinkedIn post

18 of 20 evals passed. 900 of 1000 passed. Both are 90%.

Only one of those is evidence — and neither of them clears a 90% gate. A lower
confidence bound approaches the observed rate from below and never reaches it,
so you have to beat the bar, not hit it. How much headroom you need is exactly
what sample size buys you.

I wrote opik-rigor to make that the default. It gates on the Wilson lower bound
rather than the observed rate, and it pins the LLM judge doing the grading to one
exact model version and one hashed rubric revision — because an aliased model id
re-points over time, and scores from either side of that are not comparable.

Four sessions, the plan approved before any code, and one rule that mattered more
than the rest: the agent that wrote a module never wrote its tests. Expected
values came from root-finding the definition of the interval, not from running
the implementation and trusting what came back.

That caught three real bugs. All three were in code I had written or specified.
One of them was a lower confidence bound that came out above its own point
estimate.

514 tests, no credentials, no network.

pip install opik-rigor
https://github.com/ericwehmeyer/opik-rigor

---

## 3. README badge row

Already applied to `README.md`. The PyPI badge went live with the 0.1.0 release
on 2026-08-13 and is no longer commented out.

```markdown
[![PyPI](https://img.shields.io/pypi/v/opik-rigor)](https://pypi.org/project/opik-rigor/)
[![CI](https://github.com/ericwehmeyer/opik-rigor/actions/workflows/ci.yml/badge.svg)](https://github.com/ericwehmeyer/opik-rigor/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
```
