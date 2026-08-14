# Announcement drafts

Drafts only. Nothing here has been posted anywhere. Edit freely before use.

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

I built a small Python library over four sessions, and the method is the
interesting part.

The plan came first: module contracts, session boundaries, and a test inventory
used as the acceptance contract, all written and approved before any code
existed. It is committed verbatim, including the parts it got wrong.

Then role separation — the agent that wrote a module never wrote its tests. The
test author derived expected Wilson bounds independently, by root-finding the
inequality that defines the interval, instead of checking the implementation
against itself.

That separation caught three real bugs, one of them a lower confidence bound
that came out above its own point estimate at zero successes. A fourth bug —
three tests asserting a property of the environment, not of the library — was
caught by an environment change, not by the separation.

Why single-shot assertions fail on non-deterministic systems: one run of 20 is a
sample, not a property, so a green test has told you about 20 coin flips rather
than about the system.

The library gates and pins; it is not an eval platform. 514 tests pass offline
with no credentials. On PyPI: pip install opik-rigor

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
