# opik-rigor — v0.1 Build Plan

This is the plan the bootstrap prompt's Rule 1 asks for, produced up front. Hand it to Claude Code as the approved plan ("here is the approved plan; skip planning and execute Session 1") — which saves the planning tokens and keeps you the architect.

## 1. Architecture

Five core modules, two optional integrations, one hard rule between them: **core never imports integrations; integrations only import core.**

**evidence.py** — the foundation everything logs through. One class, EvidenceLog: append-only JSONL writer with a fixed envelope (timestamp, event_type, payload). No rotation, no deletion methods — the absence of a delete API is a feature and gets a line in the README. Everything else takes an EvidenceLog instance; nothing writes evidence any other way.

**adapters/** — the provider seam. A Protocol with one method, `complete(prompt: str) -> str`, plus a `model_id` property. Ship three: AnthropicAdapter, OpenAICompatAdapter (covers Azure AI Foundry endpoints), and FakeAdapter (deterministic scripted responses, used by the entire test suite). Keys from env vars only; constructors raise if a key is passed as an argument.

**judge.py** — PinnedJudge. Takes an adapter, a rubric path, and an EvidenceLog. On init: validate the model_id against an explicit-version regex (reject "latest", bare family names, empty), hash the rubric (sha256), record both. `evaluate(input, output) -> Verdict` where Verdict is a frozen dataclass (passed: bool, score: float | None, raw: str). Strict parsing of the judge's structured response; parse failure raises JudgeOutputError and logs it — never coerced to a fail-verdict silently. Rubric drift check: if the log's last-recorded hash for this judge name differs from the current file hash, raise RubricDriftError unless accept_rubric_change=True (which logs old and new hash as a distinct event).

**sampling.py** — `sample(fn, n, concurrency=1, timeout=None) -> SampleResult`. Collects successes, failures, and exceptions separately; exceptions count as failures by default (errors_as_failures flag). SampleResult carries everything the assertions need plus wall-clock and per-run durations (free data, useful later for cost tracking).

**distribution.py** — the three assertions from the spec. assert_pass_rate uses the Wilson score interval (chosen over Clopper-Pearson: less conservative at small n, standard, closed-form — document this). assert_score_distribution gates on mean, p10, and stddev independently, any subset. assert_no_regression uses Mann-Whitney U (nonparametric because judge-score distributions are routinely non-normal and often multi-modal — document this too). Baselines: versioned JSON with embedded sha256, loaded via a Baseline class that verifies its own hash on load. All assertions raise a rich AssertionError subclass whose message includes n, statistic, interval, threshold, and verdict — the failure message is the statistical report.

Dependency policy: scipy for Mann-Whitney; Wilson interval implemented directly (ten lines, avoids statsmodels). Core deps: scipy only. Everything else stdlib.

**integrations/opik.py** (extra: `[opik]`) — two functions, not a framework: `log_sample_to_opik(sample_result, ...)` mapping runs to traces, and `log_assertion_to_opik(outcome, ...)` mapping verdicts to experiment results. Import of opik happens inside the functions with a clear error if missing. Exact API calls verified against live docs during Session 3, recorded in COMPATIBILITY.md.

**integrations/pytest_plugin.py** (extra: `[pytest]`) — `@rigor.repeat(n=...)` marker that wraps a test in sample() + assert_pass_rate, and a `rigor_judge` fixture factory. Registered via entry point; must not shadow or conflict with Opik's own pytest plugin (verify by installing both in Session 3).

## 2. Build phases — one session each, handoff between every one

**Session 1 — Core, no network.** Scaffold (pyproject, LICENSE, CI workflow, PROGRESS.md), then evidence.py, adapters (FakeAdapter fully; real adapters as thin stubs with env-var loading), judge.py, and their tests. Exit criteria: pytest green offline; rubric-drift, alias-rejection, and parse-failure tests passing; first commit history clean. This session touches no statistics and no external docs — it should stay small.

**Session 2 — Statistics.** sampling.py, distribution.py, Baseline, and the math verification tests: known-distribution fixtures proving the Wilson gate behaves correctly at n=20 vs n=200, the regression test detects a shifted distribution at alpha=0.05 and doesn't false-alarm on identical ones (seeded). Exit criteria: the library's own test suite uses assert_pass_rate on a stochastic FakeAdapter somewhere — eating its own cooking — and CI is still key-free.

**Session 3 — Integrations + example.** Fetch current Opik SDK docs first; write COMPATIBILITY.md before code. Then integrations, the pytest plugin, the end-to-end example (toy summarizer, FakeAdapter standalone path + real Opik path clearly gated behind "requires local Opik"), and the co-installation check with Opik's pytest plugin. Exit criteria: example runs offline end-to-end; both extras install cleanly.

**Session 4 — Ship.** README (problem → primitives → 10-line quickstart → SR 11-7 "designed to support" section → roadmap), rubrics/example-rubric.md, final report per the bootstrap prompt, tag v0.1.0. Shortest session; mostly writing, and the quickstart gets copy-paste tested before it's believed.

Each session ends with /handoff, the cold-start check, /clear. If any session's context approaches ~100K, split at the nearest module boundary — the module contracts above are designed so any single module is a complete, resumable unit of work.

## 3. Test inventory (the review checklist for Session 1–2 output)

Judge: rejects alias model strings; accepts explicit versions; computes and logs rubric hash; raises RubricDriftError on changed rubric; accept_rubric_change logs both hashes; parse failure raises and logs, never coerces; every evaluate() call produces exactly one evidence line.

Distribution: Wilson lower bound math against hand-computed values; pass_rate correctly fails 70%-true-rate at n=20 with min_rate=0.9; correctly passes 95%-true-rate at n=200 with min_rate=0.9; score gates trigger independently; Mann-Whitney detects a 0.5-sigma shift at reasonable n; no false alarm on identical seeded distributions; baseline hash verification rejects a tampered file.

Sampling: concurrency produces n results; timeout counted as failure; errors_as_failures=False excludes exceptions from the denominator; durations recorded.

Evidence: append-only (two writers, interleaved, no loss); envelope schema stable; log file valid JSONL under a crash-mid-line simulation (truncated final line tolerated on read).

Integrations: import errors are clear when extras absent; pytest marker runs a test n times and applies the gate; co-installation with opik's plugin collects without error.

## 4. Risks and their pre-decided answers

Opik API drift → COMPATIBILITY.md pins the verified version; integrations are two functions, cheap to update. Scope creep toward "eval platform" → the roadmap section exists precisely to park good ideas; v0.1 is two primitives done rigorously. Statistical review risk (someone checks your math in an interview) → the two "document the choice" notes in distribution.py are your prepared answers; keep them sharp. Naming → package name is one constant; decide before the v0.1.0 tag, not before Session 1.

## 5. Definition of done for v0.1

Someone who has never spoken to you can: pip install it, run the quickstart against the FakeAdapter in under five minutes, read one rubric file and one evidence line and understand the audit model, and — if they run Opik — see a verdict land in their dashboard. That's the whole bar. Everything else is roadmap.
