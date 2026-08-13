# opik-rigor

Statistical assertions and pinned-judge evaluation primitives for LLM test suites.

> **Status: v0.1 in progress.** This README is a placeholder written in Session 1 so
> the package builds. The real one — problem statement, primitives, 10-line
> quickstart, SR 11-7 notes, roadmap — is written in Session 4. See `PROGRESS.md`
> for where the build actually stands.

## What it is

Two primitives, done rigorously:

1. **A pinned judge.** An LLM-as-judge that refuses to run against an aliased model
   id, hashes its rubric, and raises when the rubric changes underneath a baseline.
2. **Statistical assertions.** `assert_pass_rate`, `assert_score_distribution`, and
   `assert_no_regression` — gates that account for the fact that you sampled a
   stochastic system n times rather than measured it once.

Everything either produces goes to an append-only evidence log with no delete API.

## License

MIT — see [LICENSE](LICENSE).
