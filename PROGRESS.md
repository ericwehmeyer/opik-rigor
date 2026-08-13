# PROGRESS

Running state of the v0.1 build. Updated at the end of every session. If you are
picking this up cold, read this file and `docs/` — nothing important lives only in
a chat transcript.

## Where the build stands

| Session | Scope | Status |
|---|---|---|
| 1 | Core, no network: scaffold, evidence, adapters, judge | **complete** — 218 passed, 1 skipped |
| 2 | Statistics: sampling, distribution, Baseline | not started |
| 3 | Integrations: Opik, pytest plugin, example | not started |
| 4 | Ship: README, rubric, tag v0.1.0 | not started |

## Session 1 — module status

| Module | State | Tests |
|---|---|---|
| `src/rigor/errors.py` | done | exercised via the modules that raise |
| `src/rigor/evidence.py` | done | `tests/test_evidence.py` — 72 |
| `src/rigor/pinning.py` | done | `tests/test_pinning.py` — 21 |
| `src/rigor/adapters/base.py` | done — Protocol, env constants, credential guards | via adapter tests |
| `src/rigor/adapters/{fake,anthropic,openai_compat}.py` | done | `tests/test_adapters.py` — 66 (+1 network-skipped) |
| `src/rigor/judge.py` | done | `tests/test_judge.py` — 46 |
| `rubrics/example-rubric.md` | done | asserted to end with `OUTPUT_FORMAT_INSTRUCTION` |
| `src/rigor/__init__.py` | done — re-exports the public surface | `tests/test_integration_session1.py` — 13 |

## Decisions made, and why

**The pin rule lives in one module.** `pinning.py` is the single definition of
"reproducible model id", imported by both the adapters and the judge. It was
written before either of them precisely so the two could not drift apart on what
the word means. An id must end in a concrete version marker (`-20250514`,
`-2024-08-06`, `-v1`, `-2.1.0`) and must not contain `latest`, `newest`,
`current`, `stable`, or `default`.

**The evidence log has no delete API.** Not an oversight — `tests/test_evidence.py`
asserts that no public name on `EvidenceLog` matches delete/remove/clear/truncate/
rotate/purge/update/pop, so adding one breaks the suite. Discarding evidence has
to be a deliberate act outside the library, which leaves a visible hole in the
timeline rather than a silent edit.

**A torn final line is tolerated on read; a malformed line anywhere else is an
error.** A process killed mid-append leaves a fragment with no trailing newline.
Dropping it costs one record; refusing to read the file costs the entire history.
A bad line in the middle is a different thing — that is corruption, and it raises.

**Line endings are normalised to LF via `.gitattributes`.** The judge hashes its
rubric file. Without this, the same rubric checked out on Windows and on a Linux CI
runner would hash differently, and the drift check would fire on every
cross-platform run — correctly, but uselessly.

**MIT license, `opik-rigor` distribution name, `rigor` import name.** The
distribution name is deliberately still open; per the build plan it is decided
before the v0.1.0 tag, not before Session 1. Changing it later touches one line of
`pyproject.toml`.

## Environment

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

Local venv is Python 3.14.4 with scipy 1.18.0 and pytest 9.1.1. CI runs Ubuntu and
Windows across Python 3.10–3.13.

## Invariants that must survive every later session

1. **Core never imports integrations.** `integrations/` may import core; nothing in
   `src/rigor/*.py` may import `rigor.integrations` or any provider SDK at module
   scope.
2. **The suite is green with no credentials present.** CI blanks `ANTHROPIC_API_KEY`
   and `OPENAI_API_KEY`. Anything needing a live endpoint is marked
   `requires_network` or `requires_opik` and deselected.
3. **`import rigor.adapters` must not require a provider SDK.** The `anthropic` and
   `openai` imports are lazy, inside `complete()`.
4. **A parse failure is never a fail-verdict.** Unparseable judge output is missing
   data; scoring it as a failure biases every statistic downstream of it.
5. **Every `evaluate()` writes exactly one verdict record** — never zero, never two.

**Unit suites test modules in isolation; one integration suite tests the seam.**
`tests/test_judge.py` scripts its own adapter rather than importing `FakeAdapter`,
and `tests/test_adapters.py` never builds a judge — which keeps them independent
but proves nothing about the two fitting together.
`tests/test_integration_session1.py` covers exactly that, plus the package-level
invariants no single module owns (notably a subprocess check that importing
`rigor` loads no provider SDK and no integration module).

**Credential guards live in `adapters/base.py`, not `fake.py`.** They started in
`fake.py` because it was the only provider-free module at the time; that left the
real adapters importing from the test double, which is backwards. `base.py` is
the seam module and has no provider dependency either.

## Known gaps entering Session 2

- **A provider exception during `evaluate()` writes no evidence.** The judge lets
  `AdapterError` propagate, so a provider outage leaves no trace in the audit
  trail. This is deliberate for now — `sampling.py` owns failure accounting and is
  where an outage should be recorded — but decide in Session 2 whether the log
  should carry a `judge.call_failure` event too. `test_a_provider_outage_
  propagates_rather_than_scoring_zero` pins the current behaviour.
- **The judge's score range is hardcoded to 1–5** (`SCORE_MIN`/`SCORE_MAX` in
  `judge.py`, interpolated into the prompt so rubric, prompt, and validator cannot
  disagree). Fine for v0.1; a configurable range is a roadmap item, not a v0.1 gap.
- Coverage is measured but no threshold is enforced in CI. Decide a floor in
  Session 2, once the statistics modules exist and the number means something.
- CI has never actually run — the workflow is written but there is no remote yet.
  Push before trusting it.
