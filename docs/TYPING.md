# Typing: what `py.typed` promises, and how it is checked

`opik_rigor/py.typed` is a PEP 561 marker. It tells a downstream user's type
checker to trust the inline annotations in this package. Before 2026-08-14
nothing had ever run a type checker against it, in this repository or outside it.

A wrong annotation behind that marker is worse than no marker at all. It does not
merely fail to help: it fails the build of every downstream project that runs
mypy or pyright in strict mode, and they cannot opt out short of an ignore rule.

## Verdict

**The claim is true, for the surface a consumer actually touches.** It was not
true when first checked. See [What was wrong](#what-was-wrong).

The evidence is a separate, unrelated project — its own `pyproject.toml`, its own
venv, this repository nowhere on `sys.path` — that imports every name in
`__all__`, implements the `Adapter` protocol, catches every exported exception,
and feeds every documented input form to every gate. Both checkers pass it in
strict mode.

| | version | consumer project, strict |
|---|---|---|
| mypy | 2.3.0 | clean |
| pyright | 1.1.411 | clean |

Checked on Python 3.14.4 (Windows) against a built wheel installed into a clean
venv, not against the source tree.

## The automated check

`scripts/verify_release.py` runs `wheel-annotations`: it extracts the built wheel
and runs `mypy --strict` against **the package as shipped**, not against `src/`.
This is the same rule every other check in that script follows — `packages = [...]`,
a build-backend change or a `.gitignore` rule can make the tree and the zip
differ, and it is the zip that ships.

The check FAILs on a bad annotation and SKIPs (never passes) when mypy is absent.
CI installs `.[typecheck]` in the build job so the row cannot silently skip; a
skip is exit 2, which that job treats as fatal.

Run it locally:

```
python -m pip install -e ".[typecheck]"
python scripts/verify_release.py
```

## pyright is a release-time check, not a CI check

pyright is distributed as a Node program — the PyPI package downloads a Node
runtime on first run. That is a network fetch and a second language runtime
inside a gate whose value depends on being hermetic, so it is not in CI.

To run it before a release:

```
python -m pip install pyright
python -m pyright --pythonpath <path-to-your-venv-python> <a-consumer-project>
```

**Point `--pythonpath` at the interpreter that has opik-rigor installed.** Without
it pyright reports `Import "opik_rigor" could not be resolved` and then ~117
cascading "type is unknown" errors, all of which are one configuration mistake
rather than 118 defects.

## What was wrong

### 1. The credential catch-all accepted credentials, statically

All three adapters annotated their `**forbidden` catch-all as `object`:

```python
def __init__(self, model_id: str, *, timeout: float = 60.0, **forbidden: object) -> None:
    reject_credential_kwargs(forbidden, type(self).__name__)
```

Every value is assignable to `object`, so this type-checked clean under **both**
checkers in strict mode:

```python
AnthropicAdapter("claude-opus-4-20250514", api_key="sk-...")   # no diagnostic
```

and raised `TypeError` at runtime. That call is the exact mistake this package
exists to prevent, written the way a user would naturally write it, and the
`py.typed` marker promised a checker would catch it.

Fixed by annotating the catch-all with the bottom type
(`base.ForbiddenKwarg = NoReturn`), which no value is assignable to — matching the
runtime rule that *every* keyword reaching the catch-all raises. `NoReturn` rather
than `Never` only because `Never` landed in `typing` in 3.11 and this package
supports 3.10; the two are the same type to both checkers.

Costs nothing at runtime (every adapter module uses
`from __future__ import annotations`) and does not break `Adapter(**config)`
forwarding, because `Any` remains assignable to `NoReturn`. Pinned by
`tests/test_adapters.py::test_the_credential_catch_all_is_annotated_as_the_bottom_type`.

### 2. `_marker_arguments` inferred a `Literal` key type

`dict(zip(_POSITIONAL, marker.args))` infers
`dict[Literal["n", "min_rate", ...], Any]`, which then rejects both the
`.update(marker.kwargs)` below it and the declared `dict[str, Any]` return.
Fixed by declaring the local. Pyright only; mypy did not flag it.

### 3. `_coerce_scores` called a value pyright typed as `object`

`callable(x)` narrows to a callable returning `object`, and `object` is not
iterable. Fixed by resolving the attribute once into a local and annotating the
call's result `Any` — which also removed a double attribute lookup, so a caller
whose `scores` was a property no longer pays for it twice.

## Diagnostics that are *not* defects

**`scipy` ships no `py.typed` and no stubs.** As of 1.18 there is no marker
anywhere in the package, so `mypy --strict` reports `import-untyped` for
`scipy.stats`. It is a hard dependency, but it is imported lazily inside
`_mannwhitneyu()`, which is annotated `-> Any` precisely so SciPy's absent types
cannot leak into `assert_no_regression`'s signature. Silenced in
`[[tool.mypy.overrides]]`; `scipy-stubs` exists but is not depended on.

**`anthropic` / `openai` / `opik` are optional extras** and are absent from a
minimal install. Same override, same reasoning.

**pyright strict reports ~90 diagnostics against the package's own internals.**
None reach a consumer — library internals are not checked by a downstream user,
which the clean consumer run demonstrates. They are, by rule:

| count | rule | verdict |
|---|---|---|
| ~39 | `reportUnknownArgumentType` | `Any` from `json.loads` and `Mapping[str, Any]` in the baseline/evidence parsers. Inherent to parsing arbitrary JSON. |
| ~21 | `reportUnnecessaryIsInstance` | **Deliberate.** The validators re-check types the annotations already assert, because a caller without a type checker can pass anything. Removing these would delete the runtime validation this library is built on. |
| ~21 | `reportUnknownVariableType` | Same origin as the argument-type cluster. |
| 2 | `reportPrivateUsage` | `_short_repr` across modules, and `pyfuncitem._fixtureinfo` — the latter is what pytest's own `pytest_pyfunc_call` uses and has no public equivalent. |

**`"Anthropic" is not a known attribute of module "anthropic"`** — pyright
resolving `import anthropic`, inside `opik_rigor/adapters/anthropic.py`, to that
same file. A resolution artifact of checking the package directory directly; it
disappears the moment a real `anthropic` (or a stub) is on the path, and Python's
absolute imports never do this at runtime (verified: the import fails cleanly with
`ImportError`, which the adapter handles). Not a defect, but the reason a module
named after the package it imports is worth avoiding.
