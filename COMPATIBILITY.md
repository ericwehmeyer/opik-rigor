# Opik compatibility

What rigor's Opik integration depends on, how it was verified, and what breaks if
Opik changes it.

**Verified against:**

| | |
|---|---|
| `opik` | **2.2.28** |
| Verified on | 2026-08-13 |
| Python used to verify | 3.14.4 (Windows) |
| Opik's own `Requires-Python` | `>=3.10` |
| Method | Installed the package into a clean venv and introspected the live objects with `inspect.signature`; cross-checked against the published docs |

The surface below was read off the installed package, not copied from a tutorial.
Where the rendered documentation disagreed with the installed code, the installed
code won — see [Corrections](#corrections-to-the-published-docs).

## The API rigor actually calls

This is the whole dependency. It is deliberately tiny: two functions' worth of
surface, so that an Opik release which moves something costs an afternoon rather
than a rewrite.

```python
opik.Opik(project_name=None, workspace=None, host=None, api_key=None, batching=True)

client.trace(
    id=None, name=None, start_time=None, end_time=None,
    input=None, output=None, metadata=None, tags=None,
    feedback_scores=None, project_name=None, error_info=None,
    thread_id=None, attachments=None, environment=None,
) -> Trace

trace.span(
    id=None, parent_span_id=None, name=None, type="general",
    start_time=None, end_time=None, metadata=None,
    input=None, output=None, tags=None, usage=None,
    feedback_scores=None, ...
) -> Span

trace.end()
span.end()

client.log_traces_feedback_scores(scores, project_name=None) -> None
client.flush(timeout=None) -> bool
```

`FeedbackScoreDict` is a `TypedDict`:

| key | required | type |
|---|---|---|
| `name` | **yes** | `str` |
| `value` | **yes** | `float` |
| `id` | no | `str` |
| `reason` | no | `str \| None` |
| `category_name` | no | `str \| None` |

`type` on a span is `Literal["general", "tool", "llm", "guardrail"]`.

### Not used, on purpose

`create_experiment` / `Experiment.insert` require a `dataset_name` and take
`ExperimentItemReferences(dataset_item_id, trace_id)` — i.e. an experiment item
must point at a **dataset item that already exists in Opik**. rigor does not own
your datasets and will not create them behind your back, so assertion outcomes are
logged as **feedback scores on a trace**, not as experiment items. If you want
rigor's verdicts inside an Opik experiment, create the dataset and experiment
yourself and pass the trace ids; that is a documented seam, not a missing feature.

## Findings that shaped the integration

### 1. Opik registers a `pytest11` entry point named `opik`

```
group='pytest11'  name='opik'  value='opik.plugins.pytest.hooks'
```

So rigor's own pytest plugin **must not** use the name `opik`, and must not
register hooks that assume they are the only plugin present. rigor registers under
the name `rigor`. Both plugins load together; the co-installation test asserts
collection succeeds with both active.

Opik's plugin is inert unless `llm_unit`-decorated tests are collected, or it is
forced on with `--opik` / `opik_pytest_enabled = true`. rigor does not set either,
so installing both changes nothing about an existing suite.

Opik's decorator is:

```python
opik.llm_unit(
    expected_output_key="expected_output",
    input_key="input",
    metadata_key="metadata",
)
```

rigor's marker is `@pytest.mark.rigor_repeat(n=..., min_rate=...)`. Different
name, different mechanism (a pytest marker rather than a function decorator), so
the two compose rather than collide.

(This line said `@rigor.repeat(...)` until it was corrected — the spelling guessed
in the build plan, written into this file *before* the plugin existed, and never
reconciled against `MARKER_NAME` once it did. Writing the compatibility document
first is still the right order; it just means the parts describing your own
unwritten code are predictions, and predictions need checking back.)

### 2. `opik.record_traces_locally()` makes offline testing real

```python
with opik.record_traces_locally() as storage:
    ...
    storage.trace_trees   # flushes the client, then returns what was recorded
    storage.span_trees
```

This is why rigor's Opik tests are not mock theatre: they run the real client
through the real code path and inspect what it actually produced, with no server.
Opik's own docstring warns the emulator is *connection-scoped* — concurrent
operations on the same connection can leak into the same handle — so tests look
traces up by id rather than assuming the handle holds only their own.

**Caveat found by running it, not by reading it: `record_traces_locally()` does
not stop the online processor.** The local emulator sits *behind* the real backend
processor in the same `ChainedMessageProcessor`, so the client still tries to reach
a server and the emulator only sees a message once the online path has given up.
Against an unreachable host that cost ~17s for the first trace; against a
reachable endpoint returning a fast 401 it is ~2s but needs internet. "No server"
is true of Opik; it is not true of the socket.

rigor's tests therefore run a loopback `ThreadingHTTPServer` that returns 401 to
everything and point `OPIK_URL_OVERRIDE` at it — fully offline, ~0.05s per trace.
`OPIK_CONFIG_PATH` is aimed at a nonexistent file as well, so a developer's real
`~/.opik.config` cannot leak a live host or workspace into the fixture.

### 2b. `log_traces_feedback_scores` drops bad batches silently

A batch that fails Opik's pydantic validation (extra keys are forbidden) is
`LOGGER.error`-ed and **discarded — no exception reaches the caller**. Nothing
would tell you your scores never arrived. rigor coerces every value to a JSON-safe
scalar before handing it over, and restricts what it sends to a documented
whitelist, rather than relying on an error it would never see.

`trace.span(error_info=...)` does exist (inside the `...` above).
`ErrorInfoDict` requires `exception_type` and `traceback`; `message` is
`NotRequired`. `log_traces_feedback_scores` takes the **trace id** as each score's
`id` key.

### 3. `inspect.signature(Trace.span)` raises on Python 3.14

```
AttributeError: 'function' object has no attribute 'Span'
```

In `opik/api_objects/trace/trace_client.py`, `Trace.span` is annotated
`-> span.Span`, but the method's own parameter list shadows the module name
`span`. Under PEP 649's lazy annotation evaluation (Python 3.13+, default in 3.14)
the annotation is resolved on demand, in a scope where `span` now names the
function, and resolution fails.

**Scope: introspection only.** Calling `trace.span(...)` works. It breaks
`inspect.signature`, `typing.get_type_hints`, and anything that walks annotations
— which includes some doc generators and runtime validators. rigor never
introspects Opik's objects, so it is unaffected, but the integration tests do not
rely on signature introspection either, and that is deliberate rather than
accidental.

This is an upstream bug in opik 2.2.28. Reported here so that a future reader who
hits it knows it is not rigor's doing.

## A correction to this file, not to the docs

**An earlier version of this section claimed Opik's published SDK reference
renders every parameter with a leading underscore (`_name`, `_input`), and warned
that code copied from it would silently produce an unnamed trace. That claim was
wrong, and Opik's documentation is fine.**

What happened: the page was read through a tool that converts HTML to markdown.
The reference italicises each parameter, and italics in markdown are `_like this_`
— so every name came back wearing one extra leading underscore. The tell was
sitting in the same output the whole time. Two genuinely private parameters,
`_use_batching` and `_show_misconfiguration_message`, came back as
`__use_batching` and `__show_misconfiguration_message`. *Every* name had gained
exactly one underscore, which is a converter artifact, not a documentation
convention.

It is left in the file rather than quietly deleted because the mistake is
instructive. Session 3's rule was "verify the vendor API by installing and
introspecting it, not by reading the docs" — and that rule worked: the introspected
signature was correct and the integration was built against it. The error was in
the *other* direction, asserting a fault in someone else's work on the strength of
a single tool's rendering. Introspection told us what the API is; it never told us
what the documentation says, and the second claim needed its own evidence.

Verified against the live page on 2026-08-13 after the fact: the parameters are
ordinary names, exactly as the installed code has them.

The one real documentation note that survives: the page
`comet.com/docs/opik/testing/pytest_integration` 404s; the live one is
`comet.com/docs/opik/v1/testing/pytest_integration`.

## Version policy

`pyproject.toml` declares `opik = ["opik>=2.0,<3"]`.

- **Lower bound 2.0** because the verified surface above is 2.x. 1.x is untested
  by this project and the extra should not claim otherwise.
- **Upper bound <3** because a major bump is exactly where `trace()` keyword names
  would change, and the failure mode of guessing wrong is a silently unnamed trace
  rather than an exception. This one does not depend on the retracted claim above:
  the real signature ends in `**ignored_kwargs`, confirmed by introspecting the
  installed package, so an unrecognised keyword is swallowed rather than raising.

## What to do when this drifts

The integration is two functions in `src/opik_rigor/integrations/opik.py`. If an Opik
release breaks them:

1. Re-run the introspection in a clean venv and update the table at the top.
2. The tests in `tests/test_integration_opik.py` run against
   `record_traces_locally`, so they fail loudly rather than silently logging
   nothing — a broken mapping shows up as a missing field, not as green tests.
3. Update the version bound and this file in the same commit as the code fix, so
   the record of what was verified never lags the code.

rigor's core never imports Opik. If the integration breaks, the assertions, the
judge, and the evidence log keep working — you lose a dashboard, not a test suite.
