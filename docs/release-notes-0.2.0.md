# opik-rigor 0.2.0

2026-08-14

One change removes input that 0.1.1 accepted, and it is first because it is the
only thing here that can break a working call. The two after it are the reason to
take this release today: on 0.1.1 this library could not judge with any Anthropic
model that Anthropic currently serves.

## Breaking — a one-sided `confidence` at or below 0.5 is refused

`wilson_lower_bound` and `assert_pass_rate` now raise `ValueError` when
`confidence <= 0.5`. They used to accept it and answer.

A confidence at or below a coin flip produces an interval that means nothing, and
the library now refuses to manufacture false evidence out of it. Below 0.5 the `z`
is negative, so the "lower bound" lands *above* the observed rate and gets worse as
the sample grows — `wilson_lower_bound(89, 100, 0.0001)` was 0.9615 and the same
rate over 1000 runs gave 0.9216, more evidence for a lower floor. At exactly 0.5
the `z` is zero and the bound *is* the raw rate, so
`assert_pass_rate((20, 20), 1.0, confidence=0.5)` passed: twenty runs proving
perfection, which is the claim this library exists to refuse. Every one of those
numbers was arithmetically correct, which is why this is a narrowed domain rather
than a corrected formula. A gate written `confidence=0.3` reads in a test file as
an act of statistical caution and was looser than comparing the raw rate.

**Migration, in one line:** raise the confidence above 0.5, or drop the argument
and take the 0.95 default; if you wanted the full two-sided range, call
`wilson_interval`, which is deliberately unaffected and still accepts any
confidence strictly inside `(0, 1)`.

**Who this actually affects.** Only callers who passed an explicit `confidence` at
or below 0.5 to `wilson_lower_bound` or `assert_pass_rate` — including through the
pytest plugin, as `@pytest.mark.rigor_repeat(..., confidence=0.4)`, where the
refusal now arrives after the runs. Everyone on the default is untouched, and no
gate's verdict moves for any input still accepted.

## `AnthropicAdapter` could not call any current Anthropic model

0.1.1 sent `temperature` on every request, with a constructor default of `0.0`, and
`temperature`/`top_p`/`top_k` were removed from the Messages API on the current
generation — Opus 5, Opus 4.8, Opus 4.7, Sonnet 5, Fable 5 and Mythos 5 return a
**400** for any of them. No configuration avoided it: the parameter had no
omit-sentinel, the constructor rejected `None`, and the value was passed
unconditionally. A judge built the way the README builds one could not complete a
single call.

`temperature` now defaults to `None`, meaning the key is **absent** from the
request rather than set to something chosen for you, and it is omitted on every
model, current or older. An explicit value is still sent to a model that accepts
one; against a model documented not to accept one it is refused at *construction*,
naming the model, rather than becoming a 400 partway through a run that has already
spent calls. The cost worth naming: an older model that still takes sampling
parameters now gets the API default rather than `0.0` unless you ask for it.

## `is_pinned` rejected every current frontier Anthropic model id

`claude-opus-5`, `claude-sonnet-5`, `claude-opus-4-8` and every other current
Anthropic id were refused, because the rule required a trailing date and
Anthropic's ids no longer carry one — while `claude-3-7-sonnet-20250219`, retired
in February 2026, was accepted. `require_pinned` runs at config load, so this
refused to start. The rule now checks the property instead of the spelling: no
alias token (`latest`, `newest`, `current`, `stable`, `default`), and an ending in
a release designator — a release number, a date stamp, or an explicit version.
Vertex and Bedrock snapshot spellings are recognised. `gpt-4.1`, an OpenAI alias
0.1.1 wrongly accepted, is now correctly refused.

With this and the adapter fix, both blockers between a fresh install and a judgement
from a current Anthropic model are closed. They were sequential: the pin rule was
the front door locked, and the 400 was there being no room behind it.

## The PyPI page

A project's long description is frozen at upload time, so pypi.org has been
rendering 0.1.1's README — dead links, an example command that only resolves in a
git checkout, and a printed key the report dict does not have. None of the
documentation fixes in this release could reach that page. Publishing 0.2.0 is what
un-stales it.

## Install

```bash
pip install --upgrade opik-rigor
```

If you pinned `>=0.1.x,<0.2`, that bound has to widen before this release can reach
you — read the breaking change above before it does.

The complete record, including the score-distribution gate that returned green on
infinite input, the `_runs_needed` correction, and the import-cost work that takes
`import opik_rigor` from about a second to a small fraction of one, is in the
[CHANGELOG](https://github.com/ericwehmeyer/opik-rigor/blob/v0.2.0/CHANGELOG.md).
