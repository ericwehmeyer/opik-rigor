"""Tests for the model-id pinning rules.

``is_pinned`` is the single definition of "reproducible model id" in opik_rigor,
so these tests fix both halves of it: an id that names one immutable version is
accepted, and anything that can silently re-point under the caller is refused --
including every token in :data:`ALIAS_TOKENS`, iterated rather than hardcoded so
a newly added token cannot slip through untested.

The rule these tests are derived from
=====================================

An id is pinned when it names **one immutable model version**. The thing being
protected is that an alias re-points over time, which silently invalidates every
score recorded against it. Three clauses, in order:

**R1 -- no alias token.** ``latest``, ``newest``, ``current``, ``stable``,
``default``: words that mean "whatever is being served right now". A provider
that wants a moving pointer has to *name* it, and these are the names.

**R2 -- ends in a release designator.** Splitting on ``-``, ``_``, ``@`` and
``:``, the final component must be nothing but a version: ``5``, ``8``,
``20251001``, ``06``, ``v1``, ``2.1.0``. A final component that is a *word*
(``sonnet``, ``mini``, ``4o``, ``large``, ``preview``, ``instruct``) names a
*kind* of model, and a kind is exactly what a provider re-points at new weights.

**R3 -- the vendor exception.** For a provider documented to publish
``<family>-<number>`` as a moving pointer to its newest dated snapshot, R2 is not
sufficient: a date stamp or an explicit ``-vN`` is also required. Today that is
OpenAI, and it lives in one table in ``pinning.py``.

The table, derived by hand from those three clauses before the predicate was
written
=======================================================================

Read the columns left to right and stop at the first one that decides. "final
component" is what remains after splitting on ``-``, ``_``, ``@``, ``:``.

===================================== ========= ================ ======= =======
id                                    R1 alias  R2 final comp.   R3      verdict
===================================== ========= ================ ======= =======
claude-opus-5                         --        ``5``   number   n/a     PINNED
claude-sonnet-5                       --        ``5``   number   n/a     PINNED
claude-fable-5                        --        ``5``   number   n/a     PINNED
claude-opus-4-8                       --        ``8``   number   n/a     PINNED
claude-opus-4-7                       --        ``7``   number   n/a     PINNED
claude-opus-4-6                       --        ``6``   number   n/a     PINNED
claude-sonnet-4-6                     --        ``6``   number   n/a     PINNED
claude-haiku-4-5                      --        ``5``   number   n/a     PINNED
claude-haiku-4-5-20251001             --        date             n/a     PINNED
claude-sonnet-4-5-20250929            --        date             n/a     PINNED
claude-opus-4-20250514                --        date             n/a     PINNED
claude-3-5-haiku-20241022             --        date             n/a     PINNED
claude-3-7-sonnet-20250219            --        date             n/a     PINNED
claude-opus-4-5@20251101              --        date (``@``)     n/a     PINNED
anthropic.claude-opus-5               --        ``5``   number   n/a     PINNED
...-sonnet-20241022-v2:0              --        ``0``   (``:``)  n/a     PINNED
gpt-4o-2024-08-06                     --        ``06``  number   dated   PINNED
gpt-4.1-2025-04-14                    --        ``14``  number   dated   PINNED
gpt-4o-mini-2024-07-18                --        ``18``  number   dated   PINNED
gpt-oss-120b-v1                       --        ``v1``           ``-vN`` PINNED
fake-scripted-v1                      --        ``v1``           n/a     PINNED
my-finetune-2.1.0                     --        ``2.1.0``        n/a     PINNED
my_finetune_v1                        --        ``v1``  (``_``)  n/a     PINNED
mistral-large-v2                      --        ``v2``           n/a     PINNED
llama-3.1-8b-instruct-v1-2            --        ``2``   number   n/a     PINNED
------------------------------------- --------- ---------------- ------- -------
""                                    --        nothing to pin   --      REFUSED
"   " / "\\t\\n"                       --        nothing to pin   --      REFUSED
None / 42 / b"..." / ["x"]            --        not a string     --      REFUSED
claude                                --        ``claude`` word  --      REFUSED
my-finetune                           --        word             --      REFUSED
mistral-large                         --        word             --      REFUSED
claude-3-5-sonnet                     --        word             --      REFUSED
gpt-4o                                --        ``4o`` not num.  --      REFUSED
gpt-4o-mini                           --        word             --      REFUSED
claude-opus-4-20250514-preview        --        word             --      REFUSED
meta-llama/Llama-3.3-70B-Instruct     --        word             --      REFUSED
claude-3-5-sonnet-latest              ``latest``  --             --      REFUSED
claude-latest-20250514                ``latest``  --             --      REFUSED
gpt-5                                 --        ``5``   number   no date REFUSED
gpt-4                                 --        ``4``   number   no date REFUSED
gpt-4.1                               --        ``4.1``          no date REFUSED
===================================== ========= ================ ======= =======

Four rows are worth their own sentence, because they are where a fix that made
everything pass would have quietly removed the safety check:

* **claude-3-7-sonnet-20250219 is PINNED although it was retired on 2026-02-19.**
  Pinned means immutable, not available. A score recorded against it in 2025 is
  still comparable to another score recorded against it in 2025, which is the
  entire question ``is_pinned`` answers. A liveness check is a different check
  and this module does not claim to be one.
* **claude-sonnet-4-5 is PINNED, where the old rule refused it.** That is the
  verdict this change moves, and it moves because the id is genuinely immutable.
* **gpt-5 is REFUSED although claude-opus-5 is PINNED**, and the two are the same
  shape. Nothing in the string separates them; the difference is one provider's
  documented policy, so R3 carries it and R3 alone.
* **claude-3-5-sonnet is REFUSED while claude-sonnet-4-5 is PINNED.** Both are
  "family + tier + generation, no date". R2 is positional and only reads the last
  component, so it lands on the right answer here for a reason narrower than the
  one a human would give (``claude-3-5-sonnet`` was never a served id).
"""

from __future__ import annotations

import pytest

from opik_rigor.errors import ModelPinError
from opik_rigor.pinning import ALIAS_TOKENS, PINNED_EXAMPLES, is_pinned, require_pinned

#: Every row of the table above whose verdict is PINNED.
PINNED_IDS = [
    # Anthropic's current convention: the release number ends the id, no date.
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    # Anthropic's dated forms, current and historical.
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-20250514",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    # Separators other than "-" that providers use around a snapshot.
    "claude-opus-4-5@20251101",
    "anthropic.claude-opus-5",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    # OpenAI: only the dated (or explicitly versioned) forms.
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
    "gpt-4o-mini-2024-07-18",
    "gpt-oss-120b-v1",
    # Everything else: an explicit version marker.
    "fake-scripted-v1",
    "my-finetune-2.1.0",
    "my_finetune_v1",
    "mistral-large-v2",
    "llama-3.1-8b-instruct-v1-2",
]

#: Every row of the table above whose verdict is REFUSED, minus the non-strings.
UNPINNED_IDS = [
    "",
    "   ",
    "\t\n",
    "claude",
    "my-finetune",
    "mistral-large",
    "claude-3-5-sonnet",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-opus-4-20250514-preview",
    "meta-llama/Llama-3.3-70B-Instruct",
    "claude-3-5-sonnet-latest",
    "gpt-5",
    "gpt-4",
    "gpt-4.1",
]

#: One otherwise-pinned id per alias token: the version marker is present, so
#: only the alias may cause the rejection.
ALIAS_IDS = [f"claude-{token}-20250514" for token in ALIAS_TOKENS]

#: The ids a caller would reach for on their first day, all of which 0.1.1
#: refused. This list is the defect, restated as an assertion.
CURRENT_ANTHROPIC_IDS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
]


@pytest.mark.parametrize("model_id", PINNED_IDS)
def test_ids_that_name_one_immutable_version_are_pinned(model_id: str) -> None:
    assert is_pinned(model_id) is True


@pytest.mark.parametrize("model_id", UNPINNED_IDS)
def test_ids_that_can_re_point_are_not_pinned(model_id: str) -> None:
    assert is_pinned(model_id) is False


@pytest.mark.parametrize("model_id", CURRENT_ANTHROPIC_IDS)
def test_every_current_anthropic_model_id_is_accepted(model_id: str) -> None:
    # The defect this file exists for: 0.1.1 refused all of these, because the
    # rule required a trailing date and Anthropic stopped shipping one.
    assert is_pinned(model_id) is True
    assert require_pinned(model_id) == model_id


def test_a_provider_alias_and_an_immutable_id_of_the_same_shape_are_separated() -> None:
    # These two strings are structurally identical -- family, dash, number. Only
    # provider policy distinguishes them, so this asserts the vendor table is
    # doing its job rather than the shape rule accidentally getting it right.
    assert is_pinned("claude-opus-5") is True
    assert is_pinned("gpt-5") is False


def test_a_retired_but_dated_id_is_still_pinned() -> None:
    # Pinned means immutable, not available. Retiring a model does not make the
    # scores recorded against it uncomparable to each other.
    assert is_pinned("claude-3-7-sonnet-20250219") is True


@pytest.mark.parametrize("model_id", ALIAS_IDS)
def test_an_alias_token_disqualifies_an_otherwise_pinned_id(model_id: str) -> None:
    assert is_pinned(model_id) is False


@pytest.mark.parametrize("token", ALIAS_TOKENS)
def test_alias_tokens_are_matched_regardless_of_case(token: str) -> None:
    assert is_pinned(f"claude-{token.upper()}-20250514") is False
    assert is_pinned(f"claude-{token.capitalize()}-20250514") is False


@pytest.mark.parametrize("model_id", PINNED_IDS)
def test_surrounding_whitespace_does_not_change_pinnedness(model_id: str) -> None:
    assert is_pinned(f"  {model_id}\n") is True


@pytest.mark.parametrize("value", [None, 42, 3.5, b"claude-opus-4-20250514", ["x"], object()])
def test_non_string_input_is_rejected_without_raising(value: object) -> None:
    assert is_pinned(value) is False  # type: ignore[arg-type]


@pytest.mark.parametrize("model_id", PINNED_IDS)
def test_require_pinned_returns_the_stripped_id(model_id: str) -> None:
    assert require_pinned(f"  {model_id}  ") == model_id


@pytest.mark.parametrize("model_id", UNPINNED_IDS + ALIAS_IDS)
def test_require_pinned_raises_for_every_unpinned_id(model_id: str) -> None:
    with pytest.raises(ModelPinError):
        require_pinned(model_id)


def test_rejection_message_explains_the_rule_and_the_cost_of_breaking_it() -> None:
    # The message is the entire user experience of this error, so it has to say
    # what was refused, what is required instead, and why the rule exists.
    with pytest.raises(ModelPinError) as excinfo:
        require_pinned("claude-3-5-sonnet-latest")

    message = str(excinfo.value)
    assert "claude-3-5-sonnet-latest" in message
    assert "version" in message
    assert "alias" in message
    assert all(token in message for token in ALIAS_TOKENS)
    assert all(example in message for example in PINNED_EXAMPLES)


def test_every_id_the_rejection_message_recommends_is_itself_pinned() -> None:
    # The 0.1.1 message told the reader to add a date suffix. For a current
    # Anthropic id that instruction produces a model that does not exist, so the
    # advice has to be checked against the predicate rather than proof-read.
    assert PINNED_EXAMPLES, "the message must offer at least one worked example"
    for example in PINNED_EXAMPLES:
        assert is_pinned(example) is True, example


def test_the_rejection_message_offers_more_than_the_dated_spelling() -> None:
    # A reader holding 'claude-opus-5' must be shown a form that resembles what
    # they hold. An example whose last component is a short number is that form;
    # a date stamp is eight digits, so requiring a shorter one excludes it.
    tails = [example.rsplit("-", 1)[-1] for example in PINNED_EXAMPLES]
    assert any(tail.isdigit() and len(tail) < 8 for tail in tails), PINNED_EXAMPLES


def test_the_rejection_message_names_the_specific_reason() -> None:
    # A reader who typed one wrong thing should be told which thing it was.
    with pytest.raises(ModelPinError) as alias_case:
        require_pinned("claude-sonnet-latest")
    assert "latest" in str(alias_case.value)

    with pytest.raises(ModelPinError) as shape_case:
        require_pinned("gpt-4o-mini")
    assert "mini" in str(shape_case.value)

    with pytest.raises(ModelPinError) as vendor_case:
        require_pinned("gpt-5")
    assert "dated" in str(vendor_case.value) or "date" in str(vendor_case.value)


def test_rejection_message_names_the_calling_context() -> None:
    with pytest.raises(ModelPinError, match="sampler refuses unpinned model id"):
        require_pinned("gpt-4o", context="sampler")

    with pytest.raises(ModelPinError, match="judge refuses unpinned model id"):
        require_pinned("gpt-4o")


@pytest.mark.parametrize("value", [None, 42, ["x"]])
def test_require_pinned_rejects_non_string_input(value: object) -> None:
    with pytest.raises(ModelPinError):
        require_pinned(value)  # type: ignore[arg-type]
