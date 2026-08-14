"""Model-id pinning rules.

A judge whose model id is an alias is not reproducible. ``claude-sonnet-latest``
names a different set of weights this month than it did last month, so a score
recorded against it cannot be compared to a score recorded before the provider
re-pointed the alias -- and nothing in the log tells you the swap happened.

This module is the single definition of "pinned" so that adapters, judges, and
tests cannot drift apart on what the word means.

The property being protected is **immutability**, not spelling. Until 0.1.1 this
module required a trailing date stamp, on the reasoning that a dated id is an
immutable one. That was a proxy, and the proxy came apart when Anthropic stopped
putting dates on its model ids: ``claude-opus-5`` was refused while
``claude-3-7-sonnet-20250219`` -- retired on 2026-02-19 -- was accepted.

The rule
--------

An id is pinned when all three of these hold.

1. **No alias token.** ``latest``, ``newest``, ``current``, ``stable``,
   ``default``: words that mean "whatever is being served right now". This half
   is provider-independent, because a provider that wants a moving pointer has
   to *name* it, and these are what those names look like.

2. **It ends in a release designator.** Splitting on ``-``, ``_``, ``@`` and
   ``:``, the last component must be nothing but a version: ``5``, ``8``,
   ``20251001``, ``06``, ``v1``, ``2.1.0``. A last component that is a *word*
   (``sonnet``, ``mini``, ``4o``, ``large``, ``preview``, ``instruct``) names a
   *kind* of model, and a kind is precisely the thing a provider re-points when
   new weights ship.

3. **Not a known moving family.** See :data:`_MOVING_FAMILIES` below.

Accepted::

    claude-opus-5                 # release number at the end, no date
    claude-opus-4-8
    claude-haiku-4-5-20251001     # YYYYMMDD date stamp
    claude-opus-4-5@20251101      # Vertex snapshot separator
    gpt-4o-2024-08-06             # YYYY-MM-DD date stamp
    fake-scripted-v1              # explicit vN
    my-finetune-2.1.0             # explicit dotted version

Rejected::

    ""                            # empty
    claude                        # bare family name
    gpt-4o                        # family + tier, no release designator
    gpt-5                         # a moving family (rule 3)
    claude-3-5-sonnet-latest      # alias token

What this rule cannot catch
---------------------------

Whether an id re-points is a **provider policy**; nothing in the string says so.
``claude-opus-5`` and ``gpt-5`` are the same shape and only one of them names one
immutable version, so no vendor-neutral rule can separate them. That irreducible
residue is isolated in :data:`_MOVING_FAMILIES` -- one table, one provider today
-- rather than smeared through the predicate. Three consequences worth stating
before someone rediscovers them:

* **A moving ``<family>-<number>`` pointer from a provider not yet in that table
  is accepted.** This is the failure mode of the fix, and it is the same class of
  failure as the defect it replaces: a rule written against the naming
  conventions that existed the day it was written. The table needs a line when a
  provider adopts the convention; the predicate does not need changing.
* **A self-hosted id whose weights never move is refused if it ends in a word.**
  Suffix it ``-v1``. This direction is the deliberate one: a false refusal is
  loud and takes one edit, a false acceptance silently invalidates every score
  recorded afterwards, which is the failure this whole module exists to prevent.
* **Pinned is not available, and not correct.** ``claude-3-7-sonnet-20250219``
  is retired and still names one immutable version -- which is exactly what a
  score recorded against it in 2025 needs. Nothing here checks that a model
  exists, or that you have access to it.
"""

from __future__ import annotations

import re

from .errors import ModelPinError

#: Tokens that mean "whatever the provider is currently serving".
ALIAS_TOKENS = ("latest", "newest", "current", "stable", "default")

#: Worked examples quoted in the rejection message. Every one of these must
#: satisfy :func:`is_pinned`, and ``tests/test_pinning.py`` asserts that it does
#: -- because the 0.1.1 message told the reader to add a date suffix, which for a
#: current Anthropic id produces a model id that does not exist.
PINNED_EXAMPLES = (
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
    "gpt-4o-2024-08-06",
    "my-finetune-v1",
)

# ---------------------------------------------------------------------------
# The vendor-specific half, deliberately kept to one table.
#
# THIS TABLE WILL NEED UPDATING. It records providers that publish a bare
# ``<family>-<number>`` id as a *moving pointer* to their newest dated snapshot,
# so that rule 2 -- "ends in a release designator" -- is not sufficient for them.
# There is no way to derive membership from the string: ``gpt-5`` and
# ``claude-opus-5`` are the same shape, and the difference is a documented
# policy. Add a pattern here when another provider adopts the convention.
#
#   OpenAI: ``gpt-4o``, ``gpt-4.1`` and ``gpt-5`` re-point to the newest dated
#   snapshot (``gpt-4o`` -> ``gpt-4o-2024-08-06``). Pinning one means naming the
#   snapshot. The ``(?:\A|[/.])`` guard keeps this from matching a finetune of
#   your own called something like ``my-gpt-tuned-3``, while still matching a
#   routed id such as ``openai/gpt-4o`` or ``azure.gpt-4o``.
# ---------------------------------------------------------------------------
_MOVING_FAMILIES = (re.compile(r"(?:\A|[/.])gpt[-.]"),)

#: Characters providers use between the parts of a model id. ``@`` is Vertex's
#: snapshot separator (``claude-opus-4-5@20251101``); ``:`` ends a Bedrock ARN
#: style id (``...-20241022-v2:0``).
_SEPARATORS = re.compile(r"[-_@:]")

#: A final component that is nothing but a version: ``5``, ``20251001``, ``v1``,
#: ``2.1.0``. Matched with ``fullmatch``, so ``4o`` and ``mini`` do not qualify.
_RELEASE_DESIGNATOR = re.compile(r"v?\d+(?:\.\d+)*")

#: A date stamp or an explicit ``vN``, anchored at the end -- what a member of
#: :data:`_MOVING_FAMILIES` must additionally carry.
_EXPLICIT_SNAPSHOT = re.compile(r"[-_@:](?:\d{8}|\d{4}-\d{2}-\d{2}|v\d+(?:[.\-]\d+)*)\Z")


def _refusal_reason(model_id: object) -> str | None:
    """Return the reason ``model_id`` is not pinned, or ``None`` if it is.

    One function so that :func:`is_pinned` and :func:`require_pinned` cannot
    disagree, and so that the error message can name the clause that decided
    rather than reciting the whole rule at a caller who broke one part of it.
    """
    if not isinstance(model_id, str):
        return f"It is a {type(model_id).__name__}, not a string."
    candidate = model_id.strip()
    if not candidate:
        return "It is empty."

    lowered = candidate.lower()
    for token in ALIAS_TOKENS:
        if token in lowered:
            return (
                f"It contains the alias token {token!r}, which names whatever the "
                f"provider is serving today rather than one fixed version."
            )

    tail = _SEPARATORS.split(lowered)[-1]
    if _RELEASE_DESIGNATOR.fullmatch(tail) is None:
        return (
            f"It ends in {tail!r}, which names a kind of model rather than one "
            f"release of it, and a kind is what a provider re-points."
        )

    moving_family = any(family.search(lowered) for family in _MOVING_FAMILIES)
    if moving_family and _EXPLICIT_SNAPSHOT.search(lowered) is None:
        return (
            "This provider publishes <family>-<number> ids as moving pointers "
            "to its newest dated snapshot, so a release number is not enough "
            "here -- name the dated snapshot instead."
        )

    return None


def is_pinned(model_id: str) -> bool:
    """True if ``model_id`` names one immutable model version.

    Immutable, not current and not available: a retired id that still names one
    fixed set of weights is pinned, because the scores recorded against it remain
    comparable to each other. See the module docstring for the rule and for what
    it cannot catch.
    """
    return _refusal_reason(model_id) is None


def require_pinned(model_id: str, *, context: str = "judge") -> str:
    """Return ``model_id`` if it is pinned, else raise :class:`ModelPinError`."""
    reason = _refusal_reason(model_id)
    if reason is None:
        return model_id.strip()
    raise ModelPinError(
        f"{context} refuses unpinned model id {model_id!r}. {reason} A pinned id "
        f"names one immutable model version: it must not contain an alias token "
        f"({', '.join(ALIAS_TOKENS)}), and it must end in a release designator -- a "
        f"release number, a date stamp, or an explicit version -- as in "
        f"{', '.join(PINNED_EXAMPLES)}. An alias re-points over time, which "
        f"silently invalidates every score recorded against it."
    )
