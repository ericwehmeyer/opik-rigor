"""Model-id pinning rules.

A judge whose model id is an alias is not reproducible. ``claude-sonnet-latest``
names a different set of weights this month than it did last month, so a score
recorded against it cannot be compared to a score recorded before the provider
re-pointed the alias -- and nothing in the log tells you the swap happened.

This module is the single definition of "pinned" so that adapters, judges, and
tests cannot drift apart on what the word means.

Accepted (the id ends in a concrete version marker)::

    claude-opus-4-20250514        # YYYYMMDD date stamp
    gpt-4o-2024-08-06             # YYYY-MM-DD date stamp
    fake-scripted-v1              # explicit vN
    my-finetune-2.1.0             # explicit dotted version

Rejected::

    ""                            # empty
    claude                        # bare family name
    gpt-4o                        # family + tier, no version
    claude-3-5-sonnet-latest      # alias token
"""

from __future__ import annotations

import re

from .errors import ModelPinError

#: Tokens that mean "whatever the provider is currently serving".
ALIAS_TOKENS = ("latest", "newest", "current", "stable", "default")

#: A concrete version marker anchored at the end of the id.
_VERSION_SUFFIX = re.compile(
    r"(?:"
    r"\d{8}"  # 20250514
    r"|\d{4}-\d{2}-\d{2}"  # 2024-08-06
    r"|v\d+(?:[.\-]\d+)*"  # v1, v2.1, v2-1
    r"|\d+\.\d+(?:\.\d+)*"  # 2.1, 2.1.0
    r")$"
)


def is_pinned(model_id: str) -> bool:
    """True if ``model_id`` names one immutable model version."""
    if not isinstance(model_id, str):
        return False
    candidate = model_id.strip()
    if not candidate:
        return False
    lowered = candidate.lower()
    if any(token in lowered for token in ALIAS_TOKENS):
        return False
    return _VERSION_SUFFIX.search(candidate) is not None


def require_pinned(model_id: str, *, context: str = "judge") -> str:
    """Return ``model_id`` if it is pinned, else raise :class:`ModelPinError`."""
    if is_pinned(model_id):
        return model_id.strip()
    raise ModelPinError(
        f"{context} refuses unpinned model id {model_id!r}. "
        f"It must end in a concrete version marker (a date such as '-20250514' or "
        f"'-2024-08-06', or an explicit version such as '-v1' or '-2.1.0') and must not "
        f"contain an alias token ({', '.join(ALIAS_TOKENS)}). An alias re-points over "
        f"time, which silently invalidates every score recorded against it."
    )
