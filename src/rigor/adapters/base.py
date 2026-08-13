"""The provider seam.

One method, one property. Everything rigor needs from a model provider is "give
me a string back for this string, and tell me exactly which model produced it."

Adapters never accept credentials as constructor arguments -- keys come from the
environment only, so a key cannot end up in a traceback, a test fixture, or a
committed config file. Constructors raise :class:`TypeError` if you try.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

#: Environment variables consulted by the shipped adapters.
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"


@runtime_checkable
class Adapter(Protocol):
    """A provider that can answer a single-turn prompt."""

    @property
    def model_id(self) -> str:
        """The exact, pinned model identifier that will serve ``complete``.

        This is the string a judge pins against, so it must name a concrete
        immutable version -- not an alias that the provider re-points over time.
        """
        ...

    def complete(self, prompt: str) -> str:
        """Return the model's response to ``prompt``."""
        ...


class AdapterError(Exception):
    """Raised when a provider call fails in a way the caller should see."""
