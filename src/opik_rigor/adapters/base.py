"""The provider seam.

One method, one property. Everything opik_rigor needs from a model provider is "give
me a string back for this string, and tell me exactly which model produced it."

Adapters never accept credentials as constructor arguments -- keys come from the
environment only, so a key cannot end up in a traceback, a test fixture, or a
committed config file. Constructors raise :class:`TypeError` if you try.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
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


#: Constructor keywords that would smuggle a secret into a call site.
CREDENTIAL_KWARGS = ("api_key", "key", "token", "secret")


def reject_credential_kwargs(kwargs: Mapping[str, object], class_name: str) -> None:
    """Turn a credential keyword into a loud ``TypeError``.

    Silently ignoring ``api_key=`` would leave the caller believing the key was
    used, and leave the key sitting in their source. Naming the offending
    keyword -- never its value -- tells them what to do instead.
    """
    if not kwargs:
        return
    offenders = sorted(name for name in kwargs if name in CREDENTIAL_KWARGS)
    if offenders:
        raise TypeError(
            f"{class_name} does not accept credentials as arguments "
            f"(got {', '.join(offenders)}). Keys are read from the environment only, "
            f"so that a key cannot end up in a traceback, a test fixture, or a "
            f"committed config file."
        )
    unexpected = ", ".join(sorted(kwargs))
    raise TypeError(f"{class_name} got unexpected keyword argument(s): {unexpected}")


def require_env_key(variable: str, class_name: str) -> str:
    """Read a credential from the environment or explain what is missing.

    Reading at construction rather than at call time means a misconfigured suite
    fails while you are still looking at it, not three hundred samples into a
    run. The value is returned and stored privately -- never logged, never put
    into a message, never shown by ``repr``.
    """
    value = os.environ.get(variable, "").strip()
    if not value:
        raise AdapterError(
            f"{class_name} needs the {variable} environment variable. "
            f"Credentials are read from the environment only -- they are never "
            f"accepted as constructor arguments -- so export {variable} before "
            f"constructing the adapter."
        )
    return value
