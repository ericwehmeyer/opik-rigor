"""Provider adapters.

Importing this package must never require a provider SDK. The adapters import
``anthropic`` and ``openai`` lazily inside the call path, so a suite that runs
entirely on :class:`FakeAdapter` -- which is most of them -- installs nothing.

No adapter accepts a credential as a constructor argument. Keys come from the
environment, so a key cannot be committed in a fixture or surface in a traceback.
"""

from __future__ import annotations

from .anthropic import AnthropicAdapter
from .base import (
    ENV_ANTHROPIC_API_KEY,
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_BASE_URL,
    Adapter,
    AdapterError,
)
from .fake import FakeAdapter
from .openai_compat import OpenAICompatAdapter

__all__ = [
    "ENV_ANTHROPIC_API_KEY",
    "ENV_OPENAI_API_KEY",
    "ENV_OPENAI_BASE_URL",
    "Adapter",
    "AdapterError",
    "AnthropicAdapter",
    "FakeAdapter",
    "OpenAICompatAdapter",
]
