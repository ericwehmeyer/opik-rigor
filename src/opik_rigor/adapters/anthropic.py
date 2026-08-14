"""Adapter for the Anthropic Messages API.

Thin on purpose: validate the configuration, read the key from the environment,
make one call, hand back one string. Anything cleverer -- retries, caching,
streaming -- would sit between the model and the evidence log, and the whole
point of this library is that what got recorded is what the model said.

There is no default ``model_id``. A default would be an alias in disguise: the
library would silently start judging with a different set of weights the day it
was bumped, and every score recorded before the bump would quietly stop being
comparable. The caller names the exact version.
"""

from __future__ import annotations

import threading
from typing import Any

from .base import (
    ENV_ANTHROPIC_API_KEY,
    AdapterError,
    ForbiddenKwarg,
    reject_credential_kwargs,
    require_env_key,
)

#: The provider SDK, imported lazily -- see :meth:`AnthropicAdapter._sdk_client`.
PACKAGE = "anthropic"


class AnthropicAdapter:
    """Single-turn completions from a pinned Anthropic model.

    Args:
        model_id: Exact model version, e.g. ``claude-sonnet-4-5-20250929``.
        max_tokens: Cap on the response length.
        temperature: Defaults to ``0.0`` -- a judge that is asked the same
            question twice should answer it the same way.
        timeout: Per-request timeout in seconds, handed to the SDK.

    Raises:
        TypeError: If a credential keyword is passed.
        AdapterError: If ``ANTHROPIC_API_KEY`` is unset or empty.
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 60.0,
        **forbidden: ForbiddenKwarg,
    ) -> None:
        reject_credential_kwargs(forbidden, type(self).__name__)

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"model_id must be a non-empty string, got {model_id!r}")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
        if not 0.0 <= float(temperature) <= 1.0:
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {temperature!r}")
        if float(timeout) <= 0:
            raise ValueError(f"timeout must be > 0 seconds, got {timeout!r}")

        self._model_id = model_id.strip()
        self._max_tokens = max_tokens
        self._temperature = float(temperature)
        self._timeout = float(timeout)
        # Private, and deliberately absent from __repr__ and from every message
        # this module raises. See _redact.
        self._api_key = require_env_key(ENV_ANTHROPIC_API_KEY, type(self).__name__)
        self._client: Any | None = None
        self._client_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def timeout(self) -> float:
        return self._timeout

    def complete(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        client = self._sdk_client()
        try:
            message = client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # provider SDKs raise their own hierarchy
            raise AdapterError(
                f"anthropic call failed for model {self._model_id!r}: "
                f"{type(exc).__name__}: {self._redact(exc)}"
            ) from exc
        return self._extract_text(message)

    def _sdk_client(self) -> Any:
        """Import the SDK and build a client on first use.

        The import is lazy because ``import opik_rigor.adapters`` must succeed on a
        machine that has never installed a provider SDK -- the fake adapter runs
        the entire test suite, and requiring ``anthropic`` to collect those tests
        would make the dependency mandatory for people who never call a provider.

        The client is cached because a sampler makes hundreds of calls from a
        thread pool, and a fresh client per call would throw away the connection
        pool each time.
        """
        with self._client_lock:
            if self._client is None:
                try:
                    import anthropic
                except ImportError as exc:
                    raise AdapterError(
                        f"{type(self).__name__} needs the {PACKAGE!r} package, which is not "
                        f"installed. Install it with: pip install {PACKAGE}"
                    ) from exc
                try:
                    self._client = anthropic.Anthropic(
                        api_key=self._api_key, timeout=self._timeout
                    )
                except Exception as exc:
                    raise AdapterError(
                        f"could not construct the anthropic client: "
                        f"{type(exc).__name__}: {self._redact(exc)}"
                    ) from exc
            return self._client

    def _extract_text(self, message: Any) -> str:
        """Flatten the response's text blocks into one string."""
        blocks = getattr(message, "content", None)
        if blocks is None:
            raise AdapterError(
                f"anthropic response for {self._model_id!r} carried no content field"
            )
        parts = [
            text
            for block in blocks
            if isinstance(text := getattr(block, "text", None), str) and text
        ]
        if not parts:
            raise AdapterError(
                f"anthropic response for {self._model_id!r} contained no text blocks "
                f"(stop_reason={getattr(message, 'stop_reason', None)!r}). A truncated or "
                f"tool-only response is missing data, not an empty answer."
            )
        return "".join(parts)

    def _redact(self, exc: BaseException) -> str:
        """Strip the key out of provider error text.

        An auth failure that echoes the offending header would otherwise put the
        key into a traceback and, from there, into CI logs.
        """
        return str(exc).replace(self._api_key, "***")

    def __repr__(self) -> str:
        return (
            f"AnthropicAdapter(model_id={self._model_id!r}, max_tokens={self._max_tokens}, "
            f"temperature={self._temperature}, timeout={self._timeout})"
        )
