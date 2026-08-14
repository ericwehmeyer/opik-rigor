"""Adapter for any endpoint that speaks the OpenAI chat-completions protocol.

Named ``openai_compat`` rather than ``openai`` because the protocol has outlived
the vendor: Azure AI Foundry, vLLM, Ollama, Together and most gateways serve the
same ``/chat/completions`` shape. Point ``base_url`` at one of those and this
adapter judges against it unchanged -- which is the difference between a library
you can run inside a corporate network and one you cannot.

As with the Anthropic adapter there is no default ``model_id``: a default would
re-point under you, and a score recorded against a moving target is not evidence.

Sampling parameters
-------------------

``temperature`` still defaults to ``0.0`` here, and ``temperature=None`` omits the
key from the request. The asymmetry with :mod:`opik_rigor.adapters.anthropic`,
where the default *had* to become "omit", is deliberate and worth stating.

Anthropic's default moved because the old one is a guaranteed 400 on every model
Anthropic currently serves -- there was no configuration a caller could supply
that worked. Nothing forces the same move here: ``0.0`` is still accepted by the
endpoints this adapter is aimed at, and dropping it by default would quietly
cost determinism everywhere it does work, in exchange for nothing.

What this adapter did lack was the *escape hatch*: some newer reasoning-style
deployments accept only their own default sampling values, and there was no way
to construct this adapter without sending one. ``None`` is that hatch. There is
deliberately **no table of which deployments those are**, for two reasons: this
project has not verified any such list against the vendor's documentation (and
does not publish vendor facts it has not checked -- see ``COMPATIBILITY.md``),
and ``model_id`` here is whatever name a gateway, vLLM server or Azure deployment
happens to use, so a table keyed on it would be guessing twice.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from .base import (
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_BASE_URL,
    AdapterError,
    reject_credential_kwargs,
    require_env_key,
)

#: The provider SDK, imported lazily -- see :meth:`OpenAICompatAdapter._sdk_client`.
PACKAGE = "openai"


class OpenAICompatAdapter:
    """Single-turn completions from a pinned OpenAI-protocol model.

    Args:
        model_id: Exact model version, e.g. ``gpt-4o-2024-08-06``, or whatever
            deployment name the compatible endpoint serves. The undated ids
            (``gpt-4o``, ``gpt-4.1``, ``gpt-5``) are refused: OpenAI re-points
            them at its newest snapshot. See :mod:`opik_rigor.pinning`.
        base_url: The endpoint root, e.g. an Azure AI Foundry deployment URL or a
            self-hosted vLLM server. Falls back to ``OPENAI_BASE_URL`` and then to
            the SDK's own default, so the same code runs against a gateway in CI
            and against api.openai.com on a laptop with nothing but an env var
            changing.
        max_tokens: Cap on the response length.
        temperature: Defaults to ``0.0`` -- a judge that is asked the same
            question twice should answer it the same way. Pass ``None`` to omit
            the parameter from the request entirely, which is what a deployment
            that accepts only its own default sampling values needs. See the
            module docstring for why this default did not move when the
            Anthropic adapter's did.
        timeout: Per-request timeout in seconds, handed to the SDK.

    Raises:
        TypeError: If a credential keyword is passed.
        AdapterError: If ``OPENAI_API_KEY`` is unset or empty.
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        timeout: float = 60.0,
        **forbidden: object,
    ) -> None:
        reject_credential_kwargs(forbidden, type(self).__name__)

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"model_id must be a non-empty string, got {model_id!r}")
        if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
            raise ValueError(f"base_url must be a non-empty string or None, got {base_url!r}")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
        if temperature is not None:
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
                raise TypeError(
                    f"temperature must be a float or None, got {temperature!r}. "
                    f"None means the parameter is omitted from the request."
                )
            if not 0.0 <= float(temperature) <= 2.0:
                raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature!r}")
        if float(timeout) <= 0:
            raise ValueError(f"timeout must be > 0 seconds, got {timeout!r}")

        self._model_id = model_id.strip()
        resolved = base_url if base_url is not None else os.environ.get(ENV_OPENAI_BASE_URL, "")
        self._base_url = resolved.strip() or None
        self._max_tokens = max_tokens
        self._temperature = None if temperature is None else float(temperature)
        self._timeout = float(timeout)
        # Private, and deliberately absent from __repr__ and from every message
        # this module raises. See _redact.
        self._api_key = require_env_key(ENV_OPENAI_API_KEY, type(self).__name__)
        self._client: Any | None = None
        self._client_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def base_url(self) -> str | None:
        """The resolved endpoint root, or ``None`` to use the SDK's default."""
        return self._base_url

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def temperature(self) -> float | None:
        """The value that will be sent, or ``None`` if no such key is sent."""
        return self._temperature

    @property
    def timeout(self) -> float:
        return self._timeout

    def complete(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        client = self._sdk_client()
        # Built as a dict rather than passed as keywords so that "omitted" means
        # the key is absent -- ``temperature=None`` on the wire is a sent
        # parameter, not an unsent one.
        request: dict[str, Any] = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:  # provider SDKs raise their own hierarchy
            raise AdapterError(
                f"openai-compatible call failed for model {self._model_id!r} at "
                f"{self._base_url or 'the SDK default endpoint'}: "
                f"{type(exc).__name__}: {self._redact(exc)}"
            ) from exc
        return self._extract_text(response)

    def _sdk_client(self) -> Any:
        """Import the SDK and build a client on first use.

        The import is lazy because ``import opik_rigor.adapters`` must succeed on a
        machine that has never installed a provider SDK -- the fake adapter runs
        the entire test suite, and requiring ``openai`` to collect those tests
        would make the dependency mandatory for people who never call a provider.

        The client is cached because a sampler makes hundreds of calls from a
        thread pool, and a fresh client per call would throw away the connection
        pool each time.
        """
        with self._client_lock:
            if self._client is None:
                try:
                    import openai
                except ImportError as exc:
                    raise AdapterError(
                        f"{type(self).__name__} needs the {PACKAGE!r} package, which is not "
                        f"installed. Install it with: pip install {PACKAGE}"
                    ) from exc
                kwargs: dict[str, Any] = {"api_key": self._api_key, "timeout": self._timeout}
                if self._base_url is not None:
                    kwargs["base_url"] = self._base_url
                try:
                    self._client = openai.OpenAI(**kwargs)
                except Exception as exc:
                    raise AdapterError(
                        f"could not construct the openai client: "
                        f"{type(exc).__name__}: {self._redact(exc)}"
                    ) from exc
            return self._client

    def _extract_text(self, response: Any) -> str:
        """Pull the first choice's message text out of the response."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise AdapterError(
                f"openai-compatible response for {self._model_id!r} carried no choices"
            )
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content:
            finish = getattr(choices[0], "finish_reason", None)
            raise AdapterError(
                f"openai-compatible response for {self._model_id!r} contained no message text "
                f"(finish_reason={finish!r}). A refusal, a truncation or a tool-only reply is "
                f"missing data, not an empty answer."
            )
        return content

    def _redact(self, exc: BaseException) -> str:
        """Strip the key out of provider error text.

        An auth failure that echoes the offending header would otherwise put the
        key into a traceback and, from there, into CI logs.
        """
        return str(exc).replace(self._api_key, "***")

    def __repr__(self) -> str:
        return (
            f"OpenAICompatAdapter(model_id={self._model_id!r}, base_url={self._base_url!r}, "
            f"max_tokens={self._max_tokens}, temperature={self._temperature}, "
            f"timeout={self._timeout})"
        )
