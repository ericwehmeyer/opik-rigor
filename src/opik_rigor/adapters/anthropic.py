"""Adapter for the Anthropic Messages API.

Thin on purpose: validate the configuration, read the key from the environment,
make one call, hand back one string. Anything cleverer -- retries, caching,
streaming -- would sit between the model and the evidence log, and the whole
point of this library is that what got recorded is what the model said.

There is no default ``model_id``. A default would be an alias in disguise: the
library would silently start judging with a different set of weights the day it
was bumped, and every score recorded before the bump would quietly stop being
comparable. The caller names the exact version.

Sampling parameters
-------------------

``temperature`` defaults to ``None``, which means **the key is absent from the
request**, not that some value is chosen for you. That default was ``0.0`` until
0.1.2, and it made this adapter unusable: ``temperature``, ``top_p`` and ``top_k``
were removed from the Messages API on the current generation of models, and
sending any of them returns a 400. A judge built the way the README builds one
could not complete a single call.

Omitting unconditionally, rather than omitting only for the models known to
refuse the parameter, is the deliberate part. A model-conditional default would
put :data:`_SAMPLING_REMOVED` on the happy path, so the day Anthropic ships a
model that table has not heard of, the zero-configuration constructor would start
returning 400s again -- the same defect, waiting on a release date. Omitting
always is correct on every model Anthropic serves, current or older, and cannot
rot.

What it costs: an older model that still accepts sampling parameters now gets the
API default rather than ``0.0`` unless you ask. If you are pinned to such a model
and want the pre-0.1.2 request, pass ``temperature=0.0`` explicitly -- it is sent
unchanged. On a current model that lever no longer exists at all, and passing a
value is refused at construction rather than turned into a 400 mid-run.
"""

from __future__ import annotations

import re
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

# ---------------------------------------------------------------------------
# The vendor-specific half, deliberately kept to one table -- the same shape,
# and for the same reason, as ``_MOVING_FAMILIES`` in :mod:`opik_rigor.pinning`.
#
# THIS TABLE WILL NEED UPDATING. It records the Anthropic model families whose
# Messages API no longer accepts ``temperature`` / ``top_p`` / ``top_k``. Nothing
# in a model id says whether its provider still takes a sampling parameter; that
# is a documented policy, and the only honest way to hold it is one table that
# says out loud it needs maintaining.
#
# Read off Anthropic's current API reference on 2026-08-14 -- see
# ``COMPATIBILITY.md`` §"Anthropic Messages API" for the sources and the exact
# per-model wording. Opus 5, Opus 4.8, Opus 4.7, Fable 5 and Mythos 5 return a
# 400 for any of the three. Sonnet 5 is narrower in the vendor's own text -- a
# *non-default* value returns 400 -- which comes to the same thing here, because
# the only reason to pass the parameter is to set a value that is not the
# default. Opus 4.6, Sonnet 4.6 and everything older still accept them.
#
# **The gap this leaves, stated rather than discovered by a consumer:** a model
# released after this line was written is not in the table. That costs exactly
# one thing -- an explicit ``temperature`` against it produces the vendor's 400
# instead of our ValueError, which is the behaviour that shipped before this
# table existed. It can never break a call that works, because the table is not
# consulted on the default path.
#
# The ``(?:\A|[/.])`` prefix admits a gateway's routing prefix
# (``anthropic.claude-opus-5`` on Bedrock) without matching a finetune of your
# own called ``my-claude-opus-5-tuned``. The trailing lookahead keeps
# ``claude-opus-5`` from swallowing an unrelated ``claude-opus-5x``.
# ---------------------------------------------------------------------------
_SAMPLING_REMOVED = re.compile(
    r"(?:\A|[/.])claude-(?:opus-5|opus-4-8|opus-4-7|sonnet-5|fable-5|mythos-5)(?=[-_@:]|\Z)"
)


def rejects_sampling_parameters(model_id: str) -> bool:
    """True if ``model_id`` names a model documented to refuse ``temperature``.

    Consulted in exactly one place -- the constructor, when a caller passes a
    ``temperature`` explicitly -- so that the impossible combination fails where
    the caller is still looking at it rather than several hundred calls into a
    sampling run. See :data:`_SAMPLING_REMOVED` for what the table covers and for
    the gap it leaves.
    """
    if not isinstance(model_id, str):
        return False
    return _SAMPLING_REMOVED.search(model_id.strip().lower()) is not None


class AnthropicAdapter:
    """Single-turn completions from a pinned Anthropic model.

    Args:
        model_id: Exact model version, e.g. ``claude-opus-5`` or the dated
            spelling ``claude-sonnet-4-5-20250929``. Anthropic's current ids
            carry no date suffix; both forms are pinned. See
            :mod:`opik_rigor.pinning`.
        max_tokens: Cap on the response length.
        temperature: ``None`` (the default) omits the parameter from the request
            entirely -- which is the only thing the current generation of models
            accepts. Pass a float to send one; it reaches the API unchanged, and
            is refused at construction if ``model_id`` names a model documented
            not to accept it. See the module docstring.
        timeout: Per-request timeout in seconds, handed to the SDK.

    Raises:
        TypeError: If a credential keyword is passed.
        ValueError: If any setting is out of range, or if a ``temperature`` is
            passed for a model whose API has removed it.
        AdapterError: If ``ANTHROPIC_API_KEY`` is unset or empty.
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
        timeout: float = 60.0,
        **forbidden: ForbiddenKwarg,
    ) -> None:
        reject_credential_kwargs(forbidden, type(self).__name__)

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"model_id must be a non-empty string, got {model_id!r}")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
        if float(timeout) <= 0:
            raise ValueError(f"timeout must be > 0 seconds, got {timeout!r}")

        self._model_id = model_id.strip()

        if temperature is not None:
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
                raise TypeError(
                    f"temperature must be a float or None, got {temperature!r}. "
                    f"None means the parameter is omitted from the request."
                )
            if not 0.0 <= float(temperature) <= 1.0:
                raise ValueError(f"temperature must be between 0.0 and 1.0, got {temperature!r}")
            if rejects_sampling_parameters(self._model_id):
                raise ValueError(
                    f"model {self._model_id!r} does not accept a temperature: Anthropic "
                    f"removed temperature, top_p and top_k from the Messages API for this "
                    f"model, and sending one returns HTTP 400. Omit the argument (the "
                    f"default) to leave the parameter out of the request. Sending it and "
                    f"letting the provider refuse would put the failure hundreds of calls "
                    f"into a run; dropping it silently would record a judgement you did "
                    f"not ask for. If you need temperature=0.0 for reproducibility, pin a "
                    f"model whose API still has the parameter."
                )

        self._max_tokens = max_tokens
        self._temperature = None if temperature is None else float(temperature)
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
    def temperature(self) -> float | None:
        """The value that will be sent, or ``None`` if no such key is sent.

        ``None`` is not "zero" and not "the provider's default as far as we
        know" -- it is the statement that this adapter puts no ``temperature``
        in the payload at all.
        """
        return self._temperature

    @property
    def timeout(self) -> float:
        return self._timeout

    def complete(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        client = self._sdk_client()
        # Built as a dict rather than passed as keywords so that "omitted" means
        # the key is absent. ``temperature=None`` on the wire is a sent parameter
        # and 400s exactly like ``0.0`` does.
        request: dict[str, Any] = {
            "model": self._model_id,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        try:
            message = client.messages.create(**request)
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
