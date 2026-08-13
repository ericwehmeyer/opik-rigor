"""A scripted adapter, so the rest of opik_rigor can be tested without a provider.

Every other module in this library -- judges, samplers, the statistical gates --
is exercised against :class:`FakeAdapter`. That makes its behaviour part of the
library's contract rather than a test detail, and it explains the three unusual
choices below.

*Seeded stochasticity.* A statistical gate is only worth testing against a model
that varies. ``seed`` makes the adapter draw from the script at random while
staying byte-identical across runs, so a flaky-looking gate is a real bug and not
a coin flip in CI.

*Thread safety.* The sampler drives an adapter from a thread pool. If the cursor
were unguarded, two threads would hand back the same scripted line and a test
that "passed" would have silently measured 199 samples instead of 200. The lock
is what makes the call log trustworthy as evidence.

*No credentials, ever.* The fake takes the same credential-rejecting constructor
as the real adapters, so a test cannot get into the habit of passing ``api_key=``
and then discover at the provider boundary that the habit was never supported.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence

from .base import AdapterError, reject_credential_kwargs

#: Pinned by construction: a default that failed ``is_pinned`` would make every
#: fixture in the suite a bad example of how to name a model.
DEFAULT_FAKE_MODEL_ID = "fake-scripted-v1"

ResponseSource = Sequence[str] | Mapping[str, str] | Callable[[str], str]


class FakeAdapter:
    """A deterministic, scripted stand-in for a provider.

    ``responses`` may be:

    * a :class:`~collections.abc.Sequence` of strings, handed out in order;
    * a :class:`~collections.abc.Mapping` from prompt to response, so a test can
      script an answer per prompt without depending on call order;
    * a callable taking the prompt and returning the response, for responses that
      have to be computed.

    Args:
        model_id: Reported verbatim; the default is pinned.
        responses: The script (see above).
        cycle: Sequence mode only -- restart at the top instead of raising when
            the script runs out.
        seed: Sequence mode only -- draw each response at random from the script
            using a private :class:`random.Random`, so the run is stochastic but
            reproducible. Private because a global ``random.seed`` in one test
            would otherwise change what another test measures.
        latency: Seconds to sleep per call, for exercising timeout handling.
        fail_with: An exception (instance or class) to start raising.
        fail_after: Number of successful calls before ``fail_with`` kicks in;
            defaults to ``0`` (fail immediately) when ``fail_with`` is given.

    Attributes:
        calls: Every prompt that reached the adapter, in arrival order. A prompt
            that then raised is still recorded -- the log is what was attempted,
            which is what a debugging test needs to see.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_FAKE_MODEL_ID,
        responses: ResponseSource,
        cycle: bool = False,
        seed: int | None = None,
        latency: float = 0.0,
        fail_with: BaseException | type[BaseException] | None = None,
        fail_after: int | None = None,
        **forbidden: object,
    ) -> None:
        reject_credential_kwargs(forbidden, type(self).__name__)

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"model_id must be a non-empty string, got {model_id!r}")
        self._model_id = model_id.strip()

        self._scripted: list[str] | None = None
        self._mapping: dict[str, str] | None = None
        self._callable: Callable[[str], str] | None = None
        self._classify(responses)

        if seed is not None and self._scripted is None:
            raise ValueError("seed is only meaningful when responses is a sequence")
        if cycle and self._scripted is None:
            raise ValueError("cycle is only meaningful when responses is a sequence")
        if latency < 0:
            raise ValueError(f"latency must be >= 0, got {latency!r}")
        if fail_after is not None and (not isinstance(fail_after, int) or fail_after < 0):
            raise ValueError(f"fail_after must be a non-negative int, got {fail_after!r}")
        if fail_after is not None and fail_with is None:
            raise ValueError("fail_after has no effect without fail_with")
        if fail_with is not None and not _is_exception(fail_with):
            raise TypeError(f"fail_with must be an exception or exception class, got {fail_with!r}")

        self._cycle = bool(cycle)
        self._seed = seed
        self._rng = random.Random(seed) if seed is not None else None
        self._latency = float(latency)
        self._fail_with = fail_with
        self._fail_after = 0 if fail_with is not None and fail_after is None else fail_after

        self._lock = threading.Lock()
        self._cursor = 0
        self._successes = 0
        self.calls: list[str] = []

    def _classify(self, responses: ResponseSource) -> None:
        if isinstance(responses, str):
            # A bare string is a Sequence of one-character responses, which is
            # never what the caller meant. Refusing beats scripting the alphabet.
            raise TypeError("responses must be a sequence of strings, not a single string")
        if isinstance(responses, Mapping):
            self._mapping = {str(k): str(v) for k, v in responses.items()}
        elif callable(responses):
            self._callable = responses
        elif isinstance(responses, Sequence):
            self._scripted = [str(item) for item in responses]
        else:
            raise TypeError(
                "responses must be a sequence of strings, a mapping of prompt to "
                f"response, or a callable, got {type(responses).__name__}"
            )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def call_count(self) -> int:
        """How many prompts have reached the adapter."""
        with self._lock:
            return len(self.calls)

    def complete(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")

        with self._lock:
            self.calls.append(prompt)
            failure = self._due_failure()
            producer = self._callable
            scripted: str | None = None
            if failure is None and producer is None:
                scripted = self._next_scripted(prompt)
                self._successes += 1

        if failure is not None:
            raise failure
        text = self._invoke_callable(producer, prompt) if scripted is None else scripted

        if self._latency:
            time.sleep(self._latency)
        return text

    def _due_failure(self) -> BaseException | None:
        """The exception this call should raise, if the fail budget is spent."""
        if self._fail_with is None or self._successes < (self._fail_after or 0):
            return None
        if isinstance(self._fail_with, BaseException):
            return self._fail_with
        return self._fail_with()

    def _next_scripted(self, prompt: str) -> str:
        """Pick the next response. Caller holds the lock."""
        if self._mapping is not None:
            try:
                return self._mapping[prompt]
            except KeyError:
                raise AdapterError(
                    f"FakeAdapter has no scripted response for prompt {prompt!r} "
                    f"({len(self._mapping)} prompt(s) scripted). Script it, or use a "
                    f"sequence or callable if the prompt is not known in advance."
                ) from None

        scripted = self._scripted
        if not scripted:
            raise AdapterError("FakeAdapter was given an empty response script")
        if self._rng is not None:
            return self._rng.choice(scripted)
        if self._cursor >= len(scripted):
            if not self._cycle:
                raise AdapterError(
                    f"FakeAdapter ran out of scripted responses after "
                    f"{len(scripted)} call(s). Script more responses, or pass "
                    f"cycle=True to repeat the script."
                )
            self._cursor = 0
        text = scripted[self._cursor]
        self._cursor += 1
        return text

    def _invoke_callable(self, producer: Callable[[str], str] | None, prompt: str) -> str:
        """Run the user callable outside the lock.

        Holding the lock across foreign code would deadlock the moment a callable
        inspected ``call_count`` -- which is exactly what a callable that decides
        based on how far along the run is would do.
        """
        if producer is None:  # pragma: no cover - complete() routes here only in callable mode
            raise AdapterError("FakeAdapter has no response callable")
        text = producer(prompt)
        if not isinstance(text, str):
            raise AdapterError(
                f"FakeAdapter response callable returned {type(text).__name__}, expected str"
            )
        with self._lock:
            self._successes += 1
        return text

    def __repr__(self) -> str:
        if self._mapping is not None:
            script = f"mapping of {len(self._mapping)}"
        elif self._callable is not None:
            script = "callable"
        else:
            script = f"sequence of {len(self._scripted or ())}"
        return (
            f"FakeAdapter(model_id={self._model_id!r}, responses={script}, "
            f"seed={self._seed!r}, calls={self.call_count})"
        )


def _is_exception(value: object) -> bool:
    return isinstance(value, BaseException) or (
        isinstance(value, type) and issubclass(value, BaseException)
    )
