"""What the adapters put in the request payload for sampling parameters.

These tests assert on **the dict handed to the provider client**, never on what a
server did with it. That is deliberate twice over: the suite is offline by
construction, and pinning the payload is the honest test anyway -- it records what
this library sends, which is the thing this library controls.

The expected values below were derived from the documented vendor surface
*before* the adapter was changed, and are written down here so that the author of
the change did not get to invent them. Every row is transcribed from Anthropic's
own current-API reference (the ``claude-api`` skill bundled with this session:
its *Thinking & Effort* per-model table, ``shared/error-codes.md`` §"Model-specific
400s", and ``shared/model-migration.md`` §"Migrating to Opus 4.7" / "Opus 4.8" /
"Claude Opus 5" / "Claude Sonnet 5" / "Claude Fable 5"), read on **2026-08-14**.

Derived payload table
---------------------

`temperature`, `top_p` and `top_k` were **removed** from the Messages API on the
current generation. The column that matters is what the adapter must put on the
wire.

==============================  ===============================  =======================
model id                        documented Messages-API          required payload
                                behaviour
==============================  ===============================  =======================
``claude-opus-5``               removed -- any of the three      no ``temperature`` key
                                returns 400
``claude-opus-4-8``             removed -- 400                   no ``temperature`` key
``claude-opus-4-7``             removed -- 400                   no ``temperature`` key
``claude-fable-5``              removed -- 400                   no ``temperature`` key
``claude-mythos-5``             same surface as Fable 5          no ``temperature`` key
``claude-sonnet-5``             a **non-default** value 400s;    no ``temperature`` key
                                omitting it is accepted
``claude-opus-4-6``             sampling params allowed          ``temperature`` iff asked
``claude-sonnet-4-6``           allowed                          ``temperature`` iff asked
``claude-opus-4-5``             allowed                          ``temperature`` iff asked
``claude-sonnet-4-5-20250929``  allowed                          ``temperature`` iff asked
``claude-haiku-4-5-20251001``   allowed                          ``temperature`` iff asked
==============================  ===============================  =======================

Two consequences of that table drive every assertion below.

1. **The default must send no ``temperature`` key at all**, for every model.
   Not "0.0 for old models, absent for new ones": a model-conditional default
   would put the vendor table on the happy path, so the day Anthropic ships a
   model the table has not heard of, the zero-configuration constructor would
   start 400ing again. An unconditional omit is correct on every row above and
   cannot rot.
2. **An explicitly-passed value is still sent**, because rows 7-11 still accept
   one and a caller who names a number meant it.

The one place the table is consulted is the impossible combination -- an explicit
``temperature`` *and* a model documented to refuse it. That is refused at
construction rather than sent, because the alternative is a 400 several hundred
calls into a sampling run. A stale table there costs the caller the vendor's own
400 instead of ours, which is exactly the behaviour that shipped before this
change; it can never break a call that works.

One disagreement inside the source, recorded rather than resolved: the skill's
per-model *Thinking & Effort* table lists Sonnet 5's sampling column as flatly
"Removed -- 400", while its Sonnet 5 migration section says only a *non-default*
value is refused. Both readings agree that omitting the parameter is accepted and
that this adapter's old ``0.0`` was not, which is all these tests turn on.
"""

from __future__ import annotations

from typing import Any

import pytest

from opik_rigor.adapters import (
    ENV_ANTHROPIC_API_KEY,
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_BASE_URL,
    AnthropicAdapter,
    OpenAICompatAdapter,
)
from opik_rigor.pinning import is_pinned

SENTINEL_KEY = "sk-sentinel-DO-NOT-LEAK-0123456789"

#: Column 1 of the table above: ids whose payload must carry no ``temperature``.
REFUSES_TEMPERATURE = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-sonnet-5",
)

#: The bottom half of the table: ids documented to still accept the parameter.
ACCEPTS_TEMPERATURE = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
)


@pytest.fixture
def anthropic_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(ENV_ANTHROPIC_API_KEY, SENTINEL_KEY)
    return SENTINEL_KEY


@pytest.fixture
def openai_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(ENV_OPENAI_API_KEY, SENTINEL_KEY)
    monkeypatch.delenv(ENV_OPENAI_BASE_URL, raising=False)
    return SENTINEL_KEY


def _recording_anthropic_client(captured: dict[str, Any]) -> Any:
    """A stand-in for ``anthropic.Anthropic`` that records the call kwargs."""

    class Block:
        text = "ok"

    class Message:
        content = [Block()]
        stop_reason = "end_turn"

    class Client:
        class messages:  # noqa: N801 - mimics the SDK's attribute shape
            @staticmethod
            def create(**kwargs: object) -> Message:
                captured.clear()
                captured.update(kwargs)
                return Message()

    return Client()


def _recording_openai_client(captured: dict[str, Any]) -> Any:
    class Message:
        content = "ok"

    class Choice:
        message = Message()
        finish_reason = "stop"

    class Response:
        choices = [Choice()]

    class Client:
        class chat:  # noqa: N801 - mimics the SDK's attribute shape
            class completions:  # noqa: N801 - mimics the SDK's attribute shape
                @staticmethod
                def create(**kwargs: object) -> Response:
                    captured.clear()
                    captured.update(kwargs)
                    return Response()

    return Client()


def _anthropic_payload(
    monkeypatch: pytest.MonkeyPatch, model_id: str, **kwargs: object
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    adapter = AnthropicAdapter(model_id, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(adapter, "_sdk_client", lambda: _recording_anthropic_client(captured))
    adapter.complete("hi")
    return captured


def _openai_payload(
    monkeypatch: pytest.MonkeyPatch, model_id: str, **kwargs: object
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    adapter = OpenAICompatAdapter(model_id, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(adapter, "_sdk_client", lambda: _recording_openai_client(captured))
    adapter.complete("hi")
    return captured


# --------------------------------------------------------------------------- #
# the table, asserted directly: absence of the key, not its value
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", REFUSES_TEMPERATURE + ACCEPTS_TEMPERATURE)
def test_the_default_payload_carries_no_temperature_key_for_any_model(
    model_id: str, anthropic_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Consequence 1 in the module docstring. `not in`, not `is None`: a `None`
    # value on the wire is a sent parameter and would 400 exactly like `0.0`.
    payload = _anthropic_payload(monkeypatch, model_id)

    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "top_k" not in payload
    assert payload["model"] == model_id
    assert payload["max_tokens"] == 1024
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("model_id", ACCEPTS_TEMPERATURE)
def test_an_explicit_temperature_is_still_sent_to_a_model_that_accepts_one(
    model_id: str, anthropic_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Consequence 2. This is the compatibility half: a caller who pinned an older
    # dated model and asked for 0.0 in 0.1.1 gets byte-identical requests.
    payload = _anthropic_payload(monkeypatch, model_id, temperature=0.0)

    assert payload["temperature"] == 0.0

    warmer = _anthropic_payload(monkeypatch, model_id, temperature=0.7)

    assert warmer["temperature"] == 0.7


@pytest.mark.parametrize("model_id", REFUSES_TEMPERATURE)
def test_an_explicit_temperature_on_a_current_model_is_refused_at_construction(
    model_id: str, anthropic_env: str
) -> None:
    # Not silently dropped, and not sent into a 400 three hundred samples into a
    # run: refused where the caller is still looking at the constructor.
    with pytest.raises(ValueError) as excinfo:
        AnthropicAdapter(model_id, temperature=0.0)

    message = str(excinfo.value)
    assert "temperature" in message
    assert model_id in message
    # The caller has to be able to tell what happened and what to do instead.
    assert "omit" in message.lower()


@pytest.mark.parametrize("model_id", REFUSES_TEMPERATURE)
def test_a_current_model_still_constructs_and_calls_with_no_temperature(
    model_id: str, anthropic_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point of the fix: the zero-configuration constructor -- the one
    # the README and every example use -- has to work on a current model.
    adapter = AnthropicAdapter(model_id)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(adapter, "_sdk_client", lambda: _recording_anthropic_client(captured))

    assert adapter.complete("hi") == "ok"
    assert adapter.temperature is None
    assert "temperature" not in captured


# --------------------------------------------------------------------------- #
# the vendor table itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", REFUSES_TEMPERATURE)
def test_the_table_recognises_every_documented_current_model(model_id: str) -> None:
    from opik_rigor.adapters.anthropic import rejects_sampling_parameters

    assert rejects_sampling_parameters(model_id)


@pytest.mark.parametrize("model_id", ACCEPTS_TEMPERATURE)
def test_the_table_leaves_models_that_still_accept_sampling_alone(model_id: str) -> None:
    from opik_rigor.adapters.anthropic import rejects_sampling_parameters

    assert not rejects_sampling_parameters(model_id)


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-opus-5",  # Amazon Bedrock's provider prefix
        "claude-opus-5@20260101",  # Vertex's snapshot separator
        "claude-opus-4-8-20260101",  # a dated snapshot of a current model
        "CLAUDE-OPUS-5",  # case is not a spelling of a different model
        "  claude-sonnet-5  ",  # the constructor strips before it decides
    ],
)
def test_the_table_matches_the_spellings_a_gateway_actually_serves(
    model_id: str, anthropic_env: str
) -> None:
    with pytest.raises(ValueError, match="temperature"):
        AnthropicAdapter(model_id, temperature=0.0)


@pytest.mark.parametrize(
    "model_id",
    [
        "my-claude-opus-5-finetune-v1",  # not Anthropic's model, just named after it
        "claude-opus-5x-v1",  # a different family that starts the same way
        "claude-opus-4-5@20251101",  # the nearest older neighbour
    ],
)
def test_the_table_does_not_capture_ids_that_merely_look_similar(
    model_id: str, anthropic_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _anthropic_payload(monkeypatch, model_id, temperature=0.25)

    assert payload["temperature"] == 0.25


def test_every_id_in_the_table_is_a_pinned_id() -> None:
    # If pinning refused one of these, the adapter could never be reached with it
    # and this whole file would be testing an unreachable path -- which is how
    # item 16 and item 19 managed to hide behind each other.
    for model_id in REFUSES_TEMPERATURE + ACCEPTS_TEMPERATURE:
        assert is_pinned(model_id), model_id


# --------------------------------------------------------------------------- #
# the surface a consumer reads
# --------------------------------------------------------------------------- #


def test_the_temperature_property_reports_omission_as_none(anthropic_env: str) -> None:
    assert AnthropicAdapter("claude-opus-5").temperature is None
    assert AnthropicAdapter("claude-opus-4-6", temperature=0.0).temperature == 0.0


def test_the_repr_says_which_of_the_two_happened(anthropic_env: str) -> None:
    assert "temperature=None" in repr(AnthropicAdapter("claude-opus-5"))
    assert "temperature=0.0" in repr(AnthropicAdapter("claude-opus-4-6", temperature=0.0))


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.0", True])
def test_an_explicitly_passed_temperature_is_still_range_checked(
    bad: object, anthropic_env: str
) -> None:
    with pytest.raises((ValueError, TypeError), match="temperature"):
        AnthropicAdapter("claude-opus-4-6", temperature=bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# the OpenAI-compatible adapter
# --------------------------------------------------------------------------- #


def test_openai_compat_still_sends_zero_by_default(
    openai_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unchanged on purpose. Anthropic's default had to move because the old one
    # is a guaranteed 400 on every model Anthropic currently serves; nothing
    # forces the same move here, and moving it would quietly cost determinism on
    # every endpoint where 0.0 still works.
    payload = _openai_payload(monkeypatch, "gpt-4o-2024-08-06")

    assert payload["temperature"] == 0.0


def test_openai_compat_accepts_none_as_omit(
    openai_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The escape hatch this adapter did not have: newer reasoning-style
    # deployments accept only their own default. Which ids those are is a vendor
    # fact this project has not verified, so there is no table here -- only the
    # ability for a caller who knows to omit the parameter.
    payload = _openai_payload(monkeypatch, "gpt-4o-2024-08-06", temperature=None)

    assert "temperature" not in payload
    assert payload["model"] == "gpt-4o-2024-08-06"
    assert payload["max_tokens"] == 1024


def test_openai_compat_temperature_property_reports_omission(openai_env: str) -> None:
    assert OpenAICompatAdapter("gpt-4o-2024-08-06", temperature=None).temperature is None
    assert "temperature=None" in repr(
        OpenAICompatAdapter("gpt-4o-2024-08-06", temperature=None)
    )
