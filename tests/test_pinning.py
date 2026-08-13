"""Tests for the model-id pinning rules.

``is_pinned`` is the single definition of "reproducible model id" in opik_rigor, so
these tests fix both halves of it: an id ending in a concrete version marker is
accepted, and anything that can silently re-point under the caller is refused --
including every token in :data:`ALIAS_TOKENS`, iterated rather than hardcoded so
a newly added token cannot slip through untested.
"""

from __future__ import annotations

import pytest

from opik_rigor.errors import ModelPinError
from opik_rigor.pinning import ALIAS_TOKENS, is_pinned, require_pinned

PINNED_IDS = [
    "claude-opus-4-20250514",
    "gpt-4o-2024-08-06",
    "fake-scripted-v1",
    "my-finetune-2.1.0",
    "claude-3-5-haiku-20241022",
    "gpt-4.1-2025-04-14",
    "mistral-large-v2",
    "llama-3.1-8b-instruct-v1-2",
]

UNPINNED_IDS = [
    "",
    "   ",
    "\t\n",
    "claude",
    "gpt-4o",
    "claude-3-5-sonnet-latest",
    "claude-sonnet-4-5",
    "gpt-4o-mini",
    "my-finetune",
    "claude-opus-4-20250514-preview",
]

#: One otherwise-pinned id per alias token: the version marker is present, so
#: only the alias may cause the rejection.
ALIAS_IDS = [f"claude-{token}-20250514" for token in ALIAS_TOKENS]


@pytest.mark.parametrize("model_id", PINNED_IDS)
def test_ids_ending_in_a_concrete_version_marker_are_pinned(model_id: str) -> None:
    assert is_pinned(model_id) is True


@pytest.mark.parametrize("model_id", UNPINNED_IDS)
def test_ids_without_a_concrete_version_marker_are_not_pinned(model_id: str) -> None:
    assert is_pinned(model_id) is False


@pytest.mark.parametrize("model_id", ALIAS_IDS)
def test_an_alias_token_disqualifies_an_otherwise_pinned_id(model_id: str) -> None:
    assert is_pinned(model_id) is False


@pytest.mark.parametrize("token", ALIAS_TOKENS)
def test_alias_tokens_are_matched_regardless_of_case(token: str) -> None:
    assert is_pinned(f"claude-{token.upper()}-20250514") is False
    assert is_pinned(f"claude-{token.capitalize()}-20250514") is False


@pytest.mark.parametrize("model_id", PINNED_IDS)
def test_surrounding_whitespace_does_not_change_pinnedness(model_id: str) -> None:
    assert is_pinned(f"  {model_id}\n") is True


@pytest.mark.parametrize("value", [None, 42, 3.5, b"claude-opus-4-20250514", ["x"], object()])
def test_non_string_input_is_rejected_without_raising(value: object) -> None:
    assert is_pinned(value) is False  # type: ignore[arg-type]


@pytest.mark.parametrize("model_id", PINNED_IDS)
def test_require_pinned_returns_the_stripped_id(model_id: str) -> None:
    assert require_pinned(f"  {model_id}  ") == model_id


@pytest.mark.parametrize("model_id", UNPINNED_IDS + ALIAS_IDS)
def test_require_pinned_raises_for_every_unpinned_id(model_id: str) -> None:
    with pytest.raises(ModelPinError):
        require_pinned(model_id)


def test_rejection_message_explains_the_rule_and_the_cost_of_breaking_it() -> None:
    # The message is the entire user experience of this error, so it has to say
    # what was refused, what is required instead, and why the rule exists.
    with pytest.raises(ModelPinError) as excinfo:
        require_pinned("claude-3-5-sonnet-latest")

    message = str(excinfo.value)
    assert "claude-3-5-sonnet-latest" in message
    assert "version" in message
    assert "alias" in message
    assert all(token in message for token in ALIAS_TOKENS)
    assert "-20250514" in message
    assert "-v1" in message


def test_rejection_message_names_the_calling_context() -> None:
    with pytest.raises(ModelPinError, match="sampler refuses unpinned model id"):
        require_pinned("gpt-4o", context="sampler")

    with pytest.raises(ModelPinError, match="judge refuses unpinned model id"):
        require_pinned("gpt-4o")


@pytest.mark.parametrize("value", [None, 42, ["x"]])
def test_require_pinned_rejects_non_string_input(value: object) -> None:
    with pytest.raises(ModelPinError):
        require_pinned(value)  # type: ignore[arg-type]
