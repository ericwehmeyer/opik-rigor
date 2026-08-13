"""Tests for the provider seam.

Three properties are load-bearing enough to test rather than assume:

* a credential never becomes a constructor argument, a ``repr``, or an exception
  message -- the tests plant a sentinel key in the environment and hunt for it;
* ``import rigor.adapters`` works with no provider SDK installed, which is tested
  for real (neither SDK is installed here) rather than by mocking an import;
* :class:`FakeAdapter` hands out exactly the scripted responses, in order, under
  concurrency, and reproducibly under a seed -- because every statistical gate in
  this library is measured against it.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rigor.adapters import (
    ENV_ANTHROPIC_API_KEY,
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_BASE_URL,
    Adapter,
    AdapterError,
    AnthropicAdapter,
    FakeAdapter,
    OpenAICompatAdapter,
)
from rigor.pinning import is_pinned

ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
OPENAI_MODEL = "gpt-4o-2024-08-06"

SENTINEL_KEY = "sk-sentinel-DO-NOT-LEAK-0123456789"

CREDENTIAL_KWARGS = ("api_key", "key", "token", "secret")


@pytest.fixture
def anthropic_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """A planted key, so no test depends on the developer's real environment."""
    monkeypatch.setenv(ENV_ANTHROPIC_API_KEY, SENTINEL_KEY)
    return SENTINEL_KEY


@pytest.fixture
def openai_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(ENV_OPENAI_API_KEY, SENTINEL_KEY)
    monkeypatch.delenv(ENV_OPENAI_BASE_URL, raising=False)
    return SENTINEL_KEY


# --------------------------------------------------------------------------- #
# package surface
# --------------------------------------------------------------------------- #


def test_package_imports_with_no_provider_sdk_installed() -> None:
    # Not a hypothetical: neither SDK is installed in this environment, which is
    # the whole point -- collecting the suite must not require a provider.
    assert importlib.util.find_spec("anthropic") is None
    assert importlib.util.find_spec("openai") is None

    import rigor.adapters as adapters

    assert set(adapters.__all__) == {
        "ENV_ANTHROPIC_API_KEY",
        "ENV_OPENAI_API_KEY",
        "ENV_OPENAI_BASE_URL",
        "Adapter",
        "AdapterError",
        "AnthropicAdapter",
        "FakeAdapter",
        "OpenAICompatAdapter",
    }
    for name in adapters.__all__:
        assert hasattr(adapters, name)


def test_every_adapter_satisfies_the_protocol(anthropic_env: str, openai_env: str) -> None:
    fake = FakeAdapter(responses=["ok"])
    real = AnthropicAdapter(ANTHROPIC_MODEL)
    compat = OpenAICompatAdapter(OPENAI_MODEL)

    assert isinstance(fake, Adapter)
    assert isinstance(real, Adapter)
    assert isinstance(compat, Adapter)


def test_shipped_model_ids_are_pinned(anthropic_env: str, openai_env: str) -> None:
    # A default that failed is_pinned would teach every reader of the fixtures
    # the wrong lesson about naming a model.
    assert is_pinned(FakeAdapter(responses=["ok"]).model_id)
    assert is_pinned(AnthropicAdapter(ANTHROPIC_MODEL).model_id)
    assert is_pinned(OpenAICompatAdapter(OPENAI_MODEL).model_id)


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("keyword", CREDENTIAL_KWARGS)
def test_constructors_reject_every_credential_keyword(
    keyword: str, anthropic_env: str, openai_env: str
) -> None:
    with pytest.raises(TypeError, match="does not accept credentials"):
        FakeAdapter(responses=["ok"], **{keyword: SENTINEL_KEY})
    with pytest.raises(TypeError, match="does not accept credentials"):
        AnthropicAdapter(ANTHROPIC_MODEL, **{keyword: SENTINEL_KEY})
    with pytest.raises(TypeError, match="does not accept credentials"):
        OpenAICompatAdapter(OPENAI_MODEL, **{keyword: SENTINEL_KEY})


@pytest.mark.parametrize("keyword", CREDENTIAL_KWARGS)
def test_credential_rejection_names_the_keyword_but_never_the_value(
    keyword: str, anthropic_env: str
) -> None:
    with pytest.raises(TypeError) as excinfo:
        AnthropicAdapter(ANTHROPIC_MODEL, **{keyword: SENTINEL_KEY})

    message = str(excinfo.value)
    assert keyword in message
    assert SENTINEL_KEY not in message


def test_unknown_keyword_is_still_a_type_error(anthropic_env: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        AnthropicAdapter(ANTHROPIC_MODEL, tempreature=0.5)


@pytest.mark.parametrize(
    ("factory", "variable"),
    [
        (lambda: AnthropicAdapter(ANTHROPIC_MODEL), ENV_ANTHROPIC_API_KEY),
        (lambda: OpenAICompatAdapter(OPENAI_MODEL), ENV_OPENAI_API_KEY),
    ],
)
def test_missing_key_fails_at_construction_naming_the_variable(
    factory, variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(AdapterError, match=variable):
        factory()


@pytest.mark.parametrize(
    ("factory", "variable"),
    [
        (lambda: AnthropicAdapter(ANTHROPIC_MODEL), ENV_ANTHROPIC_API_KEY),
        (lambda: OpenAICompatAdapter(OPENAI_MODEL), ENV_OPENAI_API_KEY),
    ],
)
def test_blank_key_is_treated_as_missing(
    factory, variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(variable, "   ")

    with pytest.raises(AdapterError, match=variable):
        factory()


def test_key_never_appears_in_repr_or_str(anthropic_env: str, openai_env: str) -> None:
    for adapter in (AnthropicAdapter(ANTHROPIC_MODEL), OpenAICompatAdapter(OPENAI_MODEL)):
        assert SENTINEL_KEY not in repr(adapter)
        assert SENTINEL_KEY not in str(adapter)
        assert SENTINEL_KEY not in str(vars(adapter).keys())


def test_key_never_appears_in_the_error_raised_by_a_failing_call(
    anthropic_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A provider that echoes the offending header into its error message would
    # otherwise put the key straight into CI logs.
    adapter = AnthropicAdapter(ANTHROPIC_MODEL)

    class Exploding:
        class messages:  # noqa: N801 - mimics the SDK's attribute shape
            @staticmethod
            def create(**_: object) -> object:
                raise RuntimeError(f"401 unauthorized for x-api-key: {SENTINEL_KEY}")

    monkeypatch.setattr(adapter, "_sdk_client", lambda: Exploding())

    with pytest.raises(AdapterError) as excinfo:
        adapter.complete("hello")

    message = str(excinfo.value)
    assert SENTINEL_KEY not in message
    assert "***" in message
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# --------------------------------------------------------------------------- #
# lazy provider imports
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("factory", "package"),
    [
        (lambda: AnthropicAdapter(ANTHROPIC_MODEL), "anthropic"),
        (lambda: OpenAICompatAdapter(OPENAI_MODEL), "openai"),
    ],
)
def test_missing_sdk_is_reported_with_the_install_command(
    factory, package: str, anthropic_env: str, openai_env: str
) -> None:
    adapter = factory()

    with pytest.raises(AdapterError) as excinfo:
        adapter.complete("hello")

    message = str(excinfo.value)
    assert package in message
    assert f"pip install {package}" in message
    assert isinstance(excinfo.value.__cause__, ImportError)


# --------------------------------------------------------------------------- #
# real adapter configuration
# --------------------------------------------------------------------------- #


def test_model_id_is_required_and_positional(anthropic_env: str) -> None:
    with pytest.raises(TypeError):
        AnthropicAdapter()  # type: ignore[call-arg]


def test_defaults_are_deterministic_and_readable(anthropic_env: str, openai_env: str) -> None:
    anthropic = AnthropicAdapter(ANTHROPIC_MODEL)
    compat = OpenAICompatAdapter(OPENAI_MODEL)

    for adapter in (anthropic, compat):
        assert adapter.temperature == 0.0  # a judge must answer the same way twice
        assert adapter.max_tokens == 1024
        assert adapter.timeout == 60.0
    assert anthropic.model_id == ANTHROPIC_MODEL
    assert compat.model_id == OPENAI_MODEL
    assert compat.base_url is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"max_tokens": 1.5},
        {"temperature": -0.1},
        {"temperature": 3.0},
        {"timeout": 0},
        {"timeout": -1},
    ],
)
def test_invalid_configuration_is_refused_at_construction(
    kwargs: dict, anthropic_env: str, openai_env: str
) -> None:
    with pytest.raises(ValueError):
        OpenAICompatAdapter(OPENAI_MODEL, **kwargs)


def test_empty_model_id_is_refused(anthropic_env: str) -> None:
    with pytest.raises(ValueError, match="model_id"):
        AnthropicAdapter("   ")


def test_base_url_falls_back_to_the_environment(
    openai_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This is how an Azure AI Foundry (or vLLM, or gateway) endpoint gets used
    # without touching the code that constructs the judge.
    monkeypatch.setenv(ENV_OPENAI_BASE_URL, "https://example-foundry.openai.azure.com/v1")

    assert OpenAICompatAdapter(OPENAI_MODEL).base_url == (
        "https://example-foundry.openai.azure.com/v1"
    )


def test_explicit_base_url_wins_over_the_environment(
    openai_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_OPENAI_BASE_URL, "https://from-env.example/v1")

    adapter = OpenAICompatAdapter(OPENAI_MODEL, base_url="https://explicit.example/v1")

    assert adapter.base_url == "https://explicit.example/v1"


def test_blank_base_url_env_is_ignored(openai_env: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OPENAI_BASE_URL, "  ")

    assert OpenAICompatAdapter(OPENAI_MODEL).base_url is None


def test_anthropic_response_text_blocks_are_joined(
    anthropic_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Block:
        def __init__(self, text: str | None) -> None:
            self.text = text

    class Message:
        content = [Block("hello "), Block(None), Block("world")]
        stop_reason = "end_turn"

    captured: dict[str, object] = {}

    class Client:
        class messages:  # noqa: N801 - mimics the SDK's attribute shape
            @staticmethod
            def create(**kwargs: object) -> Message:
                captured.update(kwargs)
                return Message()

    adapter = AnthropicAdapter(ANTHROPIC_MODEL, max_tokens=32)
    monkeypatch.setattr(adapter, "_sdk_client", lambda: Client())

    assert adapter.complete("hi") == "hello world"
    assert captured["model"] == ANTHROPIC_MODEL
    assert captured["max_tokens"] == 32
    assert captured["temperature"] == 0.0
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_tool_only_response_is_an_error_not_an_empty_string(
    anthropic_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Missing data must not be recorded as an empty answer: a blank string would
    # be scored, and a score for a response that never happened is a fabrication.
    class Message:
        content: list[object] = []
        stop_reason = "max_tokens"

    class Client:
        class messages:  # noqa: N801 - mimics the SDK's attribute shape
            @staticmethod
            def create(**_: object) -> Message:
                return Message()

    adapter = AnthropicAdapter(ANTHROPIC_MODEL)
    monkeypatch.setattr(adapter, "_sdk_client", lambda: Client())

    with pytest.raises(AdapterError, match="no text blocks"):
        adapter.complete("hi")


def test_openai_compat_extracts_the_first_choice(
    openai_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Message:
        content = "the answer"

    class Choice:
        message = Message()
        finish_reason = "stop"

    class Response:
        choices = [Choice()]

    captured: dict[str, object] = {}

    class Client:
        class chat:  # noqa: N801 - mimics the SDK's attribute shape
            class completions:  # noqa: N801 - mimics the SDK's attribute shape
                @staticmethod
                def create(**kwargs: object) -> Response:
                    captured.update(kwargs)
                    return Response()

    adapter = OpenAICompatAdapter(OPENAI_MODEL, base_url="https://gateway.example/v1")
    monkeypatch.setattr(adapter, "_sdk_client", lambda: Client())

    assert adapter.complete("hi") == "the answer"
    assert captured["model"] == OPENAI_MODEL
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_compat_empty_content_is_an_error(
    openai_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Choice:
        message = type("M", (), {"content": None})()
        finish_reason = "content_filter"

    class Client:
        class chat:  # noqa: N801 - mimics the SDK's attribute shape
            class completions:  # noqa: N801 - mimics the SDK's attribute shape
                @staticmethod
                def create(**_: object) -> object:
                    return type("R", (), {"choices": [Choice()]})()

    adapter = OpenAICompatAdapter(OPENAI_MODEL)
    monkeypatch.setattr(adapter, "_sdk_client", lambda: Client())

    with pytest.raises(AdapterError, match="content_filter"):
        adapter.complete("hi")


@pytest.mark.requires_network
@pytest.mark.skipif(
    not os.environ.get(ENV_ANTHROPIC_API_KEY, "").strip(),
    reason=f"{ENV_ANTHROPIC_API_KEY} is not set",
)
def test_live_anthropic_call_returns_text() -> None:  # pragma: no cover - needs a real endpoint
    adapter = AnthropicAdapter(ANTHROPIC_MODEL, max_tokens=16)

    assert adapter.complete("Reply with the single word: pong").strip()


# --------------------------------------------------------------------------- #
# FakeAdapter: response modes
# --------------------------------------------------------------------------- #


def test_sequence_responses_are_handed_out_in_order() -> None:
    fake = FakeAdapter(responses=["a", "b", "c"])

    assert [fake.complete(f"p{i}") for i in range(3)] == ["a", "b", "c"]
    assert fake.calls == ["p0", "p1", "p2"]
    assert fake.call_count == 3


def test_exhausted_sequence_raises_rather_than_repeating_the_last_answer() -> None:
    fake = FakeAdapter(responses=["only"])
    fake.complete("p")

    with pytest.raises(AdapterError, match="ran out of scripted responses"):
        fake.complete("p")


def test_cycle_repeats_the_script() -> None:
    fake = FakeAdapter(responses=["a", "b"], cycle=True)

    assert [fake.complete("p") for _ in range(5)] == ["a", "b", "a", "b", "a"]


def test_mapping_responses_answer_per_prompt_regardless_of_order() -> None:
    fake = FakeAdapter(responses={"ping": "pong", "hello": "world"})

    assert fake.complete("hello") == "world"
    assert fake.complete("ping") == "pong"
    assert fake.complete("hello") == "world"


def test_unknown_prompt_in_mapping_mode_raises() -> None:
    fake = FakeAdapter(responses={"ping": "pong"})

    with pytest.raises(AdapterError, match="no scripted response"):
        fake.complete("who?")

    assert fake.calls == ["who?"]  # the attempt is still recorded


def test_callable_responses_are_computed_from_the_prompt() -> None:
    fake = FakeAdapter(responses=lambda prompt: prompt.upper())

    assert fake.complete("shout") == "SHOUT"
    assert fake.call_count == 1


def test_callable_may_read_the_call_log_without_deadlocking() -> None:
    # The callable is invoked outside the lock precisely so this works: a script
    # that answers differently as a run progresses is a normal thing to write.
    fake: FakeAdapter

    def responder(prompt: str) -> str:
        return f"{prompt}-{fake.call_count}"

    fake = FakeAdapter(responses=responder)

    assert fake.complete("a") == "a-1"
    assert fake.complete("b") == "b-2"


def test_a_bare_string_script_is_refused() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        FakeAdapter(responses="abc")


def test_non_string_callable_result_is_refused() -> None:
    fake = FakeAdapter(responses=lambda _: 42)  # type: ignore[arg-type,return-value]

    with pytest.raises(AdapterError, match="expected str"):
        fake.complete("p")


def test_empty_script_raises_on_first_call() -> None:
    fake = FakeAdapter(responses=[])

    with pytest.raises(AdapterError, match="empty response script"):
        fake.complete("p")


def test_bad_response_type_is_refused_at_construction() -> None:
    with pytest.raises(TypeError, match="responses must be"):
        FakeAdapter(responses=42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# FakeAdapter: seeded determinism
# --------------------------------------------------------------------------- #


def test_same_seed_gives_two_instances_the_identical_call_sequence() -> None:
    # This is what makes a statistical gate testable: the adapter varies like a
    # real model, but a failing run can be replayed exactly.
    script = ["yes", "no", "maybe", "certainly"]
    first = FakeAdapter(responses=script, seed=1234)
    second = FakeAdapter(responses=script, seed=1234)

    left = [first.complete(f"p{i}") for i in range(50)]
    right = [second.complete(f"p{i}") for i in range(50)]

    assert left == right
    assert set(left) <= set(script)
    assert len(set(left)) > 1  # actually stochastic, not a constant


def test_different_seeds_diverge() -> None:
    script = [str(i) for i in range(10)]
    left = [FakeAdapter(responses=script, seed=1).complete("p") for _ in range(30)]
    right = [FakeAdapter(responses=script, seed=2).complete("p") for _ in range(30)]

    assert left != right


def test_seeded_draws_do_not_exhaust_the_script() -> None:
    fake = FakeAdapter(responses=["a", "b"], seed=7)

    assert len([fake.complete("p") for _ in range(100)]) == 100


def test_the_seeded_stream_is_private_to_the_adapter() -> None:
    # A global random.seed() in some other test must not change what this
    # adapter returns, or an unrelated edit would move a statistical result.
    import random

    script = ["a", "b", "c", "d", "e"]
    random.seed(0)
    first = [FakeAdapter(responses=script, seed=99).complete("p") for _ in range(20)]
    random.seed(999999)
    second = [FakeAdapter(responses=script, seed=99).complete("p") for _ in range(20)]

    assert first == second


def test_seed_and_cycle_are_refused_outside_sequence_mode() -> None:
    with pytest.raises(ValueError, match="seed is only meaningful"):
        FakeAdapter(responses={"p": "r"}, seed=1)
    with pytest.raises(ValueError, match="cycle is only meaningful"):
        FakeAdapter(responses=lambda p: p, cycle=True)


# --------------------------------------------------------------------------- #
# FakeAdapter: failure injection and latency
# --------------------------------------------------------------------------- #


def test_fail_after_lets_exactly_n_calls_succeed() -> None:
    fake = FakeAdapter(responses=["a"] * 10, fail_with=RuntimeError("provider down"), fail_after=3)

    assert [fake.complete("p") for _ in range(3)] == ["a", "a", "a"]

    with pytest.raises(RuntimeError, match="provider down"):
        fake.complete("p")
    with pytest.raises(RuntimeError):
        fake.complete("p")  # and it keeps failing


def test_fail_with_a_class_fails_from_the_first_call() -> None:
    fake = FakeAdapter(responses=["a"], fail_with=TimeoutError)

    with pytest.raises(TimeoutError):
        fake.complete("p")


def test_a_failed_attempt_is_still_recorded_in_the_call_log() -> None:
    # .calls is the log of what was attempted, not of what succeeded: a debugging
    # test needs to see the prompt that blew up.
    fake = FakeAdapter(responses=["a", "b"], fail_with=RuntimeError, fail_after=1)
    assert fake.complete("first") == "a"

    with pytest.raises(RuntimeError):
        fake.complete("second")

    assert fake.calls == ["first", "second"]
    assert fake.call_count == 2


def test_fail_after_without_fail_with_is_refused() -> None:
    with pytest.raises(ValueError, match="fail_after has no effect"):
        FakeAdapter(responses=["a"], fail_after=2)


def test_fail_with_a_non_exception_is_refused() -> None:
    with pytest.raises(TypeError, match="fail_with"):
        FakeAdapter(responses=["a"], fail_with="boom")  # type: ignore[arg-type]


def test_latency_is_spent_per_call() -> None:
    fake = FakeAdapter(responses=["a"] * 3, latency=0.02)

    started = time.perf_counter()
    for _ in range(3):
        fake.complete("p")
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.05  # three sleeps of 20ms, minus clock slop


def test_negative_latency_is_refused() -> None:
    with pytest.raises(ValueError, match="latency"):
        FakeAdapter(responses=["a"], latency=-1.0)


# --------------------------------------------------------------------------- #
# FakeAdapter: concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_calls_consume_each_response_exactly_once() -> None:
    # The sampler drives the adapter from a thread pool. An unguarded cursor
    # would hand the same line to two threads and quietly shrink the sample.
    total = 400
    script = [f"r{i}" for i in range(total)]
    fake = FakeAdapter(responses=script)
    start = threading.Barrier(8)

    def call(index: int) -> str:
        start.wait()
        return fake.complete(f"p{index}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, range(total)))

    assert sorted(results) == sorted(script)
    assert fake.call_count == total
    assert sorted(fake.calls) == sorted(f"p{i}" for i in range(total))


def test_concurrent_calls_keep_the_call_log_complete_in_cycle_mode() -> None:
    fake = FakeAdapter(responses=["a", "b", "c"], cycle=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fake.complete, [f"p{i}" for i in range(300)]))

    assert len(results) == 300
    assert fake.call_count == 300
    # Cycling 300 calls over a 3-line script deals each line exactly 100 times.
    assert [results.count(line) for line in ("a", "b", "c")] == [100, 100, 100]


def test_seeded_draws_stay_reproducible_under_concurrency() -> None:
    # Not the *order* threads finish in -- the multiset of draws, which is what a
    # statistic over the run actually depends on.
    script = ["a", "b", "c", "d"]
    prompts = [f"p{i}" for i in range(200)]

    def run() -> list[str]:
        fake = FakeAdapter(responses=script, seed=4242)
        with ThreadPoolExecutor(max_workers=8) as pool:
            return sorted(pool.map(fake.complete, prompts))

    assert run() == run()


# --------------------------------------------------------------------------- #
# FakeAdapter: misc
# --------------------------------------------------------------------------- #


def test_model_id_defaults_to_the_pinned_fake_and_can_be_overridden() -> None:
    assert FakeAdapter(responses=["a"]).model_id == "fake-scripted-v1"
    custom = FakeAdapter(responses=["a"], model_id="my-finetune-2.1.0")
    assert custom.model_id == "my-finetune-2.1.0"


def test_repr_describes_the_script_without_dumping_it() -> None:
    fake = FakeAdapter(responses=["a", "b"], seed=3)
    fake.complete("p")

    text = repr(fake)

    assert "fake-scripted-v1" in text
    assert "sequence of 2" in text
    assert "calls=1" in text


def test_non_string_prompt_is_refused() -> None:
    fake = FakeAdapter(responses=["a"])

    with pytest.raises(TypeError, match="prompt must be a string"):
        fake.complete(None)  # type: ignore[arg-type]


def test_empty_model_id_is_refused_by_the_fake() -> None:
    with pytest.raises(ValueError, match="model_id"):
        FakeAdapter(responses=["a"], model_id="")
