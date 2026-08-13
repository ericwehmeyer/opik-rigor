"""Tests for the hash-verified baseline store.

A baseline exists to answer "did we get worse?", and that answer is only worth
having if the file it comes from cannot be quietly edited. So these tests are
mostly an argument about trust: that a tampered file is refused, that a merely
*reformatted* file is not, that every way of being unreadable produces its own
message, and that the numbers survive the round trip bit for bit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from opik_rigor.baseline import (
    BASELINE_SCHEMA_VERSION,
    DIGEST_PATTERN,
    PAYLOAD_FIELDS,
    Baseline,
)
from opik_rigor.errors import BaselineError
from opik_rigor.sampling import Run, SampleResult

SEED = 20260812


def make_baseline(**overrides: Any) -> Baseline:
    """A fully populated baseline; every field set to something distinguishable."""
    fields: dict[str, Any] = {
        "name": "helpfulness",
        "scores": (4.0, 3.5, 5.0),
        "outcomes": (True, True, False),
        "model_id": "claude-sonnet-4-20250514",
        "rubric_hash": "a" * 64,
        "created": "2026-08-12T09:15:00.000000+00:00",
        "metadata": {"suite": "nightly", "runner": "ci"},
    }
    fields.update(overrides)
    return Baseline(**fields)


def read_document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_document(path: Path, document: dict[str, Any], **dump: Any) -> None:
    path.write_text(json.dumps(document, **dump), encoding="utf-8", newline="\n")


@dataclasses.dataclass(frozen=True)
class ScoredResponse:
    """Stands in for a judge Verdict: what a SampleResult carries as a run value."""

    score: float | None
    passed: bool


def sample_of_scores(pairs: list[tuple[float | None, bool]]) -> SampleResult:
    runs = tuple(
        Run(index=index, value=ScoredResponse(score, passed), outcome=passed, duration=0.0)
        for index, (score, passed) in enumerate(pairs)
    )
    return SampleResult(runs=runs, wall_clock=0.0)


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #


def test_save_then_load_returns_an_equal_baseline(tmp_path: Path) -> None:
    rng = random.Random(SEED)
    original = make_baseline(
        scores=tuple(round(rng.uniform(1.0, 5.0), 3) for _ in range(20)),
        outcomes=tuple(rng.random() < 0.8 for _ in range(20)),
    )
    path = original.save(tmp_path / "baseline.json")

    loaded = Baseline.load(path)

    assert loaded == original
    assert loaded.digest() == original.digest()
    assert loaded.scores == original.scores
    assert loaded.outcomes == original.outcomes
    assert loaded.metadata == original.metadata


def test_floats_round_trip_exactly(tmp_path: Path) -> None:
    # 0.1 + 0.2 is 0.30000000000000004, which any formatter that "tidies" floats
    # mangles into 0.3 -- a different number, silently compared against later.
    awkward = 0.1 + 0.2
    path = make_baseline(scores=(awkward, 1e-17, 1 / 3), outcomes=(True,)).save(
        tmp_path / "baseline.json"
    )

    loaded = Baseline.load(path)

    assert loaded.scores == (awkward, 1e-17, 1 / 3)
    assert loaded.scores[0] != 0.3
    assert repr(awkward) in path.read_text(encoding="utf-8")


def test_save_returns_the_path_and_creates_missing_parents(tmp_path: Path) -> None:
    target = tmp_path / "runs" / "2026" / "baseline.json"

    returned = make_baseline().save(target)

    assert returned == target
    assert target.exists()


def test_stored_document_is_the_payload_plus_a_digest(tmp_path: Path) -> None:
    baseline = make_baseline()
    path = baseline.save(tmp_path / "baseline.json")

    document = read_document(path)

    assert set(document) == {*PAYLOAD_FIELDS, "digest"}
    assert document["schema_version"] == BASELINE_SCHEMA_VERSION
    assert document["digest"] == baseline.digest()
    assert DIGEST_PATTERN.match(document["digest"])
    # Pretty-printed for review, not for hashing.
    assert path.read_text(encoding="utf-8").startswith("{\n  ")


def test_digest_is_taken_over_the_canonical_form_not_the_stored_bytes(tmp_path: Path) -> None:
    # Pins the hash contract in one readable line. If sort_keys, the separators,
    # or ensure_ascii ever change, this fails here rather than as an unexplainable
    # "tampered file" on someone's machine six months from now.
    baseline = make_baseline(name="naïve — 判定")
    canonical = json.dumps(
        baseline.payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert baseline.digest() == "sha256:" + hashlib.sha256(canonical).hexdigest()

    path = baseline.save(tmp_path / "baseline.json")
    assert Baseline.load(path) == baseline


# --------------------------------------------------------------------------- #
# tamper detection -- the reason this module exists
# --------------------------------------------------------------------------- #


def test_a_tampered_file_is_rejected(tmp_path: Path) -> None:
    # THE load-bearing test of this module. Everything else here is hygiene; this
    # is the property the whole design is for. Without it a regression can be made
    # to disappear by editing the file it is compared against, and every "no
    # regression" result the library ever reports means nothing.
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    original_digest = document["digest"]
    document["scores"][0] = 1.0  # the edit that would hide a regression
    write_document(path, document, indent=2, sort_keys=True)
    assert read_document(path)["digest"] == original_digest  # the forger kept the digest

    with pytest.raises(BaselineError) as excinfo:
        Baseline.load(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "digest mismatch" in message
    assert original_digest in message


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("name", "renamed"),
        ("scores", [1.0, 1.0, 1.0]),
        ("outcomes", [True, True, True]),
        ("model_id", "claude-opus-4-20250514"),
        ("rubric_hash", "b" * 64),
        ("created", "2020-01-01T00:00:00.000000+00:00"),
        ("metadata", {"suite": "smoke"}),
    ],
)
def test_editing_any_payload_field_invalidates_the_digest(
    tmp_path: Path, key: str, value: Any
) -> None:
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    document[key] = value
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError, match="digest mismatch"):
        Baseline.load(path)


def test_an_added_unknown_field_is_reported_rather_than_ignored(tmp_path: Path) -> None:
    # An ignored key would sit in a file that looks digest-covered while being
    # covered by nothing -- a good place to hide "approved_by: me".
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    document["approved_by"] = "me"
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError) as excinfo:
        Baseline.load(path)

    assert "unknown field(s) approved_by" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


# --------------------------------------------------------------------------- #
# reformatting is not tampering
# --------------------------------------------------------------------------- #


def test_reformatting_the_file_does_not_invalidate_it(tmp_path: Path) -> None:
    baseline = make_baseline()
    path = baseline.save(tmp_path / "baseline.json")
    document = read_document(path)
    reordered = dict(reversed(list(document.items())))
    write_document(path, reordered, indent=7, sort_keys=False, separators=(" ,", " : "))

    assert list(read_document(path)) != list(document)  # genuinely re-laid-out
    assert Baseline.load(path) == baseline


def test_minified_file_still_loads(tmp_path: Path) -> None:
    baseline = make_baseline()
    path = baseline.save(tmp_path / "baseline.json")
    write_document(path, read_document(path), separators=(",", ":"))

    assert "\n" not in path.read_text(encoding="utf-8")
    assert Baseline.load(path) == baseline


# --------------------------------------------------------------------------- #
# load failure modes -- one message each
# --------------------------------------------------------------------------- #


def test_missing_file_raises_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "never-recorded.json"

    with pytest.raises(BaselineError) as excinfo:
        Baseline.load(path)

    assert "not found" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_invalid_json_raises_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text('{"name": "helpfulness", "scores": [1.0,', encoding="utf-8")

    with pytest.raises(BaselineError) as excinfo:
        Baseline.load(path)

    assert "not valid JSON" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_json_that_is_not_an_object_raises(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(BaselineError, match="not a JSON object"):
        Baseline.load(path)


def test_missing_digest_raises(tmp_path: Path) -> None:
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    del document["digest"]
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError) as excinfo:
        Baseline.load(path)

    assert "carries no digest" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "deadbeef",
        "sha256:",
        "sha256:not-hex-at-all",
        "sha256:" + "A" * 64,  # uppercase hex is not the recorded form
        "sha256:" + "a" * 63,
        "md5:" + "a" * 32,
    ],
)
def test_malformed_digest_string_raises(tmp_path: Path, digest: str) -> None:
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    document["digest"] = digest
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError) as excinfo:
        Baseline.load(path)

    assert "malformed digest" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_digest_of_the_wrong_type_raises_as_malformed(tmp_path: Path) -> None:
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    document["digest"] = 12345
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError, match="malformed digest"):
        Baseline.load(path)


def test_future_schema_version_raises_before_anything_else_is_believed(tmp_path: Path) -> None:
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    document["schema_version"] = BASELINE_SCHEMA_VERSION + 1
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError) as excinfo:
        Baseline.load(path)

    message = str(excinfo.value)
    assert "newer than this version of opik_rigor" in message
    assert str(path) in message
    # Reported as the wrong schema, not as a forgery: the version decides how the
    # rest of the document reads, so it is checked before the digest.
    assert "digest mismatch" not in message


def test_unknown_older_schema_version_raises(tmp_path: Path) -> None:
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    document["schema_version"] = 0
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError, match="unknown schema_version"):
        Baseline.load(path)


def test_missing_schema_version_raises(tmp_path: Path) -> None:
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    del document["schema_version"]
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError, match="no schema_version"):
        Baseline.load(path)


def test_every_failure_mode_has_its_own_message(tmp_path: Path) -> None:
    # A caller staring at a red CI run has to be able to tell "someone edited the
    # baseline" from "the file is from a newer opik_rigor" without reading this module.
    good = make_baseline()

    def message_from(path: Path) -> str:
        with pytest.raises(BaselineError) as excinfo:
            Baseline.load(path)
        return str(excinfo.value).replace(str(path), "<path>")

    def edited(name: str, mutate: Any) -> Path:
        path = good.save(tmp_path / f"{name}.json")
        document = read_document(path)
        mutate(document)
        write_document(path, document, indent=2)
        return path

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")

    def tamper(document: dict[str, Any]) -> None:
        document["scores"][0] = 0.0

    messages = [
        message_from(tmp_path / "absent.json"),
        message_from(corrupt),
        message_from(edited("tampered", tamper)),
        message_from(edited("no_digest", lambda document: document.pop("digest"))),
        message_from(
            edited("bad_digest", lambda document: document.update(digest="sha256:nonsense"))
        ),
        message_from(
            edited(
                "future",
                lambda document: document.update(schema_version=BASELINE_SCHEMA_VERSION + 1),
            )
        ),
    ]

    assert len(set(messages)) == len(messages)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("name", 7, "must be a string"),
        ("scores", "4.0", "must be a list"),
        ("scores", [True], "only numbers"),
        ("scores", ["4.0"], "only numbers"),
        ("outcomes", {"a": 1}, "must be a list"),
        ("outcomes", [1], "only booleans"),
        ("metadata", [1, 2], "must be a JSON object"),
        ("model_id", 3, "must be a string"),
    ],
)
def test_wrong_field_types_are_reported_as_such(
    tmp_path: Path, key: str, value: Any, match: str
) -> None:
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    document[key] = value
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError, match=match):
        Baseline.load(path)


def test_missing_name_is_reported_as_a_missing_field(tmp_path: Path) -> None:
    path = make_baseline().save(tmp_path / "baseline.json")
    document = read_document(path)
    del document["name"]
    write_document(path, document, indent=2)

    with pytest.raises(BaselineError, match="missing required field 'name'"):
        Baseline.load(path)


# --------------------------------------------------------------------------- #
# digest stability
# --------------------------------------------------------------------------- #


def test_digest_is_stable_across_two_constructions_of_equal_data() -> None:
    assert make_baseline().digest() == make_baseline().digest()
    assert make_baseline(metadata={"runner": "ci", "suite": "nightly"}).digest() == (
        make_baseline(metadata={"suite": "nightly", "runner": "ci"}).digest()
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", "safety"),
        ("scores", (4.0, 3.5, 5.000001)),
        ("scores", (4.0, 3.5)),
        ("outcomes", (True, True, True)),
        ("outcomes", (True, True)),
        ("model_id", "claude-opus-4-20250514"),
        ("rubric_hash", "b" * 64),
        ("created", "2026-08-12T09:15:00.000001+00:00"),
        ("metadata", {"suite": "nightly", "runner": "local"}),
        ("metadata", {}),
    ],
)
def test_digest_differs_when_any_field_differs(field_name: str, value: Any) -> None:
    original = make_baseline()
    changed = dataclasses.replace(original, **{field_name: value})

    assert changed != original
    assert changed.digest() != original.digest()


def test_digest_does_not_depend_on_the_container_the_scores_arrived_in() -> None:
    from_tuple = make_baseline(scores=(4.0, 3.5, 5.0))
    from_list = make_baseline(scores=[4.0, 3.5, 5.0])
    from_ints = make_baseline(scores=[4, 3.5, 5])

    assert from_tuple.digest() == from_list.digest() == from_ints.digest()
    assert from_list.scores == (4.0, 3.5, 5.0)


def test_unserialisable_metadata_raises_baseline_error() -> None:
    baseline = make_baseline(metadata={"when": datetime(2026, 8, 12, tzinfo=timezone.utc)})

    with pytest.raises(BaselineError, match="not JSON-serialisable"):
        baseline.digest()


# --------------------------------------------------------------------------- #
# from_sample
# --------------------------------------------------------------------------- #


def test_from_sample_copies_scores_and_outcomes_off_the_sample() -> None:
    rng = random.Random(SEED)
    pairs = [(round(rng.uniform(1.0, 5.0), 2), rng.random() < 0.75) for _ in range(30)]
    result = sample_of_scores(pairs)

    baseline = Baseline.from_sample("helpfulness", result)

    assert baseline.name == "helpfulness"
    assert baseline.scores == result.scores()
    assert baseline.outcomes == result.outcomes
    assert baseline.n == result.n
    assert baseline.successes == result.successes
    assert baseline.pass_rate == result.pass_rate


def test_from_sample_stamps_created_with_an_aware_utc_instant() -> None:
    before = datetime.now(timezone.utc)
    baseline = Baseline.from_sample("helpfulness", sample_of_scores([(4.0, True)]))
    after = datetime.now(timezone.utc)

    stamped = datetime.fromisoformat(baseline.created)

    assert stamped.tzinfo is not None
    assert stamped.utcoffset() == timedelta(0)
    assert before <= stamped <= after


def test_from_sample_passes_through_the_instrument_metadata() -> None:
    baseline = Baseline.from_sample(
        "helpfulness",
        sample_of_scores([(4.0, True), (2.0, False)]),
        model_id="claude-sonnet-4-20250514",
        rubric_hash="c" * 64,
        metadata={"suite": "nightly"},
    )

    assert baseline.model_id == "claude-sonnet-4-20250514"
    assert baseline.rubric_hash == "c" * 64
    assert baseline.metadata == {"suite": "nightly"}


def test_from_sample_keeps_unscored_runs_in_the_outcomes(tmp_path: Path) -> None:
    # A verdict with score=None still observed a pass or a fail, so the two series
    # are different lengths and n follows the outcomes.
    result = sample_of_scores([(4.0, True), (None, True), (2.0, False)])

    baseline = Baseline.from_sample("helpfulness", result)

    assert baseline.scores == (4.0, 2.0)
    assert baseline.outcomes == (True, True, False)
    assert baseline.n == 3
    assert baseline.successes == 2
    assert Baseline.load(baseline.save(tmp_path / "baseline.json")) == baseline


def test_from_sample_created_can_be_overridden_for_a_reproducible_file() -> None:
    baseline = Baseline.from_sample(
        "helpfulness", sample_of_scores([(4.0, True)]), created="2026-01-01T00:00:00.000000+00:00"
    )

    assert baseline.created == "2026-01-01T00:00:00.000000+00:00"


# --------------------------------------------------------------------------- #
# immutability
# --------------------------------------------------------------------------- #


def test_baseline_is_frozen(tmp_path: Path) -> None:
    baseline = Baseline.load(make_baseline().save(tmp_path / "baseline.json"))

    with pytest.raises(dataclasses.FrozenInstanceError):
        baseline.name = "something else"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        baseline.scores = (1.0,)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        del baseline.created  # type: ignore[misc]


def test_score_and_outcome_series_are_tuples_after_a_verified_load(tmp_path: Path) -> None:
    # Verifying a digest and then handing back a mutable list would make the check
    # theatre: the digest would attest to the file, not to the object in hand.
    baseline = Baseline.load(make_baseline().save(tmp_path / "baseline.json"))

    assert isinstance(baseline.scores, tuple)
    assert isinstance(baseline.outcomes, tuple)
    with pytest.raises(TypeError):
        baseline.scores[0] = 1.0  # type: ignore[index]


# --------------------------------------------------------------------------- #
# summary statistics and the empty case
# --------------------------------------------------------------------------- #


def test_summary_statistics_count_the_outcomes() -> None:
    baseline = make_baseline(outcomes=(True, False, True, True))

    assert baseline.n == 4
    assert baseline.successes == 3
    assert baseline.pass_rate == 0.75


def test_an_empty_baseline_round_trips(tmp_path: Path) -> None:
    # Zero observations is a legitimate thing to record -- a suite whose runs all
    # errored, or a baseline created before the first run -- and the loader must
    # not divide by it or otherwise fall over.
    empty = Baseline(name="helpfulness", scores=(), outcomes=())
    path = empty.save(tmp_path / "empty.json")

    loaded = Baseline.load(path)

    assert loaded == empty
    assert loaded.n == 0
    assert loaded.successes == 0
    assert loaded.pass_rate == 0.0
    assert loaded.digest() == empty.digest()


def test_a_baseline_with_outcomes_but_no_scores_round_trips(tmp_path: Path) -> None:
    baseline = make_baseline(scores=(), outcomes=(True, False))
    path = baseline.save(tmp_path / "unscored.json")

    loaded = Baseline.load(path)

    assert loaded == baseline
    assert loaded.scores == ()
    assert loaded.pass_rate == 0.5
