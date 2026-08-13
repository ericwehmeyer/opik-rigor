"""Tests for the append-only evidence log.

The evidence log is the only place rigor records facts about a model, so these
tests are the argument that the file on disk can be trusted: a fixed envelope,
no way to rewrite history, no lost records under concurrent writers, and a
readable log after a crash mid-write.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rigor.errors import EvidenceError
from rigor.evidence import SCHEMA_VERSION, EvidenceLog, EvidenceRecord

ENVELOPE_KEYS = {"schema_version", "ts", "event_type", "payload"}

#: Names that would let a caller edit or discard recorded evidence. The absence
#: of these is a design property of the module, not an accident -- see below.
MUTATION_TOKENS = (
    "delete",
    "remove",
    "clear",
    "truncate",
    "rotate",
    "purge",
    "update",
    "pop",
)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def read_objects(path: Path) -> list[dict]:
    return [json.loads(line) for line in read_lines(path)]


# --------------------------------------------------------------------------- #
# envelope
# --------------------------------------------------------------------------- #


def test_written_line_carries_exactly_the_four_envelope_keys(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    log.append("judge.verdict", {"score": 1, "nested": {"a": [1, 2]}})

    obj = read_objects(log.path)[0]

    assert set(obj) == ENVELOPE_KEYS
    assert obj["schema_version"] == SCHEMA_VERSION
    assert obj["event_type"] == "judge.verdict"
    assert obj["payload"] == {"score": 1, "nested": {"a": [1, 2]}}


def test_timestamp_is_an_aware_utc_instant(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    before = datetime.now(timezone.utc)
    record = log.append("judge.init", {})
    after = datetime.now(timezone.utc)

    parsed = datetime.fromisoformat(read_objects(log.path)[0]["ts"])

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    assert before <= parsed <= after
    assert parsed == datetime.fromisoformat(record.ts)


def test_appended_record_round_trips_through_the_file(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    returned = log.append("assertion.evaluated", {"name": "p95", "passed": True})

    (parsed,) = log.read()

    assert parsed == EvidenceRecord(
        ts=returned.ts,
        event_type="assertion.evaluated",
        payload={"name": "p95", "passed": True},
        schema_version=SCHEMA_VERSION,
    )


def test_unicode_payloads_survive_unescaped(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    log.append("judge.verdict", {"reason": "naïve — 判定"})

    raw = log.path.read_text(encoding="utf-8")

    assert "naïve — 判定" in raw
    assert log.read()[0].payload["reason"] == "naïve — 判定"


# --------------------------------------------------------------------------- #
# append-only
# --------------------------------------------------------------------------- #


def test_log_exposes_no_way_to_rewrite_or_discard_history(tmp_path: Path) -> None:
    # Deliberate design property: an audit trail you can edit is not an audit
    # trail. Discarding evidence must be an explicit act outside the library,
    # so no public name may offer it. This test fails loudly if one is added.
    log = EvidenceLog(tmp_path / "log.jsonl")
    public_names = [name for name in dir(log) if not name.startswith("_")]

    assert set(public_names) == {"append", "last", "path", "read"}
    offenders = [
        name for name in public_names if any(token in name.lower() for token in MUTATION_TOKENS)
    ]
    assert offenders == []


def test_append_never_overwrites_earlier_records(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    for index in range(5):
        log.append("sample.completed", {"index": index})

    assert [record.payload["index"] for record in log.read()] == [0, 1, 2, 3, 4]
    assert len(read_lines(log.path)) == 5


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_threads_lose_no_records_and_tear_no_lines(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    writers, per_writer = 2, 100
    start = threading.Barrier(writers)

    def write(writer_id: int) -> None:
        start.wait()
        for index in range(per_writer):
            log.append("sample.completed", {"writer": writer_id, "index": index, "pad": "x" * 512})

    threads = [threading.Thread(target=write, args=(writer_id,)) for writer_id in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = read_lines(log.path)
    assert len(lines) == writers * per_writer
    # Every line independently parses: a torn write would show up here first.
    parsed = [json.loads(line) for line in lines]
    assert all(set(obj) == ENVELOPE_KEYS for obj in parsed)

    seen = sorted((obj["payload"]["writer"], obj["payload"]["index"]) for obj in parsed)
    expected = sorted(
        (writer_id, index) for writer_id in range(writers) for index in range(per_writer)
    )
    assert seen == expected


def test_separate_log_objects_on_one_path_both_append(tmp_path: Path) -> None:
    # Approximates two processes sharing a log file: neither may truncate the
    # other's records, because each append is an O_APPEND write of one line.
    path = tmp_path / "log.jsonl"
    first = EvidenceLog(path)
    second = EvidenceLog(path)

    for index in range(20):
        first.append("sample.completed", {"source": "first", "index": index})
        second.append("sample.completed", {"source": "second", "index": index})

    records = first.read()
    assert len(records) == 40
    assert second.read() == records
    assert [record.payload["source"] for record in records[:4]] == [
        "first",
        "second",
        "first",
        "second",
    ]


# --------------------------------------------------------------------------- #
# crash tolerance
# --------------------------------------------------------------------------- #


def test_torn_final_line_is_dropped_so_one_crash_does_not_lose_the_history(tmp_path: Path) -> None:
    # A process killed mid-append leaves a fragment with no trailing newline.
    # Tolerating it is deliberate: the alternative is an unreadable history.
    log = EvidenceLog(tmp_path / "log.jsonl")
    for index in range(3):
        log.append("sample.completed", {"index": index})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version": 1, "ts": "2026-01-01T00:00:00+00:00", "event_ty')

    records = log.read()

    assert [record.payload["index"] for record in records] == [0, 1, 2]


def test_complete_final_line_without_newline_is_still_read(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = EvidenceLog(path)
    log.append("sample.completed", {"index": 0})
    line = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "ts": "2026-01-01T00:00:00+00:00",
            "event_type": "sample.completed",
            "payload": {"index": 1},
        }
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)

    assert [record.payload["index"] for record in log.read()] == [0, 1]


def test_malformed_line_in_the_middle_is_reported_with_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = EvidenceLog(path)
    for index in range(3):
        log.append("sample.completed", {"index": index})
    lines = read_lines(path)
    lines.insert(1, '{"schema_version": 1, "ts": "2026-01')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(EvidenceError) as excinfo:
        log.read()

    message = str(excinfo.value)
    assert "line 2" in message
    assert str(path) in message


def test_line_missing_an_envelope_field_in_the_middle_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = EvidenceLog(path)
    log.append("sample.completed", {"index": 0})
    log.append("sample.completed", {"index": 1})
    lines = read_lines(path)
    lines.insert(1, json.dumps({"schema_version": 1, "ts": "2026-01-01T00:00:00+00:00"}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(EvidenceError, match="line 2"):
        log.read()


def test_blank_line_in_the_middle_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = EvidenceLog(path)
    log.append("sample.completed", {"index": 0})
    log.append("sample.completed", {"index": 1})
    lines = read_lines(path)
    lines.insert(1, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(EvidenceError, match="blank line"):
        log.read()


# --------------------------------------------------------------------------- #
# reading edge cases
# --------------------------------------------------------------------------- #


def test_reading_a_log_that_was_never_written_returns_no_records(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "never-written.jsonl")

    assert log.read() == []
    assert log.last("judge.verdict") is None
    assert not log.path.exists()


def test_reading_an_empty_file_returns_no_records(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text("", encoding="utf-8")

    assert EvidenceLog(path).read() == []


def test_appending_under_a_missing_directory_creates_the_tree(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "2026" / "log.jsonl"
    log = EvidenceLog(path)
    log.append("judge.init", {"judge": "helpfulness"})

    assert path.exists()
    assert log.read()[0].payload == {"judge": "helpfulness"}


# --------------------------------------------------------------------------- #
# last()
# --------------------------------------------------------------------------- #


def test_last_returns_the_most_recent_record_of_the_event_type(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    log.append("judge.verdict", {"judge": "helpfulness", "score": 1})
    log.append("sample.completed", {"judge": "helpfulness", "n": 30})
    log.append("judge.verdict", {"judge": "helpfulness", "score": 5})

    record = log.last("judge.verdict")

    assert record is not None
    assert record.payload["score"] == 5


def test_last_filters_on_every_payload_keyword(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    log.append("judge.verdict", {"judge": "helpfulness", "run": "a", "score": 1})
    log.append("judge.verdict", {"judge": "safety", "run": "b", "score": 2})
    log.append("judge.verdict", {"judge": "helpfulness", "run": "b", "score": 3})
    log.append("judge.verdict", {"judge": "safety", "run": "a", "score": 4})

    record = log.last("judge.verdict", judge="helpfulness", run="b")

    assert record is not None
    assert record.payload["score"] == 3


def test_last_returns_none_when_nothing_matches(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")
    log.append("judge.verdict", {"judge": "helpfulness", "score": 1})

    assert log.last("judge.parse_failure") is None
    assert log.last("judge.verdict", judge="safety") is None
    assert log.last("judge.verdict", judge="helpfulness", missing_key="x") is None


# --------------------------------------------------------------------------- #
# payload serialisation
# --------------------------------------------------------------------------- #


def test_unserialisable_payload_values_are_recorded_as_their_string_form(tmp_path: Path) -> None:
    # Pinned behaviour: json.dumps(default=str) means evidence is never dropped
    # for being un-encodable, at the cost of the value arriving back as text.
    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque object>"

    log = EvidenceLog(tmp_path / "log.jsonl")
    returned = log.append("judge.verdict", {"obj": Opaque(), "seen": {1, 2}})

    assert isinstance(returned.payload["obj"], Opaque)  # the returned record is unconverted
    payload = log.read()[0].payload
    assert payload["obj"] == "<Opaque object>"
    assert payload["seen"] in ("{1, 2}", "{2, 1}")


def test_payload_with_unorderable_mixed_keys_raises_evidence_error(tmp_path: Path) -> None:
    # sort_keys=True cannot order str against int, and default= does not apply to
    # keys, so this is one of the few payloads that is refused outright.
    log = EvidenceLog(tmp_path / "log.jsonl")

    with pytest.raises(EvidenceError, match="not JSON-serialisable"):
        log.append("judge.verdict", {1: "a", "b": 2})

    assert log.read() == []


def test_payload_with_an_unencodable_key_raises_evidence_error(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")

    with pytest.raises(EvidenceError, match="not JSON-serialisable"):
        log.append("judge.verdict", {("a", "b"): 1})


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #


def test_append_refuses_an_empty_event_type(tmp_path: Path) -> None:
    log = EvidenceLog(tmp_path / "log.jsonl")

    with pytest.raises(EvidenceError, match="event_type must be a non-empty string"):
        log.append("", {"score": 1})

    assert not log.path.exists()
