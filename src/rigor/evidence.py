"""Append-only evidence log.

Everything in rigor that produces a fact about a model -- a verdict, a rubric
hash, a rejected model id, a sampling run -- writes it here and nowhere else.

The log is JSONL with a fixed envelope::

    {"schema_version": 1, "ts": "<RFC3339 UTC>", "event_type": "<str>", "payload": {...}}

There is intentionally no delete, truncate, rotate, or update method. An audit
trail you can edit is not an audit trail. If you need to discard evidence, delete
the file yourself -- that is a deliberate act outside the library, and it leaves a
hole in the timeline that a reviewer can see.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import EvidenceError

SCHEMA_VERSION = 1

#: Event types written by the library itself. Callers may write their own.
EVENT_JUDGE_INIT = "judge.init"
EVENT_JUDGE_VERDICT = "judge.verdict"
EVENT_JUDGE_PARSE_FAILURE = "judge.parse_failure"
EVENT_RUBRIC_CHANGE_ACCEPTED = "judge.rubric_change_accepted"
EVENT_SAMPLE_COMPLETED = "sample.completed"
EVENT_ASSERTION = "assertion.evaluated"


@dataclass(frozen=True)
class EvidenceRecord:
    """One line of the log, parsed."""

    ts: str
    event_type: str
    payload: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_json(cls, line: str) -> EvidenceRecord:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise EvidenceError(f"evidence line is not a JSON object: {line[:120]!r}")
        try:
            return cls(
                ts=obj["ts"],
                event_type=obj["event_type"],
                payload=obj["payload"],
                schema_version=obj.get("schema_version", SCHEMA_VERSION),
            )
        except KeyError as exc:
            raise EvidenceError(f"evidence line missing field {exc.args[0]!r}") from exc


class EvidenceLog:
    """Append-only JSONL writer.

    Each :meth:`append` is a single ``O_APPEND`` write of one complete line, so
    concurrent writers -- threads, or separate processes sharing a path -- interleave
    whole records rather than corrupting each other.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        parent = self._path.parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event_type: str, payload: Mapping[str, Any]) -> EvidenceRecord:
        """Write one record and return it. Raises rather than dropping evidence."""
        if not event_type:
            raise EvidenceError("event_type must be a non-empty string")
        record = EvidenceRecord(
            ts=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            event_type=event_type,
            payload=dict(payload),
        )
        try:
            line = json.dumps(
                {
                    "schema_version": record.schema_version,
                    "ts": record.ts,
                    "event_type": record.event_type,
                    "payload": record.payload,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError) as exc:
            # default=str rescues most values, but it does not apply to *keys*:
            # a tuple key, or a dict mixing str and int keys under sort_keys=True,
            # still fails. Refuse the record rather than write a half-serialised one.
            raise EvidenceError(f"payload is not JSON-serialisable: {exc}") from exc

        data = (line + "\n").encode("utf-8")
        with self._lock:
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                written = 0
                while written < len(data):
                    written += os.write(fd, data[written:])
                os.fsync(fd)
            finally:
                os.close(fd)
        return record

    def read(self) -> list[EvidenceRecord]:
        """Parse the whole log.

        A truncated final line -- the signature of a crash mid-write -- is tolerated
        and dropped, because the alternative is that one interrupted run makes the
        entire history unreadable. A malformed line anywhere else is an error.
        """
        if not self._path.exists():
            return []
        raw = self._path.read_text(encoding="utf-8")
        if not raw:
            return []
        lines = raw.split("\n")
        trailing_complete = raw.endswith("\n")
        if trailing_complete:
            lines.pop()
        records: list[EvidenceRecord] = []
        for index, line in enumerate(lines):
            is_last = index == len(lines) - 1
            if not line.strip():
                if is_last and not trailing_complete:
                    continue
                raise EvidenceError(f"blank line at position {index} in {self._path}")
            try:
                records.append(EvidenceRecord.from_json(line))
            except (json.JSONDecodeError, EvidenceError):
                if is_last and not trailing_complete:
                    continue  # torn write at end of file
                raise EvidenceError(
                    f"malformed evidence at line {index + 1} of {self._path}: {line[:120]!r}"
                ) from None
        return records

    def last(self, event_type: str, **payload_match: Any) -> EvidenceRecord | None:
        """Most recent record of ``event_type`` whose payload matches every kwarg."""
        for record in reversed(self.read()):
            if record.event_type != event_type:
                continue
            if all(record.payload.get(k) == v for k, v in payload_match.items()):
                return record
        return None
