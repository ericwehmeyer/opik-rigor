"""Hash-verified baselines.

A baseline is the recorded behaviour of a system at a point in time, kept so that
a later run can answer "did we get worse?". That question is only worth asking if
the answer cannot be arranged: a baseline file that anyone can edit lets a
regression be made to disappear by editing the thing it is compared against, and
a comparison against an edited file is not evidence, it is a formality.

So every baseline carries a sha256 of its own contents, and :meth:`Baseline.load`
recomputes that digest before returning. A file whose contents no longer match its
digest is refused outright rather than loaded with a warning -- there is no
sensible way to compare against a baseline you cannot vouch for.

**The hash contract.** The digest is taken over a *canonical* serialisation of the
payload::

    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

encoded UTF-8. Key order and separators are part of the hash contract, not a
formatting preference. Change either -- drop ``sort_keys``, put a space after the
colon -- and every baseline already written to disk stops verifying, with a
"tampered" message pointing at files nobody touched. That is exactly why
:data:`BASELINE_SCHEMA_VERSION` exists: a change to the canonical form must be
made visible by bumping it, so old files are rejected as *the wrong schema*
rather than as forgeries.

The file on disk is pretty-printed with ``indent=2`` so that it diffs and reviews
like source, but the digest is never taken over those bytes. Hashing the
pretty-printed form would mean that re-indenting a file, or a tool that sorts JSON
keys, silently destroyed a baseline; hashing the canonical form means a
reformatted file still verifies, while a changed *value* does not.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import BaselineError

if TYPE_CHECKING:  # pragma: no cover - typing only; the runtime path is duck-typed
    from .sampling import SampleResult

#: Bumped whenever the canonical payload or the meaning of a field changes. A file
#: declaring a version this code does not know is refused, never guessed at.
BASELINE_SCHEMA_VERSION = 1

#: Every key of the canonical payload -- that is, every field the digest covers.
#: The stored document is exactly these plus ``digest``.
PAYLOAD_FIELDS = (
    "created",
    "metadata",
    "model_id",
    "name",
    "outcomes",
    "rubric_hash",
    "schema_version",
    "scores",
)

#: The stored digest is ``sha256:`` followed by 64 lowercase hex characters. The
#: algorithm is spelled out in the file so that a future change of hash is a
#: readable difference rather than a mismatch nobody can explain.
DIGEST_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z")


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """The exact bytes the digest is taken over.

    Public because the contract is public: anything that wants to verify a rigor
    baseline -- a CI script, another implementation -- needs this function and not
    a prose description of it.
    """
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"baseline is not JSON-serialisable: {exc}") from exc
    return text.encode("utf-8")


@dataclass(frozen=True)
class Baseline:
    """One recorded observation set, verifiable against its own digest.

    Frozen, and ``scores``/``outcomes`` are tuples, so that a baseline cannot be
    edited after :meth:`load` checked it. A mutable baseline would make the check
    theatre: the digest would attest to what was on disk, not to the object the
    assertion is about to compare against.
    """

    name: str
    scores: tuple[float, ...]
    outcomes: tuple[bool, ...]
    model_id: str = ""
    rubric_hash: str = ""
    created: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise the containers at construction time rather than trusting the
        # caller to pass tuples: two baselines with the same numbers must hash the
        # same whether they were built from a list, a generator, or a load().
        object.__setattr__(self, "scores", tuple(float(score) for score in self.scores))
        object.__setattr__(self, "outcomes", tuple(bool(outcome) for outcome in self.outcomes))

    # ----------------------------------------------------------------- #
    # summary statistics
    # ----------------------------------------------------------------- #

    @property
    def n(self) -> int:
        """Recorded outcomes.

        Counts ``outcomes``, not ``scores``: the two are separate series and a
        judge that declined to score still produced a pass/fail observation, so
        a baseline may legitimately hold fewer scores than outcomes.
        """
        return len(self.outcomes)

    @property
    def successes(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome)

    @property
    def pass_rate(self) -> float:
        """Recorded pass rate. A point estimate from ``n`` observations."""
        return self.successes / self.n if self.n else 0.0

    # ----------------------------------------------------------------- #
    # hashing and serialisation
    # ----------------------------------------------------------------- #

    def payload(self) -> dict[str, Any]:
        """Every field the digest covers, including the schema version.

        The schema version is inside the payload on purpose. It decides how the
        rest of the document is to be read, so leaving it outside the digest would
        leave the one field that changes the meaning of every other field as the
        one field an editor could change for free.
        """
        return {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "name": self.name,
            "scores": list(self.scores),
            "outcomes": list(self.outcomes),
            "model_id": self.model_id,
            "rubric_hash": self.rubric_hash,
            "created": self.created,
            "metadata": dict(self.metadata),
        }

    def digest(self) -> str:
        """``sha256:<hex>`` over :func:`canonical_bytes` of :meth:`payload`.

        Floats go through ``json``'s default ``repr``-precision formatting, which
        round-trips every finite double exactly. A lossy formatter -- ``%.6g``, or
        anything that "tidies" ``0.30000000000000004`` -- would quietly change the
        numbers a regression gate compares, which is a worse failure than a
        mismatch because it looks like a measurement.
        """
        return "sha256:" + hashlib.sha256(canonical_bytes(self.payload())).hexdigest()

    def to_json(self) -> str:
        """The pretty-printed document written to disk, digest included."""
        document = {**self.payload(), "digest": self.digest()}
        return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the baseline to ``path`` and return the path."""
        file = Path(path)
        parent = file.parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" keeps the file byte-identical across platforms. It does not
        # affect the digest -- that is the point of hashing the canonical form --
        # but a baseline that changes on checkout is noise in every diff.
        file.write_text(self.to_json() + "\n", encoding="utf-8", newline="\n")
        return file

    # ----------------------------------------------------------------- #
    # construction
    # ----------------------------------------------------------------- #

    @classmethod
    def from_sample(cls, name: str, result: SampleResult, **kwargs: Any) -> Baseline:
        """Record a :class:`~rigor.sampling.SampleResult` as a baseline.

        Takes the scores from ``result.scores()`` and the outcomes from
        ``result.outcomes``, and stamps ``created`` now. The timestamp is set here
        rather than at :meth:`save` time because it is a fact about when the
        system was observed, not about when someone got round to writing the file.

        Extra keyword arguments -- ``model_id``, ``rubric_hash``, ``metadata`` --
        are passed through, which is how the baseline records *which instrument*
        produced it. Comparing against a baseline taken with a different model or
        a different rubric is the mistake this metadata exists to expose.
        """
        kwargs.setdefault("created", datetime.now(timezone.utc).isoformat(timespec="microseconds"))
        return cls(
            name=name,
            scores=result.scores(),
            outcomes=result.outcomes,
            **kwargs,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Baseline:
        """Read, verify, and return the baseline at ``path``.

        Raises :class:`~rigor.errors.BaselineError` -- naming the file, because a
        suite compares against several -- when the file is missing, is not JSON,
        declares a schema version this code does not implement, carries no digest
        or a malformed one, or does not hash to the digest it carries.

        Verification is done against the payload rebuilt from the parsed document,
        not against the raw text, so what the digest attests to is exactly the
        object being returned -- reformatting the file changes the text and not
        the object, which is the whole point of hashing the canonical form.

        A document carrying a key this schema does not define is refused rather
        than having the key ignored. An ignored key would sit in a file that
        *looks* digest-covered while being covered by nothing at all, which is a
        good place to hide a note saying the regression was approved.
        """
        file = Path(path)
        try:
            raw = file.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise BaselineError(f"baseline file not found: {file}") from exc
        except OSError as exc:
            raise BaselineError(f"baseline file {file} could not be read: {exc}") from exc

        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BaselineError(f"baseline file {file} is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise BaselineError(
                f"baseline file {file} is not a JSON object "
                f"(top level is {type(document).__name__})"
            )

        # Schema first: the version decides how everything else is to be read, so
        # a version this code does not implement has to be reported as such rather
        # than as the digest mismatch it would otherwise surface as.
        _check_schema_version(document, file)

        unknown = sorted(set(document) - {*PAYLOAD_FIELDS, "digest"})
        if unknown:
            raise BaselineError(
                f"baseline file {file} carries unknown field(s) {', '.join(unknown)}, "
                f"which the digest does not cover; a field this schema does not define "
                f"is not part of the recorded baseline"
            )

        recorded = document.get("digest")
        if recorded is None:
            raise BaselineError(
                f"baseline file {file} carries no digest, so nothing about it can be "
                f"verified; it is an unsigned claim about what the system used to do"
            )
        if not isinstance(recorded, str) or not DIGEST_PATTERN.match(recorded):
            raise BaselineError(
                f"baseline file {file} has a malformed digest {recorded!r}; "
                f"expected 'sha256:' followed by 64 lowercase hex characters"
            )

        baseline = cls(
            name=_string(document, "name", file, required=True),
            scores=_numbers(document, "scores", file),
            outcomes=_booleans(document, "outcomes", file),
            model_id=_string(document, "model_id", file),
            rubric_hash=_string(document, "rubric_hash", file),
            created=_string(document, "created", file),
            metadata=_mapping(document, "metadata", file),
        )

        computed = baseline.digest()
        if not hmac.compare_digest(computed, recorded):
            raise BaselineError(
                f"digest mismatch for baseline file {file}: the file records {recorded} "
                f"but its contents hash to {computed}. The file has been edited since it "
                f"was recorded, so it cannot be used as a baseline -- re-record it "
                f"deliberately if the change was intended"
            )
        return baseline


# --------------------------------------------------------------------------- #
# document parsing
# --------------------------------------------------------------------------- #


def _check_schema_version(document: Mapping[str, Any], file: Path) -> None:
    version = document.get("schema_version")
    if version is None:
        raise BaselineError(f"baseline file {file} declares no schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise BaselineError(
            f"baseline file {file} has a malformed schema_version {version!r}; "
            f"expected an integer"
        )
    if version > BASELINE_SCHEMA_VERSION:
        raise BaselineError(
            f"baseline file {file} declares schema_version {version}, which is newer "
            f"than this version of rigor understands ({BASELINE_SCHEMA_VERSION}). "
            f"Upgrade rigor rather than reading it with the wrong rules"
        )
    if version != BASELINE_SCHEMA_VERSION:
        raise BaselineError(
            f"baseline file {file} declares unknown schema_version {version}; "
            f"this version of rigor reads schema_version {BASELINE_SCHEMA_VERSION}"
        )


def _string(document: Mapping[str, Any], key: str, file: Path, *, required: bool = False) -> str:
    value = document.get(key)
    if value is None:
        if required:
            raise BaselineError(f"baseline file {file} is missing required field {key!r}")
        return ""
    if not isinstance(value, str):
        raise BaselineError(
            f"baseline file {file}: {key!r} must be a string, got {type(value).__name__}"
        )
    return value


def _numbers(document: Mapping[str, Any], key: str, file: Path) -> tuple[float, ...]:
    values = _sequence(document, key, file)
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise BaselineError(
                f"baseline file {file}: {key!r} must contain only numbers, got {value!r}"
            )
    return tuple(float(value) for value in values)


def _booleans(document: Mapping[str, Any], key: str, file: Path) -> tuple[bool, ...]:
    values = _sequence(document, key, file)
    for value in values:
        if not isinstance(value, bool):
            raise BaselineError(
                f"baseline file {file}: {key!r} must contain only booleans, got {value!r}"
            )
    return tuple(bool(value) for value in values)


def _sequence(document: Mapping[str, Any], key: str, file: Path) -> list[Any]:
    value = document.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise BaselineError(
            f"baseline file {file}: {key!r} must be a list, got {type(value).__name__}"
        )
    return value


def _mapping(document: Mapping[str, Any], key: str, file: Path) -> dict[str, Any]:
    value = document.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BaselineError(
            f"baseline file {file}: {key!r} must be a JSON object, got {type(value).__name__}"
        )
    return dict(value)
