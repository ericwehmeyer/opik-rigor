"""Exception hierarchy for rigor.

Every failure mode that a caller might reasonably want to catch separately gets
its own type. Nothing in this library ever converts an error into a quiet
"failed evaluation" -- a judge that could not be parsed is a bug in the harness,
not evidence about the system under test.
"""

from __future__ import annotations


class RigorError(Exception):
    """Base class for every error raised by rigor."""


class ModelPinError(RigorError):
    """Raised when a judge is constructed with a model id that is not pinned.

    Aliases such as ``claude-sonnet-latest`` or bare family names silently change
    under you, which makes every historical verdict uncomparable. The judge
    refuses them at construction time rather than at analysis time.
    """


class RubricDriftError(RigorError):
    """Raised when the rubric file's hash differs from the last recorded hash.

    Pass ``accept_rubric_change=True`` to acknowledge the change; the acceptance
    is written to the evidence log as its own event, recording both hashes.
    """

    def __init__(self, judge_name: str, recorded_hash: str, current_hash: str) -> None:
        self.judge_name = judge_name
        self.recorded_hash = recorded_hash
        self.current_hash = current_hash
        super().__init__(
            f"rubric drift for judge {judge_name!r}: "
            f"evidence log last recorded {recorded_hash}, rubric file now hashes to "
            f"{current_hash}. Scores before and after this change are not comparable. "
            f"Pass accept_rubric_change=True to acknowledge and record the change."
        )


class JudgeOutputError(RigorError):
    """Raised when a judge's response cannot be parsed into a Verdict.

    Deliberately not a fail-verdict: an unparseable response is missing data, and
    counting missing data as a failure biases every downstream statistic.
    """

    def __init__(self, message: str, raw: str) -> None:
        self.raw = raw
        super().__init__(f"{message} (raw response recorded in evidence log)")


class EvidenceError(RigorError):
    """Raised when the evidence log cannot be written or read."""


class BaselineError(RigorError):
    """Raised when a baseline file is missing, malformed, or fails its own hash.

    A baseline whose contents no longer match its embedded digest is not a
    baseline -- it is an unsigned claim about what the system used to do.
    """


class StatisticalAssertionError(AssertionError, RigorError):
    """Base class for a failed statistical gate.

    Inherits :class:`AssertionError` so pytest reports it as an ordinary test
    failure, and carries the full statistical report on the instance so a caller
    that wants the numbers does not have to parse the message.

    The message is deliberately verbose. "assert 0.87 >= 0.9" tells a reader
    nothing about whether they sampled enough times to know that; the failure
    message of a statistical gate has to be the statistical report itself.
    """

    def __init__(self, message: str, **stats: object) -> None:
        self.stats = dict(stats)
        super().__init__(message)
