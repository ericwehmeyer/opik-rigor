"""Pinned, rubric-hashed LLM-as-judge.

A judge is only useful as a measuring instrument if the instrument itself holds
still. Two things move underneath a naive judge and silently invalidate every
number it ever produced:

* **the model**, when the id is an alias such as ``claude-3-5-sonnet-latest``;
* **the rubric**, when someone edits the prompt file between runs.

:class:`PinnedJudge` refuses the first at construction time (via
:func:`rigor.pinning.require_pinned`) and detects the second by hashing the
rubric file and comparing it to the hash recorded in the evidence log. A change
is an error, not a warning -- scores either side of a rubric edit are not
comparable, and the only honest options are "don't edit it" or "say out loud
that you did" (``accept_rubric_change=True``, which records both hashes).

Parsing is deliberately strict. An unparseable judge response is *missing data*,
never a failing verdict: see :meth:`PinnedJudge.evaluate`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.base import Adapter
from .errors import JudgeOutputError, RubricDriftError
from .evidence import (
    EVENT_JUDGE_INIT,
    EVENT_JUDGE_PARSE_FAILURE,
    EVENT_JUDGE_VERDICT,
    EVENT_RUBRIC_CHANGE_ACCEPTED,
    EvidenceLog,
)
from .pinning import require_pinned

#: Inclusive bounds of the score the rubric asks for. A response outside them is
#: an error rather than something to clamp: a model that answers 9 on a 1-5 scale
#: did not understand the rubric, and silently rewriting it to 5 manufactures a
#: measurement nobody made.
SCORE_MIN = 1.0
SCORE_MAX = 5.0

#: Accepted spellings of the boolean verdict field. ``pass`` is what the prompt
#: asks for; ``passed`` is the variant models emit anyway. Accepting a known
#: spelling of the *same declared field* is not the same as inferring a verdict
#: from prose -- the value still has to be a JSON boolean.
PASS_KEYS = ("pass", "passed")

#: The last thing the judge model reads. ``rubrics/example-rubric.md`` ends with
#: this same text verbatim so that a rubric file is readable on its own, and so
#: that editing the expected format in one place is visibly a change to both.
OUTPUT_FORMAT_INSTRUCTION = (
    "Answer with a single JSON object and nothing else, in exactly this form:\n"
    "\n"
    '{"pass": true, "score": 4, "reason": "one sentence naming the deciding criterion"}\n'
    "\n"
    '- "pass" is required and must be the JSON boolean true or false, never a string.\n'
    f'- "score" must be a number from {SCORE_MIN:g} to {SCORE_MAX:g}, or null if the\n'
    "  rubric gives you no basis to score. Do not invent a number and do not answer\n"
    "  outside that range.\n"
    '- "reason" is one sentence.\n'
    "Do not wrap the object in commentary. If you are unsure, say so in the reason\n"
    "rather than answering in prose -- an unparseable answer is discarded, which is\n"
    "safer than a guess."
)

#: Assembled by :meth:`PinnedJudge.build_prompt`. Contains no literal braces, so
#: ``.format`` can safely substitute values that do (the JSON example above).
PROMPT_TEMPLATE = """You are grading one model response against a fixed rubric.

Apply only the rubric. Your own taste, the fluency of the writing, and anything
you happen to know about the subject are out of scope unless the rubric asks for
them. Grade the response as it is, not as it could be after an edit.

=== RUBRIC ===
{rubric}
=== END RUBRIC ===

=== INPUT GIVEN TO THE MODEL ===
{input}
=== END INPUT ===

=== MODEL OUTPUT UNDER EVALUATION ===
{output}
=== END MODEL OUTPUT ===

{output_format}
"""


@dataclass(frozen=True)
class Verdict:
    """One graded response.

    Frozen because a verdict is evidence: it is written to the log at the moment
    it is produced, and a caller that could mutate it afterwards would leave the
    in-memory object and the recorded fact disagreeing with no trace.
    """

    passed: bool
    score: float | None
    raw: str
    model_id: str = ""
    rubric_hash: str = ""
    reason: str | None = None


def hash_rubric_text(data: bytes) -> str:
    """sha256 of ``data`` with CRLF line endings normalised to LF.

    Without this the same rubric file hashes differently on a Windows checkout
    (``\\r\\n``) and a Linux CI runner (``\\n``), and since CI runs both, every
    cross-platform run would raise :class:`~rigor.errors.RubricDriftError` for a
    file nobody touched. Normalising the *bytes we hash* -- not the file -- keeps
    the identity of a rubric tied to its content rather than to git's autocrlf
    setting, while leaving the file on disk exactly as checked out.
    """
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def hash_rubric_file(path: str | os.PathLike[str]) -> str:
    """sha256 of the rubric at ``path``, newline-normalised.

    A missing rubric raises :class:`FileNotFoundError` unchanged: it is an
    ordinary filesystem mistake, not a judgement about a model, so it does not
    earn a rigor-specific exception type.
    """
    return hash_rubric_text(Path(path).read_bytes())


class PinnedJudge:
    """An LLM judge pinned to one model version and one rubric revision."""

    def __init__(
        self,
        adapter: Adapter,
        rubric_path: str | os.PathLike[str],
        evidence: EvidenceLog,
        *,
        name: str = "default",
        accept_rubric_change: bool = False,
    ) -> None:
        # 1. Pinning is checked first and at construction time. Discovering at
        #    analysis time that a week of verdicts came from an alias is a week
        #    of wasted compute; discovering it here costs nothing.
        self._model_id = require_pinned(adapter.model_id, context=f"judge {name!r}")
        self._adapter = adapter
        self._name = name
        self._evidence = evidence
        self._rubric_path = Path(rubric_path)
        self._rubric_text = self._rubric_path.read_text(encoding="utf-8")
        self._rubric_hash = hash_rubric_text(self._rubric_path.read_bytes())

        # 2. Compare against what this judge last recorded. Only this judge's own
        #    history matters, so the lookup filters on the name.
        previous = evidence.last(EVENT_JUDGE_INIT, judge=name)
        recorded_hash = previous.payload.get("rubric_hash") if previous is not None else None
        if recorded_hash is not None and recorded_hash != self._rubric_hash:
            if not accept_rubric_change:
                raise RubricDriftError(name, str(recorded_hash), self._rubric_hash)
            # The acceptance is itself evidence: a reader of the log must be able
            # to see where the scale changed, and which two revisions it spans.
            evidence.append(
                EVENT_RUBRIC_CHANGE_ACCEPTED,
                {
                    "judge": name,
                    "model_id": self._model_id,
                    "rubric_path": str(self._rubric_path),
                    "recorded_hash": recorded_hash,
                    "current_hash": self._rubric_hash,
                },
            )

        evidence.append(
            EVENT_JUDGE_INIT,
            {
                "judge": name,
                "model_id": self._model_id,
                "rubric_path": str(self._rubric_path),
                "rubric_hash": self._rubric_hash,
            },
        )

    # ----------------------------------------------------------------- #
    # identity
    # ----------------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def rubric_path(self) -> Path:
        return self._rubric_path

    @property
    def rubric_hash(self) -> str:
        return self._rubric_hash

    # ----------------------------------------------------------------- #
    # evaluation
    # ----------------------------------------------------------------- #

    def build_prompt(self, input: str, output: str) -> str:  # noqa: A002 - mirrors evaluate()
        """The exact prompt sent to the adapter, exposed so tests can pin it."""
        return PROMPT_TEMPLATE.format(
            rubric=self._rubric_text.strip(),
            input=input,
            output=output,
            output_format=OUTPUT_FORMAT_INSTRUCTION,
        )

    def evaluate(self, input: str, output: str) -> Verdict:  # noqa: A002 - the domain word
        """Grade ``output`` as a response to ``input`` and record the verdict.

        Exactly one ``judge.verdict`` record is written per successful call --
        never zero (an unrecorded verdict is not evidence) and never two (a
        double-counted verdict skews every rate computed from the log).

        A response that cannot be parsed raises
        :class:`~rigor.errors.JudgeOutputError` *after* the raw text is recorded,
        so the failure survives even though the call does not return. It is
        deliberately never turned into ``passed=False``: an unparseable answer is
        missing data, and folding missing data into the failure bucket biases the
        pass rate downwards by exactly the judge's own flakiness rate -- the one
        error term you most need to be able to see separately.
        """
        raw = self._adapter.complete(self.build_prompt(input, output))
        try:
            passed, score, reason = _parse_response(raw)
        except JudgeOutputError as exc:
            self._evidence.append(
                EVENT_JUDGE_PARSE_FAILURE,
                {
                    "judge": self._name,
                    "model_id": self._model_id,
                    "rubric_hash": self._rubric_hash,
                    "error": str(exc),
                    "raw": raw,
                },
            )
            raise

        verdict = Verdict(
            passed=passed,
            score=score,
            raw=raw,
            model_id=self._model_id,
            rubric_hash=self._rubric_hash,
            reason=reason,
        )
        self._evidence.append(
            EVENT_JUDGE_VERDICT,
            {
                "judge": self._name,
                "model_id": self._model_id,
                "rubric_hash": self._rubric_hash,
                "passed": verdict.passed,
                "score": verdict.score,
                "reason": verdict.reason,
                "input": input,
                "output": output,
                "raw": raw,
            },
        )
        return verdict


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Every top-level JSON object embedded in ``text``, in order.

    Scanning rather than regex-matching is what lets a fenced ```json block or an
    object surrounded by prose parse without a special case for either, while
    still refusing text that merely *sounds* like a verdict.
    """
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    index = 0
    while True:
        start = text.find("{", index)
        if start == -1:
            return found
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:
            index = start + 1
            continue
        index = end
        if isinstance(obj, dict):
            found.append(obj)


def _pass_key(obj: dict[str, Any]) -> str | None:
    for key in PASS_KEYS:
        if key in obj:
            return key
    return None


def _parse_response(raw: str) -> tuple[bool, float | None, str | None]:
    """Strictly parse a judge response into ``(passed, score, reason)``.

    Every rejection path raises rather than guessing. In particular there is no
    fallback that looks for "yes"/"pass" in prose: a judge that did not answer in
    the requested form did not answer, and a regex over its hedging invents a
    verdict that no model actually gave.
    """
    if not isinstance(raw, str) or not raw.strip():
        text = raw if isinstance(raw, str) else ""
        raise JudgeOutputError("judge returned an empty response", text)

    objects = _json_objects(raw)
    candidates = [obj for obj in objects if _pass_key(obj) is not None]
    if not candidates:
        detail = (
            "no JSON object in the response carried a 'pass' field"
            if objects
            else "response contained no JSON object"
        )
        raise JudgeOutputError(f"judge response is not a verdict: {detail}", raw)

    # Two verdict-shaped objects that disagree is genuine ambiguity: picking one
    # would be a coin flip recorded as a measurement.
    signatures = {
        json.dumps(
            {"pass": obj[_pass_key(obj)], "score": obj.get("score")},  # type: ignore[index]
            sort_keys=True,
            default=str,
        )
        for obj in candidates
    }
    if len(signatures) > 1:
        raise JudgeOutputError(
            f"judge response contained {len(signatures)} conflicting verdict objects", raw
        )

    obj = candidates[0]
    passed = obj[_pass_key(obj)]  # type: ignore[index]
    if not isinstance(passed, bool):
        raise JudgeOutputError(
            f"'pass' must be a JSON boolean, got {type(passed).__name__} {passed!r}", raw
        )

    raw_score = obj.get("score")
    score: float | None = None
    if raw_score is not None:
        # bool is an int subclass; true would otherwise read as the score 1.
        if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
            raise JudgeOutputError(
                f"'score' must be a number or null, got {type(raw_score).__name__} {raw_score!r}",
                raw,
            )
        score = float(raw_score)
        if not SCORE_MIN <= score <= SCORE_MAX:
            raise JudgeOutputError(
                f"'score' {score:g} is outside the rubric's range "
                f"{SCORE_MIN:g}-{SCORE_MAX:g}; it is not clamped because a score the "
                f"rubric cannot express means the judge misread the rubric",
                raw,
            )

    reason = obj.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)
    return passed, score, reason
