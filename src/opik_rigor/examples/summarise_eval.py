"""End-to-end evaluation of a deliberately imperfect summariser, offline.

Run it::

    python -m opik_rigor.examples.summarise_eval --seed 7 --n 40

That address works from a bare ``pip install opik-rigor``, which is the point: the
module ships inside the wheel. It used to be ``python examples/summarise_eval.py``,
a path that exists in the git tree and in no installation.

Nothing here touches the network. The "model under test" is a plain Python
function and the judge is backed by :class:`opik_rigor.FakeAdapter`, so the whole
story below is reproducible byte for byte from the seed -- which is the same
discipline opik_rigor asks of a real suite, for the same reason: a gate you cannot
reproduce is a gate you cannot argue with.

The story it tells, in order:

1. a summariser that is *genuinely* imperfect -- it sometimes drops the caveat
   the source document stated, which is the failure a reader would actually be
   hurt by;
2. a judge pinned to one model id and one rubric revision, so the measuring
   instrument holds still while the thing being measured changes;
3. n samples, because one call to a stochastic system is an anecdote;
4. the statistical gates, showing both ways they can fail -- "you missed the
   bar" and "you did not sample enough to tell", which call for opposite
   responses;
5. a recorded baseline, then a deliberately degraded summariser, and the
   regression gate catching it;
6. the evidence log that all of the above wrote itself into.

With ``--opik`` it additionally pushes the sample and the gate result to an
Opik instance. That step -- and only that step -- needs software and a server
this script cannot conjure, so it is best-effort: it explains what is missing
and leaves the exit code alone.
"""

from __future__ import annotations

import argparse
import itertools
import random
import shutil
import sys
import textwrap
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opik_rigor import (
    Baseline,
    EvidenceLog,
    FakeAdapter,
    ModelPinError,
    PassRateError,
    PinnedJudge,
    RegressionError,
    SampleResult,
    ScoreDistributionError,
    Verdict,
    assert_no_regression,
    assert_pass_rate,
    assert_score_distribution,
    example_rubric_path,
    sample,
)

#: Default output directory, resolved against the *caller's* working directory.
#: It was ``Path(__file__).parent / ".rigor-run"`` while this script lived in the
#: repository's ``examples/``; now that it ships inside the wheel, that expression
#: would write into site-packages -- a directory the reader does not own, may not
#: be able to write to, and would never think to look in.
DEFAULT_OUT = Path(".rigor-run")

#: The rubric ships inside the package, so this example points at the same file a
#: reader gets from `pip install opik-rigor` rather than at a repository path they
#: would not have.
RUBRIC = example_rubric_path()

WIDTH = 96


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Document:
    """One source document, decomposed into the sentences that carry its weight.

    Real corpora are prose. This one is pre-split because the summariser below is
    *extractive* -- it copies sentences rather than writing them -- so the only
    interesting variable is which sentences survive. That is on purpose: omission
    is the failure mode the rubric's second criterion is about, and it is the one
    a summariser fails at quietly.
    """

    doc_id: str
    outcome: str
    reason: str
    caveat: str
    decoration: str

    @property
    def body(self) -> str:
        """The source as the judge sees it. ``decoration`` is correctly droppable."""
        return " ".join((self.outcome, self.reason, self.caveat, self.decoration))


CORPUS: tuple[Document, ...] = (
    Document(
        doc_id="memo-billing",
        outcome="The board approved moving the billing platform to Northwind.",
        reason="The deciding factor was a 40 percent lower five-year cost of ownership.",
        caveat=(
            "That saving assumes current transaction volumes; at the volumes forecast "
            "for 2027 it falls to 12 percent."
        ),
        decoration="The vendor review closed on 14 May and ran for eleven weeks.",
    ),
    Document(
        doc_id="incident-4471",
        outcome="Reconciliation will move to the new scheduler in the next release.",
        reason="The old scheduler swallowed non-zero exit codes, so six nights of failure went "
        "unreported.",
        caveat="The fix does not cover the manual month-end run, which still exits silently.",
        decoration="The incident was raised by a customer, not by monitoring.",
    ),
    Document(
        doc_id="eval-rerank",
        outcome="The team recommends adopting the retrieval rerank step in production.",
        reason="It lifted answer accuracy from 71 percent to 84 percent on the internal benchmark.",
        caveat=(
            "The benchmark holds 240 questions and the gain is reported without a "
            "confidence interval."
        ),
        decoration="The work took one engineer nine days, mostly spent on data loading.",
    ),
    Document(
        doc_id="deprecation-v1",
        outcome="Support for the v1 export API ends on 31 March 2027.",
        reason="Fewer than 2 percent of accounts still call it, and it blocks the storage "
        "migration.",
        caveat="Three enterprise contracts guarantee the endpoint until 2028 and must be "
        "renegotiated first.",
        decoration="The v2 endpoint has been generally available since 2024.",
    ),
    Document(
        doc_id="panel-staff-eng",
        outcome="The panel recommends hiring the candidate for the staff engineer role.",
        reason="She was the only candidate to reason correctly about the failure mode in the "
        "system design exercise.",
        caveat="Two of the five panellists missed the design session and abstained.",
        decoration="The loop ran over two days and included a written exercise.",
    ),
    Document(
        doc_id="finance-q3",
        outcome="Q3 revenue came in at 4.2 million dollars, ahead of the 3.9 million plan.",
        reason="A single enterprise renewal that had been forecast for Q4 closed early.",
        caveat="Pulling that renewal forward leaves Q4 roughly 600 thousand dollars short of plan.",
        decoration="Headcount was flat across the quarter.",
    ),
)


# --------------------------------------------------------------------------- #
# the system under test
# --------------------------------------------------------------------------- #

#: How often the healthy summariser drops the source's stated caveat. Not zero:
#: a system under test that never fails makes every gate below unfalsifiable.
CAVEAT_DROP_HEALTHY = 0.15

#: How often it drops the main stated reason. A secondary omission -- annoying,
#: not dangerous -- which is why the judge scores it separately below.
REASON_DROP = 0.15

#: The regression. Same code, same corpus, one constant changed, which is what
#: most real regressions look like from the outside.
CAVEAT_DROP_DEGRADED = 0.75


@dataclass
class Summariser:
    """A toy extractive summariser whose defect is omission.

    The randomness lives in an injected :class:`random.Random` rather than the
    module-level ``random``, so two summarisers in the same process cannot pull
    each other's draws around -- the same reason ``FakeAdapter`` keeps a private
    RNG.
    """

    rng: random.Random
    caveat_drop: float
    reason_drop: float = REASON_DROP

    def __call__(self, doc: Document) -> str:
        # Both draws are always taken, in a fixed order, even when the second is
        # not consulted. Otherwise the healthy and degraded runs would consume
        # different numbers of draws and would be comparing different streams as
        # well as different systems.
        drop_caveat = self.rng.random() < self.caveat_drop
        drop_reason = self.rng.random() < self.reason_drop
        kept = [doc.outcome]
        if not drop_reason:
            kept.append(doc.reason)
        if not drop_caveat:
            kept.append(doc.caveat)
        return " ".join(kept)


# --------------------------------------------------------------------------- #
# the judge
# --------------------------------------------------------------------------- #

TIER_COMPLETE = "complete"
TIER_REASON_MISSING = "reason-missing"
TIER_CAVEAT_MISSING = "caveat-missing"

#: What the fake judge answers for each tier of summary, drawn at random within
#: the tier. The verdicts are scripted per tier rather than globally so that the
#: judge's answers track the summariser's *actual* behaviour: a judge whose
#: verdicts are noise would make every number below a decoration.
#:
#: Each tier still carries disagreement, because real judges have some. Note in
#: particular the last entry under ``caveat-missing``: the judge sometimes fails
#: to notice the omission. That is not a bug in the fixture. It is the reason you
#: sample a judge instead of calling it once.
VERDICTS: dict[str, tuple[str, ...]] = {
    TIER_COMPLETE: (
        '{"pass": true, "score": 5, "reason": "faithful, complete on every load-bearing '
        'point, and tight"}',
        '{"pass": true, "score": 5, "reason": "outcome, reason and caveat all survive intact"}',
        '{"pass": true, "score": 5, "reason": "a reader could act on this without the source"}',
        '{"pass": true, "score": 4, "reason": "faithful and complete; wording looser than needed"}',
    ),
    TIER_REASON_MISSING: (
        '{"pass": true, "score": 4, "reason": "the stated reason is missing but nothing '
        'load-bearing is"}',
        '{"pass": true, "score": 4, "reason": "safe to act on; a secondary point is absent"}',
        '{"pass": false, "score": 3, "reason": "without the stated reason a reader cannot '
        'weigh the decision"}',
    ),
    TIER_CAVEAT_MISSING: (
        '{"pass": false, "score": 2, "reason": "the stated caveat is omitted, so acting on '
        'this summary is risky"}',
        '{"pass": false, "score": 2, "reason": "a load-bearing limit in the source does not '
        'appear at all"}',
        '{"pass": false, "score": 2, "reason": "omits the caveat that bounds the recommendation"}',
        '{"pass": false, "score": 3, "reason": "the caveat is gone and the attribution is thin"}',
        '{"pass": true, "score": 4, "reason": "reads as faithful and complete"}',
    ),
}


def classify(summary: str) -> str:
    """Which tier a summary falls into, by looking at what it kept.

    A real judge reads. This one matches sentences, which is the whole of its
    cheat: it is standing in for the *correlation* between a summary's content
    and a grader's verdict, not for the grading.
    """
    for doc in CORPUS:
        if doc.outcome in summary:
            if doc.caveat not in summary:
                return TIER_CAVEAT_MISSING
            return TIER_COMPLETE if doc.reason in summary else TIER_REASON_MISSING
    raise ValueError(f"summary matches no document in the corpus: {summary!r}")


#: Delimiters from opik_rigor's own judge prompt. The fake needs only the part of the
#: prompt under evaluation -- the source document is quoted higher up, and every
#: caveat appears there whether or not the summary kept it.
OUTPUT_OPEN = "=== MODEL OUTPUT UNDER EVALUATION ==="
OUTPUT_CLOSE = "=== END MODEL OUTPUT ==="


def extract_summary(prompt: str) -> str:
    head = prompt.split(OUTPUT_OPEN, 1)[-1]
    return head.split(OUTPUT_CLOSE, 1)[0].strip()


def scripted_judge(rng: random.Random) -> Callable[[str], str]:
    """A judge response function: reads the prompt, answers in the rubric's JSON."""

    def respond(prompt: str) -> str:
        return rng.choice(VERDICTS[classify(extract_summary(prompt))])

    return respond


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #


def rule(title: str) -> None:
    print()
    print(f"-- {title} ".ljust(WIDTH, "-"))


def show(label: str, value: object) -> None:
    print(f"  {label:<22}{value}".rstrip())


def paragraph(text: str, indent: str = "  ") -> None:
    for line in textwrap.wrap(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent):
        print(line)


def show_gate(report: dict[str, Any]) -> None:
    """The one line that matters, in the same shape for every gate."""
    show("gate", "PASS" if report["passed"] else "FAIL")


def show_pass_rate(report: dict[str, Any]) -> None:
    show("observed", f"{report['successes']}/{report['n']} = {report['pass_rate']:.4f}")
    show("bar (min_rate)", f"{report['min_rate']:.4f}")
    show(
        f"{report['confidence']:.0%} lower bound",
        f"{report['lower_bound']:.4f}   <- what the gate compares, not the observed rate",
    )
    show(
        f"{report['confidence']:.0%} interval",
        f"[{report['interval_lower']:.4f}, {report['interval_upper']:.4f}]",
    )
    show_gate(report)


def show_distribution(report: dict[str, Any]) -> None:
    stddev = "n/a" if report["stddev"] is None else f"{report['stddev']:.4f}"
    show("scores", report["n"])
    show("mean", f"{report['mean']:.4f}  (bar {report['min_mean']})")
    show("p10", f"{report['p10']:.4f}  (bar {report['min_p10']})")
    show("stddev (ddof=1)", f"{stddev}  (bar {report['max_stddev']})")
    show("range", f"[{report['min_score']:.1f}, {report['max_score']:.1f}]")
    show_gate(report)


def show_regression(report: dict[str, Any]) -> None:
    show("current", f"n={report['n_current']}, median={report['median_current']:.4f}")
    show("baseline", f"n={report['n_baseline']}, median={report['median_baseline']:.4f}")
    show("Mann-Whitney U", f"{report['u_statistic']:.1f}")
    show("p-value", f"{report['p_value']:.6g}  (alpha {report['alpha']:g})")
    show_gate(report)


def show_failure(exc: Exception) -> None:
    """Print a gate's failure message in full.

    The message *is* the statistical report -- it names the numbers, and it says
    which of the two failure modes occurred -- so truncating it would throw away
    the thing the library exists to produce.
    """
    paragraph(str(exc), indent="  ")


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def evaluate_corpus(
    judge: PinnedJudge,
    summariser: Summariser,
    n: int,
    log: EvidenceLog,
    label: str,
) -> SampleResult:
    """Sample the judge over the corpus, cycling documents in a fixed order.

    ``sample`` calls a zero-argument function, so which document a run sees is
    the caller's business. Cycling (rather than drawing at random) means the
    healthy and degraded runs below see the same documents in the same order, so
    the comparison between them is about the summariser and not about which
    memos each happened to draw.
    """
    documents = itertools.cycle(CORPUS)

    def run_once() -> Verdict:
        doc = next(documents)
        return judge.evaluate(doc.body, summariser(doc))

    return sample(run_once, n, evidence=log, label=label)


def print_header(args: argparse.Namespace, judge: PinnedJudge, log: EvidenceLog) -> None:
    rule("0. the setup")
    show("seed", args.seed)
    show("n per run", args.n)
    show("corpus", f"{len(CORPUS)} source documents")
    show("judge model", f"{judge.model_id}  (pinned: no alias, ends in a release designator)")
    show("rubric", RUBRIC)
    show("rubric sha256", judge.rubric_hash)
    show("evidence log", log.path)
    print()
    paragraph(
        "The judge is pinned to one model id and one rubric revision. Change either and opik_rigor "
        "raises rather than quietly comparing scores across two different instruments -- which "
        "is what the hash above is for."
    )
    try:
        # Constructed and thrown away: the point is that it never gets as far as
        # grading anything. The refusal happens at construction, not at analysis
        # time, because by analysis time a week of verdicts is already spoiled.
        aliased = FakeAdapter(model_id="fake-judge-latest", responses=["{}"])
        PinnedJudge(aliased, RUBRIC, log, name="aliased")
    except ModelPinError as exc:
        print()
        print("  an alias is refused, at construction time:")
        paragraph(str(exc), indent="    ")


def print_system_under_test() -> None:
    rule("1. the system under test")
    doc = CORPUS[0]
    paragraph(
        "A plain function, no model involved. It keeps the outcome always, the stated reason "
        f"{1 - REASON_DROP:.0%} of the time, and the stated caveat "
        f"{1 - CAVEAT_DROP_HEALTHY:.0%} of the time. Dropping the caveat is the failure that "
        "matters: a reader who acts on the summary alone is ambushed by the thing the source "
        "warned them about."
    )
    print()
    show("source document", doc.doc_id)
    paragraph(doc.body, indent="    ")
    print()
    print("  a good summary, keeping every load-bearing point:")
    paragraph(" ".join((doc.outcome, doc.reason, doc.caveat)), indent="    ")
    print()
    print("  the same summariser on the same document, having dropped the caveat:")
    paragraph(" ".join((doc.outcome, doc.reason)), indent="    ")


def print_sample(result: SampleResult, title: str) -> None:
    rule(title)
    show("runs", result.n)
    show("passed", result.successes)
    show("failed", result.failures)
    show("exceptions", f"{len(result.exceptions)}  (a system that did not run, counted separately)")
    show("observed pass rate", f"{result.pass_rate:.4f}   <- a point estimate; never gate on it")


def print_evidence(log: EvidenceLog, tail: int = 5) -> None:
    rule("7. the audit trail")
    records = log.read()
    tally = Counter(record.event_type for record in records)
    show("evidence log", log.path)
    show("records", len(records))
    for event_type, count in sorted(tally.items()):
        show(f"  {event_type}", count)
    print()
    paragraph(
        "One JSON object per line, appended and never rewritten -- there is no delete API. "
        "Each line carries a schema version, a UTC timestamp, an event type, and the payload "
        f"below. The last {tail} records of this run, timestamps elided so that the same seed "
        "prints the same bytes:"
    )
    print()
    for record in records[-tail:]:
        paragraph(f"{record.event_type}  {summarise_payload(record.event_type, record.payload)}")


def summarise_payload(event_type: str, payload: Any) -> str:
    """A few load-bearing fields per event type, so the tail stays readable."""
    keys: Sequence[str]
    if event_type == "judge.verdict":
        keys = ("passed", "score", "reason")
    elif event_type == "sample.completed":
        keys = ("label", "n", "successes", "failures", "exceptions")
    elif event_type == "assertion.evaluated":
        keys = ("gate", "label", "passed", "n")
    else:
        keys = tuple(payload)
    parts = []
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        text = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{key}={text}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# the optional Opik leg
# --------------------------------------------------------------------------- #

#: Named rather than left to the client's default, so the example's traces land
#: somewhere obvious instead of in whatever project the machine happens to have
#: configured -- and so that Opik's "no project name configured" warning is not
#: the first thing a reader sees.
OPIK_PROJECT = "opik_rigor-example"

OPIK_HELP = """\
Everything above is finished and unaffected. Sending the same run to Opik needs
two things this script cannot provide for you:

  1. The client library.
       pip install "opik-rigor[opik]"        (or simply: pip install opik)

  2. Somewhere to send it. Either a local instance, per Opik's own docs --
       git clone https://github.com/comet-ml/opik
       cd opik/deployment/docker-compose && docker compose up -d
       # then open http://localhost:5173
     -- or Opik Cloud, pointed at with: opik configure

Exiting 0 regardless. The evaluation succeeded; only the dashboard write did
not.\
"""


def explain_opik(detail: str) -> None:
    print("  the Opik leg did not complete:")
    paragraph(detail, indent="    ")
    print()
    for line in OPIK_HELP.splitlines():
        print(f"  {line}".rstrip())


def log_to_opik(
    result: SampleResult,
    report: dict[str, Any],
    *,
    judge: PinnedJudge,
    args: argparse.Namespace,
) -> None:
    """Best-effort mirror of the run into Opik. Never fails the script.

    The import is inside this function on purpose, and so is every call: with
    ``--opik`` off -- the default -- ``opik_rigor.integrations.opik`` is never
    touched, so the example runs on an install that has neither the ``opik``
    package nor a server to talk to.
    """
    rule("8. mirroring the run into Opik (--opik)")
    paragraph(
        "This is the only part of the script that leaves the process. It is also the only "
        "part allowed to fail without failing the run: opik_rigor's core never imports an "
        "integration, so a dashboard that is down costs you a dashboard, not a test suite."
    )
    print()
    try:
        from opik_rigor.integrations import opik as opik_integration
    except Exception as exc:  # noqa: BLE001 - a missing optional integration is not a failure
        explain_opik(f"{type(exc).__name__}: {exc}")
        return

    try:
        trace_id = opik_integration.log_sample_to_opik(
            result,
            name="summariser-healthy",
            judge_name=judge.name,
            project_name=OPIK_PROJECT,
            tags=("example",),
            metadata={"seed": args.seed, "rubric_hash": judge.rubric_hash},
        )
        opik_integration.log_assertion_to_opik(
            report, trace_id=trace_id, project_name=OPIK_PROJECT
        )
    except Exception as exc:  # noqa: BLE001 - the whole point is that this cannot fail the run
        explain_opik(f"{type(exc).__name__}: {exc}")
        return

    show("project", OPIK_PROJECT)
    show("sample", f"handed to Opik as trace {trace_id}")
    show("pass-rate gate", f"{result.n} spans, plus feedback scores on that trace")
    print()
    paragraph(
        "Handed to, not confirmed delivered. Opik's client batches in a background thread and "
        "does not raise when the destination is unreachable, so the trace id above is proof "
        "that opik_rigor built the trace, not that a server accepted it. If OPIK: warnings "
        "appeared on stderr, or the trace is not in your project, that is the write failing "
        "quietly -- "
        "which is exactly why the gates above do not depend on it."
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a deliberately imperfect summariser with opik_rigor: pinned judge, "
            "sampled n times, gated statistically, recorded as evidence. Offline by default."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for the summariser and the judge. Same seed, same output, byte for byte.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=40,
        help="Judge calls per run. Two runs are made: healthy and degraded.",
    )
    parser.add_argument(
        "--opik",
        action="store_true",
        help="Additionally mirror the run into Opik. Needs the opik package and a "
        "reachable instance; never fails the script if either is missing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where the evidence log and the baseline are written, relative to the "
        "current directory. Cleared on each run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n < 2:
        print("--n must be at least 2: a distribution over one observation is not one")
        return 2

    # A fresh directory each run. The log is append-only by design, so reusing a
    # directory would show you this run's records mixed with the last one's.
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)
    log = EvidenceLog(args.out / "evidence.jsonl")

    # One judge for both runs. The instrument is held fixed while the system
    # under test changes -- comparing two runs graded by two judges measures the
    # judges as much as the summaries.
    adapter = FakeAdapter(
        model_id="fake-judge-v1",
        responses=scripted_judge(random.Random(args.seed + 1)),
    )
    judge = PinnedJudge(adapter, RUBRIC, log, name="summariser")

    print("opik_rigor -- an end-to-end summarisation eval, entirely offline")
    print_header(args, judge, log)
    print_system_under_test()

    # ---- the healthy run ------------------------------------------------- #

    healthy = Summariser(random.Random(args.seed), caveat_drop=CAVEAT_DROP_HEALTHY)
    result = evaluate_corpus(judge, healthy, args.n, log, "summariser-healthy")
    print_sample(result, "2. sampling the judge over the corpus")

    rule("3. the pass-rate gate")
    paragraph(
        "The gate compares the one-sided Wilson lower bound against the bar, never the "
        "observed rate. 18/20 and 900/1000 are both 90 percent, and only one of them is "
        "evidence."
    )
    print()
    try:
        pass_report = assert_pass_rate(
            result, 0.60, evidence=log, label="summariser-healthy"
        )
    except PassRateError as exc:
        show_failure(exc)
        print()
        paragraph(
            f"The headline gate did not clear at --seed {args.seed} --n {args.n}, so the rest "
            "of the walkthrough would be recording a baseline from a run that never met the "
            "bar. If the message above says the sample was underpowered, raise --n and try "
            "again -- the default of 40 clears it."
        )
        return 1
    show_pass_rate(pass_report)

    rule("4. the same sample, against a bar it cannot defend")
    strict_bar = round(result.pass_rate - 0.025, 4)
    paragraph(
        f"Nothing about the system has changed. The bar has moved to {strict_bar:.4f}, which "
        "sits just below the observed rate -- so a plain `assert pass_rate >= bar` would pass "
        "this run without comment. It is the second failure mode, and it reads differently "
        "from the first: not 'you missed the bar' but 'you did not sample enough to tell'."
    )
    print()
    try:
        assert_pass_rate(result, strict_bar, evidence=log, label="summariser-healthy-strict")
    except PassRateError as exc:
        show_failure(exc)
        print()
        show("underpowered", exc.stats["underpowered"])
        show("runs needed", exc.stats["runs_needed"])

    rule("5. the score-distribution gate")
    paragraph(
        "A mean gate alone passes a system that is excellent four times in five and unusable "
        "the fifth time, which is exactly the failure users notice. p10 bounds the bad tail "
        "and stddev bounds the swing."
    )
    print()
    dist_report = assert_score_distribution(
        result,
        min_mean=3.5,
        min_p10=1.5,
        max_stddev=1.75,
        evidence=log,
        label="summariser-healthy",
    )
    show_distribution(dist_report)

    # ---- record the baseline --------------------------------------------- #

    rule("6. recording a baseline, then regressing against it")
    baseline_path = Baseline.from_sample(
        "summariser",
        result,
        model_id=judge.model_id,
        rubric_hash=judge.rubric_hash,
        metadata={"seed": args.seed, "n": args.n, "caveat_drop": CAVEAT_DROP_HEALTHY},
    ).save(args.out / "baseline.json")
    show("baseline written", baseline_path)
    paragraph(
        "It carries a sha256 of its own contents, the judge's model id, and the rubric hash. "
        "Editing the file to make a regression go away is refused on load: a comparison "
        "against a baseline you cannot vouch for is a formality, not evidence."
    )

    # ---- the degraded run ------------------------------------------------- #

    print()
    paragraph(
        "... time passes, and someone changes the summariser so that it drops the stated "
        f"caveat {CAVEAT_DROP_DEGRADED:.0%} of the time instead of "
        f"{CAVEAT_DROP_HEALTHY:.0%}. Same judge, same rubric, same documents in the same "
        "order. Only the system under test moved."
    )
    degraded = Summariser(random.Random(args.seed + 2), caveat_drop=CAVEAT_DROP_DEGRADED)
    after = evaluate_corpus(judge, degraded, args.n, log, "summariser-degraded")
    print_sample(after, "6a. the degraded run, sampled identically")

    rule("6b. the same pass-rate gate, on the degraded run")
    try:
        assert_pass_rate(after, 0.60, evidence=log, label="summariser-degraded")
    except PassRateError as exc:
        show_failure(exc)
        print()
        show("underpowered", f"{exc.stats['underpowered']}  <- contrast with section 4")

    rule("6c. the score-distribution gate, on the degraded run")
    try:
        assert_score_distribution(
            after,
            min_mean=3.5,
            min_p10=1.5,
            max_stddev=1.75,
            evidence=log,
            label="summariser-degraded",
        )
    except ScoreDistributionError as exc:
        show_failure(exc)

    rule("6d. the regression gate, against the recorded baseline")
    recorded = Baseline.load(baseline_path)
    show("baseline verified", f"{recorded.n} outcomes, {len(recorded.scores)} scores, digest ok")
    print()
    try:
        assert_no_regression(
            after.scores(), recorded.scores, evidence=log, label="summariser-nightly"
        )
    except RegressionError as exc:
        show_failure(exc)
        print()
        show_regression(exc.stats)

    print_evidence(log)

    if args.opik:
        log_to_opik(result, pass_report, judge=judge, args=args)

    rule("done")
    paragraph(
        "Every number above was produced inside this process, with no network and no "
        "credentials -- only the optional Opik leg leaves it. Rerun with --seed "
        f"{args.seed} and the output is identical byte for byte; change the seed and the "
        "numbers move, which is the honest thing for a measurement of a stochastic system "
        "to do."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
