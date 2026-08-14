"""opik_rigor -- statistical assertions and pinned-judge evaluation for LLM test suites.

Two primitives:

* :class:`~opik_rigor.judge.PinnedJudge` -- an LLM-as-judge that refuses an aliased
  model id, hashes its rubric, and raises when the rubric changes underneath a
  recorded baseline.
* the statistical gates in ``opik_rigor.distribution`` (Session 2) -- assertions that
  account for having sampled a stochastic system n times rather than measured it
  once.

Both write to an append-only :class:`~opik_rigor.evidence.EvidenceLog` that has no
delete API.

The judge's score range (:data:`~opik_rigor.judge.SCORE_MIN`,
:data:`~opik_rigor.judge.SCORE_MAX`) and its rubric hashers
(:func:`~opik_rigor.judge.hash_rubric_file`,
:func:`~opik_rigor.judge.hash_rubric_text`) are re-exported here as well. They are
not decoration: a consumer that imputes a score for an ungradeable response has to
know where the bottom of the scale is, and one that hashes a judge config has to
hash the rubric exactly as rigor does or the two will disagree about whether the
instrument changed. Re-deriving either in a consumer -- a hard-coded ``1.0``, a
hand-rolled sha256 -- is the drift this library exists to catch, so the names are
public rather than left to be scavenged from ``opik_rigor.judge``.

Nothing here imports an integration or a provider SDK at module scope: importing
``opik_rigor`` must work with no credentials, no network, and neither the ``anthropic``
nor ``openai`` package installed. Integrations live under ``opik_rigor.integrations``
and import core, never the other way round.
"""

from __future__ import annotations

from .adapters import (
    Adapter,
    AdapterError,
    AnthropicAdapter,
    FakeAdapter,
    OpenAICompatAdapter,
)
from .baseline import Baseline
from .distribution import (
    PassRateError,
    RegressionError,
    ScoreDistributionError,
    assert_no_regression,
    assert_pass_rate,
    assert_score_distribution,
    wilson_interval,
    wilson_lower_bound,
)
from .errors import (
    BaselineError,
    EvidenceError,
    JudgeOutputError,
    ModelPinError,
    RigorError,
    RubricDriftError,
    StatisticalAssertionError,
)
from .evidence import EvidenceLog, EvidenceRecord
from .judge import (
    SCORE_MAX,
    SCORE_MIN,
    PinnedJudge,
    Verdict,
    example_rubric_path,
    hash_rubric_file,
    hash_rubric_text,
)
from .pinning import is_pinned, require_pinned
from .sampling import Run, SampleResult, SampleTimeout, sample, sample_of

__version__ = "0.1.0"

__all__ = [
    "Adapter",
    "AdapterError",
    "AnthropicAdapter",
    "Baseline",
    "BaselineError",
    "EvidenceError",
    "EvidenceLog",
    "EvidenceRecord",
    "FakeAdapter",
    "JudgeOutputError",
    "ModelPinError",
    "OpenAICompatAdapter",
    "PassRateError",
    "PinnedJudge",
    "RegressionError",
    "RigorError",
    "RubricDriftError",
    "Run",
    "SCORE_MAX",
    "SCORE_MIN",
    "SampleResult",
    "SampleTimeout",
    "ScoreDistributionError",
    "StatisticalAssertionError",
    "Verdict",
    "__version__",
    "assert_no_regression",
    "assert_pass_rate",
    "assert_score_distribution",
    "example_rubric_path",
    "hash_rubric_file",
    "hash_rubric_text",
    "is_pinned",
    "require_pinned",
    "sample",
    "sample_of",
    "wilson_interval",
    "wilson_lower_bound",
]
