"""rigor -- statistical assertions and pinned-judge evaluation for LLM test suites.

Two primitives:

* :class:`~rigor.judge.PinnedJudge` -- an LLM-as-judge that refuses an aliased
  model id, hashes its rubric, and raises when the rubric changes underneath a
  recorded baseline.
* the statistical gates in ``rigor.distribution`` (Session 2) -- assertions that
  account for having sampled a stochastic system n times rather than measured it
  once.

Both write to an append-only :class:`~rigor.evidence.EvidenceLog` that has no
delete API.

Nothing here imports an integration or a provider SDK at module scope: importing
``rigor`` must work with no credentials, no network, and neither the ``anthropic``
nor ``openai`` package installed. Integrations live under ``rigor.integrations``
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
from .judge import PinnedJudge, Verdict
from .pinning import is_pinned, require_pinned
from .sampling import Run, SampleResult, SampleTimeout, sample, sample_of

__version__ = "0.1.0.dev0"

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
    "SampleResult",
    "SampleTimeout",
    "ScoreDistributionError",
    "StatisticalAssertionError",
    "Verdict",
    "__version__",
    "assert_no_regression",
    "assert_pass_rate",
    "assert_score_distribution",
    "is_pinned",
    "require_pinned",
    "sample",
    "sample_of",
    "wilson_interval",
    "wilson_lower_bound",
]
