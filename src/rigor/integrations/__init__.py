"""Optional integrations. Core never imports anything from here.

The dependency runs one way only: an integration may import ``rigor``, and
nothing in ``rigor`` may import an integration. That is what keeps the assertions,
the judge, and the evidence log working when a vendor SDK changes underneath you
-- you lose a dashboard, not a test suite.

Nothing is re-exported at this level and no third-party package is imported at
module scope. Import the submodule you want::

    from rigor.integrations.opik import log_sample_to_opik

Each submodule imports its own dependency lazily, inside the functions that need
it, and raises a message naming the missing extra rather than an ImportError from
three frames down.
"""

from __future__ import annotations

__all__: list[str] = []
