"""Runnable examples, shipped **inside the wheel** rather than only in the repository.

``python -m opik_rigor.examples.summarise_eval`` works from a bare
``pip install opik-rigor``, with no checkout, no network and no credentials.

That is the whole reason this subpackage exists. The README's worked example used
to be addressed as ``python examples/summarise_eval.py``, which is a path in the
git tree and not in the artifact: the published wheel contains ``opik_rigor/`` and
``opik_rigor-<version>.dist-info/`` and nothing else, so the one command the
quickstart ended on could not be run by anybody who had followed the install
instructions directly above it. The same fault, in the same document, had already
been fixed once for the example rubric -- which is why
:func:`opik_rigor.example_rubric_path` exists and why this module is here rather
than in a sibling directory.

Nothing in the core library imports this subpackage, and this subpackage imports
no integration at module scope: ``opik_rigor.integrations.opik`` is reached only
inside the ``--opik`` branch of the example, so the module can be imported and run
on an install that has neither the ``opik`` package nor a server to send to.
"""

from __future__ import annotations

__all__: list[str] = []
