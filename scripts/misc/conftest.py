"""Shared pytest setup for the ``scripts/misc/<group>/test/`` dirs.

Tests import the misc groups as top-level packages (``from hazards import ...``,
``from likelihood_breakdown import ...``), so ``scripts/misc`` must be on
``sys.path`` whichever test dir pytest is collecting. The repo-root
``pytest.ini`` makes pytest load this file even when a single test file is run.
"""

import sys
from pathlib import Path

_MISC = str(Path(__file__).resolve().parent)
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)
