"""Run provenance shared by the point-source and cluster breakdown cells.

``source_revisions`` resolves each PyAuto* library's git checkout from the
module that was actually *imported*, not from a sibling-directory guess. The
earlier ``<autolens_profiling>/../PyAutoFit`` lookup holds on the laptop
workspace, but on RAL the libraries live at ``/mnt/ral/jnightin/PyAuto/<repo>``
while the profiling checkout (or its worktree) sits elsewhere, so the guess
raised ``CalledProcessError`` after every timing had been taken and no JSON was
written (autolens_profiling#297). Resolving from ``module.__file__`` records the
code that ran, and a missing checkout is recorded as a string, never raised.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

# Library repo name -> importable top-level package.
_LIBRARIES = (
    ("PyAutoNerves", "autonerves"),
    ("PyAutoFit", "autofit"),
    ("PyAutoArray", "autoarray"),
    ("PyAutoGalaxy", "autogalaxy"),
    ("PyAutoLens", "autolens"),
)


def git_revision(path: Path) -> str:
    """``git rev-parse HEAD`` at *path*, or an ``unavailable: ...`` string."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {type(exc).__name__}"


def source_revisions(profiling_root: Path) -> dict[str, str]:
    """Git SHAs of this checkout and of every imported PyAuto* library."""
    revisions = {"autolens_profiling": git_revision(profiling_root)}
    for repo, package in _LIBRARIES:
        try:
            module = importlib.import_module(package)
        except ImportError:
            revisions[repo] = "unavailable: not importable"
            continue
        # <repo>/<package>/__init__.py -> <repo>
        revisions[repo] = git_revision(Path(module.__file__).resolve().parent.parent)
    return revisions


def thread_environment() -> dict[str, str | None]:
    """The thread-pinning variables a CPU timing depends on."""
    return {
        name: os.environ.get(name)
        for name in ("NPROC", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
