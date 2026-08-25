"""Expose the vendored COALA packages on ``sys.path``.

COALA's compression machinery (``coala.*`` and ``lib.Solvers``) uses absolute
imports such as ``import lib.Solvers`` and ``from coala.Injector import ...``, so
the directory that *contains* the ``coala/`` and ``lib/`` packages must be on
``sys.path``. We vendor those packages under ``vendor/`` (copied out of the
original COALA/ folder) so the pipeline runs standalone — the COALA/ and
Dobi-SVD/ reference folders are no longer required at runtime.

If a top-level ``COALA/``/``Dobi-SVD/`` folder is still present it is ignored;
the vendored copies under ``vendor/`` take precedence.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "vendor"
CANS_SVD_DIR = REPO_ROOT / "cans_svd"


def add_reference_paths() -> None:
    """Put the vendored package root (``vendor/``) on ``sys.path`` (idempotent).

    Also exposes the optional ``cans_svd/`` repo if present: its public package
    lives at ``cans_svd/src/cans_svd`` and internally imports the top-level
    ``src_torch``/``src_jax`` packages from the repo root, so both directories
    must be importable.
    """
    if not (VENDOR_DIR / "coala").exists():
        raise FileNotFoundError(
            f"Vendored COALA packages not found under {VENDOR_DIR}. "
            "Expected vendor/coala/ and vendor/lib/ next to the pipeline/ package."
        )
    entries = [str(VENDOR_DIR)]
    if CANS_SVD_DIR.exists():
        entries += [str(CANS_SVD_DIR), str(CANS_SVD_DIR / "src")]
    for entry in entries:
        if entry not in sys.path:
            sys.path.insert(0, entry)
