"""Deriving a reproducible generator from the identity of a piece of work.

Both the perturbation sweep and the sensitivity mapper need a random generator per *cell* —
per ``(perturbation, severity, variant)``, or per ``(feature, time-group, variant)`` — so
that a cell's draw depends only on which cell it is and not on how many cells ran before
it. That is what lets a single severity be re-run in isolation and give the numbers it had
inside a full sweep, which repo B's global seeding could not do.

The obvious implementation is wrong, and quietly so. ``hash((seed, name, severity))``
appears to work and passes any test that runs both calls in one process — but Python
randomises string hashing per process unless ``PYTHONHASHSEED`` is set, so the same seed
and the same cell produce different numbers on the next run. A sweep is then not
reproducible across runs at all, which is the entire point of recording a seed in the
result. This module existed as two lines of ``hash()`` in an earlier revision and the
defect surfaced only as an intermittently failing test.

:func:`derive_generator` uses BLAKE2b over the cell's string form instead: stable across
processes, across platforms, and across Python versions.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

__all__ = ["derive_seed", "derive_generator"]


def derive_seed(seed: int, *parts: Any) -> int:
    """A stable 64-bit seed from a base seed and the identity of one piece of work.

    Deterministic across processes, which ``hash()`` is not. Order matters, and the parts
    are separated by a byte that cannot occur in their string forms, so
    ``("ab", "c")`` and ``("a", "bc")`` derive different seeds rather than colliding.
    """
    payload = b"\x00".join(str(part).encode("utf-8") for part in (seed, *parts))
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def derive_generator(seed: int, *parts: Any) -> np.random.Generator:
    """A generator whose stream depends only on ``seed`` and the given identity."""
    return np.random.default_rng(np.random.SeedSequence(derive_seed(seed, *parts)))
