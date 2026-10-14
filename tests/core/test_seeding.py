"""Tests for cell-derived seeding.

The property that matters is one a same-process test cannot check: the same seed and the
same cell must give the same numbers *on the next run*. An earlier revision derived cell
seeds with ``hash()``, which randomises string hashing per process, so a recorded seed did
not reproduce a sweep — and the only symptom was an intermittently failing assertion
elsewhere. The cross-process test below is therefore run in a subprocess.
"""

from __future__ import annotations

import textwrap

import pytest

from tests.conftest import run_child

from themis_assess.core.seeding import derive_generator, derive_seed


def test_the_same_identity_derives_the_same_seed():
    assert derive_seed(7, "gaussian_noise", 3, 0) == derive_seed(7, "gaussian_noise", 3, 0)


def test_different_identities_derive_different_seeds():
    base = derive_seed(7, "gaussian_noise", 3, 0)
    assert derive_seed(7, "gaussian_noise", 3, 1) != base
    assert derive_seed(7, "gaussian_noise", 4, 0) != base
    assert derive_seed(7, "uniform_bias", 3, 0) != base
    assert derive_seed(8, "gaussian_noise", 3, 0) != base


def test_the_parts_are_separated_so_they_cannot_collide():
    """('ab', 'c') and ('a', 'bc') must not derive the same seed."""
    assert derive_seed(0, "ab", "c") != derive_seed(0, "a", "bc")


def test_order_matters():
    assert derive_seed(0, "a", "b") != derive_seed(0, "b", "a")


def test_the_generator_is_reproducible():
    a = derive_generator(1, "cell", 2).normal(size=5)
    b = derive_generator(1, "cell", 2).normal(size=5)
    assert a == pytest.approx(b)


def test_the_seed_is_stable_across_processes():
    """The whole reason this module exists. ``hash()`` would fail this."""
    script = textwrap.dedent(
        """
        from themis_assess.core.seeding import derive_seed
        print(derive_seed(42, "gaussian_noise", 3, 0))
        """
    )
    outputs = {run_child(script, check=True).stdout.strip() for _ in range(3)}
    assert len(outputs) == 1, f"seed differed between processes: {outputs}"


def test_a_sweep_cell_is_reproducible_across_processes():
    """The property at the level a user cares about: a recorded seed reproduces a sweep."""
    script = textwrap.dedent(
        """
        import numpy as np
        from themis_assess.assessment.robustness import RobustnessEvaluator
        from themis_assess.handlers.sklearn import CallableHandler

        evaluator = RobustnessEvaluator(
            CallableHandler(lambda X: np.zeros(X.shape[0])), seed=42
        )
        rng = evaluator._cell_rng("gaussian_noise", 3, 0)
        print(f"{float(rng.normal()):.10f}")
        """
    )
    outputs = {run_child(script, check=True).stdout.strip() for _ in range(3)}
    assert len(outputs) == 1, f"cell draw differed between processes: {outputs}"
