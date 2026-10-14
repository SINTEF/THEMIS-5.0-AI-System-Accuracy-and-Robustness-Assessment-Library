"""Tests for the four split protocols of survey §3.1, plus grading and resampling.

The property every one of these must have is *disjointness under the protocol's own
definition*, and each protocol defines it differently: no shared domain, no shared time,
no shared subclass. A split that merely has disjoint indices can still leak — a temporal
split with a windowed model is the standard example — so several tests here check the
protocol's semantics rather than just the index sets.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.protocols.splits import (
    Split,
    compositional_split,
    domain_split,
    near_far_split,
    resample_subsets,
    subpopulation_split,
    temporal_split,
)


# --------------------------------------------------------------------------
# the container
# --------------------------------------------------------------------------


def test_a_split_with_a_leaked_sample_is_rejected():
    with pytest.raises(ValueError, match="both train and test"):
        Split(
            name="leaky",
            train=np.array([0, 1, 2]),
            test=np.array([2, 3]),
            protocol="test",
        )


def test_split_counts():
    split = Split(
        name="ok", train=np.arange(7), test=np.arange(7, 10), protocol="test"
    )
    assert (split.n_train, split.n_test) == (7, 3)


# --------------------------------------------------------------------------
# domain
# --------------------------------------------------------------------------


@pytest.fixture
def domains():
    return np.repeat(["hospital_a", "hospital_b", "hospital_c"], 20)


def test_domain_split_holds_out_the_named_domain(domains):
    (split,) = domain_split(domains, held_out=["hospital_b"])
    assert split.n_train == 40
    assert split.n_test == 20
    assert set(np.asarray(domains)[split.test]) == {"hospital_b"}
    assert "hospital_b" not in set(np.asarray(domains)[split.train])


def test_domain_split_leave_one_out_yields_one_split_per_domain(domains):
    splits = domain_split(domains, leave_one_out=True)
    assert len(splits) == 3
    assert [s.n_test for s in splits] == [20, 20, 20]
    # Every domain is held out exactly once.
    held = [s.metadata["held_out_domains"][0] for s in splits]
    assert sorted(held) == ["hospital_a", "hospital_b", "hospital_c"]


def test_domain_split_train_and_test_never_share_a_domain(domains):
    labels = np.asarray(domains)
    for split in domain_split(domains, leave_one_out=True):
        assert not set(labels[split.train]) & set(labels[split.test])


def test_domain_split_needs_two_domains():
    with pytest.raises(ValueError, match="at least 2 domains"):
        domain_split(np.repeat(["only_one"], 10), leave_one_out=True)


def test_domain_split_rejects_an_unknown_held_out_domain(domains):
    with pytest.raises(ValueError, match="not present"):
        domain_split(domains, held_out=["hospital_z"])


def test_domain_split_requires_a_choice(domains):
    with pytest.raises(ValueError, match="either held_out"):
        domain_split(domains)
    with pytest.raises(ValueError, match="not both"):
        domain_split(domains, held_out=["hospital_a"], leave_one_out=True)


# --------------------------------------------------------------------------
# temporal
# --------------------------------------------------------------------------


def test_temporal_split_puts_the_past_in_training():
    times = np.arange(100)
    (split,) = temporal_split(times, cutoff=70)
    assert split.train.max() < 70
    assert split.test.min() >= 70
    assert (split.n_train, split.n_test) == (70, 30)


def test_temporal_split_by_fraction():
    times = np.arange(100)
    (split,) = temporal_split(times, test_fraction=0.25)
    assert split.n_test == 25


def test_temporal_gap_discards_exactly_gap_samples():
    """The gap is what stops a windowed model reading across the boundary."""
    times = np.arange(100)
    (no_gap,) = temporal_split(times, cutoff=70)
    (with_gap,) = temporal_split(times, cutoff=70, gap=5)
    assert with_gap.n_test == no_gap.n_test - 5
    assert with_gap.metadata["n_dropped_to_gap"] == 5
    # The dropped samples are the ones nearest the boundary, on the test side.
    assert with_gap.test.min() == 75
    assert with_gap.n_train == no_gap.n_train  # the training side is untouched


def test_temporal_split_handles_unsorted_timestamps():
    rng = np.random.default_rng(0)
    times = np.arange(100)
    order = rng.permutation(100)
    (split,) = temporal_split(times[order], cutoff=70)
    assert set(times[order][split.train]) == set(range(70))
    assert set(times[order][split.test]) == set(range(70, 100))


def test_temporal_split_accepts_datetimes():
    times = np.array(
        ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"], dtype="datetime64[D]"
    )
    (split,) = temporal_split(times, cutoff=np.datetime64("2024-03-01"))
    assert (split.n_train, split.n_test) == (2, 2)


def test_temporal_split_needs_exactly_one_boundary():
    with pytest.raises(ValueError, match="exactly one"):
        temporal_split(np.arange(10))
    with pytest.raises(ValueError, match="exactly one"):
        temporal_split(np.arange(10), cutoff=5, test_fraction=0.2)


# --------------------------------------------------------------------------
# subpopulation
# --------------------------------------------------------------------------


def test_subpopulation_split_hits_the_requested_mixture():
    groups = np.repeat([0, 1], 100)
    rng = np.random.default_rng(0)
    (split,) = subpopulation_split(
        groups, test_weights={0: 0.9, 1: 0.1}, train_fraction=0.5, rng=rng
    )
    counts = np.bincount(np.asarray(groups)[split.test], minlength=2)
    ratio = counts[0] / counts.sum()
    assert ratio == pytest.approx(0.9, abs=0.02)


def test_subpopulation_split_keeps_every_group_in_training():
    """The distinction from a domain split: nothing on the test side is unseen."""
    groups = np.repeat([0, 1, 2], 60)
    (split,) = subpopulation_split(
        groups,
        test_weights={0: 0.7, 1: 0.2, 2: 0.1},
        rng=np.random.default_rng(1),
    )
    assert set(np.asarray(groups)[split.train]) == {0, 1, 2}


def test_subpopulation_split_is_reproducible():
    groups = np.repeat([0, 1], 50)
    kwargs = {"test_weights": {0: 0.8, 1: 0.2}}
    (a,) = subpopulation_split(groups, rng=np.random.default_rng(7), **kwargs)
    (b,) = subpopulation_split(groups, rng=np.random.default_rng(7), **kwargs)
    assert np.array_equal(a.test, b.test)
    assert np.array_equal(a.train, b.train)


def test_subpopulation_split_rejects_an_unknown_group():
    with pytest.raises(ValueError, match="not present"):
        subpopulation_split(np.repeat([0, 1], 10), test_weights={0: 0.5, 9: 0.5})


def test_subpopulation_split_train_and_test_are_disjoint():
    groups = np.repeat([0, 1], 100)
    (split,) = subpopulation_split(
        groups, test_weights={0: 0.6, 1: 0.4}, rng=np.random.default_rng(3)
    )
    assert not set(split.train.tolist()) & set(split.test.tolist())


# --------------------------------------------------------------------------
# compositional
# --------------------------------------------------------------------------


@pytest.fixture
def breeds():
    """Two superclasses of three subclasses each, 10 samples per subclass."""
    sub = np.repeat(
        ["terrier", "spaniel", "retriever", "tabby", "siamese", "persian"], 10
    )
    sup = np.repeat(["dog", "cat"], 30)
    return sub, sup


def test_compositional_split_holds_out_whole_subclasses(breeds):
    sub, sup = breeds
    (split,) = compositional_split(sub, sup, rng=np.random.default_rng(0))
    assert not set(sub[split.train]) & set(sub[split.test])


def test_compositional_split_keeps_every_superclass_on_both_sides(breeds):
    """The point of the protocol: the *label* is familiar, the *examples* are not."""
    sub, sup = breeds
    (split,) = compositional_split(sub, sup, rng=np.random.default_rng(0))
    assert set(sup[split.train]) == set(sup[split.test]) == {"dog", "cat"}


def test_compositional_split_never_holds_out_every_subclass(breeds):
    """Regression: round() at three subclasses held out two, leaving one to train on."""
    sub, sup = breeds
    for seed in range(5):
        (split,) = compositional_split(
            sub, sup, test_fraction=0.5, rng=np.random.default_rng(seed)
        )
        assert len(split.metadata["held_out_subclasses"]) == 2  # one per superclass
        assert split.n_train == 40
        assert split.n_test == 20


def test_compositional_split_rejects_a_subclass_in_two_superclasses():
    sub = np.array(["a", "a", "b", "b"])
    sup = np.array(["x", "y", "x", "x"])
    with pytest.raises(ValueError, match="exactly one superclass"):
        compositional_split(sub, sup)


def test_compositional_split_needs_two_subclasses_per_superclass():
    sub = np.repeat(["only", "p", "q"], 5)
    sup = np.repeat(["x", "y", "y"], 5)
    with pytest.raises(ValueError, match="at least 2"):
        compositional_split(sub, sup)


# --------------------------------------------------------------------------
# near / far grading
# --------------------------------------------------------------------------


def test_near_far_grading_splits_the_test_side_at_the_quantile():
    split = Split(
        name="base", train=np.arange(50), test=np.arange(50, 100), protocol="domain"
    )
    shift = np.concatenate([np.zeros(50), np.linspace(0.0, 1.0, 50)])
    near, far = near_far_split(split, shift)
    assert near.n_test + far.n_test == split.n_test
    assert near.n_test == pytest.approx(far.n_test, abs=1)
    assert shift[near.test].max() <= shift[far.test].min()


def test_near_far_grading_shares_the_training_side():
    split = Split(
        name="base", train=np.arange(10), test=np.arange(10, 20), protocol="temporal"
    )
    near, far = near_far_split(split, np.arange(20, dtype=float))
    assert np.array_equal(near.train, split.train)
    assert np.array_equal(far.train, split.train)
    assert near.protocol == far.protocol == "temporal+near_far"


def test_near_far_grading_records_the_threshold():
    split = Split(
        name="base", train=np.arange(10), test=np.arange(10, 20), protocol="domain"
    )
    near, _ = near_far_split(split, np.arange(20, dtype=float), quantile=0.25)
    assert "shift_threshold" in near.metadata
    assert near.metadata["grading"] == "near"


def test_near_far_grading_refuses_a_test_side_it_cannot_halve():
    split = Split(name="tiny", train=np.arange(5), test=np.array([5]), protocol="domain")
    with pytest.raises(ValueError, match="into near and far"):
        near_far_split(split, np.arange(6, dtype=float))


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def test_resample_subsets_returns_the_requested_number_and_size():
    subsets = resample_subsets(
        100, n_resamples=20, fraction=0.8, rng=np.random.default_rng(0)
    )
    assert len(subsets) == 20
    assert all(len(s) == 80 for s in subsets)


def test_resample_subsets_vary():
    """A spread is the whole point; identical subsets would report zero variance."""
    subsets = resample_subsets(200, n_resamples=10, rng=np.random.default_rng(0))
    distinct = {tuple(s.tolist()) for s in subsets}
    assert len(distinct) == 10


def test_resample_subsets_without_replacement_has_no_duplicates():
    subsets = resample_subsets(
        100, n_resamples=5, replace=False, rng=np.random.default_rng(0)
    )
    for subset in subsets:
        assert len(set(subset.tolist())) == len(subset)


def test_stratified_resampling_preserves_group_proportions():
    """Stratifying isolates sampling noise from mixture drift."""
    groups = np.repeat([0, 1], [180, 20])  # a 90/10 imbalance
    subsets = resample_subsets(
        200, n_resamples=10, fraction=0.5, stratify_by=groups, rng=np.random.default_rng(0)
    )
    for subset in subsets:
        counts = np.bincount(groups[subset], minlength=2)
        assert counts[0] == 90
        assert counts[1] == 10


def test_unstratified_resampling_lets_the_mixture_drift():
    """Repo C's class-imbalance sensitivity depends on this being the default."""
    groups = np.repeat([0, 1], [180, 20])
    subsets = resample_subsets(
        200, n_resamples=50, fraction=0.5, rng=np.random.default_rng(0)
    )
    minority = [np.count_nonzero(groups[s] == 1) for s in subsets]
    assert len(set(minority)) > 1


def test_resample_subsets_validates_its_arguments():
    with pytest.raises(ValueError, match="n_samples must be positive"):
        resample_subsets(0)
    with pytest.raises(ValueError, match="fraction must lie"):
        resample_subsets(10, fraction=1.5)
    with pytest.raises(ValueError, match="label"):
        resample_subsets(10, stratify_by=np.zeros(5))
