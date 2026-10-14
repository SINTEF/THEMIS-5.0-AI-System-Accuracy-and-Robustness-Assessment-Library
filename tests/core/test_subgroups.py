"""Tests for subgroups as masks — the answer to repo C's objection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from themis_assess.core.subgroups import (
    SubgroupSpec,
    complement,
    from_column_values,
    from_criteria,
    resolve_masks,
)
from themis_assess.metrics.robustness.subgroup import group_scores, worst_group_accuracy


@pytest.fixture
def people():
    return pd.DataFrame(
        {
            "age": [25, 35, 38, 62, 70],
            "job": ["nurse", "nurse", "teacher", "nurse", "retired"],
            "site": ["oslo", "oslo", "bergen", "bergen", "oslo"],
        }
    )


# --------------------------------------------------------------------------
# SubgroupSpec
# --------------------------------------------------------------------------


def test_spec_evaluates_its_mask(people):
    spec = SubgroupSpec("nurses", lambda df: (df["job"] == "nurse").to_numpy())
    assert spec.mask(people).tolist() == [True, True, False, True, False]


def test_spec_accepts_a_zero_one_array_and_converts_it():
    spec = SubgroupSpec("onehot", lambda _: np.array([1, 0, 1]))
    assert spec.mask(None).dtype == bool


def test_spec_rejects_a_non_boolean_mask():
    spec = SubgroupSpec("bad", lambda _: np.array([0.3, 0.7]))
    with pytest.raises(TypeError, match="neither boolean nor 0/1"):
        spec.mask(None)


def test_spec_validates_mask_length(people):
    spec = SubgroupSpec("short", lambda _: np.array([True, False]))
    with pytest.raises(ValueError, match="mask of length 2 for 5 samples"):
        spec.mask(people, n_samples=5)


def test_spec_requires_a_name():
    with pytest.raises(ValueError, match="non-empty name"):
        SubgroupSpec("", lambda _: np.array([True]))


def test_spec_requires_a_callable():
    with pytest.raises(TypeError, match="must be callable"):
        SubgroupSpec("g", "not a function")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# resolve_masks
# --------------------------------------------------------------------------


def test_resolve_masks_returns_masks_in_spec_order(people):
    specs = [
        SubgroupSpec("z_site_oslo", lambda df: (df["site"] == "oslo").to_numpy()),
        SubgroupSpec("a_nurses", lambda df: (df["job"] == "nurse").to_numpy()),
    ]
    masks = resolve_masks(specs, people, n_samples=len(people))
    assert list(masks) == ["z_site_oslo", "a_nurses"]


def test_resolve_masks_rejects_an_empty_group_by_default(people):
    specs = [SubgroupSpec("nobody", lambda df: (df["job"] == "pilot").to_numpy())]
    with pytest.raises(ValueError, match="selects no samples"):
        resolve_masks(specs, people)


def test_resolve_masks_can_allow_an_empty_group(people):
    specs = [SubgroupSpec("nobody", lambda df: (df["job"] == "pilot").to_numpy())]
    masks = resolve_masks(specs, people, require_non_empty=False)
    assert not masks["nobody"].any()


def test_resolve_masks_rejects_duplicate_names(people):
    spec = SubgroupSpec("g", lambda df: (df["job"] == "nurse").to_numpy())
    with pytest.raises(ValueError, match="Duplicate subgroup name"):
        resolve_masks([spec, spec], people)


def test_resolve_masks_needs_at_least_one_spec(people):
    with pytest.raises(ValueError, match="at least one SubgroupSpec"):
        resolve_masks([], people)


# --------------------------------------------------------------------------
# from_criteria — repo A's dictionary form
# --------------------------------------------------------------------------


def test_from_criteria_handles_a_range(people):
    spec = from_criteria("thirties", {"age": {"min": 30, "max": 40}})
    assert spec.mask(people).tolist() == [False, True, True, False, False]


def test_from_criteria_handles_an_open_range(people):
    spec = from_criteria("seniors", {"age": {"min": 60}})
    assert spec.mask(people).tolist() == [False, False, False, True, True]


def test_from_criteria_handles_membership(people):
    spec = from_criteria("care", {"job": ["nurse", "teacher"]})
    assert spec.mask(people).tolist() == [True, True, True, True, False]


def test_from_criteria_handles_equality(people):
    spec = from_criteria("bergen", {"site": "bergen"})
    assert spec.mask(people).tolist() == [False, False, True, True, False]


def test_from_criteria_combines_conditions_with_and(people):
    """The documented example from repo A's docstring."""
    spec = from_criteria("nurses_in_their_thirties", {"age": {"min": 30, "max": 40}, "job": ["nurse"]})
    assert spec.mask(people).tolist() == [False, True, False, False, False]


def test_from_criteria_writes_a_description(people):
    spec = from_criteria("g", {"age": {"min": 30, "max": 40}, "job": ["nurse"]})
    assert "age in [30, 40]" in spec.description
    assert "job in ['nurse']" in spec.description


def test_from_criteria_rejects_an_unknown_column(people):
    spec = from_criteria("g", {"height": 180})
    with pytest.raises(ValueError, match="column\\(s\\) not in the DataFrame"):
        spec.mask(people)


def test_from_criteria_rejects_an_unknown_range_key(people):
    spec = from_criteria("g", {"age": {"minimum": 30}})
    with pytest.raises(ValueError, match="unknown range key"):
        spec.mask(people)


def test_from_criteria_needs_criteria():
    with pytest.raises(ValueError, match="no criteria"):
        from_criteria("g", {})


def test_from_criteria_explains_itself_on_non_frame_data():
    spec = from_criteria("g", {"age": 30})
    with pytest.raises(TypeError, match="supply a mask_fn directly"):
        spec.mask(np.zeros(3))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_from_column_values_builds_one_spec_per_value(people):
    specs = from_column_values("site", values=["oslo", "bergen"])
    assert [s.name for s in specs] == ["site=oslo", "site=bergen"]
    masks = resolve_masks(specs, people, n_samples=len(people))
    assert masks["site=oslo"].sum() == 3
    assert masks["site=bergen"].sum() == 2


def test_from_column_values_supports_a_name_prefix(people):
    specs = from_column_values("site", values=["oslo"], prefix="location")
    assert specs[0].name == "location=oslo"


def test_from_column_values_requires_explicit_values():
    """Lazily enumerating hides which groups a report actually covers."""
    with pytest.raises(ValueError, match="explicit `values`"):
        from_column_values("site")


def test_complement_inverts_a_group(people):
    nurses = from_criteria("nurses", {"job": "nurse"})
    others = complement(nurses)
    assert others.name == "not(nurses)"
    assert others.mask(people).tolist() == [False, False, True, False, True]


def test_complement_takes_an_explicit_name(people):
    spec = complement(from_criteria("a", {"site": "oslo"}), name="elsewhere")
    assert spec.name == "elsewhere"


# --------------------------------------------------------------------------
# the point: four mechanisms, one metric layer
# --------------------------------------------------------------------------


def test_all_four_repo_mechanisms_feed_the_same_metric(people):
    """repo C one-hot, repo D similarity, repo B category, repo A criteria dict."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])

    onehot = np.array([1, 1, 0, 0, 1])  # repo C
    similarity = np.array([0.9, 0.8, 0.1, 0.2, 0.7])  # repo D
    category = people["site"].to_numpy()  # repo B

    specs = [
        SubgroupSpec("onehot", lambda _: onehot == 1),
        SubgroupSpec("similar", lambda _: similarity > 0.5),
        SubgroupSpec("oslo", lambda _: category == "oslo"),
        from_criteria("criteria", {"site": "oslo"}),
    ]
    masks = resolve_masks(specs, people, n_samples=len(people))
    scores = group_scores(y_true, y_pred, masks, accuracy_score)

    # All four select the same three people, so all four agree.
    assert len(set(scores.values())) == 1
    assert worst_group_accuracy(scores) == pytest.approx(1.0)
