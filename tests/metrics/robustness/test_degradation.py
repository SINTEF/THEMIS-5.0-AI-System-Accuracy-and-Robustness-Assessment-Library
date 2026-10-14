"""Golden-value tests for the degradation metrics.

Every expected value here is computed by hand from the survey's equation, not captured
from the implementation. That is the point: these tests pin the equations, so a
refactor that changes a number fails loudly.
"""

from __future__ import annotations

import math

import pytest

from themis_assess.core.errors import ZeroDenominatorError
from themis_assess.metrics.robustness.degradation import (
    absolute_relative_robustness,
    accuracy_drop,
    corruption_error,
    error_based_robustness,
    error_inflation,
    error_ratio,
    macro_error,
    mean_corruption_error,
    mean_relative_corruption_error,
    mean_relative_error_inflation,
    relative_accuracy_drop,
    relative_corruption_error,
    relative_error_inflation,
    relative_error_ratio,
    relative_robustness,
    total_error_based_robustness,
    total_relative_robustness,
)

# --------------------------------------------------------------------------
# §5.1.1
# --------------------------------------------------------------------------


def test_macro_error_is_unweighted_mean_eq2():
    # (0.1 + 0.2 + 0.6) / 3 = 0.3
    assert macro_error({"a": 0.1, "b": 0.2, "c": 0.6}) == pytest.approx(0.3)


def test_macro_error_accepts_a_bare_sequence():
    assert macro_error([0.1, 0.2, 0.6]) == pytest.approx(0.3)


def test_macro_error_ignores_domain_size():
    """Two domains, equal weight, regardless of how many samples each held."""
    assert macro_error({"big": 0.0, "small": 1.0}) == pytest.approx(0.5)


def test_accuracy_drop_eq3():
    assert accuracy_drop(0.9, 0.7) == pytest.approx(0.2)


def test_accuracy_drop_is_negative_when_shift_helps():
    assert accuracy_drop(0.7, 0.9) == pytest.approx(-0.2)


def test_relative_accuracy_drop_eq4():
    # (0.9 - 0.72) / 0.9 = 0.2
    assert relative_accuracy_drop(0.9, 0.72) == pytest.approx(0.2)


def test_error_ratio_eq5():
    assert error_ratio(e_id=0.1, e_ood=0.25) == pytest.approx(2.5)


def test_error_ratio_is_one_when_unchanged():
    assert error_ratio(e_id=0.3, e_ood=0.3) == pytest.approx(1.0)


def test_zero_denominators_raise_rather_than_returning_inf():
    with pytest.raises(ZeroDenominatorError, match="in-distribution accuracy"):
        relative_accuracy_drop(0.0, 0.0)
    with pytest.raises(ZeroDenominatorError, match="in-distribution error"):
        error_ratio(e_id=0.0, e_ood=0.5)
    with pytest.raises(ZeroDenominatorError, match="clean accuracy"):
        relative_robustness(0.0, 0.5)


# --------------------------------------------------------------------------
# §5.1.3, accuracy-based
# --------------------------------------------------------------------------


def test_relative_robustness_eq11():
    assert relative_robustness(a_clean=0.8, a_perturbed=0.6) == pytest.approx(0.75)


def test_relative_robustness_is_one_on_unperturbed_data():
    """The invariant the old base.py docstring asked for: no change -> exactly 1."""
    assert relative_robustness(0.83, 0.83) == pytest.approx(1.0)


def test_relative_robustness_exceeds_one_when_perturbation_helps():
    assert relative_robustness(a_clean=0.5, a_perturbed=0.6) > 1.0


def test_total_relative_robustness_is_a_sum_eq12():
    assert total_relative_robustness({"noise": 0.9, "blur": 0.8, "shift": 0.95}) == pytest.approx(
        2.65
    )


def test_total_relative_robustness_grows_with_perturbation_count():
    """Why the survey says it is comparable only within a fixed set."""
    three = total_relative_robustness([1.0, 1.0, 1.0])
    four = total_relative_robustness([1.0, 1.0, 1.0, 1.0])
    assert four > three


def test_absolute_relative_robustness_is_symmetric():
    """Repo-A extension: |1 - R|, blind to the direction of the change."""
    worse = absolute_relative_robustness(a_clean=1.0, a_perturbed=0.8)
    better = absolute_relative_robustness(a_clean=1.0, a_perturbed=1.2)
    assert worse == pytest.approx(0.2)
    assert better == pytest.approx(0.2)


def test_absolute_relative_robustness_is_zero_when_unchanged():
    assert absolute_relative_robustness(0.77, 0.77) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# §5.1.3, error-based
# --------------------------------------------------------------------------


def test_relative_error_ratio_eq13_is_clean_over_perturbed():
    """Direction matters: clean over perturbed, so higher is better."""
    # (0.1 + 1e-6) / (0.2 + 1e-6)
    expected = (0.1 + 1e-6) / (0.2 + 1e-6)
    assert relative_error_ratio(e_clean=0.1, e_perturbed=0.2) == pytest.approx(expected)
    assert relative_error_ratio(e_clean=0.1, e_perturbed=0.2) < 1.0


def test_relative_error_ratio_eps_absorbs_a_zero_perturbed_error():
    """The one equation where the survey supplies a stabiliser."""
    assert relative_error_ratio(e_clean=0.0, e_perturbed=0.0) == pytest.approx(1.0)


def test_relative_error_ratio_with_eps_zero_raises_on_zero_denominator():
    with pytest.raises(ZeroDenominatorError):
        relative_error_ratio(e_clean=0.1, e_perturbed=0.0, eps=0.0)


def test_error_based_robustness_eq14_averages_over_severities():
    # eps=0 to keep the arithmetic exact: mean of 1.0/1.0, 1.0/2.0, 1.0/4.0
    value = error_based_robustness(e_clean=1.0, perturbed_errors=[1.0, 2.0, 4.0], eps=0.0)
    assert value == pytest.approx((1.0 + 0.5 + 0.25) / 3)


def test_total_error_based_robustness_is_a_product_eq15():
    assert total_error_based_robustness([0.9, 0.8, 0.5]) == pytest.approx(0.36)


def test_multiplicative_aggregation_is_stricter_than_additive():
    """One weak perturbation should drag the product down; the survey's rationale."""
    strong = total_error_based_robustness([1.0, 1.0, 1.0])
    one_weak = total_error_based_robustness([1.0, 1.0, 0.1])
    assert one_weak == pytest.approx(0.1)
    assert one_weak < strong / 5


def test_error_inflation_eq16():
    assert error_inflation(e_clean=0.2, e_perturbed=0.5) == pytest.approx(2.5)


def test_error_inflation_is_one_when_unchanged():
    assert error_inflation(0.2, 0.2) == pytest.approx(1.0)


def test_relative_error_inflation_eq17_subtracts_the_baseline():
    # model EI = 0.4/0.2 = 2.0 ; baseline EI = 0.9/0.3 = 3.0 ; REI = -1.0
    value = relative_error_inflation(
        e_clean=0.2, e_perturbed=0.4, e_clean_baseline=0.3, e_perturbed_baseline=0.9
    )
    assert value == pytest.approx(-1.0)


def test_relative_error_inflation_is_zero_when_model_matches_baseline():
    """A perturbation that hurts everyone equally scores 0, not 1."""
    value = relative_error_inflation(
        e_clean=0.1, e_perturbed=0.2, e_clean_baseline=0.4, e_perturbed_baseline=0.8
    )
    assert value == pytest.approx(0.0)


def test_mean_relative_error_inflation_eq18_averages_over_types_and_severities():
    # model EI: noise -> 2, 3 ; blur -> 4, 5      (clean 0.1)
    # base  EI: noise -> 1, 1 ; blur -> 2, 2      (clean 0.2)
    # REI:            1, 2 ;         2, 3   -> mean = 8/4 = 2.0
    value = mean_relative_error_inflation(
        e_clean=0.1,
        perturbed_errors={"noise": [0.2, 0.3], "blur": [0.4, 0.5]},
        e_clean_baseline=0.2,
        perturbed_errors_baseline={"noise": [0.2, 0.2], "blur": [0.4, 0.4]},
    )
    assert value == pytest.approx(2.0)


def test_mismatched_perturbation_sets_are_rejected():
    with pytest.raises(ValueError, match="same perturbations"):
        mean_relative_error_inflation(
            e_clean=0.1,
            perturbed_errors={"noise": [0.2]},
            e_clean_baseline=0.2,
            perturbed_errors_baseline={"blur": [0.2]},
        )


def test_mismatched_severity_lengths_are_rejected():
    with pytest.raises(ValueError, match="differ in length"):
        mean_relative_error_inflation(
            e_clean=0.1,
            perturbed_errors={"noise": [0.2, 0.3]},
            e_clean_baseline=0.2,
            perturbed_errors_baseline={"noise": [0.2]},
        )


# --------------------------------------------------------------------------
# §5.1.3, corruption error
# --------------------------------------------------------------------------


def test_corruption_error_eq19():
    # (0.1 + 0.2 + 0.3) / (0.2 + 0.4 + 0.6) = 0.6 / 1.2 = 0.5
    assert corruption_error([0.1, 0.2, 0.3], [0.2, 0.4, 0.6]) == pytest.approx(0.5)


def test_corruption_error_is_one_against_itself():
    curve = [0.1, 0.25, 0.4]
    assert corruption_error(curve, curve) == pytest.approx(1.0)


def test_corruption_error_normalisation_stops_hard_corruptions_dominating():
    """The survey's stated rationale for dividing by the baseline.

    An easy corruption and a hard one, on both of which the model is exactly twice as
    good as the baseline. mCE should be 0.5, not something pulled toward the hard one.
    """
    model = {"easy": [0.01, 0.02], "hard": [0.40, 0.80]}
    base = {"easy": [0.02, 0.04], "hard": [0.80, 1.60]}
    assert mean_corruption_error(model, base) == pytest.approx(0.5)


def test_mean_corruption_error_eq20():
    model = {"noise": [0.1, 0.3], "blur": [0.2, 0.2]}
    base = {"noise": [0.2, 0.2], "blur": [0.4, 0.4]}
    # CE noise = 0.4/0.4 = 1.0 ; CE blur = 0.4/0.8 = 0.5 ; mean = 0.75
    assert mean_corruption_error(model, base) == pytest.approx(0.75)


def test_relative_corruption_error_eq21_subtracts_clean_error_first():
    # numerator   (0.3-0.1) + (0.5-0.1) = 0.6
    # denominator (0.6-0.2) + (1.0-0.2) = 1.2
    value = relative_corruption_error(
        e_clean=0.1,
        perturbed_errors=[0.3, 0.5],
        e_clean_baseline=0.2,
        baseline_perturbed_errors=[0.6, 1.0],
    )
    assert value == pytest.approx(0.5)


def test_mce_and_mrce_can_disagree():
    """Why the survey insists on reporting both.

    A model with much better clean accuracy but the *same* added error under
    corruption: mCE flatters it, mRCE does not.
    """
    model_clean, base_clean = 0.05, 0.30
    model = {"noise": [0.15, 0.25]}
    base = {"noise": [0.40, 0.50]}

    mce = mean_corruption_error(model, base)
    mrce = mean_relative_corruption_error(model_clean, model, base_clean, base)

    assert mce < 0.5  # looks strongly robust
    assert mrce == pytest.approx(1.0)  # identical added error -> no robustness advantage


def test_mean_relative_corruption_error_eq22():
    model = {"a": [0.3], "b": [0.5]}
    base = {"a": [0.6], "b": [1.0]}
    # RCE a = (0.3-0.1)/(0.6-0.2) = 0.5 ; RCE b = (0.5-0.1)/(1.0-0.2) = 0.5
    value = mean_relative_corruption_error(0.1, model, 0.2, base)
    assert value == pytest.approx(0.5)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: macro_error([]),
        lambda: total_relative_robustness([]),
        lambda: total_error_based_robustness([]),
        lambda: error_based_robustness(1.0, []),
        lambda: mean_corruption_error({}, {}),
    ],
)
def test_empty_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


def test_curves_of_different_length_are_rejected_by_corruption_error():
    with pytest.raises(ValueError, match="differ in length"):
        corruption_error([0.1, 0.2], [0.1])


def test_no_metric_silently_returns_nan():
    """A guard against the class of bug this module replaces."""
    value = corruption_error([0.1, 0.2], [0.3, 0.4])
    assert math.isfinite(value)


# --------------------------------------------------------------------------
# severity curves as mappings
# --------------------------------------------------------------------------


class TestCurvesAsMappings:
    """A severity curve may be a ``{severity: value}`` mapping or a sequence.

    Regression tests for a defect the first end-to-end run exposed: the helper that
    normalised a curve iterated the mapping directly, which yields its *keys*. Every
    curve passed as a dict silently became ``[1, 2, 3, 4, 5]``, identical for model and
    baseline, so ``CE`` came out at exactly 1.0 — a plausible "matches the baseline"
    rather than an error.
    """

    def test_a_mapping_and_a_sequence_agree(self):
        as_sequence = [0.10, 0.20, 0.30]
        as_mapping = {1: 0.10, 2: 0.20, 3: 0.30}
        base_sequence = [0.05, 0.10, 0.15]
        base_mapping = {1: 0.05, 2: 0.10, 3: 0.15}
        assert corruption_error(as_mapping, base_mapping) == pytest.approx(
            corruption_error(as_sequence, base_sequence)
        )

    def test_corruption_error_uses_the_values_not_the_keys(self):
        """0.6 / 0.3 = 2.0. Iterating the keys would give (1+2+3)/(1+2+3) = 1.0."""
        model = {1: 0.1, 2: 0.2, 3: 0.3}
        baseline = {1: 0.05, 2: 0.10, 3: 0.15}
        assert corruption_error(model, baseline) == pytest.approx(2.0)

    def test_a_mapping_is_read_in_severity_order_not_insertion_order(self):
        """Out-of-order construction must not pair severity 3 with severity 1."""
        ordered = {1: 1.0, 2: 2.0, 3: 6.0}
        shuffled = {3: 6.0, 1: 1.0, 2: 2.0}
        baseline = {1: 1.0, 2: 1.0, 3: 1.0}
        assert relative_corruption_error(
            0.5, shuffled, 0.5, baseline
        ) == pytest.approx(relative_corruption_error(0.5, ordered, 0.5, baseline))

    def test_error_based_robustness_reads_a_mapping(self):
        assert error_based_robustness(1.0, {1: 1.0, 2: 1.0}) == pytest.approx(
            error_based_robustness(1.0, [1.0, 1.0])
        )

    def test_mean_relative_error_inflation_reads_mappings(self):
        model = {"noise": {1: 2.0, 2: 4.0}}
        baseline = {"noise": {1: 1.5, 2: 2.5}}
        by_mapping = mean_relative_error_inflation(1.0, model, 1.0, baseline)
        by_sequence = mean_relative_error_inflation(
            1.0, {"noise": [2.0, 4.0]}, 1.0, {"noise": [1.5, 2.5]}
        )
        assert by_mapping == pytest.approx(by_sequence)
        # (2/1 - 1.5/1) + (4/1 - 2.5/1) over 2 cells = (0.5 + 1.5) / 2 = 1.0
        assert by_mapping == pytest.approx(1.0)
