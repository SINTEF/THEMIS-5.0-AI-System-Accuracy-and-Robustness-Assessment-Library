"""Tests for the results bundle — the one dictionary.

The bundle is a handover format, so the tests are about the properties a handover needs:
it must serialise (numpy defeats ``json.dumps`` by default), it must be complete enough
to reconstruct the claim without the objects that made it, it must say what its metrics
mean, and its shape must not depend on which parts of an assessment were run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from themis_assess.report.bundle import SCHEMA, AssessmentReport, build_report, to_plain


# --------------------------------------------------------------------------
# to_plain — the reason a results dictionary from any of the four repos failed
# to serialise
# --------------------------------------------------------------------------


def test_numpy_scalars_and_arrays_become_json():
    payload = to_plain(
        {
            "float": np.float64(0.25),
            "int": np.int64(3),
            "bool": np.bool_(True),
            "array": np.arange(4).reshape(2, 2),
            "nested": [np.float32(1.5), {"k": np.int32(2)}],
        }
    )
    assert json.loads(json.dumps(payload, allow_nan=False)) == {
        "float": 0.25,
        "int": 3,
        "bool": True,
        "array": [[0, 1], [2, 3]],
        "nested": [1.5, {"k": 2}],
    }


def test_non_finite_values_become_null_rather_than_invalid_json():
    """``NaN`` is a literal most JSON parsers reject, so a report full of them is
    unusable as a handover format even though ``json.dumps`` will emit it."""
    payload = to_plain({"a": float("nan"), "b": np.inf, "c": -np.inf, "d": 1.0})
    assert payload == {"a": None, "b": None, "c": None, "d": 1.0}
    json.dumps(payload, allow_nan=False)


def test_integer_mapping_keys_become_strings_as_json_requires():
    """A severity curve is keyed by int. JSON object keys are strings, so this is a real
    type change and every reader of a loaded curve has to sort numerically."""
    assert to_plain({1: 0.9, 2: 0.8, 10: 0.5}) == {"1": 0.9, "2": 0.8, "10": 0.5}


def test_an_enum_serialises_as_its_name():
    from themis_assess.core.types import AccessLevel

    assert to_plain(AccessLevel.PROBABILISTIC) == "PROBABILISTIC"


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def test_a_report_of_accuracy_alone_is_valid(accuracy):
    payload = AssessmentReport(model_name="m", accuracy=accuracy).to_dict()
    assert payload["schema"] == SCHEMA
    assert "accuracy" in payload
    assert "robustness" not in payload
    json.dumps(payload, allow_nan=False)


def test_a_report_of_robustness_alone_is_valid(robustness):
    payload = AssessmentReport(model_name="m", robustness=robustness).to_dict()
    assert "robustness" in payload
    assert "accuracy" not in payload
    json.dumps(payload, allow_nan=False)


def test_an_empty_report_still_serialises():
    payload = AssessmentReport().to_dict()
    assert payload["summary"]["n_findings"] == 0
    assert payload["summary"]["accuracy_passed"] is None
    json.dumps(payload, allow_nan=False)


def test_the_whole_report_round_trips_through_json(accuracy, robustness):
    report = build_report(
        model_name="random_forest",
        accuracy=accuracy,
        robustness=robustness,
        context={"split": "holdout", "ran_by": "test"},
    )
    loaded = json.loads(report.to_json())
    assert loaded["model_name"] == "random_forest"
    assert loaded["context"]["split"] == "holdout"
    assert loaded["accuracy"]["overall"]["metrics"]["accuracy"] == accuracy["accuracy"]
    assert loaded["robustness"]["clean"] == robustness.clean


def test_the_claim_can_be_reconstructed_from_the_dictionary_alone(robustness):
    """§8's four elements — what was varied, what was monitored, how it was measured,
    and how much is tolerated — must all be readable without the Python objects."""
    section = AssessmentReport(robustness=robustness).to_dict()["robustness"]
    assert section["perturbations"]  # what was varied
    assert section["severities"]
    assert section["accuracy_metric"]  # what was monitored
    assert section["scored"]  # how the change was measured
    assert section["tolerance"]["verdicts"]  # how much is tolerated
    assert section["seed"] is not None  # and how to reproduce it


def test_the_curves_survive_with_their_severities(robustness):
    section = AssessmentReport(robustness=robustness).to_dict()["robustness"]
    for perturbation, curve in section["curves"].items():
        assert perturbation in section["perturbations"]
        assert sorted(int(k) for k in curve) == section["severities"]


def test_values_are_recorded_not_rounded(accuracy):
    """A rounded record cannot be checked against the run that produced it."""
    payload = AssessmentReport(accuracy=accuracy).to_dict()
    assert payload["accuracy"]["overall"]["metrics"]["accuracy"] == pytest.approx(
        accuracy["accuracy"], abs=0
    )


# --------------------------------------------------------------------------
# the glossary travels with the numbers
# --------------------------------------------------------------------------


def test_every_metric_in_the_report_has_a_card(accuracy, robustness):
    payload = build_report(accuracy=accuracy, robustness=robustness).to_dict()
    glossary = payload["glossary"]
    for metric in payload["accuracy"]["overall"]["metrics"]:
        assert glossary[metric]["description"]
        assert glossary[metric]["higher_is_better"] is not None or glossary[metric]["target"] is not None
    for metric in payload["robustness"]["scored"]:
        assert metric in glossary


def test_the_glossary_can_be_switched_off_for_size(accuracy):
    report = AssessmentReport(accuracy=accuracy)
    assert "glossary" not in report.to_dict(glossary=False)
    assert len(report.to_json(glossary=False)) < len(report.to_json())


# --------------------------------------------------------------------------
# distances are kept out of the robustness scores — §8 rule 9
# --------------------------------------------------------------------------


def test_a_distance_lands_in_its_own_section(accuracy):
    report = AssessmentReport(accuracy=accuracy).add_distance("wasserstein_distance", 0.4)
    payload = report.to_dict()
    assert payload["distances"] == {"wasserstein_distance": 0.4}
    assert "wasserstein_distance" not in payload.get("robustness", {}).get("scored", {})
    assert "wasserstein_distance" in payload["glossary"]


# --------------------------------------------------------------------------
# findings and summary
# --------------------------------------------------------------------------


def test_the_summary_counts_the_findings_it_carries(accuracy, robustness):
    report = build_report(accuracy=accuracy, robustness=robustness)
    payload = report.to_dict()
    assert payload["summary"]["n_findings"] == len(payload["findings"])
    assert sum(payload["summary"]["by_severity"].values()) == len(payload["findings"])
    assert payload["summary"]["headlines"] == [
        f["headline"] for f in payload["findings"][:5]
    ]


def test_the_findings_are_severity_ordered_in_the_payload(accuracy, robustness):
    order = {"critical": 0, "serious": 1, "warning": 2, "info": 3}
    severities = [
        order[f["severity"]]
        for f in build_report(accuracy=accuracy, robustness=robustness).to_dict()["findings"]
    ]
    assert severities == sorted(severities)


def test_reporting_issues_join_the_model_findings(accuracy_bare, robustness_bare):
    """"No tolerance was supplied" belongs beside "the worst subgroup is behind": both
    are things a reader has to decide about before the assessment means anything."""
    payload = build_report(
        accuracy=accuracy_bare, robustness=robustness_bare
    ).to_dict()
    areas = {f["area"] for f in payload["findings"]}
    assert "reporting" in areas
    assert "accuracy" in areas


def test_saving_writes_json_to_disk(accuracy, tmp_path):
    path = AssessmentReport(accuracy=accuracy).save(tmp_path / "nested" / "report.json")
    assert path.exists()
    assert json.loads(path.read_text())["schema"] == SCHEMA


def test_sensitivity_maps_may_be_passed_as_a_sequence_and_are_named_by_axis(problem):
    from themis_assess.sensitivity.maps import SensitivityMapper

    smap = SensitivityMapper(
        problem["handler"],
        perturbation="gaussian_noise",
        severities=(1, 2),
        measure="degradation",
        metric="accuracy",
    ).feature_severity_map(problem["data"], problem["actual"])

    report = build_report(sensitivity=[smap])
    assert list(report.sensitivity) == ["feature_severity"]
    payload = report.to_dict()
    section = payload["sensitivity"]["feature_severity"]
    assert section["row_labels"] == list(smap.row_labels)
    assert np.asarray(section["grid"]).shape == smap.shape
    assert section["measure"] == "degradation"
    # a sensitivity map names the input the model leans on, which is a finding
    assert any(f["area"] == "sensitivity" for f in payload["findings"])
