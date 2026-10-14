"""Tests for the YAML config layer.

Two things carry the weight. A misspelt key must be an error, because the alternative is
a run that proceeds with the default while the file says otherwise — the failure hydra's
strict mode exists for and which repo B had switched off. And an absent criteria section
must produce ``None`` rather than an empty criteria set, so that "nothing was checked"
stays distinguishable from "everything passed".
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from themis_assess.config import (
    AssessmentConfig,
    dump_config,
    load_config,
)
from themis_assess.core.adapters import ArrayAdapter
from themis_assess.handlers.sklearn import SklearnHandler


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] * 1.5 + X[:, 1] > 0).astype(int)
    return {
        "data": ArrayAdapter(
            X, feature_names=["a", "b", "c", "d"], categorical_features=[]
        ),
        "actual": y,
        "handler": SklearnHandler(LogisticRegression().fit(X, y), name="logistic"),
    }


def write(tmp_path, text: str):
    path = tmp_path / "assessment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def test_an_empty_file_is_a_valid_config(tmp_path):
    """All-or-nothing configuration is what stops people using a config at all."""
    config = load_config(write(tmp_path, ""))
    assert config.model_name == "model"
    assert config.accuracy.acceptance_criteria() is None
    assert config.robustness.tolerance_criteria() is None


def test_a_comments_only_file_is_also_valid(tmp_path):
    config = load_config(write(tmp_path, "# nothing set yet\n"))
    assert config.accuracy.per_class is True


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not contain a YAML mapping"):
        load_config(write(tmp_path, "- just\n- a list\n"))


def test_a_misspelt_key_is_an_error_not_a_silent_default(tmp_path):
    """``severity`` rather than ``severities`` would sweep all five levels while the
    file said three, and nothing would report the discrepancy."""
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(write(tmp_path, "robustness:\n  severity: [1, 2, 3]\n"))


def test_the_error_names_the_keys_that_do_exist(tmp_path):
    with pytest.raises(ValueError, match="severities"):
        load_config(write(tmp_path, "robustness:\n  severity: [1]\n"))


def test_a_misspelt_top_level_section_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(write(tmp_path, "robustnes:\n  severities: [1]\n"))


def test_the_full_shape_reads_back(tmp_path):
    path = write(
        tmp_path,
        """
model_name: plovdiv_lightgbm
context:
  split: temporal 2019-2022 / 2023
  owner: clinical analytics
accuracy:
  metrics: [accuracy, balanced_accuracy]
  per_class: false
  diagnose_outliers: true
  criteria:
    - metric: balanced_accuracy
      acceptable: 0.85
      rationale: the triage queue has to be staffable
      critical: true
robustness:
  perturbations: [gaussian_noise]
  severities: [1, 2, 3]
  n_variants: 2
  seed: 7
  accuracy_metric: balanced_accuracy
  criteria:
    - metric: relative_robustness
      acceptable: 0.9
      basis: worst_cell
      rationale: within a year of sensor drift
report:
  output: out/report.json
""",
    )
    config = load_config(path)
    assert config.model_name == "plovdiv_lightgbm"
    assert config.context["owner"] == "clinical analytics"
    assert config.accuracy.metrics == ("accuracy", "balanced_accuracy")
    assert config.accuracy.per_class is False
    assert config.accuracy.diagnose_outliers is True
    assert config.robustness.severities == (1, 2, 3)
    assert config.robustness.seed == 7
    assert config.report.output == "out/report.json"
    assert config.path == str(path)


# --------------------------------------------------------------------------
# criteria — the part that belongs in a file
# --------------------------------------------------------------------------


def test_criteria_become_requirements_with_their_rationale(tmp_path):
    config = load_config(
        write(
            tmp_path,
            """
accuracy:
  source: clinical sign-off 2024-03
  criteria:
    - metric: accuracy
      acceptable: 0.9
      rationale: agreed with the clinic
      critical: false
""",
        )
    )
    criteria = config.accuracy.acceptance_criteria()
    assert criteria.metrics() == ["accuracy"]
    requirement = criteria.get("accuracy")
    assert requirement.acceptable == 0.9
    assert requirement.critical is False
    assert requirement.rationale == "agreed with the clinic"
    assert criteria.provenance()["source"] == "clinical sign-off 2024-03"


def test_no_criteria_means_nothing_was_checked_not_everything_passed(tmp_path, problem):
    config = load_config(write(tmp_path, "accuracy:\n  metrics: [accuracy]\n"))
    result = config.run_accuracy(problem["handler"], problem["data"], problem["actual"])
    assert result.passed is None
    assert result.acceptance is None


def test_a_tolerance_from_the_file_produces_a_verdict(tmp_path, problem):
    config = load_config(
        write(
            tmp_path,
            """
robustness:
  perturbations: [gaussian_noise]
  severities: [1, 2]
  accuracy_metric: accuracy
  criteria:
    - metric: relative_robustness
      acceptable: 0.99
      basis: worst_cell
      rationale: deliberately strict for the test
""",
        )
    )
    assessment = config.assess_robustness(
        problem["handler"], problem["data"], problem["actual"]
    )
    assert assessment.passed is False
    assert assessment.tolerance.failures


def test_an_unknown_requirement_field_is_refused_by_the_requirement_itself(tmp_path):
    config = load_config(
        write(
            tmp_path,
            "accuracy:\n  criteria:\n    - metric: accuracy\n      acceptible: 0.9\n",
        )
    )
    with pytest.raises(TypeError):
        config.accuracy.acceptance_criteria()


# --------------------------------------------------------------------------
# what the config builds
# --------------------------------------------------------------------------


def test_the_evaluator_is_built_from_the_file(tmp_path, problem):
    config = load_config(
        write(tmp_path, "accuracy:\n  metrics: [accuracy, f1]\n  per_class: false\n")
    )
    result = config.run_accuracy(problem["handler"], problem["data"], problem["actual"])
    assert sorted(result.overall.metrics) == ["accuracy", "f1"]
    assert result.per_class == {}


def test_the_diagnostic_flags_reach_the_run(tmp_path, problem):
    config = load_config(
        write(tmp_path, "accuracy:\n  metrics: [accuracy]\n  diagnose_outliers: true\n")
    )
    result = config.run_accuracy(problem["handler"], problem["data"], problem["actual"])
    assert "outliers" in result.diagnostics


def test_a_call_can_override_the_file(tmp_path, problem):
    """The file holds the agreed settings; an exploratory call varies one of them
    without editing it."""
    config = load_config(write(tmp_path, "accuracy:\n  metrics: [accuracy, f1]\n"))
    result = config.run_accuracy(
        problem["handler"], problem["data"], problem["actual"], metrics=["accuracy"]
    )
    assert sorted(result.overall.metrics) == ["accuracy"]


def test_the_sweep_settings_reach_the_evaluator(tmp_path, problem):
    config = load_config(
        write(
            tmp_path,
            "robustness:\n  perturbations: [gaussian_noise]\n"
            "  severities: [1, 2]\n  n_variants: 1\n  seed: 3\n",
        )
    )
    plan = config.robustness_evaluator(problem["handler"]).plan(
        problem["data"], problem["actual"]
    )
    assert plan.perturbations == ("gaussian_noise",)
    assert plan.severities == (1, 2)


def test_the_seed_from_the_file_makes_the_sweep_reproducible(tmp_path, problem):
    text = (
        "robustness:\n  perturbations: [gaussian_noise]\n"
        "  severities: [2]\n  n_variants: 2\n  seed: 11\n  accuracy_metric: accuracy\n"
    )
    config = load_config(write(tmp_path, text))
    first = config.assess_robustness(
        problem["handler"], problem["data"], problem["actual"]
    )
    second = config.assess_robustness(
        problem["handler"], problem["data"], problem["actual"]
    )
    assert first.curves() == second.curves()


def test_the_context_reaches_the_report(tmp_path, problem):
    config = load_config(
        write(
            tmp_path,
            "model_name: m\ncontext:\n  split: holdout\n  vintage: '2024-03'\n"
            "accuracy:\n  metrics: [accuracy]\n",
        )
    )
    accuracy = config.run_accuracy(
        problem["handler"], problem["data"], problem["actual"]
    )
    payload = config.build_report(accuracy=accuracy).to_dict()
    assert payload["model_name"] == "m"
    assert payload["context"]["split"] == "holdout"
    # and the config that produced the run is named in the report
    assert payload["context"]["config"].endswith("assessment.yaml")


def test_the_report_is_written_where_the_file_says(tmp_path, problem):
    import json

    config = load_config(
        write(
            tmp_path,
            f"accuracy:\n  metrics: [accuracy]\n"
            f"report:\n  output: {tmp_path / 'out' / 'report.json'}\n",
        )
    )
    accuracy = config.run_accuracy(
        problem["handler"], problem["data"], problem["actual"]
    )
    written = config.save_report(config.build_report(accuracy=accuracy))
    assert written.exists()
    assert json.loads(written.read_text())["accuracy"]["overall"]["metrics"]["accuracy"]


def test_no_output_configured_writes_nothing(tmp_path, problem):
    config = load_config(write(tmp_path, ""))
    assert config.save_report(config.build_report()) is None


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_a_config_round_trips_through_yaml(tmp_path):
    original = load_config(
        write(
            tmp_path,
            """
model_name: m
context: {split: holdout}
accuracy:
  metrics: [accuracy]
  per_class: false
  diagnose_ambiguity: true
  criteria:
    - metric: accuracy
      acceptable: 0.9
      rationale: because
robustness:
  perturbations: [gaussian_noise]
  severities: [1, 2]
  seed: 5
  criteria:
    - metric: relative_robustness
      acceptable: 0.9
      basis: aggregate
report:
  output: out.json
  glossary: false
""",
        )
    )
    text = dump_config(original)
    round_trip = tmp_path / "round_trip.yaml"
    round_trip.write_text(text, encoding="utf-8")
    again = load_config(round_trip)
    assert again.to_mapping() == original.to_mapping()
    assert again.accuracy.criteria == original.accuracy.criteria
    assert again.robustness.severities == (1, 2)
    assert again.report.glossary is False



def test_a_config_built_in_python_can_be_written_out(tmp_path):
    from themis_assess.config import AccuracyConfig

    config = AssessmentConfig(
        model_name="built",
        accuracy=AccuracyConfig(
            metrics=("accuracy",),
            criteria=({"metric": "accuracy", "acceptable": 0.8, "rationale": "r"},),
        ),
    )
    text = dump_config(config, tmp_path / "written.yaml")
    assert "acceptable: 0.8" in text
    assert load_config(tmp_path / "written.yaml").model_name == "built"
