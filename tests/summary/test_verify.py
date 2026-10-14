"""Tests for the verifier — the layer that makes a generated narrative trustworthy.

This is the module the whole design leans on, so the tests are adversarial: each one
plays the part of a model that has produced something plausible and wrong, and asserts it
is caught. A verifier that only passes good text is worthless.
"""

from __future__ import annotations

import pytest

from themis_assess.report.findings import Finding
from themis_assess.summary.verify import (
    CAUSAL_PHRASES,
    allowed_numbers,
    verify_narrative,
)


@pytest.fixture
def findings():
    return [
        Finding(
            "critical",
            "accuracy.criterion_failed",
            "balanced_accuracy is below its requirement",
            "detail",
            evidence={
                "metric": "balanced_accuracy",
                "value": 0.9266,
                "acceptable": 0.95,
                "margin": -0.0234,
            },
        ),
        Finding(
            "serious",
            "robustness.worst_cell",
            "uniform_bias at severity 5 costs 46%",
            "detail",
            evidence={
                "perturbation": "uniform_bias",
                "severity": 5,
                "clean": 0.9266,
                "worst": 0.5016,
                "retained": 0.5413,
            },
        ),
    ]


def check(text, findings, **kwargs):
    kwargs.setdefault("allowed_metrics", {"balanced_accuracy", "accuracy"})
    return verify_narrative(text, findings, **kwargs)


# --------------------------------------------------------------------------
# numeric grounding — the highest-value check
# --------------------------------------------------------------------------


def test_a_value_from_the_evidence_passes(findings):
    assert check("Balanced accuracy is 0.9266 [F1].", findings).ok


def test_an_invented_value_is_caught(findings):
    """The most damaging error a summary can make, and the one that gets quoted."""
    result = check("Balanced accuracy is 0.8912 [F1].", findings)
    assert not result.ok
    assert any(p.kind == "ungrounded_number" for p in result.problems)


def test_a_value_the_audience_layer_reformatted_passes(findings):
    """0.9266 may legitimately appear as 93% or 92.7%: the audience decides the form."""
    for text in ("It is right 93% of the time [F1].", "Accuracy is 92.7% [F1].",
                 "Accuracy is 0.927 [F1]."):
        assert check(text, findings).ok, text


def test_the_complement_of_a_proportion_passes(findings):
    """'loses 46%' is the natural way to say a retained fraction of 0.5413, and a reader
    should not be denied it just because the subtraction happened in prose."""
    assert check("It loses 46% of what it had [F2].", findings).ok


def test_a_number_near_a_real_one_is_still_caught(findings):
    """The reason the allowed set is built as strings rather than compared with a
    tolerance: a tolerance would accept 0.93 as 'close enough' to 0.9266, and 0.93 was
    never in the report."""
    result = check("Balanced accuracy is 0.9300 [F1].", findings)
    assert not result.ok


def test_nested_evidence_contributes_its_values():
    finding = Finding(
        "serious",
        "accuracy.subgroup_gap",
        "h",
        "d",
        evidence={"subgroup": "medium", "metrics": {"f1": {"in_subgroup": 0.7694}}},
    )
    assert "0.7694" in allowed_numbers([finding])
    assert check("The medium group is at 0.7694 [F1].", [finding],
                 allowed_metrics={"f1"}).ok


def test_small_integers_are_allowed_so_prose_can_count(findings):
    """'three findings', 'level 5 of 5' — a narrative needs these to be readable, and no
    misquoted metric hides in a single digit."""
    assert check("There are 2 findings, the first of 2 [F1].", findings).ok


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------


def test_a_metric_the_report_does_not_contain_is_caught(findings):
    """A model reaching for something it knows from training rather than from the input."""
    result = check("The auroc is 0.9266 [F1].", findings)
    assert not result.ok
    assert any(p.kind == "metric_not_in_report" for p in result.problems)


def test_a_metric_the_report_does_contain_passes(findings):
    assert check("The auroc is 0.9266 [F1].", findings,
                 allowed_metrics={"balanced_accuracy", "auroc"}).ok


def test_a_single_word_metric_name_is_checked(findings):
    """The first version only checked snake_case, so `f1` and `auc` slipped through."""
    assert not check("The f1 score is 0.9266 [F1].", findings).ok
    assert not check("Its auc is 0.9266 [F1].", findings).ok


def test_the_three_metric_names_that_are_english_words_are_exempt(findings):
    """'accuracy', 'recall' and 'precision' appear in plain prose constantly. Flagging
    them would make the check unusable, so they are exempt as bare words — while their
    compounds stay checked."""
    assert check("Its accuracy is 0.9266 [F1].", findings, allowed_metrics=set()).ok
    assert check("Recall matters here [F1].", findings, allowed_metrics=set()).ok
    assert not check("Its worst_group_accuracy is 0.9266 [F1].", findings,
                     allowed_metrics=set()).ok


def test_citation_markers_are_not_read_as_content(findings):
    """The bug this pins: `[F1]` lowercased contains `f1`, so the vocabulary check
    rejected every correctly-cited sentence that happened to cite finding one."""
    assert check("Balanced accuracy is 0.9266 [F1].", findings).ok
    assert check("Both agree [F1, F2].", findings).ok


# --------------------------------------------------------------------------
# causal language
# --------------------------------------------------------------------------


def test_a_causal_claim_is_rejected(findings):
    """Nothing in an assessment establishes why a model fails. A summary that says
    otherwise has invented a mechanism, and no model quality prevents that."""
    result = check("Accuracy is 0.9266 because the training set was small [F1].", findings)
    assert not result.ok
    assert any(p.kind == "causal_claim" for p in result.problems)


@pytest.mark.parametrize("phrase", CAUSAL_PHRASES)
def test_every_banned_phrase_is_actually_detected(phrase, findings):
    """A banned-phrase list nobody checks is a list with typos in it."""
    result = check(f"Accuracy is 0.9266, {phrase} something [F1].", findings)
    assert not result.ok, phrase
    assert any(p.kind == "causal_claim" for p in result.problems)


def test_a_word_merely_containing_a_banned_one_is_not_flagged(findings):
    """'becomes' contains 'become', not 'because', but a substring check would fire on
    near misses. The check is word-bounded."""
    assert check("Accuracy becomes 0.9266 at that point [F1].", findings).ok


def test_stating_two_findings_are_consistent_is_allowed(findings):
    """The prompt permits noting consistency; it forbids asserting production. The
    verifier has to draw the same line, or the instruction is unfollowable."""
    assert check(
        "The degradation is consistent with the accuracy shortfall [F1, F2].", findings
    ).ok


# --------------------------------------------------------------------------
# citations
# --------------------------------------------------------------------------


def test_an_uncited_sentence_is_dropped_not_fatal(findings):
    """Partial credit is right: one unsupported aside should not cost the reader eight
    good paragraphs."""
    result = check(
        "Balanced accuracy is 0.9266 [F1]. This model is basically fine.", findings
    )
    assert result.ok
    assert "basically fine" not in result.kept
    assert any(p.kind == "uncited" for p in result.problems)


def test_a_narrative_with_no_citations_at_all_is_rejected(findings):
    result = check("The model looks good to me.", findings)
    assert not result.ok
    assert result.kept == ""


def test_citations_can_be_switched_off(findings):
    """For a model that cannot follow the instruction. The other three checks still run,
    which is the point — citations are the weakest of the four."""
    assert check("The model is at 0.9266.", findings, require_citations=False).ok
    assert not check("The model is at 0.1234.", findings, require_citations=False).ok


def test_empty_output_is_rejected_rather_than_passed(findings):
    for text in ("", "   ", "\n"):
        result = check(text, findings)
        assert not result.ok
        assert any(p.kind == "empty" for p in result.problems)


# --------------------------------------------------------------------------
# the result object
# --------------------------------------------------------------------------


def test_the_verification_serialises_and_describes_itself(findings):
    result = check("Balanced accuracy is 0.1234 [F1].", findings)
    payload = result.as_dict()
    assert payload["ok"] is False
    assert payload["problems"][0]["kind"] == "ungrounded_number"
    assert "REJECTED" in result.describe()


def test_a_dropped_sentence_is_not_counted_as_fatal(findings):
    result = check("Accuracy is 0.9266 [F1]. An aside.", findings)
    assert result.ok
    assert result.problems  # the drop is reported
    assert not result.fatal  # but it did not invalidate anything


# --------------------------------------------------------------------------
# per-sentence scoping — the hole the tutorial found
# --------------------------------------------------------------------------


def test_a_number_from_another_finding_is_caught(findings):
    """The defect this pins, and it was found by executing the documentation rather than
    by a test.

    A model wrote "its correctness is 98%, which is excellent [F3]" and the first version
    of the grounding check **passed it**, because 0.98 was in the evidence — as a recall
    *threshold* belonging to a different finding. The number existed; the attribution was
    invented, which is the more insidious error.

    So a cited sentence is now scoped to its citations. That makes the citation
    requirement load-bearing rather than decorative.
    """
    result = check("The retained fraction is 0.5413 [F1].", findings)
    assert not result.ok
    assert "finding(s) [1]" in result.problems[0].detail
    # and the same number, cited correctly, passes
    assert check("The retained fraction is 0.5413 [F2].", findings).ok


def test_a_sentence_citing_several_findings_may_use_any_of_their_numbers(findings):
    assert check("It is 0.9266 clean and retains 0.5413 [F1, F2].", findings).ok


def test_small_integers_stay_available_inside_a_scoped_sentence(findings):
    """Scoping must not cost a narrative the ability to count."""
    assert check("This is the 1 of 2 requirements at issue [F1].", findings).ok


def test_an_unscoped_sentence_still_gets_the_global_check(findings):
    """With citations switched off there is nothing to scope to, so the check falls back
    to 'anywhere in the evidence' — weaker, and better than nothing."""
    assert check("Retention was 0.5413.", findings, require_citations=False).ok
    assert not check("Retention was 0.1234.", findings, require_citations=False).ok


def test_numbers_per_finding_partitions_the_evidence(findings):
    from themis_assess.summary.verify import numbers_per_finding

    per = numbers_per_finding(findings)
    assert "0.9266" in per[1]
    assert "0.5413" in per[2]
    assert "0.5413" not in per[1]
