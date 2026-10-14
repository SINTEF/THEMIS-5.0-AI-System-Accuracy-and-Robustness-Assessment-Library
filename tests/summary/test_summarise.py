"""Tests for the summary layer: the audience, the catalogue, and the fallback.

The property that carries the most weight is the last one. A summary must never become
*wrong* by adding a language model — the worst case has to be the no-model case. Several
tests below exist only to pin that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import themis_assess as ta
from themis_assess.summary import (
    AUDIENCES,
    CLINICIAN,
    ML_ENGINEER,
    Audience,
    EchoModel,
    build_prompt,
    default_catalogue,
    get_audience,
    render_finding,
    summarise,
)
from themis_assess.report.findings import Finding


@pytest.fixture(scope="module")
def report():
    """One assessment with several kinds of finding in it."""
    rng = np.random.default_rng(0)
    n = 400
    X = np.column_stack([rng.uniform(20, 85, n), rng.normal(size=n), rng.normal(size=n)])
    y = (0.06 * (X[:, 0] - 50) + 1.4 * X[:, 1] + rng.normal(scale=0.9, size=n) > 0).astype(int)
    data = ta.ArrayAdapter(
        X[200:], feature_names=["age", "severity", "noise"], categorical_features=[]
    )
    actual = y[200:]
    site = rng.choice(["north", "south"], 200, p=[0.7, 0.3])
    handler = ta.SklearnHandler(
        RandomForestClassifier(n_estimators=30, random_state=0).fit(X[:200], y[:200]),
        name="forest",
    )
    subgroups = [
        ta.SubgroupSpec(name, (lambda name: lambda d: site == name)(name))
        for name in ("north", "south")
    ]
    accuracy = ta.AccuracyEvaluator(
        handler,
        subgroups=subgroups,
        criteria=ta.AcceptanceCriteria(
            [ta.Requirement("balanced_accuracy", acceptable=0.99, rationale="strict, for the test")],
            source="test",
        ),
    ).run(data, actual)
    robustness = ta.RobustnessEvaluator(
        handler, perturbations=["gaussian_noise"], severities=(1, 2, 3)
    ).assess(
        data,
        actual,
        accuracy_metric="balanced_accuracy",
        criteria=ta.RobustnessCriteria(
            [ta.ToleranceRequirement("relative_robustness", acceptable=0.99, basis="worst_cell")]
        ),
    )
    return ta.build_report(model_name="forest", accuracy=accuracy, robustness=robustness)


# --------------------------------------------------------------------------
# the audience
# --------------------------------------------------------------------------


def test_a_percentage_is_offered_only_where_the_metric_is_bounded():
    """The defect this pins: formatting by magnitude alone. A negative log-likelihood of
    0.2916 rendered as "29%" is confident nonsense, and plenty of unbounded metrics land
    between 0 and 1 — so the registry decides, not the number's size."""
    assert CLINICIAN.format_value("balanced_accuracy", 0.9266) == "93%"
    assert CLINICIAN.format_value("negative_log_likelihood", 0.2916) == "0.2916"
    assert ML_ENGINEER.format_value("balanced_accuracy", 0.9266) == "0.9266"


def test_an_unregistered_metric_never_becomes_a_percentage():
    assert CLINICIAN.format_value("house_score", 0.5) == "0.5000"


def test_a_missing_value_says_so_rather_than_printing_none():
    assert CLINICIAN.format_value("accuracy", None) == "not measured"


def test_the_vocabulary_is_partial_and_degrades_readably():
    """A metric with no entry must not raise; it becomes its own name, spaced."""
    assert CLINICIAN.name_metric("balanced_accuracy") != "balanced_accuracy"
    assert CLINICIAN.name_metric("some_new_metric") == "some new metric"


def test_a_vocabulary_phrase_becomes_a_grammatical_subject():
    """The defect this pins, found by running the example script rather than by a test:
    templates prefixed every phrase with "The model's", so a vocabulary entry that
    already carries an article produced "The model's the share of positive cases it
    finds". The three phrase shapes each have to survive."""
    assert CLINICIAN.metric_subject("accuracy") == "The model's overall correctness"
    assert CLINICIAN.metric_subject("recall") == (
        "The model's share of positive cases it finds"
    )
    assert CLINICIAN.metric_subject("auc") == "The model's ability to rank cases by risk"
    assert CLINICIAN.metric_subject("flip_rate").startswith("How often")
    assert CLINICIAN.metric_subject("f1").startswith("A combined")
    assert CLINICIAN.metric_subject(None) == "The model's metric"


def test_every_vocabulary_phrase_reads_as_a_subject_for_every_reader():
    """Cheap insurance on a hand-written table: a new entry beginning with a word that
    is neither an article nor a clause opener is the one that reads wrongly, and nobody
    re-reads all four vocabularies when adding one."""
    for audience in AUDIENCES.values():
        for metric in audience.vocabulary:
            subject = audience.metric_subject(metric)
            assert not subject.startswith("The model's the ")
            assert not subject.startswith("The model's its ")
            assert not subject.startswith("The model's how ")
            assert not subject.startswith("The model's whether ")
            assert not subject.startswith("The model's a ")


def test_two_values_that_round_alike_are_told_apart():
    """0.9474 against a required 0.95: both are "95%" to a clinician, so the sentence
    read "is 95%, short of the 95% that was agreed" — which looks like a broken
    assessment rather than a near miss. Precision is added only until they differ."""
    value, required = CLINICIAN.format_pair("balanced_accuracy", 0.9474, 0.95)
    assert (value, required) == ("94.7%", "95.0%")
    # and no precision is added when there was never a collision
    assert CLINICIAN.format_pair("balanced_accuracy", 0.80, 0.95) == ("80%", "95%")
    # equal values are equal: nothing to disambiguate, and no infinite refinement
    assert CLINICIAN.format_pair("balanced_accuracy", 0.95, 0.95) == ("95%", "95%")


def test_a_disambiguated_pair_stays_inside_the_grounded_set():
    """The added precision must not make the verifier reject the library's own
    template — the two layers agree on which renderings of a number are legitimate."""
    from themis_assess.summary.verify import allowed_numbers

    finding = Finding(
        "critical",
        "accuracy.criterion_failed",
        "h",
        "d",
        evidence={"metric": "balanced_accuracy", "value": 0.9474, "acceptable": 0.95},
    )
    allowed = allowed_numbers([finding])
    for rendered in CLINICIAN.format_pair("balanced_accuracy", 0.9474, 0.95):
        assert rendered in allowed, rendered


def test_the_reader_chooses_which_metric_to_lead_with():
    """A finding spans several metrics and only one goes in the sentence. For a reader
    who works in percentages, a metric that cannot be one is the wrong choice even when
    its gap is widest."""
    candidates = ["negative_log_likelihood", "balanced_accuracy"]
    assert sorted(candidates, key=CLINICIAN.prefers_metric)[0] == "balanced_accuracy"


def test_the_reader_changes_the_order_but_never_hides_severity():
    """A clinician meets the subgroup findings first. What must not happen is an
    informational finding outranking a critical one from the same area."""
    findings = [
        Finding("info", "sensitivity.hottest_cell", "h", "d", area="sensitivity"),
        Finding("critical", "accuracy.criterion_failed", "h", "d", area="accuracy"),
        Finding("serious", "accuracy.subgroup_gap", "h", "d", area="subgroups"),
        Finding("critical", "accuracy.no_criteria", "h", "d", area="accuracy"),
    ]
    ordered = CLINICIAN.order(findings)
    assert ordered[0].area == "subgroups"
    accuracy_findings = [f for f in ordered if f.area == "accuracy"]
    assert all(f.severity == "critical" for f in accuracy_findings)


def test_an_unknown_audience_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="Built in"):
        get_audience("nurse")


def test_an_audience_can_be_passed_as_an_object():
    custom = Audience(name="mine", label="me")
    assert get_audience(custom) is custom
    assert set(AUDIENCES) == {"clinician", "ml_engineer", "manager", "regulator"}


# --------------------------------------------------------------------------
# the templates
# --------------------------------------------------------------------------


def test_the_same_finding_reads_differently_for_two_readers():
    finding = Finding(
        "critical",
        "accuracy.criterion_failed",
        "h",
        "d",
        evidence={
            "metric": "balanced_accuracy",
            "value": 0.9266,
            "acceptable": 0.95,
            "margin": -0.0234,
            "rationale": "both diagnoses matter",
        },
    )
    clinical = render_finding(finding, CLINICIAN)
    technical = render_finding(finding, ML_ENGINEER)
    assert clinical != technical
    assert "93%" in clinical and "balanced_accuracy" not in clinical
    assert "balanced_accuracy" in technical and "0.9266" in technical


def test_a_rule_with_no_template_falls_back_to_the_findings_own_detail():
    """A rule can be added to the library before anyone writes prose for it. The
    degradation is to something correct and dry, never to something missing."""
    finding = Finding("info", "a.rule.nobody.templated", "h", "the detail sentence")
    assert render_finding(finding, CLINICIAN) == "the detail sentence"


def test_a_template_missing_a_value_falls_back_rather_than_raising():
    """Evidence is populated by the finding generators, but a hand-built finding may not
    have every key. A summary must not die of it."""
    finding = Finding("critical", "accuracy.criterion_failed", "h", "fallback detail",
                      evidence={})
    assert render_finding(finding, CLINICIAN) == "fallback detail"


def test_every_perturbation_in_the_registry_has_a_plain_language_gloss():
    """A clinician should never be shown a perturbation's internal name. Anything
    unglossed degrades to the spaced name, which is legible but not an explanation — so
    this test names the gaps rather than asserting there are none."""
    from themis_assess.perturbations.registry import default_registry
    from themis_assess.summary.templates import PERTURBATION_MEANING

    unglossed = sorted(set(default_registry().names()) - set(PERTURBATION_MEANING))
    # text perturbations are not reachable from a tabular assessment; the rest should be
    text_only = {"character_noise", "token_masking", "word_dropout"}
    assert set(unglossed) <= text_only | {"alpha_stable_noise", "exponential_damping",
                                          "input_window_shift", "window_slice"}


# --------------------------------------------------------------------------
# the catalogue
# --------------------------------------------------------------------------


def test_every_rule_the_library_emits_has_at_least_one_action():
    """A finding with no advice attached is a finding nobody can act on. This is the
    test that stops a new rule shipping without an entry."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent / "src" / "themis_assess" / "report"
    source = "".join((root / f"{name}.py").read_text() for name in ("findings", "rules"))
    emitted = set(re.findall(r'rule="([a-z_.]+)"', source))
    emitted |= {f"reporting.{m}" for m in re.findall(r'_issue\(\s*\n\s*"([a-z_]+)"', source)}
    assert emitted, "the rule inventory came back empty; the regex needs updating"
    assert default_catalogue().uncovered(emitted) == []


def test_no_catalogue_entry_is_orphaned():
    """The reverse direction: an action keyed to a rule nothing emits is dead weight, and
    usually a typo in the rule id."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent / "src" / "themis_assess" / "report"
    source = "".join((root / f"{name}.py").read_text() for name in ("findings", "rules"))
    emitted = set(re.findall(r'rule="([a-z_.]+)"', source))
    emitted |= {f"reporting.{m}" for m in re.findall(r'_issue\(\s*\n\s*"([a-z_]+)"', source)}
    assert sorted(set(default_catalogue().rules) - emitted) == []


def test_most_actions_carry_a_caveat():
    """An action list with no caveats reads as a recipe and gets followed rather than
    considered. Not every entry needs one, but the great majority should."""
    catalogue = default_catalogue()
    every = [a for rule in catalogue.rules for a in catalogue.for_rule(rule)]
    with_caveat = [a for a in every if a.caveat]
    assert len(with_caveat) / len(every) > 0.9


def test_file_order_is_preserved_because_it_is_meaningful():
    """Entries are written cheapest-and-most-diagnostic first: 'check whether the
    subgroup axis is confounded' before 'reweight the training data'."""
    actions = default_catalogue().for_rule("accuracy.subgroup_gap")
    assert "confounded" in actions[0].action
    assert "Reweight" in actions[-1].action


def test_a_misspelt_field_in_the_catalogue_is_refused(tmp_path):
    """Silently dropping an unknown key would mean a caveat somebody wrote never
    appearing, with nothing to say so."""
    from themis_assess.summary.actions import ActionCatalogue

    path = tmp_path / "actions.yaml"
    path.write_text("some.rule:\n  - action: do a thing\n    caveats: note the plural\n")
    with pytest.raises(ValueError, match="unknown field"):
        ActionCatalogue.from_yaml(path)


def test_an_entry_without_an_action_is_refused(tmp_path):
    from themis_assess.summary.actions import ActionCatalogue

    path = tmp_path / "actions.yaml"
    path.write_text("some.rule:\n  - caveat: no action here\n")
    with pytest.raises(ValueError, match="needs at least"):
        ActionCatalogue.from_yaml(path)


def test_an_unknown_rule_yields_no_actions_and_no_error():
    """The catalogue is expected to be incomplete for third-party rules."""
    assert default_catalogue().for_rule("someone.elses.rule") == ()


def test_actions_are_deduplicated_across_repeated_rules():
    findings = [
        Finding("serious", "accuracy.subgroup_gap", "h", "d"),
        Finding("warning", "accuracy.subgroup_gap", "h", "d"),
    ]
    actions = default_catalogue().for_findings(findings)
    assert len(actions) == len(default_catalogue().for_rule("accuracy.subgroup_gap"))


# --------------------------------------------------------------------------
# summarise, without a model
# --------------------------------------------------------------------------


def test_a_summary_without_a_model_is_complete(report):
    summary = summarise(report, audience="clinician")
    assert summary.headline
    assert summary.themes
    assert summary.statements
    assert summary.provenance["narrative_source"] == "templates"
    assert not summary.used_model
    assert "no language model was used" in summary.to_markdown()


def test_the_two_readers_get_materially_different_documents(report):
    clinical = summarise(report, audience="clinician").to_markdown()
    technical = summarise(report, audience="ml_engineer").to_markdown()
    assert clinical != technical
    assert "%" in clinical
    assert "balanced_accuracy" in technical
    assert "balanced_accuracy" not in clinical.split("## The findings")[0]


def test_the_severity_tally_is_ordered_by_severity_not_by_reading_order(report):
    """Insertion order follows the reader's area preference, which would put 'serious'
    before 'critical' and read as a mistake."""
    counts = summarise(report, audience="clinician").counts()
    order = ["critical", "serious", "warning", "info"]
    positions = [order.index(k) for k in counts]
    assert positions == sorted(positions)


def test_every_statement_is_cited_in_some_theme(report):
    summary = summarise(report, audience="ml_engineer")
    cited = {fid for theme in summary.themes for fid in theme.finding_ids}
    assert cited == set(summary.statements)


def test_the_summary_serialises(report):
    payload = summarise(report, audience="manager").to_dict()
    json.dumps(payload)
    assert payload["schema"] == "themis-assess/summary@1"
    assert payload["audience"] == "manager"
    assert payload["provenance"]["narrative_source"] == "templates"


def test_an_empty_report_summarises_to_an_honest_nothing():
    summary = summarise([], audience="clinician")
    assert "nothing" in summary.headline.lower()
    assert summary.themes == ()
    assert summary.actions == ()


def test_a_plain_list_of_findings_works_as_input():
    findings = [Finding("critical", "accuracy.no_criteria", "h", "d", area="accuracy")]
    summary = summarise(findings, audience="regulator")
    assert summary.statements
    assert summary.actions


def test_capping_the_findings_records_what_it_dropped(report):
    summary = summarise(report, audience="clinician", max_findings=2)
    assert len(summary.statements) == 2
    assert summary.provenance["findings_dropped"] == len(report.findings()) - 2


# --------------------------------------------------------------------------
# summarise, with a model — and the fallback
# --------------------------------------------------------------------------


def _response(themes, headline="A headline."):
    return json.dumps({"headline": headline, "themes": themes})


def test_a_well_behaved_model_is_used(report):
    statements = summarise(report, audience="clinician").statements
    first = next(iter(statements.values()))
    number = next((w for w in first.split() if "%" in w or "0." in w), "")
    model = EchoModel(
        _response([{"title": "T", "findings": [1], "prose": f"It reports {number} [F1]."}])
    )
    summary = summarise(report, audience="clinician", model=model)
    assert summary.provenance["narrative_source"] == "model"
    assert summary.used_model
    assert summary.provenance["verification"]["ok"] is True


def test_a_model_that_invents_a_number_is_discarded(report):
    """The property the whole design rests on: adding a model cannot make the summary
    wrong, only better or unchanged."""
    bad = EchoModel(
        _response([{"title": "T", "findings": [1], "prose": "Accuracy is 0.8137 [F1]."}])
    )
    with_model = summarise(report, audience="clinician", model=bad)
    without = summarise(report, audience="clinician")

    assert with_model.provenance["narrative_source"] == "templates_after_rejection"
    # the invented figure reaches no paragraph...
    assert all("0.8137" not in theme.prose for theme in with_model.themes)
    assert [t.prose for t in with_model.themes] == [t.prose for t in without.themes]
    # ...but it *is* named in the provenance and in the note the document carries, which
    # is the point: a discarded narrative should be diagnosable, not silently vanished
    assert "0.8137" in str(with_model.provenance["verification"])
    assert "0.8137" in with_model.to_markdown()


def test_a_model_that_asserts_a_cause_is_discarded(report):
    bad = EchoModel(
        _response([{"title": "T", "findings": [1],
                    "prose": "It fails because the data was thin [F1]."}])
    )
    summary = summarise(report, audience="clinician", model=bad)
    assert summary.provenance["narrative_source"] == "templates_after_rejection"
    assert "because" in summary.provenance["rejected_because"]


def test_a_model_returning_junk_falls_back_rather_than_raising(report):
    for junk in ("I cannot help with that.", "", "{not json", "[]"):
        summary = summarise(report, audience="clinician", model=EchoModel(junk))
        assert summary.provenance["narrative_source"] == "templates_after_rejection"
        assert summary.themes  # still a usable document


def test_a_model_that_raises_falls_back(report):
    class Broken:
        name = "broken"

        def complete(self, prompt, *, schema=None):
            raise RuntimeError("the daemon is not running")

    summary = summarise(report, audience="clinician", model=Broken())
    assert summary.provenance["narrative_source"] == "templates_after_rejection"
    assert "daemon" in summary.provenance["rejected_because"]


def test_the_fallback_is_reported_in_the_document_not_hidden(report):
    """A reader has to be able to tell which path wrote what they are reading."""
    bad = EchoModel(_response([{"title": "T", "findings": [1], "prose": "Bad 0.4242 [F1]."}]))
    text = summarise(report, audience="clinician", model=bad).to_markdown()
    assert "rejected by verification" in text


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------


def test_the_prompt_carries_the_findings_and_the_reader(report):
    findings = CLINICIAN.order(report.findings())
    prompt = build_prompt(findings, CLINICIAN)
    assert CLINICIAN.reads_for in prompt
    assert CLINICIAN.tone in prompt
    assert "F1" in prompt
    for rule in ("Do not write any number", "Do not explain WHY", "Do not recommend actions"):
        assert rule in prompt


def test_the_prompt_is_small_because_it_carries_findings_not_the_report(report):
    """~630 tokens of findings against ~7,900 of full report, and 40 numbers against 142.
    Feeding the dictionary would triple the numeric surface for twelve times the context."""
    prompt = build_prompt(CLINICIAN.order(report.findings()), CLINICIAN)
    assert len(prompt) < 8000
    assert len(prompt) < len(json.dumps(report.to_dict()))


def test_the_prompt_shows_the_reader_facing_sentences(report):
    """The model is asked to arrange sentences that are already correct, not to compose
    them from raw numbers. That is what makes a small model sufficient."""
    findings = CLINICIAN.order(report.findings())
    prompt = build_prompt(findings, CLINICIAN)
    assert render_finding(findings[0], CLINICIAN)[:40] in prompt


# --------------------------------------------------------------------------
# the model's own output, kept for inspection
# --------------------------------------------------------------------------


def test_the_models_raw_output_is_kept_when_it_is_accepted(report):
    statements = summarise(report, audience="clinician").statements
    first = next(iter(statements.values()))
    number = next((w for w in first.split() if "%" in w or "0." in w), "")
    model = EchoModel(
        _response([{"title": "T", "findings": [1], "prose": f"It reports {number} [F1]."}])
    )
    summary = summarise(report, audience="clinician", model=model)
    assert json.loads(summary.provenance["model_output"])["themes"][0]["title"] == "T"


def test_the_models_raw_output_is_kept_when_it_is_rejected(report):
    """The question asked first when a summary looks thin is "what did the model actually
    say?". Discarding the text along with the verdict makes that unanswerable, and hides
    the difference between a model that needs a better prompt and one that needs
    replacing."""
    bad = EchoModel(
        _response([{"title": "T", "findings": [1], "prose": "Accuracy is 0.8137 [F1]."}])
    )
    summary = summarise(report, audience="clinician", model=bad)
    assert summary.provenance["narrative_source"] == "templates_after_rejection"
    assert "0.8137" in summary.provenance["model_output"]
    # and it survives serialisation, so a stored report can still be audited
    assert "0.8137" in json.dumps(summary.to_dict())


def test_no_model_output_is_recorded_when_no_model_was_called(report):
    assert "model_output" not in summarise(report, audience="clinician").provenance


def test_the_prompt_is_stored_only_when_asked_for(report):
    model = EchoModel("not json")
    lean = summarise(report, audience="clinician", model=model)
    full = summarise(report, audience="clinician", model=model, keep_prompt=True)
    assert "prompt" not in lean.provenance
    assert lean.provenance["prompt_characters"] == len(full.provenance["prompt"])


# --------------------------------------------------------------------------
# finding a model that is installed
# --------------------------------------------------------------------------


def test_no_ollama_means_an_empty_list_and_no_model_rather_than_an_error():
    """Runs for real in CI, where ollama is neither installed nor running. The
    documentation and the example script both call this on machines that have nothing
    pulled, so returning None is the load-bearing behaviour, not an edge case."""
    from themis_assess.summary import find_local_model, list_ollama_models

    if list_ollama_models():
        pytest.skip("this machine has models pulled")
    assert list_ollama_models() == []
    assert find_local_model(["qwen3.5:9b"]) is None


@pytest.fixture
def fake_ollama(monkeypatch):
    """A stand-in for the ollama package, listing three models."""
    import sys
    import types

    module = types.ModuleType("ollama")

    class Client:
        def __init__(self, host=None):
            self.host = host

        def list(self):
            return {"models": [{"model": "llama3.2:3b"}, {"model": "qwen3.5:9b"},
                               {"model": "phi4:latest"}]}

    module.Client = Client
    monkeypatch.setitem(sys.modules, "ollama", module)
    return module


def test_an_installed_model_is_found_by_family_or_by_tag(fake_ollama):
    from themis_assess.summary import find_local_model, list_ollama_models

    assert list_ollama_models() == ["llama3.2:3b", "qwen3.5:9b", "phi4:latest"]
    assert find_local_model(["qwen3.5:9b"]).model == "qwen3.5:9b"
    # a family name matches whatever tag is installed, which is what a document that
    # cannot know the reader's tags needs
    assert find_local_model(["qwen3.5"]).model == "qwen3.5:9b"
    assert find_local_model(["gemma3", "phi4"]).model == "phi4:latest"


def test_an_unavailable_preference_falls_back_to_something_installed(fake_ollama):
    """Better than returning None: an unexpected model's output is verified before it is
    trusted, so the cost of guessing wrong is a rejected narrative, not a wrong one."""
    from themis_assess.summary import find_local_model

    assert find_local_model(["nothing-like-this"]).model == "llama3.2:3b"
    assert find_local_model().model == "llama3.2:3b"


def test_a_daemon_that_is_not_running_looks_the_same_as_no_ollama(monkeypatch):
    import sys
    import types

    module = types.ModuleType("ollama")

    class Client:
        def __init__(self, host=None):
            raise ConnectionError("connection refused")

    module.Client = Client
    monkeypatch.setitem(sys.modules, "ollama", module)

    from themis_assess.summary import find_local_model, list_ollama_models

    assert list_ollama_models() == []
    assert find_local_model() is None


# --------------------------------------------------------------------------
# the headline is checked, but not for citations
# --------------------------------------------------------------------------


def test_an_uncited_headline_is_not_a_problem(report):
    """The defect this pins, found by executing the notebook: the headline was checked
    together with the body, so every narrative — including every accepted one — came back
    with a spurious "uncited" problem against its first sentence. A headline is a claim
    about the assessment as a whole and has no single finding to cite; the template
    fallback's own headline carries no citation either."""
    statements = summarise(report, audience="clinician").statements
    first = next(iter(statements.values()))
    number = next((w for w in first.split() if "%" in w or "0." in w), "")
    model = EchoModel(
        _response(
            [{"title": "T", "findings": [1], "prose": f"It reports {number} [F1]."}],
            headline="Two things stand out about this model.",
        )
    )
    summary = summarise(report, audience="clinician", model=model)
    assert summary.provenance["narrative_source"] == "model"
    assert summary.headline == "Two things stand out about this model."
    assert summary.provenance["verification"]["problems"] == []


def test_a_headline_may_not_invent_a_number_either(report):
    """Exempting the headline from citations must not exempt it from grounding — it is
    the one sentence certain to be read and quoted."""
    model = EchoModel(
        _response(
            [{"title": "T", "findings": [1], "prose": "Something happened [F1]."}],
            headline="Accuracy reaches 0.7531.",
        )
    )
    summary = summarise(report, audience="clinician", model=model)
    assert summary.provenance["narrative_source"] == "templates_after_rejection"
    assert "0.7531" in summary.provenance["rejected_because"]


def test_a_missing_headline_keeps_the_template_one_without_rejecting_the_body(report):
    statements = summarise(report, audience="clinician").statements
    first = next(iter(statements.values()))
    number = next((w for w in first.split() if "%" in w or "0." in w), "")
    model = EchoModel(
        _response(
            [{"title": "T", "findings": [1], "prose": f"It reports {number} [F1]."}],
            headline="",
        )
    )
    summary = summarise(report, audience="clinician", model=model)
    assert summary.provenance["narrative_source"] == "model"
    assert summary.headline.startswith("This model does not meet")
