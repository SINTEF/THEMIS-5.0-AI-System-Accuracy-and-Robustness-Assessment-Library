"""Run an assessment end to end and summarise it — the shortest complete path.

    python examples/try_summary.py                          # templates only, no model
    python examples/try_summary.py --audience ml_engineer
    python examples/try_summary.py --model                  # use whichever Ollama model
                                                           # this machine has installed
    python examples/try_summary.py --model qwen3.5:9b       # or name one
    python examples/try_summary.py --fast                   # skip the robustness sweep

``report`` is an :class:`~themis_assess.report.AssessmentReport`: the one dictionary that
holds every result. It is what ``summarise`` reads, and it is built by ``ta.build_report``
from whatever assessments you happened to run — every argument to it is optional.

The full version of this walkthrough, with the explanation of each number, is
``docs/tutorial.ipynb`` (§12 is the summary section).
"""

from __future__ import annotations

import argparse

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import themis_assess as ta
from themis_assess.summary import OllamaModel, find_local_model, summarise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audience", default="clinician",
                        help="clinician | ml_engineer | manager | regulator")
    parser.add_argument("--model", nargs="?", const="auto", default=None,
                        help="an Ollama model to write the narrative (e.g. qwen3.5:9b), "
                             "or bare --model to use whichever one is installed")
    parser.add_argument("--fast", action="store_true",
                        help="skip the robustness sweep (seconds instead of a minute)")
    args = parser.parse_args()

    # ---------------------------------------------------------------- the model
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.3, random_state=0, stratify=cancer.target
    )
    model = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)

    handler = ta.SklearnHandler(model, name="random_forest")
    data = ta.ArrayAdapter(X_test, feature_names=list(cancer.feature_names))

    # ------------------------------------------------- what "good enough" means
    # The tolerance is the one thing no assessment can supply; it comes from the
    # deployment. These three are illustrative.
    criteria = ta.AcceptanceCriteria(
        [
            ta.Requirement("balanced_accuracy", acceptable=0.95, critical=True,
                           rationale="both classes matter and the split is 63/37"),
            ta.Requirement("recall", acceptable=0.98,
                           rationale="recall is on the benign class here"),
            ta.Requirement("expected_calibration_error", acceptable=0.05,
                           rationale="the score is shown to a clinician as a probability"),
        ],
        source="example script — illustrative, not a clinical sign-off",
    )

    # ------------------------------------------------------------- the assessments
    accuracy = ta.AccuracyEvaluator(handler, criteria=criteria).run(
        data, y_test, diagnose_outliers=True, diagnose_ambiguity=True
    )

    robustness = None
    if not args.fast:
        robustness = ta.RobustnessEvaluator(
            handler,
            perturbations=["gaussian_noise", "uniform_bias", "missing_values"],
            severities=(1, 2, 3, 4, 5),
            n_variants=3,
            seed=0,
        ).assess(
            data, y_test,
            accuracy_metric="balanced_accuracy",
            criteria=ta.RobustnessCriteria(
                [
                    ta.ToleranceRequirement(
                        "relative_robustness", acceptable=0.90, basis="worst_cell",
                        critical=True,
                        rationale="the worst case is the one a clinic has to survive",
                    )
                ]
            ),
        )

    # ----------------------------------------------------- >>> here is `report` <<<
    report = ta.build_report(
        model_name="random_forest",
        accuracy=accuracy,
        robustness=robustness,
        context={
            "dataset": "breast cancer Wisconsin (diagnostic), UCI id 17",
            "split": "stratified random holdout, 30% test, seed 0",
        },
    )
    print(report.summary())
    print()

    # ------------------------------------------------------------- the summary
    language_model = None
    if args.model == "auto":
        language_model = find_local_model(
            ["qwen3.5:9b", "qwen3", "llama3.2", "gemma3", "phi4"]
        )
        if language_model is None:
            print("no local model found (is ollama running, and `pip install ollama`?) "
                  "— using the templates\n")
    elif args.model:
        language_model = OllamaModel(args.model)

    summary = summarise(report, audience=args.audience, model=language_model)
    print(summary.to_markdown())
    print()
    print("narrative written by:", summary.provenance["narrative_source"])

    # The model's own words, kept whether or not they were accepted — so a rejection is
    # diagnosable rather than a shrug.
    raw = summary.provenance.get("model_output")
    if raw is not None:
        print(f"\n--- {summary.provenance['model']} returned ---\n{raw.strip()}")
        check = summary.provenance.get("verification", {})
        print(f"\n--- verification: accepted={check.get('ok')} ---")
        for problem in check.get("problems", []):
            print(f"  [{problem['kind']}] {problem['detail']}")


if __name__ == "__main__":
    main()
