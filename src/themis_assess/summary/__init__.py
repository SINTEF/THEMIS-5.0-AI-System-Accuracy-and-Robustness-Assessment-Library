"""Turning an assessment into prose someone will read — in three parts, only one of
which involves a language model.

The split matters more than any of the pieces, so it is worth stating before the API:

**(a) State what was measured.** :mod:`~themis_assess.summary.templates`. Sentences that
exist in a file, with the numbers substituted from each finding's evidence. Nothing is
generated, so nothing can be invented. This is the layer everything else falls back to,
which is why it is written to be shippable rather than to be a placeholder.

**(b) Write it as a narrative for a particular reader.**
:mod:`~themis_assess.summary.narrative`. The one job a small local model does well:
grouping eight findings into three themes and phrasing them for a clinician rather than
an engineer. Its output is checked by :mod:`~themis_assess.summary.verify` and discarded
on failure, so adding a model can make the summary better or leave it unchanged — never
make it wrong.

**(c) Recommend what to do.** :mod:`~themis_assess.summary.actions`, from a YAML file
people write. A model may select and order entries; it may never author one. What to do
about a weak model is knowledge about *your deployment*, and a model asked to invent it
produces fluent, unfounded advice — the hallucination that matters most, because
recommendations are what people act on.

Who the reader is
-----------------

:mod:`~themis_assess.summary.audience` makes that a parameter, and it shapes the
templates as well as the prompt. A clinician gets percentages, no metric names, and the
per-class and subgroup findings first; an engineer gets four decimal places, the metric
names they will grep for, and robustness first. Since the audience reaches the
deterministic layer too, the fallback is fitted rather than generic — which is what keeps
the model genuinely optional.

Getting started
---------------

.. code-block:: python

    from themis_assess.summary import summarise

    # no model needed
    print(summarise(report, audience="clinician").to_markdown())

    # with a local model, verified and with a fallback
    from themis_assess.summary import OllamaModel
    summary = summarise(report, audience="clinician", model=OllamaModel("qwen3.5:9b"))
    print(summary.provenance["narrative_source"])   # model | templates | templates_after_rejection
"""

from themis_assess.summary.actions import (
    CATALOGUE_PATH,
    Action,
    ActionCatalogue,
    default_catalogue,
)
from themis_assess.summary.audience import (
    AUDIENCES,
    CLINICIAN,
    MANAGER,
    ML_ENGINEER,
    REGULATOR,
    Audience,
    get_audience,
)
from themis_assess.summary.narrative import (
    NARRATIVE_SCHEMA,
    EchoModel,
    LanguageModel,
    OllamaModel,
    build_prompt,
    find_local_model,
    findings_for_prompt,
    list_ollama_models,
)
from themis_assess.summary.summarise import Summary, Theme, summarise
from themis_assess.summary.templates import render_finding
from themis_assess.summary.verify import (
    CAUSAL_PHRASES,
    Problem,
    Verification,
    allowed_numbers,
    verify_narrative,
)

__all__ = [
    # the entry point
    "summarise",
    "Summary",
    "Theme",
    # (a) who is reading, and the deterministic sentences
    "Audience",
    "AUDIENCES",
    "get_audience",
    "CLINICIAN",
    "ML_ENGINEER",
    "MANAGER",
    "REGULATOR",
    "render_finding",
    # (b) the model socket
    "LanguageModel",
    "OllamaModel",
    "EchoModel",
    "NARRATIVE_SCHEMA",
    "build_prompt",
    "findings_for_prompt",
    "list_ollama_models",
    "find_local_model",
    # the verifier that makes (b) safe
    "verify_narrative",
    "Verification",
    "Problem",
    "allowed_numbers",
    "CAUSAL_PHRASES",
    # (c) the catalogue
    "Action",
    "ActionCatalogue",
    "default_catalogue",
    "CATALOGUE_PATH",
]
