"""Putting (a), (b) and (c) together, with the fallback that makes it safe.

One entry point, :func:`summarise`. It takes a report, a reader, and optionally a model,
and returns a :class:`Summary` that records not just what it says but **how it was
produced** — which path wrote the prose, and what the verifier found. That provenance is
not decoration: a summary whose origin is unrecorded is a summary nobody can audit.

The order of operations
-----------------------

1. Rank the findings, then reorder them for this reader — severity still breaks ties
   within an area, so the reader's priorities change what they meet first, never whether
   a critical finding can hide behind an informational one.
2. Render each one deterministically (part **a**).
3. If a model is configured: build the prompt, call it, parse, and verify. On any failure,
   keep the templates and record why (part **b**).
4. Retrieve actions from the catalogue for the rules that fired (part **c**).

Step 3 is the only fallible step, and its failure mode is "the summary is the one step 2
already produced".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from themis_assess.summary.actions import Action, ActionCatalogue, default_catalogue
from themis_assess.summary.audience import Audience, get_audience
from themis_assess.summary.narrative import (
    NARRATIVE_SCHEMA,
    LanguageModel,
    build_prompt,
    parse_narrative,
)
from themis_assess.summary.templates import render_finding
from themis_assess.summary.verify import verify_narrative

__all__ = ["Summary", "Theme", "summarise"]


@dataclass(frozen=True, slots=True)
class Theme:
    """A group of findings and the paragraph about them."""

    title: str
    prose: str
    finding_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"title": self.title, "prose": self.prose, "findings": list(self.finding_ids)}


@dataclass(frozen=True, slots=True)
class Summary:
    """A readable account of an assessment, and the record of how it was made.

    Parameters
    ----------
    headline:
        One sentence. The thing to read if nothing else is read.
    themes:
        The narrative, grouped.
    statements:
        ``{finding_id: sentence}`` — the deterministic rendering of every finding,
        always present whether or not a model was used. This is what makes the summary
        checkable: a reader who distrusts a paragraph can find the sentences behind it.
    actions:
        Retrieved from the catalogue, never generated.
    provenance:
        Who it was written for, which model (if any), whether verification passed, what
        it objected to, and — when a model was called — the model's raw output under
        ``model_output``, kept whether or not it was accepted. Never omitted and never
        summarised away.
    """

    model_name: str
    audience: Audience
    headline: str
    themes: tuple[Theme, ...]
    statements: Mapping[str, str]
    findings: tuple[Any, ...]
    actions: tuple[Action, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    # -- reading ------------------------------------------------------------

    @property
    def used_model(self) -> bool:
        """Whether the prose came from a model or from the templates."""
        return self.provenance.get("narrative_source") == "model"

    def counts(self) -> dict[str, int]:
        """Findings per severity, most severe first — not in the reader's area order,
        which is what an insertion-ordered tally would give and reads as a mistake."""
        order = {"critical": 0, "serious": 1, "warning": 2, "info": 3}
        out: dict[str, int] = {}
        for finding in self.findings:
            severity = getattr(finding, "severity", "unknown")
            out[severity] = out.get(severity, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: order.get(kv[0], 9)))

    def to_markdown(self, *, include_actions: bool = True, include_evidence: bool = True) -> str:
        """The summary as markdown."""
        counts = self.counts()
        tally = ", ".join(f"{n} {sev}" for sev, n in counts.items()) or "no findings"

        lines = [
            f"# {self.model_name}: assessment summary",
            "",
            f"*Written for {self.audience.label}.*",
            "",
            f"**{self.headline}**",
            "",
            f"Findings: {tally}.",
        ]

        for theme in self.themes:
            lines += ["", f"## {theme.title}", "", theme.prose]

        if include_evidence and self.statements:
            lines += ["", "## The findings behind this", ""]
            for finding_id, statement in self.statements.items():
                lines.append(f"- **{finding_id}** {statement}")

        if include_actions and self.actions:
            lines += [
                "",
                "## What could be done",
                "",
                "*Retrieved from a reviewed catalogue, not generated. Each carries the "
                "condition under which it applies and what it trades away.*",
                "",
            ]
            for action in self.actions:
                lines.append(f"- **{action.action}**")
                if action.owner:
                    lines.append(f"  - who: {action.owner}")
                if action.precondition:
                    lines.append(f"  - when: {action.precondition}")
                if action.cost:
                    lines.append(f"  - cost: {action.cost}")
                if action.caveat:
                    lines.append(f"  - caveat: {action.caveat}")

        source = self.provenance.get("narrative_source")
        note = {
            "model": f"Narrative written by {self.provenance.get('model')} and verified: "
            f"{self.provenance.get('verification', {}).get('ok')}.",
            "templates": "Narrative assembled from templates; no language model was used.",
            "templates_after_rejection": (
                f"A language model's narrative was rejected by verification and "
                f"discarded. Reason: {self.provenance.get('rejected_because')}. "
                f"This text is the template fallback."
            ),
        }.get(source, "")
        if note:
            lines += ["", "---", "", f"*{note}*"]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The serialisable form, for the results bundle."""
        from themis_assess.report.bundle import to_plain

        return {
            "schema": "themis-assess/summary@1",
            "model_name": self.model_name,
            "audience": self.audience.name,
            "headline": self.headline,
            "themes": [t.as_dict() for t in self.themes],
            "statements": dict(self.statements),
            "actions": [a.as_dict() for a in self.actions],
            "counts": self.counts(),
            "provenance": to_plain(dict(self.provenance)),
        }

    def __str__(self) -> str:
        return self.to_markdown()


def _template_narrative(
    findings: Sequence[Any], statements: Mapping[str, str], audience: Audience
) -> tuple[str, tuple[Theme, ...]]:
    """Part (a) standing alone: one theme per finding area, sentences in order.

    Not a placeholder. This is what ships when no model is configured, and what a
    rejected model's output is replaced by, so it has to read acceptably on its own.
    """
    by_area: dict[str, list[tuple[str, Any]]] = {}
    ids = list(statements)
    for finding_id, finding in zip(ids, findings, strict=True):
        by_area.setdefault(getattr(finding, "area", "other"), []).append((finding_id, finding))

    titles = {
        "accuracy": "How accurate it is, and whether that is enough",
        "classes": "Which categories it gets wrong",
        "subgroups": "Whether it works equally well for everyone",
        "calibration": "Whether its stated confidence can be believed",
        "diagnostics": "Where its mistakes cluster",
        "robustness": "What happens when the data changes",
        "sensitivity": "Which inputs it depends on",
        "attribution": "What the cases it fails on have in common",
        "reporting": "Where this assessment is incomplete",
    }

    themes = []
    for area, entries in by_area.items():
        prose = " ".join(f"{statements[fid]} [{fid}]" for fid, _ in entries)
        themes.append(
            Theme(
                title=titles.get(area, area.replace("_", " ").capitalize()),
                prose=prose,
                finding_ids=tuple(fid for fid, _ in entries),
            )
        )

    return _template_headline(findings, statements, audience), tuple(themes)


def _template_headline(
    findings: Sequence[Any], statements: Mapping[str, str], audience: Audience
) -> str:
    """One sentence, in this reader's register.

    The first version of this used the top finding's own ``headline`` verbatim, which put
    "subgroup 'large' is the worst served, on 3 of 10 metric(s)" at the top of a
    clinician's summary. A finding's headline is written for the findings list, where the
    reader is technical by construction; it is not a sentence for an arbitrary audience.

    So the headline is built from severity instead — a claim about *this assessment* that
    is true by counting, rather than a restatement of one finding — and the first
    finding's own rendered statement follows it.
    """
    if not findings:
        return "No findings: nothing in this assessment reached a reporting threshold."

    counts: dict[str, int] = {}
    for finding in findings:
        severity = getattr(finding, "severity", "unknown")
        counts[severity] = counts.get(severity, 0) + 1

    unmet = sum(
        1 for f in findings if getattr(f, "rule", "").endswith(("criterion_failed",
                                                                "tolerance_exceeded"))
    )
    if unmet:
        lead = (
            f"This model does not meet {unmet} of the requirements that were set for it"
            if unmet > 1
            else "This model does not meet one of the requirements that was set for it"
        )
    elif counts.get("serious"):
        lead = (
            f"No stated requirement is unmet, but {counts['serious']} substantial "
            f"weakness(es) were found"
        )
    elif counts.get("warning"):
        lead = "Nothing serious was found, but some checks need a human decision"
    else:
        lead = "Nothing in this assessment reached a level that needs action"

    first = next(iter(statements.values()), "")
    opening = first.split(". ")[0].rstrip(".")
    return f"{lead}. Most immediately: {opening[0].lower() + opening[1:]}." if opening else f"{lead}."


def summarise(
    report: Any,
    *,
    audience: str | Audience = "ml_engineer",
    model: LanguageModel | None = None,
    catalogue: ActionCatalogue | None = None,
    max_findings: int | None = None,
    require_citations: bool = True,
    keep_prompt: bool = False,
) -> Summary:
    """Summarise an assessment report for one reader.

    Parameters
    ----------
    report:
        An :class:`~themis_assess.report.bundle.AssessmentReport`, or any object with a
        ``findings()`` method — a plain list of findings also works.
    audience:
        A name from :data:`~themis_assess.summary.audience.AUDIENCES`, or an
        :class:`~themis_assess.summary.audience.Audience`.
    model:
        Optional. Without one the summary is built from templates and is complete.
    max_findings:
        Cap on how many findings reach the narrative, most severe first. A cap keeps the
        prompt small; it also silently drops findings, so it is off by default and the
        number kept is recorded in the provenance when it is on.
    keep_prompt:
        Store the full prompt in the provenance as well as its length. Off by default
        because it is several kilobytes that end up in the serialised report, and it is
        reproducible at any time from
        :func:`~themis_assess.summary.narrative.build_prompt`. Worth turning on when a
        summary will be reviewed later by someone who cannot re-run it.

    Returns
    -------
    Summary
        Whose ``provenance["narrative_source"]`` is ``"model"``, ``"templates"``, or
        ``"templates_after_rejection"``. Read it: the three are not interchangeable.
    """
    reader = get_audience(audience)
    catalogue = catalogue if catalogue is not None else default_catalogue()

    if hasattr(report, "findings") and callable(report.findings):
        findings = list(report.findings())
        model_label = getattr(report, "model_name", "model")
    else:
        findings = list(report)
        model_label = "model"

    ordered = reader.order(findings)
    dropped = 0
    if max_findings is not None and len(ordered) > max_findings:
        dropped = len(ordered) - max_findings
        ordered = ordered[:max_findings]

    statements = {
        f"F{index}": render_finding(finding, reader)
        for index, finding in enumerate(ordered, start=1)
    }

    provenance: dict[str, Any] = {
        "audience": reader.name,
        "n_findings": len(ordered),
        "findings_dropped": dropped,
        "catalogue_actions": len(catalogue),
    }

    headline, themes = _template_narrative(ordered, statements, reader)
    provenance["narrative_source"] = "templates"

    if model is not None and ordered:
        provenance["model"] = model.name
        prompt = build_prompt(ordered, reader)
        provenance["prompt_characters"] = len(prompt)
        if keep_prompt:
            provenance["prompt"] = prompt
        try:
            raw = model.complete(prompt, schema=NARRATIVE_SCHEMA)
            # Kept whether or not it survives. A rejected narrative that is thrown away
            # unrecorded leaves nobody able to answer "what did the model actually say?"
            # — which is the first question asked when a summary looks thin, and the
            # only way to tell a model that needs a better prompt from one that needs
            # replacing. It is untrusted text held as data, never as an instruction.
            provenance["model_output"] = raw
            payload = parse_narrative(raw)
            candidate_themes = tuple(
                Theme(
                    title=str(t.get("title", "")).strip(),
                    prose=str(t.get("prose", "")).strip(),
                    finding_ids=tuple(f"F{int(i)}" for i in t.get("findings", [])),
                )
                for t in payload.get("themes", [])
            )
            # The headline is checked apart from the body, and without the citation
            # rule. A headline is a claim about the assessment as a whole — "this model
            # does not meet 3 of its requirements" — and there is no single finding for
            # it to cite; the template fallback's own headline carries no citation
            # either. Checked together, every narrative came back with a spurious
            # "uncited" problem against its first sentence, which is exactly the kind of
            # noise that teaches a reader to stop reading the problem list.
            #
            # Its numbers are still checked, against the whole evidence rather than one
            # finding's. That is the weaker form of the grounding check, and it is the
            # most that is available where there is nothing to scope to.
            metrics = _metrics_in(report)
            verification = verify_narrative(
                " ".join(t.prose for t in candidate_themes),
                ordered,
                allowed_metrics=metrics,
                require_citations=require_citations,
            )
            candidate_headline = str(payload.get("headline", "")).strip()
            if candidate_headline:
                verification = verify_narrative(
                    candidate_headline,
                    ordered,
                    allowed_metrics=metrics,
                    require_citations=False,
                ).merge(verification)
            provenance["verification"] = verification.as_dict()
            if verification.ok and candidate_themes:
                headline = candidate_headline or headline
                themes = candidate_themes
                provenance["narrative_source"] = "model"
            else:
                provenance["narrative_source"] = "templates_after_rejection"
                provenance["rejected_because"] = verification.describe()
        except Exception as exc:  # noqa: BLE001 - any model failure falls back
            provenance["narrative_source"] = "templates_after_rejection"
            provenance["rejected_because"] = f"{type(exc).__name__}: {exc}"

    return Summary(
        model_name=model_label,
        audience=reader,
        headline=headline,
        themes=themes,
        statements=statements,
        findings=tuple(ordered),
        actions=tuple(catalogue.for_findings(ordered)),
        provenance=provenance,
    )


def _metrics_in(report: Any) -> set[str]:
    """Metric names the assessment actually computed, for the vocabulary check."""
    try:
        return set(report.metric_names)
    except AttributeError:
        return set()
