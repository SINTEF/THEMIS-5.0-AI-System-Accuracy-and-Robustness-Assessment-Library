"""Part (b): the language model, and the socket it plugs into.

The only module here that a model touches, and it is deliberately the smallest. Its job
is grouping and phrasing — turning eight findings into three themes with a paragraph each,
in the register the reader needs. It does not compute anything, decide anything, or
recommend anything.

Three properties are worth stating, because they are what make the arrangement safe:

**The model is optional.** With ``model=None`` the summary is built from the templates and
is complete. Nothing degrades to a placeholder.

**The model is removable.** Its output goes through
:func:`~themis_assess.summary.verify.verify_narrative` and is discarded on failure, in
favour of the same templates. So the worst case is the no-model case, and the summary
cannot become *wrong* by adding a model — only better or unchanged.

**The model sees findings, never the report.** On the tutorial's assessment that is ~630
tokens against ~7,900, and 40 numbers against 142. Feeding the whole results dictionary
would cost twelve times the context and triple the numeric surface a model could
misquote, for no information it needs.

Bring your own model
--------------------

:class:`LanguageModel` is a two-method protocol. Anything satisfying it works —
:class:`OllamaModel` is included because it is the shortest path on a laptop, and
:class:`EchoModel` because tests need a model whose output is known.

.. code-block:: python

    from themis_assess.summary import summarise, OllamaModel

    summary = summarise(report, audience="clinician", model=OllamaModel("qwen3.5:9b"))
    print(summary.to_markdown())
    print(summary.provenance["narrative_source"])   # "model" or "templates"
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from themis_assess.summary.audience import Audience

__all__ = [
    "LanguageModel",
    "EchoModel",
    "OllamaModel",
    "NARRATIVE_SCHEMA",
    "build_prompt",
    "findings_for_prompt",
    "parse_narrative",
    "list_ollama_models",
    "find_local_model",
]


@runtime_checkable
class LanguageModel(Protocol):
    """What this package needs from a model. Two methods, no framework."""

    @property
    def name(self) -> str:
        """Which model this is, recorded in the summary's provenance."""
        ...

    def complete(self, prompt: str, *, schema: Mapping[str, Any] | None = None) -> str:
        """Return the model's response. ``schema`` is a JSON schema to constrain it."""
        ...


#: The shape the model must return. Constrained decoding is asked for where the backend
#: supports it, and the result is validated either way — a schema the backend merely
#: *encourages* is not a guarantee, and this package does not rely on one.
NARRATIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "One sentence: the single most important thing about this model.",
        },
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "findings": {"type": "array", "items": {"type": "integer"}},
                    "prose": {"type": "string"},
                },
                "required": ["title", "findings", "prose"],
            },
        },
    },
    "required": ["headline", "themes"],
}


# --------------------------------------------------------------------------
# what the model is shown
# --------------------------------------------------------------------------


def findings_for_prompt(
    findings: Sequence[Any], audience: Audience, *, registry: Any = None
) -> list[dict[str, Any]]:
    """The findings as the model should see them: numbered, compact, pre-phrased.

    Each entry carries the deterministic sentence from
    :mod:`~themis_assess.summary.templates` as ``statement``. That is not redundant with
    the evidence — it is the point. The model is being asked to *arrange and connect*
    sentences that are already correct, which is a far smaller job than composing them
    from raw numbers, and a far smaller job is what a 3-to-9-billion-parameter model does
    reliably.
    """
    from themis_assess.summary.templates import render_finding

    out = []
    for index, finding in enumerate(findings, start=1):
        out.append(
            {
                "id": f"F{index}",
                "severity": getattr(finding, "severity", ""),
                "area": getattr(finding, "area", ""),
                "rule": getattr(finding, "rule", ""),
                "statement": render_finding(finding, audience),
            }
        )
    return out


_INSTRUCTIONS = """\
You are writing the narrative section of a machine-learning model assessment.

THE READER
{label}
They are reading in order to: {reads_for}
Register: {tone}

YOUR TASK
Group the findings below into {theme_target} themes and write one short paragraph per
theme, in the register above. Then write a single-sentence headline naming the most
important thing about this model.

RULES — these are checked mechanically after you answer, and a violation causes your
entire answer to be discarded and replaced by a plain template version.

1. Use ONLY the findings listed below. Do not add observations, metrics, comparisons or
   context from anywhere else, including your own knowledge of similar models.
2. Do not write any number that does not already appear in a finding's statement. You may
   restate a number from a statement; you may not compute, round differently, average,
   or estimate one.
3. End every sentence with the finding it rests on, in square brackets: [F1] or [F1, F3].
   A sentence with no citation is deleted.
4. Do not explain WHY anything happened. The assessment measures what is, not what caused
   it. The words "because", "due to", "causes", "the reason" and their kin are forbidden.
   If two findings look related, you may say they are consistent with one another; you may
   not say one produced the other.
5. Do not recommend actions. Recommendations come from a separate reviewed catalogue and
   are added after your text.
6. Do not name a metric that does not appear in the findings below.

THE FINDINGS
{findings}

Answer with JSON only: {{"headline": "...", "themes": [{{"title": "...",
"findings": [1, 3], "prose": "..."}}]}}"""


def build_prompt(
    findings: Sequence[Any],
    audience: Audience,
    *,
    theme_target: str = "two to four",
    registry: Any = None,
) -> str:
    """The whole prompt. One function, so it can be printed, diffed and reviewed.

    Deliberately not assembled from fragments scattered across the module: a prompt is
    the specification of the model's task, and a specification you cannot read in one
    place is one nobody audits.
    """
    entries = findings_for_prompt(findings, audience, registry=registry)
    rendered = "\n".join(
        f"{e['id']} ({e['severity']}, {e['area']}): {e['statement']}" for e in entries
    )
    return _INSTRUCTIONS.format(
        label=audience.label,
        reads_for=audience.reads_for,
        tone=audience.tone,
        theme_target=theme_target,
        findings=rendered,
    )


# --------------------------------------------------------------------------
# what comes back
# --------------------------------------------------------------------------


def parse_narrative(raw: str) -> dict[str, Any]:
    """Pull the JSON object out of a model's response.

    Tolerant of the two things small models do even under a schema: wrapping the object
    in a `````json`` fence, and prefixing it with a sentence of commentary. Anything
    beyond that raises, and the caller falls back to templates — guessing at malformed
    output is how a summariser starts inventing.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in the model's response: {raw[:200]!r}")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, Mapping):
        raise ValueError("The model returned JSON that is not an object.")
    if "themes" not in payload:
        raise ValueError("The model's JSON has no 'themes' key.")
    return dict(payload)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


class EchoModel:
    """A model that returns whatever it was constructed with. For tests.

    Every property of the pipeline that matters — the prompt, the parse, the verifier,
    the fallback — is testable without a model running anywhere, which is what keeps this
    package's tests fast and deterministic. The model itself is the one part that cannot
    be unit-tested, so it is also the one part kept trivial.
    """

    def __init__(self, response: str, *, name: str = "echo") -> None:
        self._response = response
        self._name = name
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(self, prompt: str, *, schema: Mapping[str, Any] | None = None) -> str:
        self.prompts.append(prompt)
        return self._response


class OllamaModel:
    """A local model served by Ollama.

    The shortest path to a working local summariser: `ollama pull qwen3.5:9b`, then pass
    ``OllamaModel("qwen3.5:9b")``. The ``ollama`` package is imported lazily and is not a
    dependency of this library — nothing here requires a model to exist.

    ``temperature`` defaults to 0. This is a rewriting task with one right answer per
    input; sampling buys nothing and costs reproducibility.

    A note on the schema: Ollama accepts a JSON schema in ``format=``, and how strictly it
    is enforced during decoding varies by version and backend. This class passes it and
    then validates the result anyway. That is the design position throughout — a
    constraint the backend offers is an optimisation, never the guarantee.
    """

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        *,
        host: str | None = None,
        temperature: float = 0.0,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def complete(self, prompt: str, *, schema: Mapping[str, Any] | None = None) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "OllamaModel needs the `ollama` package: pip install ollama. It is not a "
                "dependency of themis-assess, because the summary works without any "
                "model — see summarise(report, audience=...) with model=None."
            ) from exc

        client = ollama.Client(host=self.host) if self.host else ollama.Client()
        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=dict(schema) if schema else None,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]


# --------------------------------------------------------------------------
# finding a model that is actually installed
# --------------------------------------------------------------------------


def list_ollama_models(*, host: str | None = None) -> list[str]:
    """The models this machine has pulled, or an empty list.

    Empty for all three reasons a caller cannot distinguish and does not need to: the
    ``ollama`` package is not installed, the daemon is not running, or nothing has been
    pulled. Never raises — a summariser that crashes because a model is absent has
    defeated the point of the model being optional.
    """
    try:
        import ollama
    except ImportError:
        return []
    try:
        client = ollama.Client(host=host) if host else ollama.Client()
        listing = client.list()
    except Exception:  # noqa: BLE001 - a daemon that is not running is a normal state
        return []

    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, Mapping):
        models = listing.get("models", [])
    names = []
    for entry in models or []:
        name = getattr(entry, "model", None) or getattr(entry, "name", None)
        if name is None and isinstance(entry, Mapping):
            name = entry.get("model") or entry.get("name")
        if name:
            names.append(str(name))
    return names


def find_local_model(
    prefer: Sequence[str] = (), *, host: str | None = None, **kwargs: Any
) -> OllamaModel | None:
    """An :class:`OllamaModel` for a model this machine already has, or ``None``.

    Written for documentation and scripts that should run identically on a laptop with a
    model and in CI without one: ``model=find_local_model()`` uses whatever is installed
    and falls back to the template path when nothing is. Because the argument is a
    *preference*, the same notebook produces real model output on one machine and honest
    template output on another, and says which — rather than failing on the machine that
    has nothing pulled.

    ``prefer`` entries match a full tag (``"qwen3.5:9b"``) or a family
    (``"qwen3.5"`` matches ``qwen3.5:9b``). The first match wins; if none match, the
    first installed model is used, since a summary from an unexpected model is still
    verified before it is trusted.
    """
    installed = list_ollama_models(host=host)
    if not installed:
        return None
    for wanted in prefer:
        for name in installed:
            if name == wanted or name.split(":", 1)[0] == wanted.split(":", 1)[0]:
                return OllamaModel(name, host=host, **kwargs)
    return OllamaModel(installed[0], host=host, **kwargs)
