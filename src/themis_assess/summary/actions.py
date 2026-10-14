"""The action catalogue: retrieved, never generated.

Part (c) of the summary, and the part most worth being strict about. What to do about a
weak model is domain knowledge about a *deployment* — what retraining costs, whether the
imaging pipeline can be changed, who signs off, what a false negative is worth. None of
that is in any language model's weights, and none of it is in the assessment either.

So the actions live in :data:`CATALOGUE_PATH`, a YAML file written by people, and a
language model's only permitted role is to **select and order** entries from it. It may
not write one. Asked for "ways to improve this model", a small model produces fluent,
plausible, unfounded advice — and that is the hallucination that matters most, because
recommendations are the part of a report people act on.

The catalogue is keyed by the ``rule`` identifier of a
:class:`~themis_assess.report.findings.Finding`, so an action is always attached to the
specific observation that occasioned it. There is no "general advice" bucket, deliberately:
advice with no finding behind it has nothing to be checked against.

The ``suggestion`` field on each :class:`Finding` is the ancestor of this file — the same
knowledge, embedded in Python where only a programmer could edit it. It stays as the
one-line version; this is where the alternatives, the preconditions and the caveats live.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["Action", "ActionCatalogue", "CATALOGUE_PATH", "default_catalogue"]

#: The catalogue that ships with the library. Edit it, or point
#: :class:`ActionCatalogue` at your own.
CATALOGUE_PATH = Path(__file__).parent / "actions.yaml"


@dataclass(frozen=True, slots=True)
class Action:
    """One thing that could be done, with the conditions and the cost attached.

    Every field beyond ``action`` and ``rule`` is optional, because a catalogue entry
    written in a hurry is better than one not written at all — but ``caveat`` is present
    on most shipped entries on purpose. An action list without caveats reads as a recipe,
    and gets followed rather than considered.
    """

    rule: str
    action: str
    precondition: str = ""
    owner: str = ""
    cost: str = ""
    caveat: str = ""
    evidence: str = ""

    @property
    def is_unconditional(self) -> bool:
        """Whether this applies whenever its finding fires."""
        return self.precondition.strip().lower().startswith("always")

    def as_dict(self) -> dict[str, Any]:
        """The serialisable form, for the results bundle."""
        return {
            "rule": self.rule,
            "action": self.action,
            "precondition": self.precondition,
            "owner": self.owner,
            "cost": self.cost,
            "caveat": self.caveat,
            "evidence": self.evidence,
        }

    def describe(self, *, addressed_to_reader: bool = False) -> str:
        """One line. ``addressed_to_reader`` decides whether the owner is named.

        A summary written for a clinician should say "the ML team could retrain with
        class weights", not imply the reader will be doing it themselves.
        """
        if addressed_to_reader or not self.owner:
            return self.action
        return f"{self.action} ({self.owner})"


class ActionCatalogue:
    """Actions by finding rule, loaded from YAML.

    Lookup is by exact rule id. A finding whose rule has no entry yields no actions and
    no error: the catalogue is expected to be incomplete, since a rule can be added to
    the library before anyone has decided what to do about it. :meth:`uncovered` is how
    you find out which those are.
    """

    def __init__(self, actions: Mapping[str, Sequence[Action]]) -> None:
        self._actions = {rule: tuple(items) for rule, items in actions.items()}

    # -- construction -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> ActionCatalogue:
        """Load a catalogue from a YAML file."""
        import yaml

        target = Path(path)
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
        if raw is None:
            return cls({})
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{target} should be a mapping of rule id -> list of actions; got "
                f"{type(raw).__name__}."
            )

        known = {f.name for f in Action.__dataclass_fields__.values()} - {"rule"}
        actions: dict[str, list[Action]] = {}
        for rule, entries in raw.items():
            if not isinstance(entries, Sequence) or isinstance(entries, str):
                raise ValueError(
                    f"Rule {rule!r} in {target.name} should map to a *list* of actions, "
                    f"even when there is only one. Got {type(entries).__name__}."
                )
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping) or "action" not in entry:
                    raise ValueError(
                        f"Entry {index} under {rule!r} in {target.name} needs at least "
                        f"an `action:` field."
                    )
                unknown = sorted(set(entry) - known)
                if unknown:
                    raise ValueError(
                        f"Entry {index} under {rule!r} in {target.name} has unknown "
                        f"field(s) {unknown}. Known: {sorted(known)}. A misspelt field "
                        f"would otherwise be silently dropped."
                    )
                actions.setdefault(str(rule), []).append(
                    Action(rule=str(rule), **{k: str(v).strip() for k, v in entry.items()})
                )
        return cls(actions)

    # -- lookup -------------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(items) for items in self._actions.values())

    def __contains__(self, rule: object) -> bool:
        return rule in self._actions

    @property
    def rules(self) -> list[str]:
        return sorted(self._actions)

    def for_rule(self, rule: str) -> tuple[Action, ...]:
        """Actions for one rule, in the order the file lists them.

        File order is meaningful and is preserved: the entries are written cheapest and
        most-diagnostic first, so "check whether the subgroup axis is confounded with the
        label" comes before "reweight the training data".
        """
        return self._actions.get(rule, ())

    def for_findings(self, findings: Iterable[Any]) -> list[Action]:
        """Actions for a set of findings, de-duplicated, in finding order.

        Findings are expected pre-ranked, so the actions come out most-severe-first. A
        rule appearing twice — two subgroups with a gap, say — contributes its actions
        once.
        """
        out: list[Action] = []
        seen: set[tuple[str, str]] = set()
        for finding in findings:
            for action in self.for_rule(getattr(finding, "rule", "")):
                key = (action.rule, action.action)
                if key in seen:
                    continue
                seen.add(key)
                out.append(action)
        return out

    def uncovered(self, rules: Iterable[str]) -> list[str]:
        """Which of ``rules`` the catalogue says nothing about.

        Used by a test over every rule the library can emit, so a new finding cannot ship
        with no advice attached without somebody noticing.
        """
        return sorted({rule for rule in rules if rule not in self._actions})


@lru_cache(maxsize=1)
def default_catalogue() -> ActionCatalogue:
    """The catalogue shipped with the library. Cached — the file does not change at run time."""
    return ActionCatalogue.from_yaml(CATALOGUE_PATH)
