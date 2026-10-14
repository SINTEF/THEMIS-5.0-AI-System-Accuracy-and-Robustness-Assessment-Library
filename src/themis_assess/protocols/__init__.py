"""Split protocols — how a shift is introduced (survey §3.1)."""

from themis_assess.protocols.splits import (
    Split,
    compositional_split,
    domain_split,
    near_far_split,
    resample_subsets,
    subpopulation_split,
    temporal_split,
)

__all__ = [
    "Split",
    "domain_split",
    "temporal_split",
    "subpopulation_split",
    "compositional_split",
    "near_far_split",
    "resample_subsets",
]
