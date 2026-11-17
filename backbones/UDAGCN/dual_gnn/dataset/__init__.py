"""Citation graph loading and deterministic source splitting."""

from .DomainData import (
    DomainData,
    apply_fixed_split,
    load_target_labels_for_final_evaluation,
)

__all__ = [
    "DomainData",
    "apply_fixed_split",
    "load_target_labels_for_final_evaluation",
]
