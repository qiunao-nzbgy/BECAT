"""Citation .mat loading and source splitting for the DANN release."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch


CITATION_DOMAINS = ("acmv9", "citationv1", "dblpv7")


@dataclass
class GraphDomainData:
    name: str
    features: np.ndarray
    adjacency: object
    labels: Optional[np.ndarray] = None


def _dataset_path(name: str, data_root: str) -> Path:
    if name not in CITATION_DOMAINS:
        raise ValueError(
            "citation domain must be one of {} (got {!r})".format(
                ", ".join(CITATION_DOMAINS), name
            )
        )
    return Path(data_root).expanduser().resolve() / "{}.mat".format(name)


def _to_csr_if_sparse(value):
    if sp.issparse(value):
        return value.tocsr()
    return np.asarray(value)


def _row_normalize_features(features):
    """Preserve the row-normalization semantics of the supplied graph DANN."""
    if sp.issparse(features):
        features = features.tocsr().astype(np.float32)
        row_sum = np.asarray(features.sum(axis=1)).ravel()
        inv_row_sum = np.zeros_like(row_sum, dtype=np.float32)
        nonzero = row_sum > 0
        inv_row_sum[nonzero] = 1.0 / row_sum[nonzero]
        normalized = sp.diags(inv_row_sum).dot(features).tocsr()
        return normalized.toarray().astype(np.float32, copy=False)

    features = np.asarray(features, dtype=np.float32)
    row_sum = features.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(features, dtype=np.float32)
    np.divide(features, row_sum, out=normalized, where=row_sum != 0)
    return normalized


def _labels_to_indices(labels):
    """Preserve one-hot and scalar-label conversion in unchanged node order."""
    if sp.issparse(labels):
        labels = labels.toarray()
    labels = np.asarray(labels)

    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(axis=1)
    else:
        labels = labels.reshape(-1)
        unique_labels = np.unique(labels)
        label_to_index = {
            label: index for index, label in enumerate(unique_labels)
        }
        labels = np.array([label_to_index[label] for label in labels])

    return labels.astype(np.int64, copy=False)


def load_graph_domain(
    name: str,
    data_root: str,
    load_labels: bool,
) -> GraphDomainData:
    """Load graph/features and, only when requested, source labels.

    ``variable_names`` is deliberately used so target ``group`` is neither
    loaded nor deserialized during training.
    """
    path = _dataset_path(name, data_root)
    if not path.exists():
        raise FileNotFoundError("Could not find dataset file: {}".format(path))

    variable_names = ("network", "attrb", "group") if load_labels else (
        "network",
        "attrb",
    )
    mat = sio.loadmat(path, variable_names=variable_names)
    missing_keys = [key for key in variable_names if key not in mat]
    if missing_keys:
        raise KeyError("{} is missing required keys: {}".format(path, missing_keys))

    features = _row_normalize_features(mat["attrb"])
    adjacency = _to_csr_if_sparse(mat["network"])
    labels = _labels_to_indices(mat["group"]) if load_labels else None

    num_nodes = features.shape[0]
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError(
            "{}: adjacency shape {} does not match node count {}".format(
                name, adjacency.shape, num_nodes
            )
        )
    if labels is not None and labels.shape[0] != num_nodes:
        raise ValueError(
            "{}: feature rows ({}) do not match labels ({})".format(
                name, num_nodes, labels.shape[0]
            )
        )

    return GraphDomainData(
        name=name,
        features=features,
        adjacency=adjacency,
        labels=labels,
    )


def load_target_labels_for_final_evaluation(
    name: str,
    data_root: str,
    expected_num_nodes: int,
) -> np.ndarray:
    """Load only ``group``; this function is legal only after restoration."""
    path = _dataset_path(name, data_root)
    if not path.exists():
        raise FileNotFoundError("Could not find dataset file: {}".format(path))
    mat = sio.loadmat(path, variable_names=("group",))
    if "group" not in mat:
        raise KeyError("{} is missing required key: group".format(path))
    labels = _labels_to_indices(mat["group"])
    if labels.shape[0] != expected_num_nodes:
        raise ValueError(
            "{}: target labels ({}) do not match node count ({})".format(
                name, labels.shape[0], expected_num_nodes
            )
        )
    return labels


def create_source_split(
    num_nodes: int,
    split_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic label-independent 80/10/remainder indices."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    generator = torch.Generator()
    generator.manual_seed(split_seed)
    indices = torch.randperm(num_nodes, generator=generator)

    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    combined = torch.cat((train_idx, val_idx, test_idx), dim=0)
    if combined.numel() != num_nodes or torch.unique(combined).numel() != num_nodes:
        raise RuntimeError("source split must cover every node exactly once")
    return train_idx, val_idx, test_idx
