from pathlib import Path

import numpy as np
import torch
from dgl import DGLGraph


CITATION_DOMAIN_ALIASES = {
    "A": "ACMv9",
    "D": "DBLPv7",
    "C": "Citationv1",
    "ACMv9": "ACMv9",
    "DBLPv7": "DBLPv7",
    "Citationv1": "Citationv1",
}
CITATION_FEATURE_DIM = 6775


def normalize_citation_domain(domain_name):
    try:
        return CITATION_DOMAIN_ALIASES[domain_name]
    except KeyError:
        accepted = ", ".join(CITATION_DOMAIN_ALIASES)
        raise ValueError(
            "Unknown Citation domain '{}'. Expected one of: {}.".format(
                domain_name, accepted
            )
        )


def _detect_delimiter(file_path):
    with file_path.open() as input_file:
        for line in input_file:
            stripped = line.strip()
            if stripped:
                return "," if "," in stripped else None
    raise ValueError("Citation file is empty: {}".format(file_path))


def _split_citation_line(line, delimiter):
    if delimiter == ",":
        return [value.strip() for value in line.split(",")]
    return line.split()


def _read_citation_labels(label_path, num_nodes):
    delimiter = _detect_delimiter(label_path)
    rows = []
    indexed_rows = None

    with label_path.open() as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = _split_citation_line(stripped, delimiter)
            if [value.lower() for value in values] == ["node_id", "label"]:
                continue
            if len(values) not in (1, 2):
                raise ValueError(
                    "{}:{} must contain either 'label' or 'node_id,label'.".format(
                        label_path, line_number
                    )
                )
            is_indexed = len(values) == 2
            if indexed_rows is None:
                indexed_rows = is_indexed
            elif indexed_rows != is_indexed:
                raise ValueError(
                    "{} mixes one-column and node_id,label rows.".format(label_path)
                )
            try:
                rows.append(tuple(int(value) for value in values))
            except ValueError:
                raise ValueError(
                    "{}:{} contains a non-integer label row.".format(
                        label_path, line_number
                    )
                )

    if len(rows) != num_nodes:
        raise ValueError(
            "{} contains {} labels, but the feature matrix contains {} nodes.".format(
                label_path, len(rows), num_nodes
            )
        )

    if indexed_rows:
        node_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
        label_values = np.asarray([row[1] for row in rows], dtype=np.int64)
        if np.any(node_ids < 0) or np.any(node_ids >= num_nodes):
            raise ValueError(
                "{} contains node IDs outside [0, {}].".format(
                    label_path, num_nodes - 1
                )
            )
        if np.unique(node_ids).size != num_nodes:
            raise ValueError(
                "{} must contain each node ID exactly once.".format(label_path)
            )
        labels = np.empty(num_nodes, dtype=np.int64)
        labels[node_ids] = label_values
    else:
        labels = np.asarray([row[0] for row in rows], dtype=np.int64)

    if np.any(labels < 0) or np.any(labels > 4):
        raise ValueError(
            "{} contains labels outside the expected range [0, 4].".format(
                label_path
            )
        )
    return torch.from_numpy(labels).long()


def _read_citation_edges(edge_path, num_nodes):
    delimiter = _detect_delimiter(edge_path)
    undirected_edges = set()

    with edge_path.open() as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = _split_citation_line(stripped, delimiter)
            if len(values) != 2:
                raise ValueError(
                    "{}:{} must contain exactly two node IDs.".format(
                        edge_path, line_number
                    )
                )
            try:
                source, target = (int(value) for value in values)
            except ValueError:
                raise ValueError(
                    "{}:{} contains a non-integer edge.".format(
                        edge_path, line_number
                    )
                )
            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                raise ValueError(
                    "{}:{} contains edge ({}, {}) outside [0, {}].".format(
                        edge_path, line_number, source, target, num_nodes - 1
                    )
                )
            if source != target:
                undirected_edges.add((min(source, target), max(source, target)))

    graph = DGLGraph()
    graph.add_nodes(num_nodes)
    node_ids = list(range(num_nodes))
    graph.add_edges(node_ids, node_ids)
    if undirected_edges:
        ordered_edges = sorted(undirected_edges)
        sources = [source for source, target in ordered_edges]
        targets = [target for source, target in ordered_edges]
        graph.add_edges(sources + targets, targets + sources)
    return graph


def read_citation_labels(data_root, domain_name, num_nodes):
    """Load labels for one Citation domain."""
    domain_name = normalize_citation_domain(domain_name)
    label_path = (
        Path(data_root)
        / domain_name
        / "raw"
        / "{}_labels.txt".format(domain_name)
    )
    if not label_path.is_file():
        raise FileNotFoundError(
            "Citation data file does not exist: {}".format(label_path)
        )
    return _read_citation_labels(label_path, num_nodes)


def read_citation_domain(data_root, domain_name, load_labels=True):
    """Load one Citation domain as (graph, optional labels, float32 features)."""
    domain_name = normalize_citation_domain(domain_name)
    raw_path = Path(data_root) / domain_name / "raw"
    feature_path = raw_path / "{}_docs.txt".format(domain_name)
    edge_path = raw_path / "{}_edgelist.txt".format(domain_name)

    for file_path in (feature_path, edge_path):
        if not file_path.is_file():
            raise FileNotFoundError(
                "Citation data file does not exist: {}".format(file_path)
            )

    feature_delimiter = _detect_delimiter(feature_path)
    features_array = np.loadtxt(
        feature_path,
        dtype=np.float32,
        delimiter=feature_delimiter,
        ndmin=2,
    )
    features = torch.from_numpy(features_array).float()
    num_nodes = features.shape[0]
    if features.shape[1] != CITATION_FEATURE_DIM:
        raise ValueError(
            "{} has feature dimension {}; expected {}.".format(
                feature_path, features.shape[1], CITATION_FEATURE_DIM
            )
        )

    labels = None
    if load_labels:
        labels = read_citation_labels(data_root, domain_name, num_nodes)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                "Feature/label row mismatch for {}: {} != {}.".format(
                    domain_name, features.shape[0], labels.shape[0]
                )
            )

    graph = _read_citation_edges(edge_path, num_nodes)
    return graph, labels, features


def create_citation_source_split(source_labels, split_seed):
    """Create the DGSDA-compatible deterministic 80/10/10 source split."""
    if source_labels.ndim != 1:
        raise ValueError(
            "Citation source labels must be one-dimensional; got shape {}.".format(
                tuple(source_labels.shape)
            )
        )

    num_nodes = int(source_labels.shape[0])
    generator = torch.Generator()
    generator.manual_seed(int(split_seed))
    indices = torch.randperm(num_nodes, generator=generator)
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    source_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    source_val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    source_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    source_train_mask[train_indices] = True
    source_val_mask[val_indices] = True
    source_test_mask[test_indices] = True

    assigned_count = (
        source_train_mask.to(torch.int8)
        + source_val_mask.to(torch.int8)
        + source_test_mask.to(torch.int8)
    )
    if not torch.all(assigned_count == 1):
        raise RuntimeError(
            "Citation source masks must be mutually exclusive and cover every node."
        )
    return source_train_mask, source_val_mask, source_test_mask
