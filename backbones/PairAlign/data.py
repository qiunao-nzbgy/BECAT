"""Citation-only data loading for the PairAlign release.

The source loader reads labels and creates the fixed source partition.  The
target training loader intentionally has no code path that opens a labels file;
target labels are loaded by :func:`load_target_labels` only after checkpoint
restoration.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array


_DOMAIN_ALIASES = {
    "ACMv9": "ACMv9",
    "Citationv1": "Citationv1",
    "DBLPv7": "DBLPv7",
    "acm": "ACMv9",
    "citation": "Citationv1",
    "dblp": "DBLPv7",
}


def canonical_domain_name(name):
    try:
        return _DOMAIN_ALIASES[name]
    except KeyError as error:
        supported = ", ".join(sorted(_DOMAIN_ALIASES))
        raise ValueError(
            f"Unsupported Citation domain '{name}'. Supported names: {supported}."
        ) from error


def _citation_paths(data_root, name, include_labels):
    """Resolve only the files needed for the requested loading phase."""
    canonical_name = canonical_domain_name(name)
    raw_dir = Path(data_root).expanduser() / canonical_name / "raw"
    paths = {
        "docs": raw_dir / f"{canonical_name}_docs.txt",
        "edges": raw_dir / f"{canonical_name}_edgelist.txt",
    }
    if include_labels:
        paths["labels"] = raw_dir / f"{canonical_name}_labels.txt"

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Citation input file(s): " + ", ".join(missing)
        )
    return canonical_name, paths


def _load_features_and_graph(data_root, name):
    # Crucially, this resolver neither resolves nor checks a labels path.
    canonical_name, paths = _citation_paths(
        data_root,
        name,
        include_labels=False,
    )
    features = np.loadtxt(
        paths["docs"],
        delimiter=",",
        dtype=np.float32,
        ndmin=2,
    )
    x = torch.from_numpy(features)
    edge_index = read_txt_array(
        str(paths["edges"]),
        sep=",",
        dtype=torch.long,
    ).t().contiguous()

    num_nodes = int(x.size(0))
    adjacency_values = np.ones(edge_index.size(1), dtype=np.float32)
    adj = sp.csr_matrix(
        (
            adjacency_values,
            (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()),
        ),
        shape=(num_nodes, num_nodes),
    )
    # Preserve the original PairAlign undirected-adjacency semantics.
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    graph = Data(x=x, edge_index=edge_index)
    graph.adj = adj
    graph.edge_weight = torch.ones(graph.num_edges, dtype=torch.float32)
    graph.domain_name = canonical_name
    return graph


def deterministic_source_masks(num_nodes, split_seed=42):
    """Return label-independent 80/10/10 masks without touching global RNG."""
    generator = torch.Generator()
    generator.manual_seed(int(split_seed))
    indices = torch.randperm(num_nodes, generator=generator)
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    masks = []
    for selected in (train_idx, val_idx, test_idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[selected] = True
        masks.append(mask)

    train_mask, val_mask, test_mask = masks
    coverage = train_mask.to(torch.int8) + val_mask.to(torch.int8) + test_mask.to(
        torch.int8
    )
    if not bool(torch.all(coverage == 1)):
        raise RuntimeError("Source partition must cover every node exactly once.")
    return train_mask, val_mask, test_mask


def _validate_graph_structure(graph, name):
    if graph.x.dim() != 2 or graph.x.size(0) != graph.num_nodes:
        raise ValueError(f"{name}: invalid feature matrix shape {tuple(graph.x.shape)}.")
    if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
        raise ValueError(
            f"{name}: edge_index must have shape [2, num_edges], got "
            f"{tuple(graph.edge_index.shape)}."
        )
    if graph.edge_index.numel() == 0:
        raise ValueError(f"{name}: edge list must not be empty.")
    if int(graph.edge_index.min()) < 0:
        raise ValueError(f"{name}: edge indices must be non-negative.")
    if int(graph.edge_index.max()) >= graph.num_nodes:
        raise ValueError(f"{name}: edge index is outside the node range.")
    if graph.edge_weight.numel() != graph.num_edges:
        raise ValueError(f"{name}: one edge weight is required per edge.")
    if graph.adj.shape != (graph.num_nodes, graph.num_nodes):
        raise ValueError(f"{name}: adjacency shape does not match node count.")


def _validate_labels(labels, num_nodes, num_classes, name):
    if labels.dim() != 1 or labels.numel() != num_nodes:
        raise ValueError(
            f"{name}: expected {num_nodes} labels, got shape {tuple(labels.shape)}."
        )
    unique = torch.unique(labels).cpu()
    expected = torch.arange(num_classes, dtype=unique.dtype)
    if not torch.equal(unique, expected):
        raise ValueError(
            f"{name}: labels must be contiguous in [0, {num_classes - 1}], "
            f"got {unique.tolist()}."
        )


def _attach_source_masks(graph, split_seed):
    train_mask, val_mask, test_mask = deterministic_source_masks(
        graph.num_nodes,
        split_seed=split_seed,
    )
    graph.source_train_mask = train_mask
    graph.source_val_mask = val_mask
    graph.source_test_mask = test_mask

    # Compatibility aliases used by the original PairAlign training code.
    graph.source_training_mask = train_mask
    graph.source_validation_mask = val_mask
    graph.source_testing_mask = test_mask
    graph.source_mask = torch.arange(graph.num_nodes, dtype=torch.long)


def _build_source_edge_class(graph):
    edge_class = np.zeros(
        (graph.num_edges, graph.num_classes, graph.num_classes),
        dtype=np.float32,
    )
    source_labels = graph.y[graph.edge_index[0]].cpu().numpy()
    destination_labels = graph.y[graph.edge_index[1]].cpu().numpy()
    edge_class[np.arange(graph.num_edges), source_labels, destination_labels] = 1.0
    graph.edge_class = edge_class


def load_source_citation(data_root, name, split_seed=42):
    """Load a labeled source Citation graph and fixed source masks."""
    graph = _load_features_and_graph(data_root, name)
    _, paths = _citation_paths(data_root, name, include_labels=True)
    labels = np.loadtxt(paths["labels"], dtype=np.int64, ndmin=1)
    graph.y = torch.from_numpy(labels).to(torch.long)
    unique = torch.unique(graph.y)
    graph.num_classes = int(unique.numel())
    _validate_graph_structure(graph, graph.domain_name)
    _validate_labels(
        graph.y,
        graph.num_nodes,
        graph.num_classes,
        graph.domain_name,
    )
    _attach_source_masks(graph, split_seed)
    _build_source_edge_class(graph)
    return graph


def load_target_citation_unlabeled(data_root, name, source_num_classes):
    """Load target features/edges only; no target label path is accessed."""
    graph = _load_features_and_graph(data_root, name)
    graph.num_classes = int(source_num_classes)
    graph.target_mask = torch.arange(graph.num_nodes, dtype=torch.long)
    graph.target_test_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    graph.target_testing_mask = graph.target_test_mask
    _validate_graph_structure(graph, graph.domain_name)
    if "y" in graph:
        raise RuntimeError("Unlabeled target graph unexpectedly contains labels.")
    if "edge_class" in graph:
        raise RuntimeError("Unlabeled target graph must not contain edge_class.")
    return graph


def validate_citation_pair(source_graph, target_graph):
    """Training-time checks that never inspect target labels."""
    _validate_graph_structure(source_graph, source_graph.domain_name)
    _validate_graph_structure(target_graph, target_graph.domain_name)
    _validate_labels(
        source_graph.y,
        source_graph.num_nodes,
        source_graph.num_classes,
        source_graph.domain_name,
    )
    if source_graph.num_node_features != target_graph.num_node_features:
        raise ValueError(
            "Source and target feature dimensions differ: "
            f"{source_graph.num_node_features} vs "
            f"{target_graph.num_node_features}."
        )
    if target_graph.num_classes != source_graph.num_classes:
        raise ValueError("Target model label space must come from the source graph.")
    if "y" in target_graph or "edge_class" in target_graph:
        raise RuntimeError(
            "Training target must contain neither target labels nor target edge_class."
        )


def load_target_labels(data_root, name, num_nodes, source_num_classes):
    """Load and validate target labels for final, all-node evaluation only."""
    canonical_name, paths = _citation_paths(data_root, name, include_labels=True)
    labels = np.loadtxt(paths["labels"], dtype=np.int64, ndmin=1)
    labels = torch.from_numpy(labels).to(torch.long)
    _validate_labels(labels, num_nodes, source_num_classes, canonical_name)
    return labels
