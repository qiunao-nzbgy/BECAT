"""Citation graph loading with graph-only caches and deterministic source masks."""

import os
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array


CITATION_DOMAINS = ("ACMv9", "Citationv1", "DBLPv7")
GRAPH_CACHE_VERSION = 1


def _raw_path(root, name, suffix):
    path = osp.join(root, "raw", "{}_{}.txt".format(name, suffix))
    if not osp.isfile(path):
        raise FileNotFoundError(
            "Missing raw file for exact dataset name {!r}: {}. Dataset "
            "directories and filename prefixes are case-sensitive.".format(name, path)
        )
    return path


def _line_count(path):
    with open(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())


def _read_features(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(line.rstrip().split(","))
    return torch.from_numpy(np.asarray(rows, dtype=float)).to(torch.float)


def _read_labels(path, expected_num_nodes=None):
    labels = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            labels.append(line.strip())
    y = torch.from_numpy(np.asarray(labels, dtype=int)).to(torch.int64)
    if expected_num_nodes is not None and y.numel() != int(expected_num_nodes):
        raise ValueError(
            "label count does not match graph node count: {} != {} ({})".format(
                y.numel(),
                int(expected_num_nodes),
                path,
            )
        )
    return y


class DomainData:
    """Load one citation graph while keeping its processed cache label-free.

    ``load_labels=True`` is used only for the source domain. The target training
    loader passes ``False``, so neither the raw label file nor an old
    ``processed/data.pt`` label cache is read.
    """

    def __init__(self, root, name, load_labels=True):
        if name not in CITATION_DOMAINS:
            raise ValueError(
                "citation domain must be one of {} (got {!r})".format(
                    ", ".join(CITATION_DOMAINS),
                    name,
                )
            )
        self.root = osp.abspath(osp.expanduser(root))
        self.name = name
        self.raw_dir = osp.join(self.root, "raw")
        self.processed_dir = osp.join(self.root, "processed")
        self.processed_path = osp.join(self.processed_dir, "graph.pt")
        self._validate_raw_graph_files()

        cache = self._load_graph_cache()
        self.data = Data(
            x=cache["x"],
            edge_index=cache["edge_index"],
            num_nodes=int(cache["num_nodes"]),
        )
        if load_labels:
            self.data.y = _read_labels(
                _raw_path(self.root, self.name, "labels"),
                expected_num_nodes=self.data.num_nodes,
            )

    def _validate_raw_graph_files(self):
        _raw_path(self.root, self.name, "docs")
        _raw_path(self.root, self.name, "edgelist")

    def _process_graph(self):
        docs_path = _raw_path(self.root, self.name, "docs")
        edge_path = _raw_path(self.root, self.name, "edgelist")
        x = _read_features(docs_path)
        edge_index = read_txt_array(edge_path, sep=",", dtype=torch.long).t()
        cache = {
            "version": GRAPH_CACHE_VERSION,
            "x": x,
            "edge_index": edge_index,
            "num_nodes": int(x.size(0)),
        }
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(cache, self.processed_path)
        return cache

    def _load_graph_cache(self):
        if not osp.isfile(self.processed_path):
            return self._process_graph()

        cache = torch.load(self.processed_path, map_location="cpu")
        required = {"version", "x", "edge_index", "num_nodes"}
        if not isinstance(cache, dict) or set(cache) != required:
            raise RuntimeError(
                "{} is not a graph-only UDAGCN cache. Remove only this graph.pt "
                "so it can be rebuilt; processed/data.pt is ignored.".format(
                    self.processed_path
                )
            )
        if cache["version"] != GRAPH_CACHE_VERSION:
            raise RuntimeError(
                "unsupported graph cache version in {}".format(self.processed_path)
            )
        self._validate_graph_cache(cache)
        return cache

    def _validate_graph_cache(self, cache):
        docs_path = _raw_path(self.root, self.name, "docs")
        edge_path = _raw_path(self.root, self.name, "edgelist")
        raw_num_nodes = _line_count(docs_path)
        raw_num_edges = _line_count(edge_path)
        with open(docs_path, "r", encoding="utf-8") as handle:
            first_line = next((line for line in handle if line.strip()), None)
        raw_num_features = (
            0 if first_line is None else len(first_line.rstrip().split(","))
        )
        expected_shape = (raw_num_nodes, raw_num_features)
        cached_shape = tuple(cache["x"].shape)
        cached_edge_count = int(cache["edge_index"].shape[1])
        if (
            cached_shape != expected_shape
            or int(cache["num_nodes"]) != raw_num_nodes
            or cached_edge_count != raw_num_edges
        ):
            raise RuntimeError(
                "graph cache for {!r} is inconsistent with raw graph files: "
                "cached x={}, raw x={}, cached edges={}, raw edges={}. Remove "
                "only {} so it can be rebuilt.".format(
                    self.name,
                    cached_shape,
                    expected_shape,
                    cached_edge_count,
                    raw_num_edges,
                    self.processed_path,
                )
            )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.data

    @property
    def num_features(self):
        return int(self.data.x.shape[1])

    @property
    def num_classes(self):
        if getattr(self.data, "y", None) is None:
            raise RuntimeError("num_classes is unavailable for an unlabeled graph")
        return int(self.data.y.max().item()) + 1


def apply_fixed_split(data, split_seed=42):
    """Regenerate a label-independent deterministic 80/10/remainder split."""
    num_nodes = int(data.num_nodes)
    generator = torch.Generator()
    generator.manual_seed(int(split_seed))
    indices = torch.randperm(num_nodes, generator=generator)
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    coverage = (
        train_mask.to(torch.int8)
        + val_mask.to(torch.int8)
        + test_mask.to(torch.int8)
    )
    if not bool(torch.all(coverage == 1)):
        raise RuntimeError("source masks must cover every node exactly once")

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def load_target_labels_for_final_evaluation(
    root,
    name,
    expected_num_nodes,
    num_classes,
):
    """Load target labels in raw node order after best-state restoration only."""
    y = _read_labels(
        _raw_path(osp.abspath(osp.expanduser(root)), name, "labels"),
        expected_num_nodes=expected_num_nodes,
    )
    if y.numel() == 0:
        raise ValueError("target label file is empty")
    if int(y.min()) < 0 or int(y.max()) >= int(num_classes):
        raise ValueError(
            "target labels fall outside the source classifier label space [0, {})".format(
                int(num_classes)
            )
        )
    return y
