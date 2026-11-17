"""Citation-only data, split, metric, and A2GNN-MMD utilities."""

import os
import os.path as osp
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array


CITATION_DOMAINS = ("ACMv9", "Citationv1", "DBLPv7")
GRAPH_CACHE_VERSION = 1


def seed_training(seed):
    """Seed training RNGs; source masks use a separate local generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_source_masks(num_nodes, split_seed):
    """Create deterministic 80/10/remainder masks without global RNG use."""
    generator = torch.Generator()
    generator.manual_seed(split_seed)
    indices = torch.randperm(num_nodes, generator=generator)

    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    coverage = train_mask.to(torch.int8) + val_mask + test_mask
    if not bool(torch.all(coverage == 1)):
        raise RuntimeError("source masks must cover every node exactly once")
    return train_mask, val_mask, test_mask


def _domain_root(data_root, name):
    if name not in CITATION_DOMAINS:
        raise ValueError(
            "citation domain must be one of {} (got {!r})".format(
                ", ".join(CITATION_DOMAINS),
                name,
            )
        )
    return osp.abspath(osp.join(osp.expanduser(data_root), name))


def _read_features(path):
    rows = []
    with open(path, "rb") as handle:
        for line in handle.readlines():
            rows.append(str(line, encoding="utf-8").split(","))
    return torch.from_numpy(np.asarray(rows, dtype=float)).to(torch.float)


def load_citation_labels(data_root, name, expected_num_nodes=None):
    """Read labels in raw-file order; callers control when this is legal."""
    label_path = osp.join(
        _domain_root(data_root, name),
        "raw",
        "{}_labels.txt".format(name),
    )
    labels = []
    with open(label_path, "rb") as handle:
        for line in handle.readlines():
            value = str(line, encoding="utf-8").replace("\r", "").replace("\n", "")
            labels.append(value)
    y = torch.from_numpy(np.asarray(labels, dtype=int)).to(torch.int64)
    if expected_num_nodes is not None and y.numel() != expected_num_nodes:
        raise ValueError(
            "{} labels do not match graph node count: {} != {}".format(
                name,
                y.numel(),
                expected_num_nodes,
            )
        )
    return y


class CitationGraphDataset:
    """Cache citation graph/features only; never cache or deserialize labels."""

    def __init__(self, data_root, name):
        self.name = name
        self.root = _domain_root(data_root, name)
        self.raw_dir = osp.join(self.root, "raw")
        self.processed_dir = osp.join(self.root, "processed")
        self.processed_path = osp.join(self.processed_dir, "graph.pt")
        self._validate_raw_graph_files()

    def _validate_raw_graph_files(self):
        for suffix in ("docs", "edgelist"):
            path = osp.join(
                self.raw_dir,
                "{}_{}.txt".format(self.name, suffix),
            )
            if not osp.isfile(path):
                raise FileNotFoundError(path)

    def _process_graph(self):
        edge_path = osp.join(
            self.raw_dir,
            "{}_edgelist.txt".format(self.name),
        )
        edge_index = read_txt_array(edge_path, sep=",", dtype=torch.long).t()
        docs_path = osp.join(self.raw_dir, "{}_docs.txt".format(self.name))
        x = _read_features(docs_path)

        cache = {
            "version": GRAPH_CACHE_VERSION,
            "x": x,
            "edge_index": edge_index,
            "num_nodes": int(x.size(0)),
        }
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(cache, self.processed_path)
        return cache

    def _load_cache(self):
        if not osp.isfile(self.processed_path):
            return self._process_graph()
        cache = torch.load(self.processed_path, map_location="cpu")
        required = {"version", "x", "edge_index", "num_nodes"}
        if not isinstance(cache, dict) or set(cache) != required:
            raise RuntimeError(
                "{} is not a graph-only A2GNN cache; remove only this invalid "
                "graph.pt so it can be rebuilt".format(self.processed_path)
            )
        if cache["version"] != GRAPH_CACHE_VERSION:
            raise RuntimeError(
                "unsupported graph cache version in {}".format(self.processed_path)
            )
        return cache

    def load_graph(self):
        cache = self._load_cache()
        data = Data(
            x=cache["x"],
            edge_index=cache["edge_index"],
            num_nodes=cache["num_nodes"],
        )
        if getattr(data, "y", None) is not None:
            raise RuntimeError("training graph cache must not contain labels")
        return data


def load_source_data(data_root, name, split_seed):
    """Load labeled source data and regenerate masks for every invocation."""
    data = CitationGraphDataset(data_root, name).load_graph()
    data.y = load_citation_labels(data_root, name, data.num_nodes)
    data.train_mask, data.val_mask, data.test_mask = make_source_masks(
        data.num_nodes,
        split_seed,
    )
    return data


def load_unlabeled_target_data(data_root, name):
    """Load only x/edge_index/num_nodes for target-domain training."""
    return CitationGraphDataset(data_root, name).load_graph()


def load_target_labels_for_final_evaluation(data_root, name, num_nodes):
    """The sole target-label entrypoint; call only after best-state restore."""
    return load_citation_labels(data_root, name, num_nodes)


def evaluate_logits(logits, labels, mask=None):
    """Evaluate logits on exactly the requested nodes (all when mask is None)."""
    labels = labels.to(logits.device, non_blocking=True)
    if mask is not None:
        mask = mask.to(logits.device, non_blocking=True)
        logits = logits[mask]
        labels = labels[mask]
    if labels.numel() == 0:
        raise ValueError("evaluation mask selected no nodes")

    predictions = logits.argmax(dim=1)
    accuracy = predictions.eq(labels).float().mean().item()
    predictions_cpu = predictions.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    return {
        "accuracy": accuracy,
        "micro_f1": f1_score(labels_cpu, predictions_cpu, average="micro"),
        "macro_f1": f1_score(labels_cpu, predictions_cpu, average="macro"),
        "weighted_f1": f1_score(labels_cpu, predictions_cpu, average="weighted"),
    }


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Original A2GNN-MMD Gaussian kernel implementation."""
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_values = [
        torch.exp(-l2_distance / bandwidth_temp)
        for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_values)


def MMD(source_feat, target_feat, sampling_num=1000, times=5):
    """Original A2GNN-MMD sampling objective over all raw embeddings."""
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)
    source_sample = torch.randint(source_num, (times, sampling_num))
    target_sample = torch.randint(target_num, (times, sampling_num))

    mmd = 0
    for index in range(times):
        source_sample_feat = source_feat[source_sample[index]]
        target_sample_feat = target_feat[target_sample[index]]
        mmd = mmd + get_MMD(source_sample_feat, target_sample_feat)
    return mmd / times


def get_MMD(source_feat, target_feat, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Original A2GNN-MMD empirical kernel loss."""
    kernels = guassian_kernel(
        source_feat,
        target_feat,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    batch_size = min(int(source_feat.size(0)), int(target_feat.size(0)))
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)
