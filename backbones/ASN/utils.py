"""Citation data utilities for the ASN release protocol."""

import os

import numpy as np
import scipy
import scipy.io as sio
import scipy.sparse as sp
import torch
from scipy.sparse import csc_matrix, lil_matrix


def normalize(mx):
    """Row-normalize a dense or sparse matrix."""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def MyScaleSimMat(W):
    """L1 row normalization used by the original ASN PPMI implementation."""
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(W)


def AggTranProbMat(G, step):
    """Compute the original aggregated K-step transition matrix."""
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k
    return A


def ComputePPMI(A):
    """Compute PPMI using the original ASN mathematics."""
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    p, _ = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0
    return PPMI


def _mat_path(data_root, dataset):
    """Return the external citation MAT path without assuming a data/ folder."""
    return os.path.join(os.path.abspath(os.path.expanduser(data_root)), str(dataset))


def _load_source_fields(data_root, dataset):
    net = sio.loadmat(
        _mat_path(data_root, dataset),
        variable_names=("network", "attrb", "group"),
    )
    features = net["attrb"]
    adjacency = net["network"]
    if not isinstance(features, scipy.sparse.lil_matrix):
        features = lil_matrix(features)
    return adjacency, features, net["group"]


def _load_unlabeled_graph_fields(data_root, dataset):
    net = sio.loadmat(
        _mat_path(data_root, dataset),
        variable_names=("network", "attrb"),
    )
    features = net["attrb"]
    adjacency = net["network"]
    if not isinstance(features, scipy.sparse.lil_matrix):
        features = lil_matrix(features)
    return adjacency, features


def _preprocess_graph(adjacency, features):
    """Apply the unchanged ASN PPMI, feature, and adjacency preprocessing."""
    A_k = AggTranProbMat(adjacency, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)
    X_n = sparse_mx_to_torch_sparse_tensor(lil_matrix(n_PPMI_))

    adjacency = (
        adjacency
        + adjacency.T.multiply(adjacency.T > adjacency)
        - adjacency.multiply(adjacency.T > adjacency)
    )
    normalized_features = normalize(features.toarray())
    normalized_adjacency = normalize(adjacency + sp.eye(adjacency.shape[0]))

    feature_tensor = torch.FloatTensor(normalized_features)
    adjacency_tensor = sparse_mx_to_torch_sparse_tensor(normalized_adjacency)
    return adjacency_tensor, feature_tensor, X_n, int(feature_tensor.shape[0])


def _convert_labels(group):
    """Preserve the original one-hot group-to-class-index conversion."""
    labels = np.asarray(group)
    labels = np.argmax(labels, axis=1)
    return torch.LongTensor(labels)


def load_data_citation(data_root, dataset):
    """Load a labeled source graph from network, attrb, and group fields."""
    adjacency, features, group = _load_source_fields(data_root, dataset)
    adjacency, features, X_n, num_nodes = _preprocess_graph(adjacency, features)
    labels = _convert_labels(group)
    return adjacency, features, labels, num_nodes, X_n


def load_unlabeled_data_citation(data_root, dataset):
    """Load target network and attrb fields without deserializing group."""
    adjacency, features = _load_unlabeled_graph_fields(data_root, dataset)
    adjacency, features, X_n, num_nodes = _preprocess_graph(adjacency, features)
    return adjacency, features, num_nodes, X_n


def load_target_labels(data_root, dataset, expected_num_nodes=None):
    """Load only target group for final evaluation after checkpoint restore."""
    net = sio.loadmat(
        _mat_path(data_root, dataset),
        variable_names=("group",),
    )
    labels = _convert_labels(net["group"])
    if expected_num_nodes is not None and labels.numel() != int(expected_num_nodes):
        raise ValueError(
            "Target label count {} does not match graph node count {}.".format(
                labels.numel(),
                int(expected_num_nodes),
            )
        )
    return labels


def split_source_indices(num_nodes, split_seed=42):
    """Create a label-independent deterministic 80/10/remainder source split."""
    generator = torch.Generator()
    generator.manual_seed(int(split_seed))
    indices = torch.randperm(int(num_nodes), generator=generator)
    train_size = int(0.8 * int(num_nodes))
    val_size = int(0.1 * int(num_nodes))
    idx_train = indices[:train_size]
    idx_val = indices[train_size : train_size + val_size]
    idx_test = indices[train_size + val_size :]
    return idx_train, idx_val, idx_test


def load_adj_label_for_reconstruction(data_root, dataset_name):
    """Build reconstruction targets from network only; group is never loaded."""
    net = sio.loadmat(
        _mat_path(data_root, dataset_name),
        variable_names=("network",),
    )
    A = net["network"]

    adj_label = A + sp.eye(A.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
    pos_weight = np.array(pos_weight).reshape(1, 1)
    pos_weight = torch.from_numpy(pos_weight)
    norm = A.shape[0] * A.shape[0] / float(
        (A.shape[0] * A.shape[0] - A.sum()) * 2
    )

    return adj_label, pos_weight, norm
