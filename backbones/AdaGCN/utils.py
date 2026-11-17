import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse import csc_matrix


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()

        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape

        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.

    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))

    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features_t, support_t, labels_t, labels_mask_t,
                        features_s, support_s, labels_s, labels_mask_s,
                        placeholders, lr_gen, lr_dis):
    """Construct the unlabeled-target training feed.

    ``labels_t`` and ``labels_mask_t`` must be dummy all-zero/all-false arrays.
    The function deliberately has no target ground-truth argument and no
    label-derived top-k inputs.
    """
    feed_dict = dict()

    feed_dict.update({placeholders['labels_t']: labels_t})
    feed_dict.update({placeholders['labels_mask_t']: labels_mask_t})
    feed_dict.update({placeholders['features_t']: features_t})
    feed_dict.update({placeholders['support_t'][i]: support_t[i] for i in range(len(support_t))})
    feed_dict.update({placeholders['num_features_nonzero_t']: features_t[1].shape})

    feed_dict.update({placeholders['labels_s']: labels_s})
    feed_dict.update({placeholders['labels_mask_s']: labels_mask_s})
    feed_dict.update({placeholders['features_s']: features_s})
    feed_dict.update({placeholders['support_s'][i]: support_s[i] for i in range(len(support_s))})
    feed_dict.update({placeholders['num_features_nonzero_s']: features_s[1].shape})

    feed_dict.update({placeholders['lr_gen']: lr_gen})
    feed_dict.update({placeholders['lr_dis']: lr_dis})

    return feed_dict


def construct_feed_dict_target(features, support, labels, labels_mask, placeholders):
    """Construct a target-only final-evaluation feed."""
    feed_dict = dict()

    feed_dict.update({placeholders['labels_t']: labels})
    feed_dict.update({placeholders['labels_mask_t']: labels_mask})
    feed_dict.update({placeholders['features_t']: features})
    feed_dict.update({placeholders['support_t'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero_t']: features[1].shape})
    feed_dict.update({placeholders['dropout']: 0.0})
    return feed_dict


def construct_feed_dict_source(features, support, labels, labels_mask, placeholders):
    """Construct a source-only validation/test feed."""
    feed_dict = dict()

    feed_dict.update({placeholders['labels_s']: labels})
    feed_dict.update({placeholders['labels_mask_s']: labels_mask})
    feed_dict.update({placeholders['features_s']: features})
    feed_dict.update({placeholders['support_s'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero_s']: features[1].shape})
    feed_dict.update({placeholders['dropout']: 0.0})
    return feed_dict


def deterministic_source_split(num_nodes, split_seed=42):
    """Return disjoint 80/10/10 masks from a local, label-free RNG.

    The split depends only on ``num_nodes`` and ``split_seed``.  It never reads
    labels and never modifies NumPy's global RNG state.
    """
    rng = np.random.RandomState(int(split_seed))
    indices = rng.permutation(int(num_nodes))
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_mask = sample_mask(train_idx, num_nodes)
    val_mask = sample_mask(val_idx, num_nodes)
    test_mask = sample_mask(test_idx, num_nodes)

    if not np.all((train_mask.astype(np.int8)
                   + val_mask.astype(np.int8)
                   + test_mask.astype(np.int8)) == 1):
        raise AssertionError('Source split masks must partition every node exactly once.')

    return train_mask, val_mask, test_mask


def masked_labels(labels, mask):
    """Copy labels only at selected nodes; all other rows are zero."""
    selected = np.zeros(labels.shape, dtype=labels.dtype)
    selected[mask] = labels[mask]
    return selected


def load_source_citation_data(file, split_seed=42):
    """Load a labeled source graph and create the release 80/10/10 split."""
    net = sio.loadmat(file, variable_names=['attrb', 'network', 'group'])
    X, A = net['attrb'], net['network']
    Y = np.asarray(net['group'], dtype=np.float32)
    if not sp.isspmatrix_csc(X):
        X = csc_matrix(X)

    train_mask, val_mask, test_mask = deterministic_source_split(
        Y.shape[0], split_seed
    )
    y_train = masked_labels(Y, train_mask)
    return A, X, Y, y_train, train_mask, val_mask, test_mask


def load_target_citation_graph(file):
    """Load only target features and adjacency; never deserialize ``group``."""
    net = sio.loadmat(file, variable_names=['attrb', 'network'])
    X, A = net['attrb'], net['network']
    if not sp.isspmatrix_csc(X):
        X = csc_matrix(X)
    if X.shape[0] != A.shape[0]:
        raise ValueError('Target feature and adjacency node counts do not match.')
    return A, X


def load_target_citation_labels(file, expected_nodes, expected_classes):
    """Load target labels for final evaluation after checkpoint restoration."""
    labels = np.asarray(
        sio.loadmat(file, variable_names=['group'])['group'],
        dtype=np.float32
    )
    expected_shape = (int(expected_nodes), int(expected_classes))
    if labels.shape != expected_shape:
        raise ValueError(
            'Target label shape {} does not match source-defined shape {}.'.format(
                labels.shape, expected_shape
            )
        )
    return labels
