"""Backbone-agnostic latent geometry adapter for graph domain adaptation.

Adapter contract
----------------
Input:
    H_s, H_t: source and target latent node embeddings with shapes [N_s, d]
Output:
    O_s, O_t: refined latent embeddings with the same shapes as H_s and H_t
    geometry_loss: scalar latent curvature alignment loss
    diagnostics: curvature, alignment, and attention-bias statistics

This module deliberately has no dependency on AdaIGCN, graph convolutions,
original adjacency matrices, raw node features, or labels.  It only uses
TensorFlow's TF1-compatible graph APIs because the host project is TF1-style.
"""

from __future__ import division

import numpy as np
import tensorflow as tf


def _finite_or_zero(tensor):
    """Replace non-finite values without changing finite numerical values."""
    return tf.where(tf.math.is_finite(tensor), tensor, tf.zeros_like(tensor))


def _safe_scalar_mean_std(values):
    """Current-model-compatible mean and standard deviation for edge values."""
    eps = tf.constant(1e-8, dtype=tf.float32)
    n_values = tf.shape(values)[0]

    def non_empty():
        mean = tf.reduce_mean(values)
        std = tf.sqrt(tf.reduce_mean(tf.square(values - mean)) + eps)
        return mean, std

    def empty():
        return tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)

    mean, std = tf.cond(n_values > 0, non_empty, empty)
    return _finite_or_zero(mean), _finite_or_zero(std)


class GeometryAdapterConfig(object):
    """Configuration for :class:`LatentGeometryAdapter`.

    Defaults reproduce the requested final Hybrid BEC-RPB setting.  The two
    ``*_active`` fields may be Python booleans or scalar bool TensorFlow
    tensors, allowing a trainer to retain its existing warm-up schedule.
    """

    _DEFAULTS = {
        'use_latent_topk_bec': True,
        'latent_topk_k': 10,
        'latent_topk_tau': 0.5,
        'lambda_latent_bec': 0.001,
        'lambda_latent_bec_stat': 0.1,
        'latent_bec_use_mmd': False,
        'latent_bec_sample_size': 128,
        'latent_bec_N': -1.0,
        'latent_bec_mmd_sigma': 1.0,
        'use_sparse_gt_head': True,
        'sparse_gt_num_heads': 2,
        'sparse_gt_num_layers': 1,
        'sparse_gt_dropout': 0.3,
        'sparse_gt_use_curv_bias': True,
        'sparse_gt_curv_bias_beta': 0.05,
        'sparse_gt_curv_bias_mode': 'hybrid',
        'sparse_gt_fourier_m': 4,
        'sparse_gt_fourier_sigma': 1.0,
        'sparse_gt_fourier_seed': 1234,
        'sparse_gt_eta_abs': 1.0,
        'sparse_gt_eta_fourier': 0.1,
        'sparse_gt_bias_norm': 'scale',
        'sparse_gt_attention_k': 10,
        'sparse_gt_use_self_loop': True,
        'sparse_gt_residual_alpha': 0.1,
        'sparse_gt_use_layernorm': False,
        # Retained as integration metadata.  The host model decides whether
        # the critic consumes raw H or refined O.
        'sparse_gt_to_critic': False,
        'loss_active': True,
        'curvature_bias_active': True,
    }

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(self._DEFAULTS)
        if unknown:
            raise TypeError('Unknown geometry adapter config keys: {}'.format(
                ', '.join(sorted(unknown))
            ))
        values = dict(self._DEFAULTS)
        values.update(kwargs)
        for name, value in values.items():
            setattr(self, name, value)

    def as_dict(self):
        return {
            name: getattr(self, name)
            for name in self._DEFAULTS
        }


class LatentTopKGraph(object):
    """Construct a symmetric, weighted TopK graph from latent embeddings."""

    def __init__(self, topk=10, tau=0.5):
        self.topk = topk
        self.tau = tau

    @staticmethod
    def add_self_loops(edge_src, edge_dst, edge_weights, n_nodes):
        self_ids = tf.range(n_nodes)
        return (
            tf.concat([edge_src, self_ids], axis=0),
            tf.concat([edge_dst, self_ids], axis=0),
            tf.concat([edge_weights, tf.ones([n_nodes], dtype=tf.float32)], axis=0)
        )

    def __call__(self, hiddens, topk=None, add_self_loops=False):
        """Return ``(edge_src, edge_dst, edge_weights)`` from ``H [N, d]``."""
        eps = tf.constant(1e-8, dtype=tf.float32)
        n_nodes = tf.shape(hiddens)[0]
        n_nodes64 = tf.cast(n_nodes, tf.int64)
        topk = self.topk if topk is None else topk
        k = tf.minimum(
            tf.maximum(tf.cast(topk, tf.int32), 0),
            tf.maximum(n_nodes - 1, 0)
        )

        norms = tf.sqrt(tf.reduce_sum(tf.square(hiddens), axis=1, keepdims=True) + eps)
        normalized = hiddens / (norms + eps)
        cosine = _finite_or_zero(tf.matmul(normalized, normalized, transpose_b=True))
        cosine = tf.linalg.set_diag(
            cosine,
            tf.fill([n_nodes], tf.constant(-1e9, dtype=tf.float32))
        )

        def no_edges():
            return (
                tf.zeros([0], dtype=tf.int32),
                tf.zeros([0], dtype=tf.int32),
                tf.zeros([0], dtype=tf.float32)
            )

        def build_edges():
            selected = tf.nn.top_k(cosine, k=k, sorted=False)
            rows = tf.tile(tf.expand_dims(tf.range(n_nodes), 1), [1, k])
            src = tf.reshape(rows, [-1])
            dst = tf.reshape(selected.indices, [-1])
            tau = tf.maximum(tf.cast(self.tau, tf.float32), eps)
            weights = tf.sigmoid(tf.reshape(selected.values, [-1]) / tau)

            sym_src = tf.concat([src, dst], axis=0)
            sym_dst = tf.concat([dst, src], axis=0)
            sym_weights = tf.concat([weights, weights], axis=0)

            keys = tf.cast(sym_src, tf.int64) * n_nodes64 + tf.cast(sym_dst, tf.int64)
            unique_keys, segment_ids = tf.unique(keys)
            unique_weights = tf.math.unsorted_segment_max(
                sym_weights,
                segment_ids,
                tf.shape(unique_keys)[0]
            )
            unique_src = tf.cast(tf.math.floordiv(unique_keys, n_nodes64), tf.int32)
            unique_dst = tf.cast(tf.math.mod(unique_keys, n_nodes64), tf.int32)
            return unique_src, unique_dst, _finite_or_zero(unique_weights)

        edge_src, edge_dst, edge_weights = tf.cond(k > 0, build_edges, no_edges)
        if add_self_loops:
            edge_src, edge_dst, edge_weights = self.add_self_loops(
                edge_src, edge_dst, edge_weights, n_nodes
            )
        return edge_src, edge_dst, edge_weights


class BakryEmeryCurvature(object):
    """Compute the current latent weighted-graph BEC signature node-wise."""

    def __init__(self, dimension=-1.0):
        self.dimension = dimension

    def __call__(self, hiddens, edge_src, edge_dst, edge_weights):
        """Return ``kappa [N]`` for an already-built latent graph."""
        eps = tf.constant(1e-8, dtype=tf.float32)
        n_nodes = tf.shape(hiddens)[0]

        def zero_kappa():
            return tf.zeros([n_nodes], dtype=tf.float32)

        def compute_kappa():
            degree = _finite_or_zero(
                tf.math.unsorted_segment_sum(edge_weights, edge_src, n_nodes)
            )
            sparse_indices = tf.cast(tf.stack([edge_src, edge_dst], axis=1), tf.int64)
            sparse_shape = tf.cast(tf.stack([n_nodes, n_nodes]), tf.int64)
            adjacency = tf.sparse.reorder(
                tf.SparseTensor(sparse_indices, edge_weights, sparse_shape)
            )

            sample_rows = tf.scatter_nd(
                tf.stack([edge_src, edge_dst], axis=1),
                edge_weights,
                tf.stack([n_nodes, n_nodes])
            )
            sample_rows = _finite_or_zero(sample_rows)
            neighbor_degree = tf.squeeze(
                tf.matmul(sample_rows, tf.expand_dims(degree, 1)),
                axis=1
            )

            path2 = tf.transpose(
                tf.sparse.sparse_dense_matmul(adjacency, tf.transpose(sample_rows))
            )
            path2 = _finite_or_zero(tf.maximum(path2, 0.0))
            triangle = 0.5 * tf.reduce_sum(path2 * sample_rows, axis=1)
            off_diag = 1.0 - tf.one_hot(tf.range(n_nodes), n_nodes, dtype=tf.float32)
            sphere2 = tf.reduce_sum(
                (1.0 - sample_rows) * (1.0 - tf.exp(-path2)) * off_diag,
                axis=1
            )

            if self.dimension < 0 or not np.isfinite(self.dimension):
                degree_coef = 1.0
            else:
                degree_coef = 1.0 - 2.0 / (self.dimension + 1e-8)

            scalar_bec = (
                degree_coef * degree
                - 0.5 * tf.square(degree)
                + 1.5 * neighbor_degree
                + triangle
                - 2.0 * sphere2
            )
            return _finite_or_zero(scalar_bec / (degree + eps))

        return tf.cond(n_nodes <= 1, zero_kappa, compute_kappa)


class CurvatureStatAlign(object):
    """Align source and target sampled BEC distributions by mean/std and MMD."""

    _METRIC_NAMES = (
        'latent_bec_loss', 'latent_bec_mmd', 'latent_bec_stat',
        'latent_kappa_src_mean', 'latent_kappa_src_std',
        'latent_kappa_src_min', 'latent_kappa_src_max',
        'latent_kappa_tgt_mean', 'latent_kappa_tgt_std',
        'latent_kappa_tgt_min', 'latent_kappa_tgt_max'
    )

    def __init__(self, config, topk_graph, curvature):
        self.config = config
        self.topk_graph = topk_graph
        self.curvature = curvature

    def _sample_kappa(self, hiddens):
        n_full = tf.shape(hiddens)[0]
        sample_size = tf.minimum(
            tf.maximum(tf.cast(self.config.latent_bec_sample_size, tf.int32), 1),
            n_full
        )
        sample_ids = tf.random.shuffle(tf.range(n_full))[:sample_size]
        sampled_hiddens = tf.gather(hiddens, sample_ids)
        edge_src, edge_dst, edge_weights = self.topk_graph(
            sampled_hiddens,
            topk=self.config.latent_topk_k,
            add_self_loops=False
        )
        return self.curvature(sampled_hiddens, edge_src, edge_dst, edge_weights)

    @staticmethod
    def _mean_std(values):
        eps = tf.constant(1e-8, dtype=tf.float32)
        mean = tf.reduce_mean(values)
        std = tf.sqrt(tf.reduce_mean(tf.square(values - mean)) + eps)
        return _finite_or_zero(mean), _finite_or_zero(std)

    def __call__(self, hiddens_s, hiddens_t, active=None):
        """Return weighted geometry loss and legacy-compatible diagnostics."""
        zero = tf.constant(0.0, dtype=tf.float32)
        active = self.config.loss_active if active is None else active
        active = tf.cast(tf.convert_to_tensor(active), tf.bool)

        def active_values():
            kappa_s = self._sample_kappa(hiddens_s)
            kappa_t = self._sample_kappa(hiddens_t)
            eps = tf.constant(1e-8, dtype=tf.float32)
            sigma = tf.maximum(tf.cast(self.config.latent_bec_mmd_sigma, tf.float32), eps)
            denom = 2.0 * tf.square(sigma) + eps
            diff_ss = tf.expand_dims(kappa_s, 1) - tf.expand_dims(kappa_s, 0)
            diff_tt = tf.expand_dims(kappa_t, 1) - tf.expand_dims(kappa_t, 0)
            diff_st = tf.expand_dims(kappa_s, 1) - tf.expand_dims(kappa_t, 0)
            mmd = _finite_or_zero(
                tf.reduce_mean(tf.exp(-tf.square(diff_ss) / denom))
                + tf.reduce_mean(tf.exp(-tf.square(diff_tt) / denom))
                - 2.0 * tf.reduce_mean(tf.exp(-tf.square(diff_st) / denom))
            )
            src_mean, src_std = self._mean_std(kappa_s)
            tgt_mean, tgt_std = self._mean_std(kappa_t)
            stat = _finite_or_zero(
                tf.square(src_mean - tgt_mean) + tf.square(src_std - tgt_std)
            )
            stat_loss = tf.cast(self.config.lambda_latent_bec_stat, tf.float32) * stat
            loss = mmd + stat_loss if self.config.latent_bec_use_mmd else stat_loss
            loss = _finite_or_zero(loss)
            weighted = tf.cast(self.config.lambda_latent_bec, tf.float32) * loss
            return [
                _finite_or_zero(weighted), loss, mmd, stat,
                src_mean, src_std, _finite_or_zero(tf.reduce_min(kappa_s)),
                _finite_or_zero(tf.reduce_max(kappa_s)), tgt_mean, tgt_std,
                _finite_or_zero(tf.reduce_min(kappa_t)),
                _finite_or_zero(tf.reduce_max(kappa_t))
            ]

        def inactive_values():
            return [zero for _ in range(len(self._METRIC_NAMES) + 1)]

        if self.config.use_latent_topk_bec:
            values = tf.cond(active, active_values, inactive_values)
        else:
            values = inactive_values()

        diagnostics = {'latent_bec_weighted': values[0]}
        diagnostics.update({
            name: values[index + 1]
            for index, name in enumerate(self._METRIC_NAMES)
        })
        diagnostics['geometry_loss'] = values[0]
        diagnostics['curvature_src_mean'] = diagnostics['latent_kappa_src_mean']
        diagnostics['curvature_src_std'] = diagnostics['latent_kappa_src_std']
        diagnostics['curvature_tgt_mean'] = diagnostics['latent_kappa_tgt_mean']
        diagnostics['curvature_tgt_std'] = diagnostics['latent_kappa_tgt_std']
        return values[0], diagnostics


class HybridCurvatureRPB(object):
    """Build scalar or hybrid scalar/Fourier curvature attention biases."""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _normalize(edge_bias, normalization):
        eps = tf.constant(1e-8, dtype=tf.float32)
        edge_bias = _finite_or_zero(edge_bias)
        if normalization == 'none':
            return edge_bias
        mean, std = _safe_scalar_mean_std(edge_bias)
        if normalization == 'scale':
            return _finite_or_zero(edge_bias / (std + eps))
        if normalization == 'zscore':
            return _finite_or_zero((edge_bias - mean) / (std + eps))
        raise ValueError("sparse_gt_bias_norm must be 'none', 'scale', or 'zscore'.")

    @staticmethod
    def _standardize(kappa):
        eps = tf.constant(1e-8, dtype=tf.float32)
        mean = tf.reduce_mean(kappa)
        std = tf.sqrt(tf.reduce_mean(tf.square(kappa - mean)) + eps)
        return _finite_or_zero((kappa - mean) / (std + eps))

    def _fourier_bias(self, kappa_norm, edge_src, edge_dst):
        m = int(self.config.sparse_gt_fourier_m)
        sigma = float(self.config.sparse_gt_fourier_sigma)
        if m <= 0:
            raise ValueError('sparse_gt_fourier_m must be positive.')
        if sigma < 0.0:
            raise ValueError('sparse_gt_fourier_sigma must be non-negative.')

        rng = np.random.RandomState(int(self.config.sparse_gt_fourier_seed))
        omega = tf.constant(rng.normal(0.0, sigma, size=(m,)).astype(np.float32))
        phase = tf.expand_dims(kappa_norm, axis=1) * tf.reshape(omega, [1, -1])
        phi = tf.concat([tf.cos(phase), tf.sin(phase)], axis=1)
        phi = phi * tf.sqrt(1.0 / tf.cast(m, tf.float32))
        diff = tf.gather(phi, edge_src) - tf.gather(phi, edge_dst)
        return _finite_or_zero(-tf.reduce_sum(tf.square(diff), axis=1))

    def __call__(self, kappa, edge_src, edge_dst):
        """Return the selected edge bias plus its scalar/Fourier components."""
        mode = str(self.config.sparse_gt_curv_bias_mode).lower()
        if mode not in ('abs', 'hybrid'):
            raise ValueError("sparse_gt_curv_bias_mode must be 'abs' or 'hybrid'.")

        kappa_norm = self._standardize(kappa)
        abs_raw = -tf.abs(
            tf.gather(kappa_norm, edge_src) - tf.gather(kappa_norm, edge_dst)
        )
        abs_bias = self._normalize(abs_raw, self.config.sparse_gt_bias_norm)
        zero_bias = tf.zeros_like(abs_bias)
        if mode == 'abs':
            return {
                'edge_bias': _finite_or_zero(abs_bias),
                'abs_bias': _finite_or_zero(abs_bias),
                'fourier_bias': zero_bias,
                'hybrid_bias': _finite_or_zero(abs_bias)
            }

        fourier_raw = self._fourier_bias(kappa_norm, edge_src, edge_dst)
        fourier_bias = self._normalize(fourier_raw, self.config.sparse_gt_bias_norm)
        hybrid_bias = (
            tf.cast(self.config.sparse_gt_eta_abs, tf.float32) * abs_bias
            + tf.cast(self.config.sparse_gt_eta_fourier, tf.float32) * fourier_bias
        )
        return {
            'edge_bias': _finite_or_zero(hybrid_bias),
            'abs_bias': _finite_or_zero(abs_bias),
            'fourier_bias': _finite_or_zero(fourier_bias),
            'hybrid_bias': _finite_or_zero(hybrid_bias)
        }


def _glorot_variable(shape, name):
    """Use the project's original Glorot-uniform numerical initialization."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(
        shape,
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )
    return tf.Variable(initial, name=name)


def _zeros_variable(shape, name):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32), name=name)


class ResidualSparseGraphTransformer(object):
    """Residual sparse multi-head Transformer operating on a latent TopK graph."""

    def __init__(self, config, hidden_dim, rpb, name='residual_sparse_gt'):
        self.config = config
        self.hidden_dim = int(hidden_dim)
        self.rpb = rpb
        self.name = name
        self.variables = {}
        self._create_variables()

    @property
    def trainable_variables(self):
        return list(self.variables.values())

    def _create_variables(self):
        num_heads = int(self.config.sparse_gt_num_heads)
        num_layers = int(self.config.sparse_gt_num_layers)
        if num_heads <= 0:
            raise ValueError('sparse_gt_num_heads must be positive.')
        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                'sparse_gt_num_heads ({}) must divide hidden_dim ({}).'.format(
                    num_heads, self.hidden_dim
                )
            )
        if num_layers < 1:
            raise ValueError('sparse_gt_num_layers must be at least 1.')

        with tf.compat.v1.variable_scope(self.name):
            for layer_idx in range(num_layers):
                prefix = 'sparse_gt_{}'.format(layer_idx)
                self.variables['{}_q_weights'.format(prefix)] = _glorot_variable(
                    [self.hidden_dim, self.hidden_dim], '{}_q_weights'.format(prefix)
                )
                self.variables['{}_k_weights'.format(prefix)] = _glorot_variable(
                    [self.hidden_dim, self.hidden_dim], '{}_k_weights'.format(prefix)
                )
                self.variables['{}_v_weights'.format(prefix)] = _glorot_variable(
                    [self.hidden_dim, self.hidden_dim], '{}_v_weights'.format(prefix)
                )
                self.variables['{}_out_weights'.format(prefix)] = _glorot_variable(
                    [self.hidden_dim, self.hidden_dim], '{}_out_weights'.format(prefix)
                )
                self.variables['{}_ln_scale'.format(prefix)] = tf.Variable(
                    tf.ones([self.hidden_dim], dtype=tf.float32),
                    name='{}_ln_scale'.format(prefix)
                )
                self.variables['{}_ln_bias'.format(prefix)] = _zeros_variable(
                    [self.hidden_dim], '{}_ln_bias'.format(prefix)
                )

    def _dropout(self, tensor, training):
        dropout_rate = min(max(float(self.config.sparse_gt_dropout), 0.0), 0.95)
        if dropout_rate == 0.0:
            return tensor
        training = tf.cast(tf.convert_to_tensor(training), tf.bool)
        return tf.cond(
            training,
            lambda: tf.compat.v1.nn.dropout(tensor, keep_prob=1.0 - dropout_rate),
            lambda: tensor
        )

    def _layer_norm(self, tensor, layer_idx):
        eps = tf.constant(1e-8, dtype=tf.float32)
        mean = tf.reduce_mean(tensor, axis=1, keepdims=True)
        variance = tf.reduce_mean(tf.square(tensor - mean), axis=1, keepdims=True)
        normalized = (tensor - mean) / tf.sqrt(variance + eps)
        prefix = 'sparse_gt_{}'.format(layer_idx)
        return (
            normalized * self.variables['{}_ln_scale'.format(prefix)]
            + self.variables['{}_ln_bias'.format(prefix)]
        )

    def _layer(self, hiddens, edge_src, edge_dst, kappa, layer_idx,
               training, curvature_bias_active):
        num_heads = int(self.config.sparse_gt_num_heads)
        head_dim = self.hidden_dim // num_heads
        n_nodes = tf.shape(hiddens)[0]
        scale = tf.sqrt(tf.cast(head_dim, tf.float32))
        prefix = 'sparse_gt_{}'.format(layer_idx)

        q = tf.reshape(
            tf.matmul(hiddens, self.variables['{}_q_weights'.format(prefix)]),
            [-1, num_heads, head_dim]
        )
        k = tf.reshape(
            tf.matmul(hiddens, self.variables['{}_k_weights'.format(prefix)]),
            [-1, num_heads, head_dim]
        )
        v = tf.reshape(
            tf.matmul(hiddens, self.variables['{}_v_weights'.format(prefix)]),
            [-1, num_heads, head_dim]
        )

        q_src = tf.gather(q, edge_src)
        k_dst = tf.gather(k, edge_dst)
        scores = tf.reduce_sum(q_src * k_dst, axis=2) / scale
        edge_bias = tf.zeros([tf.shape(edge_src)[0]], dtype=tf.float32)
        abs_edge_bias = tf.zeros_like(edge_bias)
        fourier_edge_bias = tf.zeros_like(edge_bias)
        hybrid_edge_bias = tf.zeros_like(edge_bias)

        if self.config.sparse_gt_use_curv_bias:
            bias_parts = self.rpb(kappa, edge_src, edge_dst)
            edge_bias = bias_parts['edge_bias']
            abs_edge_bias = bias_parts['abs_bias']
            fourier_edge_bias = bias_parts['fourier_bias']
            hybrid_edge_bias = bias_parts['hybrid_bias']
            curvature_bias_active = tf.cast(
                tf.convert_to_tensor(curvature_bias_active), tf.bool
            )
            beta = tf.cond(
                curvature_bias_active,
                lambda: tf.constant(self.config.sparse_gt_curv_bias_beta, tf.float32),
                lambda: tf.constant(0.0, tf.float32)
            )
            scores = scores + beta * tf.expand_dims(edge_bias, axis=1)

        scores = _finite_or_zero(scores)
        segment_max = tf.math.unsorted_segment_max(scores, edge_src, n_nodes)
        shifted = scores - tf.gather(segment_max, edge_src)
        exp_scores = _finite_or_zero(tf.exp(shifted))
        segment_sum = tf.math.unsorted_segment_sum(exp_scores, edge_src, n_nodes)
        attn = _finite_or_zero(exp_scores / (tf.gather(segment_sum, edge_src) + 1e-8))

        v_dst = tf.gather(v, edge_dst)
        messages = v_dst * tf.expand_dims(attn, axis=2)
        head_outputs = tf.math.unsorted_segment_sum(messages, edge_src, n_nodes)
        delta = tf.reshape(head_outputs, [-1, self.hidden_dim])
        delta = tf.matmul(delta, self.variables['{}_out_weights'.format(prefix)])
        delta = self._dropout(delta, training)
        residual_alpha = float(self.config.sparse_gt_residual_alpha)
        output = hiddens if residual_alpha == 0.0 else (
            hiddens + tf.constant(residual_alpha, tf.float32) * delta
        )
        if self.config.sparse_gt_use_layernorm:
            output = self._layer_norm(output, layer_idx)

        bias_mean, bias_std = _safe_scalar_mean_std(edge_bias)
        abs_mean, abs_std = _safe_scalar_mean_std(abs_edge_bias)
        fourier_mean, fourier_std = _safe_scalar_mean_std(fourier_edge_bias)
        hybrid_mean, hybrid_std = _safe_scalar_mean_std(hybrid_edge_bias)
        return output, {
            'sparse_gt_curv_bias_mean': bias_mean,
            'sparse_gt_curv_bias_std': bias_std,
            'sparse_gt_abs_bias_mean': abs_mean,
            'sparse_gt_abs_bias_std': abs_std,
            'sparse_gt_fourier_bias_mean': fourier_mean,
            'sparse_gt_fourier_bias_std': fourier_std,
            'sparse_gt_hybrid_bias_mean': hybrid_mean,
            'sparse_gt_hybrid_bias_std': hybrid_std
        }

    def __call__(self, hiddens, edge_src, edge_dst, kappa, training,
                 curvature_bias_active):
        output = hiddens
        diagnostics = None
        for layer_idx in range(int(self.config.sparse_gt_num_layers)):
            output, diagnostics = self._layer(
                output, edge_src, edge_dst, kappa, layer_idx,
                training, curvature_bias_active
            )
        return output, diagnostics


class LatentGeometryAdapter(object):
    """Reusable latent-only geometry module.

    Public interface:

    ``O_s, O_t, geometry_loss, diagnostics = adapter(H_s, H_t, training=True)``

    The adapter does not accept original adjacency, raw features, labels, or an
    AdaIGCN model object.  Its sole required inputs are source and target latent
    embedding matrices with the same feature dimension.
    """

    _BIAS_METRIC_NAMES = (
        'sparse_gt_curv_bias_mean', 'sparse_gt_curv_bias_std',
        'sparse_gt_abs_bias_mean', 'sparse_gt_abs_bias_std',
        'sparse_gt_fourier_bias_mean', 'sparse_gt_fourier_bias_std',
        'sparse_gt_hybrid_bias_mean', 'sparse_gt_hybrid_bias_std'
    )

    def __init__(self, config=None, name='geometry_adapter'):
        if config is None:
            config = GeometryAdapterConfig()
        if isinstance(config, dict):
            config = GeometryAdapterConfig(**config)
        if not isinstance(config, GeometryAdapterConfig):
            raise TypeError('config must be a GeometryAdapterConfig or dict.')
        self.config = config
        self.name = name
        self.topk_graph = LatentTopKGraph(
            topk=config.latent_topk_k,
            tau=config.latent_topk_tau
        )
        self.curvature = BakryEmeryCurvature(config.latent_bec_N)
        self.stat_align = CurvatureStatAlign(config, self.topk_graph, self.curvature)
        self.rpb = HybridCurvatureRPB(config)
        self.transformer = None
        self._hidden_dim = None

    @property
    def trainable_variables(self):
        if self.transformer is None:
            return []
        return self.transformer.trainable_variables

    @property
    def variables(self):
        if self.transformer is None:
            return {}
        return dict(self.transformer.variables)

    def _ensure_transformer(self, hiddens_s, hiddens_t):
        hidden_dim_s = hiddens_s.get_shape().as_list()[-1]
        hidden_dim_t = hiddens_t.get_shape().as_list()[-1]
        if hidden_dim_s is None or hidden_dim_t is None:
            raise ValueError('LatentGeometryAdapter requires a statically known embedding width.')
        if hidden_dim_s != hidden_dim_t:
            raise ValueError(
                'H_s and H_t must have the same embedding width; got {} and {}.'.format(
                    hidden_dim_s, hidden_dim_t
                )
            )
        if self.transformer is None:
            self._hidden_dim = hidden_dim_s
            self.transformer = ResidualSparseGraphTransformer(
                self.config,
                hidden_dim_s,
                self.rpb,
                name=self.name
            )
        elif self._hidden_dim != hidden_dim_s:
            raise ValueError('LatentGeometryAdapter cannot be reused with another embedding width.')

    def _refine_one_domain(self, hiddens, training):
        zero = tf.constant(0.0, dtype=tf.float32)
        zero_kappa = tf.zeros([tf.shape(hiddens)[0]], dtype=tf.float32)
        zero_bias_metrics = {name: zero for name in self._BIAS_METRIC_NAMES}
        if not self.config.use_sparse_gt_head:
            return hiddens, zero_kappa, zero_bias_metrics

        attention_k = int(self.config.sparse_gt_attention_k)
        if attention_k <= 0:
            attention_k = int(self.config.latent_topk_k)
        edge_src, edge_dst, edge_weights = self.topk_graph(
            hiddens,
            topk=attention_k,
            add_self_loops=False
        )
        kappa = zero_kappa
        if self.config.sparse_gt_use_curv_bias:
            kappa = self.curvature(hiddens, edge_src, edge_dst, edge_weights)
        if self.config.sparse_gt_use_self_loop:
            edge_src, edge_dst, edge_weights = LatentTopKGraph.add_self_loops(
                edge_src, edge_dst, edge_weights, tf.shape(hiddens)[0]
            )

        output, bias_metrics = self.transformer(
            hiddens,
            edge_src,
            edge_dst,
            kappa,
            training=training,
            curvature_bias_active=self.config.curvature_bias_active
        )
        return output, kappa, bias_metrics

    def __call__(self, hiddens_s, hiddens_t, training=True):
        """Refine ``H_s``/``H_t`` and return a scalar geometry regularizer."""
        self._ensure_transformer(hiddens_s, hiddens_t)
        geometry_loss, diagnostics = self.stat_align(hiddens_s, hiddens_t)
        outputs_s, kappa_s, bias_s = self._refine_one_domain(hiddens_s, training)
        outputs_t, kappa_t, bias_t = self._refine_one_domain(hiddens_t, training)

        src_mean, src_std = _safe_scalar_mean_std(kappa_s)
        tgt_mean, tgt_std = _safe_scalar_mean_std(kappa_t)
        diagnostics.update({
            'sparse_gt_kappa_s': kappa_s,
            'sparse_gt_kappa_t': kappa_t,
            'sparse_gt_kappa_src_mean': src_mean,
            'sparse_gt_kappa_src_std': src_std,
            'sparse_gt_kappa_tgt_mean': tgt_mean,
            'sparse_gt_kappa_tgt_std': tgt_std,
        })
        for name in self._BIAS_METRIC_NAMES:
            diagnostics[name] = _finite_or_zero(0.5 * (bias_s[name] + bias_t[name]))

        return outputs_s, outputs_t, geometry_loss, diagnostics
