from layers import *
from metrics import *
from becat.tensorflow import GeometryAdapterConfig, LatentGeometryAdapter
import sys

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


def define_variables(hiddens, weight_name, bias_name, flag=False):
    variables = {}
    for i in range(len(hiddens) - 1):
        variables[weight_name.format(i)] = glorot(
            [hiddens[i], hiddens[i + 1]],
            name=weight_name.format(i)
        )
        if flag:
            variables[bias_name.format(i)] = zeros(
                [hiddens[i + 1]],
                name=bias_name.format(i)
            )

    return variables


class GCN(object):
    def __init__(self,
                 placeholders,
                 input_dim,
                 N_s,
                 N_t,
                 bias_flag=False,
                 c_type='multi-label',
                 **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()

        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.da_method = FLAGS.da_method
        self.c_type = c_type
        self.bias_flag = bias_flag

        self.inputs_t = placeholders['features_t']
        self.inputs_s = placeholders['features_s']
        self.output_dim = placeholders['labels_t'].get_shape().as_list()[1]

        self.N = N_s if N_s < N_t else N_t
        self.N_s = N_s
        self.N_t = N_t

        self.hiddens_gcn = [input_dim] + [int(h) for h in FLAGS.hiddens_gcn.split('|')]
        self.hiddens_clf = (
            [self.hiddens_gcn[-1]]
            + [int(h) for h in FLAGS.hiddens_clf.split('|') if h != '']
            + [self.output_dim]
        )
        self.hiddens_dis = (
            [self.hiddens_gcn[-1]]
            + [int(h) for h in FLAGS.hiddens_dis.split('|') if h != '']
            + [1]
        )

        self.vars = {}

        self.layers_t = []
        self.activations_t = []
        self.layers_s = []
        self.activations_s = []

        self.clf_outputs_t = None
        self.hiddens_t = None

        self.clf_outputs_s = None
        self.hiddens_s = None
        self.sparse_gt_outputs_t = None
        self.sparse_gt_outputs_s = None
        self.geometry_adapter = None
        self.geometry_adapter_config = None
        self.geometry_adapter_loss = tf.constant(0.0, dtype=tf.float32)
        self.geometry_adapter_diagnostics = {}

        self.clf_loss = 0
        self.clf_loss_t = 0
        self.clf_loss_s = 0

        self.opt_op = None
        self.opt_op_t = None
        self.opt_op_s = None

        self.placeholders = placeholders
        self.build()

    @staticmethod
    def _finite_or_zero(tensor):
        return tf.where(tf.math.is_finite(tensor), tensor, tf.zeros_like(tensor))

    @staticmethod
    def _flag_enabled(name, default=False):
        try:
            return bool(getattr(FLAGS, name))
        except (AttributeError, KeyError):
            return default

    @staticmethod
    def _sparse_gt_attention_k_value():
        try:
            attention_k = int(FLAGS.sparse_gt_attention_k)
        except (AttributeError, KeyError):
            attention_k = -1

        if attention_k <= 0:
            attention_k = int(FLAGS.latent_topk_k)

        return attention_k

    @staticmethod
    def _safe_scalar_mean_std(values):
        eps = tf.constant(1e-8, dtype=tf.float32)
        n_values = tf.shape(values)[0]

        def non_empty():
            mean = tf.reduce_mean(values)
            std = tf.sqrt(tf.reduce_mean(tf.square(values - mean)) + eps)
            return mean, std

        def empty():
            return (
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32)
            )

        mean, std = tf.cond(n_values > 0, non_empty, empty)
        return GCN._finite_or_zero(mean), GCN._finite_or_zero(std)

    @staticmethod
    def _flag_explicitly_set(name):
        try:
            return getattr(FLAGS[name], 'present', 0) > 0
        except (AttributeError, KeyError):
            pass

        flag_prefix = '--{}'.format(name)
        for arg in sys.argv[1:]:
            if arg == flag_prefix or arg.startswith(flag_prefix + '='):
                return True

        return False

    def _geometry_adapter_config(self):
        """Build adapter configuration without changing legacy flag defaults.

        The adapter owns the requested final Hybrid BEC-RPB defaults.  An
        explicitly passed legacy geometry flag still overrides its matching
        adapter field, which keeps command-line experimentation compatible.
        """
        config = GeometryAdapterConfig()
        flag_to_config = {
            'use_latent_topk_bec': 'use_latent_topk_bec',
            'latent_topk_k': 'latent_topk_k',
            'latent_topk_tau': 'latent_topk_tau',
            'lambda_latent_bec': 'lambda_latent_bec',
            'lambda_latent_bec_stat': 'lambda_latent_bec_stat',
            'latent_bec_use_mmd': 'latent_bec_use_mmd',
            'latent_bec_sample_size': 'latent_bec_sample_size',
            'latent_bec_N': 'latent_bec_N',
            'latent_bec_mmd_sigma': 'latent_bec_mmd_sigma',
            'use_sparse_gt_head': 'use_sparse_gt_head',
            'sparse_gt_num_heads': 'sparse_gt_num_heads',
            'sparse_gt_num_layers': 'sparse_gt_num_layers',
            'sparse_gt_dropout': 'sparse_gt_dropout',
            'sparse_gt_use_curv_bias': 'sparse_gt_use_curv_bias',
            'sparse_gt_curv_bias_beta': 'sparse_gt_curv_bias_beta',
            'sparse_gt_curv_bias_mode': 'sparse_gt_curv_bias_mode',
            'sparse_gt_fourier_m': 'sparse_gt_fourier_m',
            'sparse_gt_fourier_sigma': 'sparse_gt_fourier_sigma',
            'sparse_gt_fourier_seed': 'sparse_gt_fourier_seed',
            'sparse_gt_eta_abs': 'sparse_gt_eta_abs',
            'sparse_gt_eta_fourier': 'sparse_gt_eta_fourier',
            'sparse_gt_bias_norm': 'sparse_gt_bias_norm',
            'sparse_gt_attention_k': 'sparse_gt_attention_k',
            'sparse_gt_use_self_loop': 'sparse_gt_use_self_loop',
            'sparse_gt_residual_alpha': 'sparse_gt_residual_alpha',
            'sparse_gt_use_layernorm': 'sparse_gt_use_layernorm',
            'sparse_gt_to_critic': 'sparse_gt_to_critic'
        }
        for flag_name, config_name in flag_to_config.items():
            if self._flag_explicitly_set(flag_name):
                setattr(config, config_name, getattr(FLAGS, flag_name))

        # Keep the existing trainer's warm-up/interval controls outside the
        # reusable adapter while making them available as scalar graph inputs.
        config.loss_active = self.placeholders['latent_bec_active']
        config.curvature_bias_active = self.placeholders['sparse_gt_curv_bias_active']
        return config

    def _latent_topk_edges(self, hiddens, topk=None, add_self_loops=False):
        eps = tf.constant(1e-8, dtype=tf.float32)
        n = tf.shape(hiddens)[0]
        n64 = tf.cast(n, tf.int64)
        if topk is None:
            topk = FLAGS.latent_topk_k
        k = tf.minimum(
            tf.maximum(tf.cast(topk, tf.int32), 0),
            tf.maximum(n - 1, 0)
        )

        norms = tf.sqrt(tf.reduce_sum(tf.square(hiddens), axis=1, keepdims=True) + eps)
        normalized = hiddens / (norms + eps)
        cosine = tf.matmul(normalized, normalized, transpose_b=True)
        cosine = self._finite_or_zero(cosine)
        if self._flag_enabled('debug_latent_bec_shapes'):
            cosine = tf.compat.v1.Print(
                cosine,
                [tf.shape(cosine)],
                message='latent_topk cosine_shape='
            )
        cosine = tf.linalg.set_diag(cosine, tf.fill([n], tf.constant(-1e9, tf.float32)))

        def no_edges():
            return (
                tf.zeros([0], dtype=tf.int32),
                tf.zeros([0], dtype=tf.int32),
                tf.zeros([0], dtype=tf.float32)
            )

        def build_edges():
            topk = tf.nn.top_k(cosine, k=k, sorted=False)
            rows = tf.tile(tf.expand_dims(tf.range(n), 1), [1, k])
            src = tf.reshape(rows, [-1])
            dst = tf.reshape(topk.indices, [-1])
            tau = tf.maximum(tf.cast(FLAGS.latent_topk_tau, tf.float32), eps)
            weights = tf.sigmoid(tf.reshape(topk.values, [-1]) / tau)

            sym_src = tf.concat([src, dst], axis=0)
            sym_dst = tf.concat([dst, src], axis=0)
            sym_weights = tf.concat([weights, weights], axis=0)

            keys = tf.cast(sym_src, tf.int64) * n64 + tf.cast(sym_dst, tf.int64)
            unique_keys, segment_ids = tf.unique(keys)
            unique_weights = tf.math.unsorted_segment_max(
                sym_weights,
                segment_ids,
                tf.shape(unique_keys)[0]
            )
            unique_src = tf.cast(tf.math.floordiv(unique_keys, n64), tf.int32)
            unique_dst = tf.cast(tf.math.mod(unique_keys, n64), tf.int32)

            return unique_src, unique_dst, self._finite_or_zero(unique_weights)

        edge_src, edge_dst, edge_weights = tf.cond(k > 0, build_edges, no_edges)
        if add_self_loops:
            edge_src, edge_dst, edge_weights = self._add_self_loops(
                edge_src,
                edge_dst,
                edge_weights,
                n
            )

        return edge_src, edge_dst, edge_weights

    def _add_self_loops(self, edge_src, edge_dst, edge_weights, n):
        self_ids = tf.range(n)
        return (
            tf.concat([edge_src, self_ids], axis=0),
            tf.concat([edge_dst, self_ids], axis=0),
            tf.concat([edge_weights, tf.ones([n], dtype=tf.float32)], axis=0)
        )

    def _latent_bec_kappa_from_edges(self, hiddens, edge_src, edge_dst, edge_weights):
        eps = tf.constant(1e-8, dtype=tf.float32)
        n = tf.shape(hiddens)[0]

        def zero_kappa():
            return tf.zeros([n], dtype=tf.float32)

        def compute_kappa():
            degree = tf.math.unsorted_segment_sum(edge_weights, edge_src, n)
            degree = self._finite_or_zero(degree)

            sparse_indices = tf.cast(tf.stack([edge_src, edge_dst], axis=1), tf.int64)
            sparse_shape = tf.cast(tf.stack([n, n]), tf.int64)
            adjacency = tf.SparseTensor(sparse_indices, edge_weights, sparse_shape)
            adjacency = tf.sparse.reorder(adjacency)

            sample_rows = tf.scatter_nd(
                tf.stack([edge_src, edge_dst], axis=1),
                edge_weights,
                tf.stack([n, n])
            )
            sample_rows = self._finite_or_zero(sample_rows)

            neighbor_degree = tf.squeeze(
                tf.matmul(sample_rows, tf.expand_dims(degree, 1)),
                axis=1
            )

            path2 = tf.transpose(
                tf.sparse.sparse_dense_matmul(adjacency, tf.transpose(sample_rows))
            )
            path2 = self._finite_or_zero(tf.maximum(path2, 0.0))

            triangle = 0.5 * tf.reduce_sum(path2 * sample_rows, axis=1)
            off_diag = 1.0 - tf.one_hot(tf.range(n), n, dtype=tf.float32)
            sphere2 = tf.reduce_sum(
                (1.0 - sample_rows) * (1.0 - tf.exp(-path2)) * off_diag,
                axis=1
            )

            if FLAGS.latent_bec_N < 0 or not np.isfinite(FLAGS.latent_bec_N):
                degree_coef = 1.0
            else:
                degree_coef = 1.0 - 2.0 / (FLAGS.latent_bec_N + 1e-8)

            scalar_bec = (
                degree_coef * degree
                - 0.5 * tf.square(degree)
                + 1.5 * neighbor_degree
                + triangle
                - 2.0 * sphere2
            )
            kappa = scalar_bec / (degree + eps)

            return self._finite_or_zero(kappa)

        return tf.cond(n <= 1, zero_kappa, compute_kappa)

    def _latent_soft_bec_signature(self, hiddens, edge_src=None, edge_dst=None,
                                   edge_weights=None, topk=None, nodes=None):
        if edge_src is None or edge_dst is None or edge_weights is None:
            edge_src, edge_dst, edge_weights = self._latent_topk_edges(
                hiddens,
                topk=topk,
                add_self_loops=False
            )

        kappa_full = self._latent_bec_kappa_from_edges(
            hiddens,
            edge_src,
            edge_dst,
            edge_weights
        )
        if nodes is None:
            return kappa_full

        return tf.gather(kappa_full, nodes)

    def _latent_bec_kappa(self, hiddens):
        n_full = tf.shape(hiddens)[0]
        sample_size = tf.minimum(
            tf.maximum(tf.cast(FLAGS.latent_bec_sample_size, tf.int32), 1),
            n_full
        )
        sample_ids = tf.random.shuffle(tf.range(n_full))[:sample_size]
        sampled_hiddens = tf.gather(hiddens, sample_ids)
        n = tf.shape(sampled_hiddens)[0]
        if self._flag_enabled('debug_latent_bec_shapes'):
            k_eff = tf.minimum(
                tf.cast(FLAGS.latent_topk_k, tf.int32),
                tf.maximum(n - 1, 0)
            )
            sampled_hiddens = tf.compat.v1.Print(
                sampled_hiddens,
                [n_full, n, k_eff, tf.shape(sampled_hiddens)],
                message='latent_bec n_full,n_sample,k_eff,sampled_hiddens_shape='
            )

        return self._latent_soft_bec_signature(sampled_hiddens, topk=FLAGS.latent_topk_k)

    def _sparse_gt_dropout(self, tensor):
        dropout_rate = tf.minimum(
            tf.maximum(self.placeholders['sparse_gt_dropout'], 0.0),
            0.95
        )
        keep_prob = 1.0 - dropout_rate

        return tf.compat.v1.nn.dropout(tensor, keep_prob=keep_prob)

    def _sparse_gt_layer_norm(self, tensor, layer_idx):
        eps = tf.constant(1e-8, dtype=tf.float32)
        mean = tf.reduce_mean(tensor, axis=1, keepdims=True)
        variance = tf.reduce_mean(tf.square(tensor - mean), axis=1, keepdims=True)
        normalized = (tensor - mean) / tf.sqrt(variance + eps)

        return (
            normalized * self.vars['sparse_gt_{}_ln_scale'.format(layer_idx)]
            + self.vars['sparse_gt_{}_ln_bias'.format(layer_idx)]
        )

    def _curvature_edge_bias(self, kappa, edge_src, edge_dst):
        eps = tf.constant(1e-8, dtype=tf.float32)
        mean = tf.reduce_mean(kappa)
        std = tf.sqrt(tf.reduce_mean(tf.square(kappa - mean)) + eps)
        kappa_norm = self._finite_or_zero((kappa - mean) / (std + eps))
        edge_bias = -tf.abs(
            tf.gather(kappa_norm, edge_src) - tf.gather(kappa_norm, edge_dst)
        )

        return self._finite_or_zero(edge_bias)

    @staticmethod
    def _sparse_gt_curv_bias_mode():
        try:
            mode = str(FLAGS.sparse_gt_curv_bias_mode).lower()
        except (AttributeError, KeyError):
            mode = 'abs'
        if mode not in ('abs', 'hybrid'):
            raise ValueError("sparse_gt_curv_bias_mode must be 'abs' or 'hybrid'.")

        return mode

    def _sparse_gt_effective_bias_norm(self):
        try:
            norm = str(FLAGS.sparse_gt_bias_norm).lower()
        except (AttributeError, KeyError):
            norm = 'none'
        if norm not in ('none', 'scale', 'zscore'):
            raise ValueError("sparse_gt_bias_norm must be 'none', 'scale', or 'zscore'.")

        if (
            self._sparse_gt_curv_bias_mode() == 'abs'
            and not self._flag_explicitly_set('sparse_gt_bias_norm')
        ):
            return 'none'

        return norm

    def _normalize_edge_bias(self, edge_bias, norm):
        eps = tf.constant(1e-8, dtype=tf.float32)
        edge_bias = self._finite_or_zero(edge_bias)
        if norm == 'none':
            return edge_bias

        mean, std = self._safe_scalar_mean_std(edge_bias)
        if norm == 'scale':
            return self._finite_or_zero(edge_bias / (std + eps))

        return self._finite_or_zero((edge_bias - mean) / (std + eps))

    def _fourier_curvature_edge_bias(self, kappa_norm, edge_src, edge_dst):
        m = int(FLAGS.sparse_gt_fourier_m)
        sigma = float(FLAGS.sparse_gt_fourier_sigma)
        if m <= 0:
            raise ValueError("sparse_gt_fourier_m must be positive.")
        if sigma < 0.0:
            raise ValueError("sparse_gt_fourier_sigma must be non-negative.")

        rng = np.random.RandomState(int(FLAGS.sparse_gt_fourier_seed))
        omega = rng.normal(loc=0.0, scale=sigma, size=(m,)).astype(np.float32)
        omega = tf.constant(omega, dtype=tf.float32)
        phase = tf.expand_dims(kappa_norm, axis=1) * tf.reshape(omega, [1, -1])
        phi = tf.concat([tf.cos(phase), tf.sin(phase)], axis=1)
        phi = phi * tf.sqrt(1.0 / tf.cast(m, tf.float32))
        diff = tf.gather(phi, edge_src) - tf.gather(phi, edge_dst)
        edge_bias = -tf.reduce_sum(tf.square(diff), axis=1)

        return self._finite_or_zero(edge_bias)

    def _hybrid_curvature_edge_bias_components(self, kappa, edge_src, edge_dst):
        eps = tf.constant(1e-8, dtype=tf.float32)
        mean = tf.reduce_mean(kappa)
        std = tf.sqrt(tf.reduce_mean(tf.square(kappa - mean)) + eps)
        kappa_norm = self._finite_or_zero((kappa - mean) / (std + eps))

        abs_raw = -tf.abs(
            tf.gather(kappa_norm, edge_src) - tf.gather(kappa_norm, edge_dst)
        )
        fourier_raw = self._fourier_curvature_edge_bias(kappa_norm, edge_src, edge_dst)
        norm = self._sparse_gt_effective_bias_norm()
        abs_bias = self._normalize_edge_bias(abs_raw, norm)
        fourier_bias = self._normalize_edge_bias(fourier_raw, norm)
        hybrid_bias = (
            tf.constant(float(FLAGS.sparse_gt_eta_abs), dtype=tf.float32) * abs_bias
            + tf.constant(float(FLAGS.sparse_gt_eta_fourier), dtype=tf.float32) * fourier_bias
        )

        return (
            self._finite_or_zero(abs_bias),
            self._finite_or_zero(fourier_bias),
            self._finite_or_zero(hybrid_bias)
        )

    def _curvature_edge_bias_components(self, kappa, edge_src, edge_dst):
        abs_bias = self._curvature_edge_bias(kappa, edge_src, edge_dst)
        zero_bias = tf.zeros_like(abs_bias)

        if self._sparse_gt_curv_bias_mode() == 'abs':
            abs_bias = self._normalize_edge_bias(
                abs_bias,
                self._sparse_gt_effective_bias_norm()
            )
            return abs_bias, abs_bias, zero_bias, abs_bias

        abs_bias, fourier_bias, hybrid_bias = self._hybrid_curvature_edge_bias_components(
            kappa,
            edge_src,
            edge_dst
        )
        return hybrid_bias, abs_bias, fourier_bias, hybrid_bias

    def _sparse_graph_transformer_layer(self, hiddens, edge_src, edge_dst,
                                        kappa, layer_idx):
        hidden_dim = self.hiddens_gcn[-1]
        num_heads = int(FLAGS.sparse_gt_num_heads)
        head_dim = hidden_dim // num_heads
        n = tf.shape(hiddens)[0]
        scale = tf.sqrt(tf.cast(head_dim, tf.float32))

        q = tf.matmul(hiddens, self.vars['sparse_gt_{}_q_weights'.format(layer_idx)])
        k = tf.matmul(hiddens, self.vars['sparse_gt_{}_k_weights'.format(layer_idx)])
        v = tf.matmul(hiddens, self.vars['sparse_gt_{}_v_weights'.format(layer_idx)])

        q = tf.reshape(q, [-1, num_heads, head_dim])
        k = tf.reshape(k, [-1, num_heads, head_dim])
        v = tf.reshape(v, [-1, num_heads, head_dim])

        q_src = tf.gather(q, edge_src)
        k_dst = tf.gather(k, edge_dst)
        scores = tf.reduce_sum(q_src * k_dst, axis=2) / scale
        edge_bias = tf.zeros([tf.shape(edge_src)[0]], dtype=tf.float32)
        abs_edge_bias = tf.zeros_like(edge_bias)
        fourier_edge_bias = tf.zeros_like(edge_bias)
        hybrid_edge_bias = tf.zeros_like(edge_bias)

        if self._flag_enabled('sparse_gt_use_curv_bias'):
            edge_bias, abs_edge_bias, fourier_edge_bias, hybrid_edge_bias = (
                self._curvature_edge_bias_components(kappa, edge_src, edge_dst)
            )
            beta = tf.cond(
                self.placeholders['sparse_gt_curv_bias_active'],
                lambda: tf.constant(FLAGS.sparse_gt_curv_bias_beta, dtype=tf.float32),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            scores = scores + beta * tf.expand_dims(edge_bias, axis=1)

        scores = self._finite_or_zero(scores)
        segment_max = tf.math.unsorted_segment_max(scores, edge_src, n)
        shifted = scores - tf.gather(segment_max, edge_src)
        exp_scores = self._finite_or_zero(tf.exp(shifted))
        segment_sum = tf.math.unsorted_segment_sum(exp_scores, edge_src, n)
        attn = exp_scores / (tf.gather(segment_sum, edge_src) + 1e-8)
        attn = self._finite_or_zero(attn)

        v_dst = tf.gather(v, edge_dst)
        messages = v_dst * tf.expand_dims(attn, axis=2)
        head_outputs = tf.math.unsorted_segment_sum(messages, edge_src, n)
        delta = tf.reshape(head_outputs, [-1, hidden_dim])
        delta = tf.matmul(
            delta,
            self.vars['sparse_gt_{}_out_weights'.format(layer_idx)]
        )
        delta = self._sparse_gt_dropout(delta)
        residual_alpha = float(FLAGS.sparse_gt_residual_alpha)
        if residual_alpha == 0.0:
            output = hiddens
        else:
            output = hiddens + tf.constant(
                residual_alpha,
                dtype=tf.float32
            ) * delta
        if self._flag_enabled('sparse_gt_use_layernorm'):
            output = self._sparse_gt_layer_norm(output, layer_idx)
        bias_mean, bias_std = self._safe_scalar_mean_std(edge_bias)
        abs_bias_mean, abs_bias_std = self._safe_scalar_mean_std(abs_edge_bias)
        fourier_bias_mean, fourier_bias_std = self._safe_scalar_mean_std(fourier_edge_bias)
        hybrid_bias_mean, hybrid_bias_std = self._safe_scalar_mean_std(hybrid_edge_bias)

        return (
            output,
            bias_mean,
            bias_std,
            abs_bias_mean,
            abs_bias_std,
            fourier_bias_mean,
            fourier_bias_std,
            hybrid_bias_mean,
            hybrid_bias_std
        )

    def _sparse_graph_transformer_head(self, hiddens):
        topk = self._sparse_gt_attention_k_value()
        edge_src, edge_dst, edge_weights = self._latent_topk_edges(
            hiddens,
            topk=topk,
            add_self_loops=False
        )
        kappa = tf.zeros([tf.shape(hiddens)[0]], dtype=tf.float32)
        if self._flag_enabled('sparse_gt_use_curv_bias'):
            kappa = self._latent_soft_bec_signature(
                hiddens,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_weights=edge_weights
            )

        if self._flag_enabled('sparse_gt_use_self_loop', True):
            edge_src, edge_dst, edge_weights = self._add_self_loops(
                edge_src,
                edge_dst,
                edge_weights,
                tf.shape(hiddens)[0]
            )

        output = hiddens
        bias_mean = tf.constant(0.0, dtype=tf.float32)
        bias_std = tf.constant(0.0, dtype=tf.float32)
        abs_bias_mean = tf.constant(0.0, dtype=tf.float32)
        abs_bias_std = tf.constant(0.0, dtype=tf.float32)
        fourier_bias_mean = tf.constant(0.0, dtype=tf.float32)
        fourier_bias_std = tf.constant(0.0, dtype=tf.float32)
        hybrid_bias_mean = tf.constant(0.0, dtype=tf.float32)
        hybrid_bias_std = tf.constant(0.0, dtype=tf.float32)
        for layer_idx in range(int(FLAGS.sparse_gt_num_layers)):
            (
                output,
                bias_mean,
                bias_std,
                abs_bias_mean,
                abs_bias_std,
                fourier_bias_mean,
                fourier_bias_std,
                hybrid_bias_mean,
                hybrid_bias_std
            ) = self._sparse_graph_transformer_layer(
                output,
                edge_src,
                edge_dst,
                kappa,
                layer_idx
            )

        return (
            output,
            kappa,
            bias_mean,
            bias_std,
            abs_bias_mean,
            abs_bias_std,
            fourier_bias_mean,
            fourier_bias_std,
            hybrid_bias_mean,
            hybrid_bias_std
        )

    def _mean_std(self, values):
        eps = tf.constant(1e-8, dtype=tf.float32)
        mean = tf.reduce_mean(values)
        std = tf.sqrt(tf.reduce_mean(tf.square(values - mean)) + eps)

        return self._finite_or_zero(mean), self._finite_or_zero(std)

    def _latent_topk_bec_alignment_loss(self):
        eps = tf.constant(1e-8, dtype=tf.float32)
        kappa_s = self._latent_bec_kappa(self.hiddens_s)
        kappa_t = self._latent_bec_kappa(self.hiddens_t)

        sigma = tf.maximum(tf.cast(FLAGS.latent_bec_mmd_sigma, tf.float32), eps)
        denom = 2.0 * tf.square(sigma) + eps
        diff_ss = tf.expand_dims(kappa_s, 1) - tf.expand_dims(kappa_s, 0)
        diff_tt = tf.expand_dims(kappa_t, 1) - tf.expand_dims(kappa_t, 0)
        diff_st = tf.expand_dims(kappa_s, 1) - tf.expand_dims(kappa_t, 0)
        mmd = (
            tf.reduce_mean(tf.exp(-tf.square(diff_ss) / denom))
            + tf.reduce_mean(tf.exp(-tf.square(diff_tt) / denom))
            - 2.0 * tf.reduce_mean(tf.exp(-tf.square(diff_st) / denom))
        )
        mmd = self._finite_or_zero(mmd)

        src_mean, src_std = self._mean_std(kappa_s)
        tgt_mean, tgt_std = self._mean_std(kappa_t)
        stat = tf.square(src_mean - tgt_mean) + tf.square(src_std - tgt_std)
        stat = self._finite_or_zero(stat)
        stat_loss = FLAGS.lambda_latent_bec_stat * stat
        if self._flag_enabled('latent_bec_use_mmd'):
            loss = mmd + stat_loss
        else:
            loss = stat_loss
        loss = self._finite_or_zero(loss)

        return {
            'loss': loss,
            'mmd': mmd,
            'stat': stat,
            'src_mean': src_mean,
            'src_std': src_std,
            'src_min': self._finite_or_zero(tf.reduce_min(kappa_s)),
            'src_max': self._finite_or_zero(tf.reduce_max(kappa_s)),
            'tgt_mean': tgt_mean,
            'tgt_std': tgt_std,
            'tgt_min': self._finite_or_zero(tf.reduce_min(kappa_t)),
            'tgt_max': self._finite_or_zero(tf.reduce_max(kappa_t))
        }

    def _build_latent_topk_bec_branch(self):
        zero = tf.constant(0.0, dtype=tf.float32)
        metric_names = [
            'loss', 'mmd', 'stat',
            'src_mean', 'src_std', 'src_min', 'src_max',
            'tgt_mean', 'tgt_std', 'tgt_min', 'tgt_max'
        ]

        if not FLAGS.use_latent_topk_bec:
            self.latent_bec_weighted = zero
            metrics = {name: zero for name in metric_names}
            if (
                self._flag_enabled('use_sparse_gt_head')
                and self._flag_enabled('sparse_gt_use_curv_bias')
            ):
                src_mean, src_std = self._mean_std(self.sparse_gt_kappa_s)
                tgt_mean, tgt_std = self._mean_std(self.sparse_gt_kappa_t)
                metrics.update({
                    'src_mean': src_mean,
                    'src_std': src_std,
                    'src_min': self._finite_or_zero(tf.reduce_min(self.sparse_gt_kappa_s)),
                    'src_max': self._finite_or_zero(tf.reduce_max(self.sparse_gt_kappa_s)),
                    'tgt_mean': tgt_mean,
                    'tgt_std': tgt_std,
                    'tgt_min': self._finite_or_zero(tf.reduce_min(self.sparse_gt_kappa_t)),
                    'tgt_max': self._finite_or_zero(tf.reduce_max(self.sparse_gt_kappa_t))
                })
        else:
            def active_values():
                metrics_active = self._latent_topk_bec_alignment_loss()
                weighted = FLAGS.lambda_latent_bec * metrics_active['loss']
                return [weighted] + [metrics_active[name] for name in metric_names]

            def inactive_values():
                return [zero for _ in range(len(metric_names) + 1)]

            values = tf.cond(
                self.placeholders['latent_bec_active'],
                active_values,
                inactive_values
            )
            self.latent_bec_weighted = values[0]
            metrics = {
                name: values[i + 1]
                for i, name in enumerate(metric_names)
            }

        self.recon_loss = zero
        self.latent_bec_loss = metrics['loss']
        self.latent_bec_mmd = metrics['mmd']
        self.latent_bec_stat = metrics['stat']
        self.latent_kappa_src_mean = metrics['src_mean']
        self.latent_kappa_src_std = metrics['src_std']
        self.latent_kappa_src_min = metrics['src_min']
        self.latent_kappa_src_max = metrics['src_max']
        self.latent_kappa_tgt_mean = metrics['tgt_mean']
        self.latent_kappa_tgt_std = metrics['tgt_std']
        self.latent_kappa_tgt_min = metrics['tgt_min']
        self.latent_kappa_tgt_max = metrics['tgt_max']

    def _build_geometry_adapter_branch(self):
        """Expose adapter diagnostics through the legacy training attributes."""
        diagnostics = self.geometry_adapter_diagnostics
        zero = tf.constant(0.0, dtype=tf.float32)
        self.recon_loss = zero
        self.latent_bec_weighted = self.geometry_adapter_loss
        self.latent_bec_loss = diagnostics['latent_bec_loss']
        self.latent_bec_mmd = diagnostics['latent_bec_mmd']
        self.latent_bec_stat = diagnostics['latent_bec_stat']
        self.latent_kappa_src_mean = diagnostics['latent_kappa_src_mean']
        self.latent_kappa_src_std = diagnostics['latent_kappa_src_std']
        self.latent_kappa_src_min = diagnostics['latent_kappa_src_min']
        self.latent_kappa_src_max = diagnostics['latent_kappa_src_max']
        self.latent_kappa_tgt_mean = diagnostics['latent_kappa_tgt_mean']
        self.latent_kappa_tgt_std = diagnostics['latent_kappa_tgt_std']
        self.latent_kappa_tgt_min = diagnostics['latent_kappa_tgt_min']
        self.latent_kappa_tgt_max = diagnostics['latent_kappa_tgt_max']

    def build(self):
        """Wrapper for _build()."""
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        theta_C = [v for v in tf.compat.v1.global_variables() if 'clf' in v.name]
        theta_D = [v for v in tf.compat.v1.global_variables() if 'dis' in v.name]
        theta_G = [
            v for v in tf.compat.v1.global_variables()
            if 'gcn' in v.name or 'sparse_gt' in v.name
        ]

        self.activations_t.append(self.inputs_t)
        for layer in self.layers_t:
            hidden = layer(self.activations_t[-1])
            self.activations_t.append(hidden)
        self.hiddens_t = self.activations_t[-1]

        self.activations_s.append(self.inputs_s)
        for layer in self.layers_s:
            hidden = layer(self.activations_s[-1])
            self.activations_s.append(hidden)
        self.hiddens_s = self.activations_s[-1]

        classifier_hiddens_t = self.hiddens_t
        classifier_hiddens_s = self.hiddens_s
        critic_hiddens_t = self.hiddens_t
        critic_hiddens_s = self.hiddens_s
        zero = tf.constant(0.0, dtype=tf.float32)
        self.sparse_gt_kappa_t = tf.zeros([tf.shape(self.hiddens_t)[0]], dtype=tf.float32)
        self.sparse_gt_kappa_s = tf.zeros([tf.shape(self.hiddens_s)[0]], dtype=tf.float32)
        self.sparse_gt_curv_bias_mean = zero
        self.sparse_gt_curv_bias_std = zero
        self.sparse_gt_abs_bias_mean = zero
        self.sparse_gt_abs_bias_std = zero
        self.sparse_gt_fourier_bias_mean = zero
        self.sparse_gt_fourier_bias_std = zero
        self.sparse_gt_hybrid_bias_mean = zero
        self.sparse_gt_hybrid_bias_std = zero

        if self._flag_enabled('use_geometry_adapter'):
            self.geometry_adapter_config = self._geometry_adapter_config()
            self.geometry_adapter = LatentGeometryAdapter(
                self.geometry_adapter_config,
                name='geometry_adapter'
            )
            (
                self.sparse_gt_outputs_s,
                self.sparse_gt_outputs_t,
                self.geometry_adapter_loss,
                self.geometry_adapter_diagnostics
            ) = self.geometry_adapter(
                self.hiddens_s,
                self.hiddens_t,
                training=self.placeholders['geometry_adapter_training']
            )
            self.vars.update(self.geometry_adapter.variables)
            self.sparse_gt_kappa_s = self.geometry_adapter_diagnostics['sparse_gt_kappa_s']
            self.sparse_gt_kappa_t = self.geometry_adapter_diagnostics['sparse_gt_kappa_t']
            classifier_hiddens_t = self.sparse_gt_outputs_t
            classifier_hiddens_s = self.sparse_gt_outputs_s
            self.sparse_gt_curv_bias_mean = self.geometry_adapter_diagnostics[
                'sparse_gt_curv_bias_mean'
            ]
            self.sparse_gt_curv_bias_std = self.geometry_adapter_diagnostics[
                'sparse_gt_curv_bias_std'
            ]
            self.sparse_gt_abs_bias_mean = self.geometry_adapter_diagnostics[
                'sparse_gt_abs_bias_mean'
            ]
            self.sparse_gt_abs_bias_std = self.geometry_adapter_diagnostics[
                'sparse_gt_abs_bias_std'
            ]
            self.sparse_gt_fourier_bias_mean = self.geometry_adapter_diagnostics[
                'sparse_gt_fourier_bias_mean'
            ]
            self.sparse_gt_fourier_bias_std = self.geometry_adapter_diagnostics[
                'sparse_gt_fourier_bias_std'
            ]
            self.sparse_gt_hybrid_bias_mean = self.geometry_adapter_diagnostics[
                'sparse_gt_hybrid_bias_mean'
            ]
            self.sparse_gt_hybrid_bias_std = self.geometry_adapter_diagnostics[
                'sparse_gt_hybrid_bias_std'
            ]
            if self.geometry_adapter_config.sparse_gt_to_critic:
                critic_hiddens_t = self.sparse_gt_outputs_t
                critic_hiddens_s = self.sparse_gt_outputs_s
            theta_G += self.geometry_adapter.trainable_variables

        elif self._flag_enabled('use_sparse_gt_head'):
            (
                self.sparse_gt_outputs_t,
                self.sparse_gt_kappa_t,
                bias_mean_t,
                bias_std_t,
                abs_bias_mean_t,
                abs_bias_std_t,
                fourier_bias_mean_t,
                fourier_bias_std_t,
                hybrid_bias_mean_t,
                hybrid_bias_std_t
            ) = self._sparse_graph_transformer_head(self.hiddens_t)
            (
                self.sparse_gt_outputs_s,
                self.sparse_gt_kappa_s,
                bias_mean_s,
                bias_std_s,
                abs_bias_mean_s,
                abs_bias_std_s,
                fourier_bias_mean_s,
                fourier_bias_std_s,
                hybrid_bias_mean_s,
                hybrid_bias_std_s
            ) = self._sparse_graph_transformer_head(self.hiddens_s)
            classifier_hiddens_t = self.sparse_gt_outputs_t
            classifier_hiddens_s = self.sparse_gt_outputs_s
            self.sparse_gt_curv_bias_mean = self._finite_or_zero(
                0.5 * (bias_mean_s + bias_mean_t)
            )
            self.sparse_gt_curv_bias_std = self._finite_or_zero(
                0.5 * (bias_std_s + bias_std_t)
            )
            self.sparse_gt_abs_bias_mean = self._finite_or_zero(
                0.5 * (abs_bias_mean_s + abs_bias_mean_t)
            )
            self.sparse_gt_abs_bias_std = self._finite_or_zero(
                0.5 * (abs_bias_std_s + abs_bias_std_t)
            )
            self.sparse_gt_fourier_bias_mean = self._finite_or_zero(
                0.5 * (fourier_bias_mean_s + fourier_bias_mean_t)
            )
            self.sparse_gt_fourier_bias_std = self._finite_or_zero(
                0.5 * (fourier_bias_std_s + fourier_bias_std_t)
            )
            self.sparse_gt_hybrid_bias_mean = self._finite_or_zero(
                0.5 * (hybrid_bias_mean_s + hybrid_bias_mean_t)
            )
            self.sparse_gt_hybrid_bias_std = self._finite_or_zero(
                0.5 * (hybrid_bias_std_s + hybrid_bias_std_t)
            )
            if self._flag_enabled('sparse_gt_to_critic'):
                critic_hiddens_t = self.sparse_gt_outputs_t
                critic_hiddens_s = self.sparse_gt_outputs_s

        self.clf_outputs_t = self._classifier(classifier_hiddens_t)
        self.clf_outputs_s = self._classifier(classifier_hiddens_s)

        if self.da_method == 'WD':
            if self.N_s < self.N_t:
                hiddens_s = tf.slice(critic_hiddens_s, [0, 0], [self.N, -1])
                hiddens_t_1 = tf.slice(critic_hiddens_t, [0, 0], [self.N, -1])
                hiddens_t_2 = tf.slice(
                    critic_hiddens_t,
                    [self.N_t - self.N - 1, 0],
                    [self.N, -1]
                )

                hiddens_s = tf.concat([hiddens_s, hiddens_s], axis=0)
                hiddens_t = tf.concat([hiddens_t_1, hiddens_t_2], axis=0)

                alpha = tf.random.uniform(shape=[2 * self.N, 1], minval=0., maxval=1.)
                difference = hiddens_s - hiddens_t
                interpolates = hiddens_t + (alpha * difference)

            elif self.N_s > self.N_t:
                hiddens_s_1 = tf.slice(critic_hiddens_s, [0, 0], [self.N, -1])
                hiddens_s_2 = tf.slice(
                    critic_hiddens_s,
                    [self.N_s - self.N - 1, 0],
                    [self.N, -1]
                )
                hiddens_t = tf.slice(critic_hiddens_t, [0, 0], [self.N, -1])

                hiddens_s = tf.concat([hiddens_s_1, hiddens_s_2], axis=0)
                hiddens_t = tf.concat([hiddens_t, hiddens_t], axis=0)

                alpha = tf.random.uniform(shape=[2 * self.N, 1], minval=0., maxval=1.)
                difference = hiddens_s - hiddens_t
                interpolates = hiddens_t + (alpha * difference)

            else:
                hiddens_s = tf.slice(critic_hiddens_s, [0, 0], [self.N, -1])
                hiddens_t = tf.slice(critic_hiddens_t, [0, 0], [self.N, -1])

                alpha = tf.random.uniform(shape=[self.N, 1], minval=0., maxval=1.)
                difference = hiddens_s - hiddens_t
                interpolates = hiddens_t + (alpha * difference)

            hiddens_whole = tf.concat(
                [critic_hiddens_s, critic_hiddens_t, interpolates],
                axis=0
            )

            critic_out = self._discriminator(hiddens_whole)
            critic_s = tf.slice(critic_out, [0, 0], [self.N_s, -1])
            critic_t = tf.slice(critic_out, [self.N_s, 0], [self.N_t, -1])

            self.wd_loss = tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t)

            gradients = tf.gradients(critic_out, [hiddens_whole])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
            self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            self.dis_loss_total = -self.wd_loss + FLAGS.gp_param * self.gradient_penalty

            self.wd_d_op = tf.compat.v1.train.AdamOptimizer(
                self.placeholders['lr_dis']
            ).minimize(
                self.dis_loss_total,
                var_list=theta_D
            )

        else:
            raise RuntimeError('Wrong DA Type!')

        self.l2_loss = FLAGS.l2_param * tf.add_n([
            tf.nn.l2_loss(v)
            for v in self.vars.values()
            if (
                'bias' not in v.name
                and 'dis' not in v.name
                and 'ln_scale' not in v.name
            )
        ])

        if self.c_type == 'multi-label':
            self.clf_loss_t = self.l2_loss + masked_sigmoid_cross_entropy(
                self.clf_outputs_t,
                self.placeholders['labels_t'],
                self.placeholders['labels_mask_t']
            )

            self.clf_loss_s = self.l2_loss + masked_sigmoid_cross_entropy(
                self.clf_outputs_s,
                self.placeholders['labels_s'],
                self.placeholders['labels_mask_s']
            )

            self.clf_loss = self.clf_loss_t + self.clf_loss_s - self.l2_loss
            self.clf_loss_pure = self.clf_loss - self.l2_loss

        if self.da_method == 'WD':
            if self._flag_enabled('use_geometry_adapter'):
                self._build_geometry_adapter_branch()
            else:
                self._build_latent_topk_bec_branch()
            self.total_loss_s_base = self.clf_loss_s + FLAGS.da_param * self.wd_loss
            self.total_loss_s_t_base = self.clf_loss + FLAGS.da_param * self.wd_loss
            self.total_loss_s = (
                self.total_loss_s_base + self.latent_bec_weighted
            )
            self.total_loss_s_t = (
                self.total_loss_s_t_base + self.latent_bec_weighted
            )
        else:
            raise RuntimeError('Wrong DA Type!')

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.placeholders['lr_gen'])

        self.opt_op_total_s = self.optimizer.minimize(
            self.total_loss_s,
            var_list=theta_G + theta_C
        )
        self.opt_op_total_s_base = self.optimizer.minimize(
            self.total_loss_s_base,
            var_list=theta_G + theta_C
        )

        self.opt_op_total_s_t = self.optimizer.minimize(
            self.total_loss_s_t,
            var_list=theta_G + theta_C
        )
        self.opt_op_total_s_t_base = self.optimizer.minimize(
            self.total_loss_s_t_base,
            var_list=theta_G + theta_C
        )

        if FLAGS.with_metrics:
            self._accuracy()

    def _classifier(self, input_tensor, act=tf.nn.relu):
        """Classification."""
        hiddens = input_tensor

        for i in range(len(self.hiddens_clf) - 2):
            hiddens = act(
                tf.matmul(hiddens, self.vars['clf_{}_weights'.format(i)])
                + self.vars['clf_{}_bias'.format(i)]
            )

        layer = len(self.hiddens_clf) - 2
        outputs = (
            tf.matmul(hiddens, self.vars['clf_{}_weights'.format(layer)])
            + self.vars['clf_{}_bias'.format(layer)]
        )

        return outputs

    def _discriminator(self, input_tensor, act=tf.nn.tanh):
        """Discriminator."""
        hiddens = input_tensor

        for i in range(len(self.hiddens_dis) - 2):
            hiddens = act(
                tf.matmul(hiddens, self.vars['dis_{}_weights'.format(i)])
                + self.vars['dis_{}_bias'.format(i)]
            )

        layer = len(self.hiddens_dis) - 2
        outputs = (
            tf.matmul(hiddens, self.vars['dis_{}_weights'.format(layer)])
            + self.vars['dis_{}_bias'.format(layer)]
        )

        return outputs

    def _build(self):
        self._create_variables()

        if FLAGS.gnn == 'gcn':
            self._create_generator()
        elif FLAGS.gnn == 'igcn':
            self._create_generator_igcn()
        else:
            raise RuntimeError('Wrong GNN Type!')

    def _create_variables(self):
        vars_gcn = define_variables(
            self.hiddens_gcn,
            weight_name='gcn_{}_weights',
            bias_name='gcn_{}_bias',
            flag=self.bias_flag
        )

        vars_clf = define_variables(
            self.hiddens_clf,
            weight_name='clf_{}_weights',
            bias_name='clf_{}_bias',
            flag=True
        )

        vars_dis = define_variables(
            self.hiddens_dis,
            weight_name='dis_{}_weights',
            bias_name='dis_{}_bias',
            flag=True
        )

        vars_sparse_gt = {}
        if (
            self._flag_enabled('use_sparse_gt_head')
            and not self._flag_enabled('use_geometry_adapter')
        ):
            hidden_dim = self.hiddens_gcn[-1]
            num_heads = int(FLAGS.sparse_gt_num_heads)
            num_layers = int(FLAGS.sparse_gt_num_layers)
            if num_heads <= 0:
                raise ValueError("sparse_gt_num_heads must be positive.")
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    "sparse_gt_num_heads ({}) must divide hidden_dim ({}).".format(
                        num_heads,
                        hidden_dim
                    )
                )
            if num_layers < 1:
                raise ValueError("sparse_gt_num_layers must be at least 1.")

            for layer_idx in range(num_layers):
                prefix = 'sparse_gt_{}'.format(layer_idx)
                vars_sparse_gt['{}_q_weights'.format(prefix)] = glorot(
                    [hidden_dim, hidden_dim],
                    name='{}_q_weights'.format(prefix)
                )
                vars_sparse_gt['{}_k_weights'.format(prefix)] = glorot(
                    [hidden_dim, hidden_dim],
                    name='{}_k_weights'.format(prefix)
                )
                vars_sparse_gt['{}_v_weights'.format(prefix)] = glorot(
                    [hidden_dim, hidden_dim],
                    name='{}_v_weights'.format(prefix)
                )
                vars_sparse_gt['{}_out_weights'.format(prefix)] = glorot(
                    [hidden_dim, hidden_dim],
                    name='{}_out_weights'.format(prefix)
                )
                vars_sparse_gt['{}_ln_scale'.format(prefix)] = tf.Variable(
                    tf.ones([hidden_dim], dtype=tf.float32),
                    name='{}_ln_scale'.format(prefix)
                )
                vars_sparse_gt['{}_ln_bias'.format(prefix)] = zeros(
                    [hidden_dim],
                    name='{}_ln_bias'.format(prefix)
                )

        self.vars = dict(vars_gcn)
        self.vars.update(vars_clf)
        self.vars.update(vars_dis)
        self.vars.update(vars_sparse_gt)

    def _create_generator(self):
        for i in range(len(self.hiddens_gcn) - 1):
            if i == 0:
                sparse = True
                act = tf.nn.relu
            else:
                sparse = False
                act = tf.nn.relu

            if self.bias_flag:
                bias = self.vars['gcn_{}_bias'.format(i)]
            else:
                bias = None

            self.layers_t.append(
                GraphConvolution(
                    input_dim=self.hiddens_gcn[i],
                    output_dim=self.hiddens_gcn[i + 1],
                    placeholder_dropout=self.placeholders['dropout'],
                    placeholder_support=self.placeholders['support_t'],
                    placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_t'],
                    weights=self.vars['gcn_{}_weights'.format(i)],
                    bias=bias,
                    act=act,
                    dropout=True,
                    sparse_inputs=sparse,
                    logging=self.logging
                )
            )

            self.layers_s.append(
                GraphConvolution(
                    input_dim=self.hiddens_gcn[i],
                    output_dim=self.hiddens_gcn[i + 1],
                    placeholder_dropout=self.placeholders['dropout'],
                    placeholder_support=self.placeholders['support_s'],
                    placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_s'],
                    weights=self.vars['gcn_{}_weights'.format(i)],
                    bias=bias,
                    act=act,
                    dropout=True,
                    sparse_inputs=sparse,
                    logging=self.logging
                )
            )

    def _create_generator_igcn(self):
        for i in range(len(self.hiddens_gcn) - 1):
            if i == 0:
                sparse = True
                act = tf.nn.relu
            else:
                sparse = False
                act = tf.nn.relu

            if self.bias_flag:
                bias = self.vars['gcn_{}_bias'.format(i)]
            else:
                bias = None

            if i < FLAGS.num_gcn_layers:
                self.layers_t.append(
                    GraphConvolution(
                        input_dim=self.hiddens_gcn[i],
                        output_dim=self.hiddens_gcn[i + 1],
                        placeholder_dropout=self.placeholders['dropout'],
                        placeholder_support=self.placeholders['support_t'],
                        placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_t'],
                        weights=self.vars['gcn_{}_weights'.format(i)],
                        bias=bias,
                        act=act,
                        dropout=True,
                        sparse_inputs=sparse,
                        logging=self.logging
                    )
                )

                self.layers_s.append(
                    GraphConvolution(
                        input_dim=self.hiddens_gcn[i],
                        output_dim=self.hiddens_gcn[i + 1],
                        placeholder_dropout=self.placeholders['dropout'],
                        placeholder_support=self.placeholders['support_s'],
                        placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_s'],
                        weights=self.vars['gcn_{}_weights'.format(i)],
                        bias=bias,
                        act=act,
                        dropout=True,
                        sparse_inputs=sparse,
                        logging=self.logging
                    )
                )

            else:
                self.layers_t.append(
                    Dense(
                        placeholder_dropout=self.placeholders['dropout'],
                        placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_t'],
                        weights=self.vars['gcn_{}_weights'.format(i)],
                        bias=bias,
                        act=act,
                        dropout=True,
                        sparse_inputs=sparse,
                        logging=self.logging
                    )
                )

                self.layers_s.append(
                    Dense(
                        placeholder_dropout=self.placeholders['dropout'],
                        placeholder_num_features_nonzero=self.placeholders['num_features_nonzero_s'],
                        weights=self.vars['gcn_{}_weights'.format(i)],
                        bias=bias,
                        act=act,
                        dropout=True,
                        sparse_inputs=sparse,
                        logging=self.logging
                    )
                )

    def _accuracy(self):
        if self.c_type == 'multi-label':
            target_true = tf.argmax(self.placeholders['labels_t'], axis=1)
            target_pred = tf.argmax(self.clf_outputs_t, axis=1)
            self.accuracy_t = tf.reduce_mean(tf.cast(tf.equal(
                tf.boolean_mask(target_pred, self.placeholders['labels_mask_t']),
                tf.boolean_mask(target_true, self.placeholders['labels_mask_t'])
            ), tf.float32))

            predictions_t = multi_label_hot(tf.sigmoid(self.clf_outputs_t))

            self.micro_f1_t, self.macro_f1_t, self.weighted_f1_t, self.TP_t, self.FP_t, self.FN_t = f1_score(
                self.placeholders['labels_t'],
                predictions_t,
                self.placeholders['labels_mask_t']
            )

            source_true = tf.argmax(self.placeholders['labels_s'], axis=1)
            source_pred = tf.argmax(self.clf_outputs_s, axis=1)
            self.accuracy_s = tf.reduce_mean(tf.cast(tf.equal(
                tf.boolean_mask(source_pred, self.placeholders['labels_mask_s']),
                tf.boolean_mask(source_true, self.placeholders['labels_mask_s'])
            ), tf.float32))

            predictions_s = multi_label_hot(tf.sigmoid(self.clf_outputs_s))

            self.micro_f1_s, self.macro_f1_s, self.weighted_f1_s, self.TP_s, self.FP_s, self.FN_s = f1_score(
                self.placeholders['labels_s'],
                predictions_s,
                self.placeholders['labels_mask_s']
            )
