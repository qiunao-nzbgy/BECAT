"""Canonical PyTorch BECAT latent geometry adapter.

This file is a PyTorch conversion of the project-local TensorFlow
``geometry_adapter.py``.  It is intentionally backbone-agnostic:

Input:
    H_s, H_t: source and target latent node embeddings, each shaped [N, d]

Output:
    O_s, O_t: refined embeddings with the same shapes as H_s and H_t
    geometry_loss: scalar tensor on the same device/dtype family as H_s
    diagnostics: Python dict of scalar curvature/alignment/bias diagnostics

The adapter does not know about source labels, target labels, classifiers,
domain critics, original graph adjacency matrices, raw node features, AdaIGCN,
or PyGDA.  It only consumes latent embeddings and can therefore be plugged into
any backbone that exposes H_s and H_t.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


BoolLike = Union[bool, torch.Tensor]


@dataclass
class GeometryAdapterConfig:
    """Configuration for :class:`LatentGeometryAdapter`.

    Defaults reproduce the current Hybrid BEC-RPB setting used by the TF1
    adapter.  ``loss_active`` and ``curvature_bias_active`` can be plain booleans
    or scalar tensors; tensor values are read as runtime switches.
    """

    use_latent_topk_bec: bool = True
    latent_topk_k: int = 10
    latent_topk_tau: float = 0.5
    lambda_latent_bec: float = 0.001
    lambda_latent_bec_stat: float = 0.1
    latent_bec_use_mmd: bool = False
    latent_bec_sample_size: int = 128
    latent_bec_N: float = -1.0
    latent_bec_mmd_sigma: float = 1.0

    use_sparse_gt_head: bool = True
    sparse_gt_num_heads: int = 2
    sparse_gt_num_layers: int = 1
    sparse_gt_dropout: float = 0.3
    sparse_gt_use_curv_bias: bool = True
    sparse_gt_curv_bias_beta: float = 0.05
    sparse_gt_curv_bias_mode: str = "hybrid"
    sparse_gt_fourier_m: int = 4
    sparse_gt_fourier_sigma: float = 1.0
    sparse_gt_fourier_seed: int = 1234
    sparse_gt_eta_abs: float = 1.0
    sparse_gt_eta_fourier: float = 0.1
    sparse_gt_bias_norm: str = "scale"
    sparse_gt_attention_k: int = 10
    sparse_gt_use_self_loop: bool = True
    sparse_gt_residual_alpha: float = 0.1
    sparse_gt_use_layernorm: bool = False

    loss_active: BoolLike = True
    curvature_bias_active: BoolLike = True
    eps: float = 1e-8


class LatentGeometryAdapter(nn.Module):
    """Backbone-agnostic latent geometry module.

    Public interface:

        adapter = LatentGeometryAdapter(config)
        O_s, O_t, geometry_loss, diagnostics = adapter(H_s, H_t, training=True)

    Parameters for the sparse transformer head are initialized lazily on the
    first forward pass because the latent dimension is only known from H_s/H_t.
    If you plan to optimize this module directly, create the optimizer after a
    first dry forward pass, or call the adapter once with representative input.
    """

    _BIAS_METRIC_NAMES = (
        "sparse_gt_curv_bias_mean",
        "sparse_gt_curv_bias_std",
        "sparse_gt_abs_bias_mean",
        "sparse_gt_abs_bias_std",
        "sparse_gt_fourier_bias_mean",
        "sparse_gt_fourier_bias_std",
        "sparse_gt_hybrid_bias_mean",
        "sparse_gt_hybrid_bias_std",
    )

    _LATENT_METRIC_NAMES = (
        "latent_bec_loss",
        "latent_bec_mmd",
        "latent_bec_stat",
        "latent_kappa_src_mean",
        "latent_kappa_src_std",
        "latent_kappa_src_min",
        "latent_kappa_src_max",
        "latent_kappa_tgt_mean",
        "latent_kappa_tgt_std",
        "latent_kappa_tgt_min",
        "latent_kappa_tgt_max",
    )

    def __init__(self, config: Optional[Union[GeometryAdapterConfig, Dict]] = None):
        super().__init__()
        if config is None:
            config = GeometryAdapterConfig()
        if isinstance(config, dict):
            config = GeometryAdapterConfig(**config)
        if not isinstance(config, GeometryAdapterConfig):
            raise TypeError("config must be a GeometryAdapterConfig, dict, or None.")

        self.config = config
        self._validate_config()

        omega = self._make_fourier_omega()
        self.register_buffer("fourier_omega", omega, persistent=False)

        self.q_layers = nn.ModuleList()
        self.k_layers = nn.ModuleList()
        self.v_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self._hidden_dim: Optional[int] = None

    def forward(
        self,
        H_s: torch.Tensor,
        H_t: torch.Tensor,
        training: bool = True,
        compute_geometry_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Refine source/target embeddings and optionally compute geometry loss.

        ``compute_geometry_loss=False`` is intended for evaluation/inference:
        BECAT refinement is still applied, but the source-target geometry
        alignment loss is skipped because it is not needed for prediction.
        """
        self._validate_inputs(H_s, H_t)

        H_s_safe = self._sanitize_tensor(H_s)
        H_t_safe = self._sanitize_tensor(H_t)
        self._ensure_transformer_parameters(
            hidden_dim=H_s_safe.shape[1],
            device=H_s_safe.device,
            dtype=H_s_safe.dtype,
        )

        if compute_geometry_loss:
            geometry_loss, diagnostics = self._curvature_stat_loss(
                H_s_safe,
                H_t_safe,
            )
        else:
            geometry_loss = H_s_safe.new_zeros(())
            diagnostics = {}

        O_s, sparse_kappa_s, bias_s = self._sparse_transformer_refine(
            H_s_safe,
            training=training,
        )
        O_t, sparse_kappa_t, bias_t = self._sparse_transformer_refine(
            H_t_safe,
            training=training,
        )

        src_mean, src_std = self._safe_mean_std(sparse_kappa_s)
        tgt_mean, tgt_std = self._safe_mean_std(sparse_kappa_t)
        diagnostics.update(
            {
                "sparse_gt_kappa_src_mean": src_mean,
                "sparse_gt_kappa_src_std": src_std,
                "sparse_gt_kappa_src_min": self._safe_min(sparse_kappa_s),
                "sparse_gt_kappa_src_max": self._safe_max(sparse_kappa_s),
                "sparse_gt_kappa_tgt_mean": tgt_mean,
                "sparse_gt_kappa_tgt_std": tgt_std,
                "sparse_gt_kappa_tgt_min": self._safe_min(sparse_kappa_t),
                "sparse_gt_kappa_tgt_max": self._safe_max(sparse_kappa_t),
            }
        )

        for name in self._BIAS_METRIC_NAMES:
            diagnostics[name] = self._sanitize_tensor(0.5 * (bias_s[name] + bias_t[name]))

        diagnostics["geometry_loss"] = self._sanitize_tensor(geometry_loss)
        diagnostics = self._detach_diagnostics(diagnostics)
        return self._sanitize_tensor(O_s), self._sanitize_tensor(O_t), geometry_loss, diagnostics

    def forward_chunked(
        self,
        H_s: torch.Tensor,
        H_t: torch.Tensor,
        max_nodes_per_call: int = 256,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Refine each domain with query-chunked, full-reference TopK.

        For inputs that fit in one call this delegates directly to ``forward``.
        For larger inputs, chunks limit only the rows of dense similarity and
        curvature intermediates.  Every query still uses its complete domain as
        the reference set, and the sparse Transformer operates on the resulting
        global graph.
        """
        self._validate_inputs(H_s, H_t)
        if int(max_nodes_per_call) <= 0:
            raise ValueError("max_nodes_per_call must be positive.")

        self._ensure_transformer_parameters(
            hidden_dim=H_s.shape[1],
            device=H_s.device,
            dtype=H_s.dtype,
        )
        if max(H_s.size(0), H_t.size(0)) <= int(max_nodes_per_call):
            return self(H_s, H_t, training=training)

        H_s_safe = self._sanitize_tensor(H_s)
        H_t_safe = self._sanitize_tensor(H_t)
        O_s, _, _ = self._sparse_transformer_refine(
            H_s_safe,
            training=training,
            query_chunk_size=int(max_nodes_per_call),
        )
        O_t, _, _ = self._sparse_transformer_refine(
            H_t_safe,
            training=training,
            query_chunk_size=int(max_nodes_per_call),
        )
        geometry_loss, diagnostics = self._curvature_stat_loss(H_s_safe, H_t_safe)
        return O_s, O_t, geometry_loss, diagnostics

    # ------------------------------------------------------------------
    # Configuration, validation, and numerical safety helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if int(self.config.sparse_gt_num_heads) <= 0:
            raise ValueError("sparse_gt_num_heads must be positive.")
        if int(self.config.sparse_gt_num_layers) < 1:
            raise ValueError("sparse_gt_num_layers must be at least 1.")
        if float(self.config.sparse_gt_fourier_sigma) < 0.0:
            raise ValueError("sparse_gt_fourier_sigma must be non-negative.")
        if int(self.config.sparse_gt_fourier_m) <= 0:
            raise ValueError("sparse_gt_fourier_m must be positive.")
        if str(self.config.sparse_gt_curv_bias_mode).lower() not in ("abs", "hybrid"):
            raise ValueError("sparse_gt_curv_bias_mode must be 'abs' or 'hybrid'.")
        if str(self.config.sparse_gt_bias_norm).lower() not in ("none", "scale", "zscore"):
            raise ValueError("sparse_gt_bias_norm must be 'none', 'scale', or 'zscore'.")

    def _validate_inputs(self, H_s: torch.Tensor, H_t: torch.Tensor) -> None:
        if not isinstance(H_s, torch.Tensor) or not isinstance(H_t, torch.Tensor):
            raise TypeError("H_s and H_t must both be torch.Tensor instances.")
        if H_s.dim() != 2 or H_t.dim() != 2:
            raise ValueError("H_s and H_t must both have shape [num_nodes, hidden_dim].")
        if H_s.shape[1] != H_t.shape[1]:
            raise ValueError(
                f"H_s and H_t must have the same hidden_dim; got {H_s.shape[1]} and {H_t.shape[1]}."
            )
        if H_s.device != H_t.device:
            raise ValueError(
                f"H_s and H_t must be on the same device; got {H_s.device} and {H_t.device}."
            )
        if H_s.dtype != H_t.dtype:
            raise ValueError(
                f"H_s and H_t must have the same dtype; got {H_s.dtype} and {H_t.dtype}."
            )
        if not torch.is_floating_point(H_s):
            raise TypeError("H_s and H_t must be floating point tensors.")

    def _make_fourier_omega(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.sparse_gt_fourier_seed))
        omega = torch.randn(
            int(self.config.sparse_gt_fourier_m),
            generator=generator,
            dtype=torch.float32,
        )
        return omega * float(self.config.sparse_gt_fourier_sigma)

    def _ensure_transformer_parameters(
        self,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not self.config.use_sparse_gt_head:
            return

        hidden_dim = int(hidden_dim)
        num_heads = int(self.config.sparse_gt_num_heads)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"sparse_gt_num_heads ({num_heads}) must divide hidden_dim ({hidden_dim})."
            )

        if self._hidden_dim is not None:
            if self._hidden_dim != hidden_dim:
                raise ValueError(
                    f"LatentGeometryAdapter was initialized for hidden_dim={self._hidden_dim}, "
                    f"but received hidden_dim={hidden_dim}."
                )
            self.to(device=device, dtype=dtype)
            return

        self._hidden_dim = hidden_dim
        for _ in range(int(self.config.sparse_gt_num_layers)):
            self.q_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.k_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.v_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.out_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.layer_norms.append(
                nn.LayerNorm(hidden_dim, eps=float(self.config.eps), elementwise_affine=True)
            )

        for module in list(self.q_layers) + list(self.k_layers) + list(self.v_layers) + list(self.out_layers):
            self._reset_glorot_linear(module)

        self.to(device=device, dtype=dtype)

    def initialize_parameters(
        self,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Publicly initialize lazy transformer parameters before optimization."""
        self._ensure_transformer_parameters(
            hidden_dim=hidden_dim,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _reset_glorot_linear(module: nn.Linear) -> None:
        fan_in, fan_out = module.weight.shape[1], module.weight.shape[0]
        bound = math.sqrt(6.0 / float(fan_in + fan_out))
        with torch.no_grad():
            module.weight.uniform_(-bound, bound)

    def _active_bool(self, value: BoolLike) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("active switches must be bools or scalar tensors.")
            return bool(value.detach().to(device="cpu").item())
        return bool(value)

    def _eps_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.new_tensor(float(self.config.eps))

    def _zero_scalar_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.new_tensor(0.0)

    def _sanitize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Replace NaN/Inf values with zero while preserving finite values."""
        if not torch.is_floating_point(tensor):
            return tensor
        return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))

    def _safe_std(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return self._zero_scalar_like(values)
        values = self._sanitize_tensor(values)
        mean = values.mean()
        std = torch.sqrt(torch.mean((values - mean).pow(2)) + self._eps_like(values))
        return self._sanitize_tensor(std)

    def _safe_mean_std(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if values.numel() == 0:
            zero = self._zero_scalar_like(values)
            return zero, zero
        values = self._sanitize_tensor(values)
        mean = self._sanitize_tensor(values.mean())
        std = self._safe_std(values)
        return mean, std

    def _safe_min(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return self._zero_scalar_like(values)
        return self._sanitize_tensor(values.min())

    def _safe_max(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return self._zero_scalar_like(values)
        return self._sanitize_tensor(values.max())

    def _safe_zscore(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values
        values = self._sanitize_tensor(values)
        mean, std = self._safe_mean_std(values)
        z = (values - mean) / (std + self._eps_like(values))
        return self._sanitize_tensor(z)

    def _detach_diagnostics(self, diagnostics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        detached = {}
        for name, value in diagnostics.items():
            if isinstance(value, torch.Tensor):
                detached[name] = self._sanitize_tensor(value).detach()
            else:
                detached[name] = value
        return detached

    # ------------------------------------------------------------------
    # Latent TopK graph
    # ------------------------------------------------------------------

    def _build_latent_topk_edges(
        self,
        hiddens: torch.Tensor,
        topk: Optional[int] = None,
        add_self_loops: bool = False,
        query_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a symmetric weighted TopK graph from cosine similarity.

        When ``query_chunk_size`` is set, only the query rows are chunked.  The
        normalized ``hiddens`` tensor remains the reference for every chunk, so
        returned source/destination indices always use global node numbering.
        """
        hiddens = self._sanitize_tensor(hiddens)
        n_nodes = int(hiddens.shape[0])
        device = hiddens.device
        dtype = hiddens.dtype

        empty_src = torch.empty(0, dtype=torch.long, device=device)
        empty_dst = torch.empty(0, dtype=torch.long, device=device)
        empty_weight = torch.empty(0, dtype=dtype, device=device)

        k_requested = int(self.config.latent_topk_k if topk is None else topk)
        k = min(max(k_requested, 0), max(n_nodes - 1, 0))

        if n_nodes <= 1 or k <= 0:
            edge_src, edge_dst, edge_weight = empty_src, empty_dst, empty_weight
        else:
            eps = self._eps_like(hiddens)
            norms = torch.sqrt(torch.sum(hiddens.pow(2), dim=1, keepdim=True) + eps)
            normalized = hiddens / (norms + eps)
            normalized_t = normalized.transpose(0, 1)

            chunk_size = n_nodes if query_chunk_size is None else int(query_chunk_size)
            if chunk_size <= 0:
                raise ValueError("query_chunk_size must be positive when provided.")

            directed_rows = []
            directed_cols = []
            for start in range(0, n_nodes, chunk_size):
                stop = min(start + chunk_size, n_nodes)
                cosine_chunk = self._sanitize_tensor(normalized[start:stop] @ normalized_t)
                local_rows = torch.arange(stop - start, device=device, dtype=torch.long)
                global_rows = local_rows + start
                cosine_chunk = cosine_chunk.clone()
                cosine_chunk[local_rows, global_rows] = hiddens.new_tensor(-1.0e9)
                _, indices = torch.topk(
                    cosine_chunk,
                    k=k,
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                rows = global_rows.view(-1, 1).expand(stop - start, k)
                directed_rows.append(rows.reshape(-1))
                directed_cols.append(indices.reshape(-1))

            rows = torch.cat(directed_rows, dim=0)
            cols = torch.cat(directed_cols, dim=0)
            symmetric_keys = torch.cat(
                [rows * n_nodes + cols, cols * n_nodes + rows],
                dim=0,
            )
            symmetric_keys = torch.unique(symmetric_keys, sorted=True)
            edge_src = torch.div(symmetric_keys, n_nodes, rounding_mode="floor")
            edge_dst = torch.remainder(symmetric_keys, n_nodes)

            tau = max(float(self.config.latent_topk_tau), float(self.config.eps))
            edge_cosine = torch.sum(
                normalized.index_select(0, edge_src)
                * normalized.index_select(0, edge_dst),
                dim=1,
            )
            edge_cosine = self._sanitize_tensor(edge_cosine)
            edge_weight = torch.sigmoid(edge_cosine / hiddens.new_tensor(tau))
            edge_weight = self._sanitize_tensor(edge_weight)

        if add_self_loops and n_nodes > 0:
            self_ids = torch.arange(n_nodes, device=device, dtype=torch.long)
            self_weight = torch.ones(n_nodes, dtype=dtype, device=device)
            edge_src = torch.cat([edge_src, self_ids], dim=0)
            edge_dst = torch.cat([edge_dst, self_ids], dim=0)
            edge_weight = torch.cat([edge_weight, self_weight], dim=0)

        return edge_src, edge_dst, edge_weight

    # ------------------------------------------------------------------
    # Soft Bakry-Émery curvature on the latent TopK graph
    # ------------------------------------------------------------------

    def _compute_soft_bec_curvature(
        self,
        hiddens: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_weight: torch.Tensor,
        row_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the current weighted latent BEC signature node-wise."""
        hiddens = self._sanitize_tensor(hiddens)
        n_nodes = int(hiddens.shape[0])
        if n_nodes == 0:
            return hiddens.new_zeros((0,))
        if n_nodes <= 1 or edge_src.numel() == 0:
            return hiddens.new_zeros((n_nodes,))

        edge_weight = self._sanitize_tensor(edge_weight.to(dtype=hiddens.dtype, device=hiddens.device))
        degree = hiddens.new_zeros((n_nodes,)).scatter_add(0, edge_src, edge_weight)
        degree = self._sanitize_tensor(degree)

        chunk_size = n_nodes if row_chunk_size is None else int(row_chunk_size)
        if chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive when provided.")

        if chunk_size >= n_nodes:
            sample_rows = hiddens.new_zeros((n_nodes, n_nodes))
            sample_rows = sample_rows.index_put(
                (edge_src, edge_dst), edge_weight, accumulate=False
            )
            sample_rows = self._sanitize_tensor(sample_rows)

            neighbor_degree = self._sanitize_tensor(sample_rows @ degree)
            path2 = self._sanitize_tensor(
                torch.clamp(sample_rows @ sample_rows.transpose(0, 1), min=0.0)
            )

            triangle = 0.5 * torch.sum(path2 * sample_rows, dim=1)
            off_diag = (~torch.eye(n_nodes, dtype=torch.bool, device=hiddens.device)).to(
                dtype=hiddens.dtype
            )
            sphere2 = torch.sum(
                (1.0 - sample_rows) * (1.0 - torch.exp(-path2)) * off_diag,
                dim=1,
            )
        else:
            adjacency = torch.sparse_coo_tensor(
                torch.stack([edge_src, edge_dst], dim=0),
                edge_weight,
                size=(n_nodes, n_nodes),
                dtype=hiddens.dtype,
                device=hiddens.device,
            ).coalesce()
            neighbor_degree_parts = []
            triangle_parts = []
            sphere2_parts = []
            for start in range(0, n_nodes, chunk_size):
                stop = min(start + chunk_size, n_nodes)
                edge_mask = (edge_src >= start) & (edge_src < stop)
                local_src = edge_src[edge_mask] - start
                local_dst = edge_dst[edge_mask]
                local_weight = edge_weight[edge_mask]

                sample_rows = hiddens.new_zeros((stop - start, n_nodes))
                sample_rows = sample_rows.index_put(
                    (local_src, local_dst), local_weight, accumulate=False
                )
                sample_rows = self._sanitize_tensor(sample_rows)

                neighbor_degree_parts.append(
                    self._sanitize_tensor(sample_rows @ degree)
                )
                path2 = torch.sparse.mm(adjacency, sample_rows.transpose(0, 1))
                path2 = self._sanitize_tensor(
                    torch.clamp(path2.transpose(0, 1), min=0.0)
                )
                triangle_parts.append(0.5 * torch.sum(path2 * sample_rows, dim=1))

                sphere_terms = (1.0 - sample_rows) * (1.0 - torch.exp(-path2))
                local_rows = torch.arange(stop - start, device=hiddens.device)
                global_rows = local_rows + start
                sphere2_parts.append(
                    torch.sum(sphere_terms, dim=1)
                    - sphere_terms[local_rows, global_rows]
                )

            neighbor_degree = self._sanitize_tensor(
                torch.cat(neighbor_degree_parts, dim=0)
            )
            triangle = self._sanitize_tensor(torch.cat(triangle_parts, dim=0))
            sphere2 = self._sanitize_tensor(torch.cat(sphere2_parts, dim=0))

        bec_dimension = float(self.config.latent_bec_N)
        if bec_dimension < 0.0 or not math.isfinite(bec_dimension):
            degree_coef = 1.0
        else:
            degree_coef = 1.0 - 2.0 / (bec_dimension + float(self.config.eps))

        scalar_bec = (
            hiddens.new_tensor(degree_coef) * degree
            - 0.5 * degree.pow(2)
            + 1.5 * neighbor_degree
            + triangle
            - 2.0 * sphere2
        )
        kappa = scalar_bec / (degree + self._eps_like(degree))
        return self._sanitize_tensor(kappa)

    # ------------------------------------------------------------------
    # Curvature statistic alignment and optional MMD
    # ------------------------------------------------------------------

    def _sample_kappa(self, hiddens: torch.Tensor) -> torch.Tensor:
        n_nodes = int(hiddens.shape[0])
        if n_nodes == 0:
            return hiddens.new_empty((0,))
        sample_size = min(max(int(self.config.latent_bec_sample_size), 1), n_nodes)
        sample_ids = torch.randperm(n_nodes, device=hiddens.device)[:sample_size]
        sampled = hiddens.index_select(0, sample_ids)
        edge_src, edge_dst, edge_weight = self._build_latent_topk_edges(
            sampled,
            topk=int(self.config.latent_topk_k),
            add_self_loops=False,
        )
        return self._compute_soft_bec_curvature(sampled, edge_src, edge_dst, edge_weight)

    def _mmd_loss(self, kappa_s: torch.Tensor, kappa_t: torch.Tensor) -> torch.Tensor:
        if kappa_s.numel() == 0:
            return self._zero_scalar_like(kappa_t if kappa_t.numel() else kappa_s)
        if kappa_t.numel() == 0:
            return self._zero_scalar_like(kappa_s)

        sigma = max(float(self.config.latent_bec_mmd_sigma), float(self.config.eps))
        denom = kappa_s.new_tensor(2.0 * sigma * sigma + float(self.config.eps))

        diff_ss = kappa_s.view(-1, 1) - kappa_s.view(1, -1)
        diff_tt = kappa_t.view(-1, 1) - kappa_t.view(1, -1)
        diff_st = kappa_s.view(-1, 1) - kappa_t.view(1, -1)
        mmd = (
            torch.exp(-diff_ss.pow(2) / denom).mean()
            + torch.exp(-diff_tt.pow(2) / denom).mean()
            - 2.0 * torch.exp(-diff_st.pow(2) / denom).mean()
        )
        return self._sanitize_tensor(mmd)

    def _zero_latent_diagnostics(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = self._zero_scalar_like(reference)
        diagnostics = {"latent_bec_weighted": zero}
        diagnostics.update({name: zero for name in self._LATENT_METRIC_NAMES})
        diagnostics.update(
            {
                "geometry_loss": zero,
                "curvature_src_mean": zero,
                "curvature_src_std": zero,
                "curvature_tgt_mean": zero,
                "curvature_tgt_std": zero,
            }
        )
        return diagnostics

    def _curvature_stat_loss(
        self,
        H_s: torch.Tensor,
        H_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Align source/target sampled BEC distributions by mean/std and MMD."""
        if not bool(self.config.use_latent_topk_bec) or not self._active_bool(self.config.loss_active):
            diagnostics = self._zero_latent_diagnostics(H_s)
            return diagnostics["geometry_loss"], diagnostics

        kappa_s = self._sample_kappa(H_s)
        kappa_t = self._sample_kappa(H_t)

        src_mean, src_std = self._safe_mean_std(kappa_s)
        tgt_mean, tgt_std = self._safe_mean_std(kappa_t)
        stat = self._sanitize_tensor((src_mean - tgt_mean).pow(2) + (src_std - tgt_std).pow(2))

        mmd = self._mmd_loss(kappa_s, kappa_t)
        stat_loss = H_s.new_tensor(float(self.config.lambda_latent_bec_stat)) * stat
        latent_loss = mmd + stat_loss if bool(self.config.latent_bec_use_mmd) else stat_loss
        latent_loss = self._sanitize_tensor(latent_loss)
        weighted = H_s.new_tensor(float(self.config.lambda_latent_bec)) * latent_loss
        weighted = self._sanitize_tensor(weighted)

        diagnostics = {
            "latent_bec_weighted": weighted,
            "latent_bec_loss": latent_loss,
            "latent_bec_mmd": mmd,
            "latent_bec_stat": stat,
            "latent_kappa_src_mean": src_mean,
            "latent_kappa_src_std": src_std,
            "latent_kappa_src_min": self._safe_min(kappa_s),
            "latent_kappa_src_max": self._safe_max(kappa_s),
            "latent_kappa_tgt_mean": tgt_mean,
            "latent_kappa_tgt_std": tgt_std,
            "latent_kappa_tgt_min": self._safe_min(kappa_t),
            "latent_kappa_tgt_max": self._safe_max(kappa_t),
            "geometry_loss": weighted,
            "curvature_src_mean": src_mean,
            "curvature_src_std": src_std,
            "curvature_tgt_mean": tgt_mean,
            "curvature_tgt_std": tgt_std,
        }
        return weighted, diagnostics

    # ------------------------------------------------------------------
    # Hybrid curvature relative positional bias
    # ------------------------------------------------------------------

    def _normalize_edge_bias(self, edge_bias: torch.Tensor) -> torch.Tensor:
        edge_bias = self._sanitize_tensor(edge_bias)
        if edge_bias.numel() == 0:
            return edge_bias

        normalization = str(self.config.sparse_gt_bias_norm).lower()
        if normalization == "none":
            return edge_bias

        mean, std = self._safe_mean_std(edge_bias)
        if normalization == "scale":
            return self._sanitize_tensor(edge_bias / (std + self._eps_like(edge_bias)))
        if normalization == "zscore":
            return self._sanitize_tensor((edge_bias - mean) / (std + self._eps_like(edge_bias)))
        raise ValueError("sparse_gt_bias_norm must be 'none', 'scale', or 'zscore'.")

    def _fourier_encoding(self, kappa_norm: torch.Tensor) -> torch.Tensor:
        m = int(self.config.sparse_gt_fourier_m)
        if kappa_norm.numel() == 0:
            return kappa_norm.new_zeros((0, 2 * m))
        omega = self.fourier_omega.to(device=kappa_norm.device, dtype=kappa_norm.dtype)
        phase = kappa_norm.view(-1, 1) * omega.view(1, -1)
        phi = torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)
        phi = phi * math.sqrt(1.0 / float(m))
        return self._sanitize_tensor(phi)

    def _compute_hybrid_bias(
        self,
        kappa: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        edge_count = int(edge_src.numel())
        if edge_count == 0:
            zero = kappa.new_zeros((0,))
            return {
                "edge_bias": zero,
                "abs_bias": zero,
                "fourier_bias": zero,
                "hybrid_bias": zero,
            }

        mode = str(self.config.sparse_gt_curv_bias_mode).lower()
        kappa_norm = self._safe_zscore(kappa)

        abs_raw = -torch.abs(kappa_norm.index_select(0, edge_src) - kappa_norm.index_select(0, edge_dst))
        abs_bias = self._normalize_edge_bias(abs_raw)

        if mode == "abs":
            zero = torch.zeros_like(abs_bias)
            return {
                "edge_bias": self._sanitize_tensor(abs_bias),
                "abs_bias": self._sanitize_tensor(abs_bias),
                "fourier_bias": zero,
                "hybrid_bias": self._sanitize_tensor(abs_bias),
            }

        phi = self._fourier_encoding(kappa_norm)
        phi_diff = phi.index_select(0, edge_src) - phi.index_select(0, edge_dst)
        fourier_raw = -torch.sum(phi_diff.pow(2), dim=1)
        fourier_bias = self._normalize_edge_bias(fourier_raw)

        hybrid_bias = (
            kappa.new_tensor(float(self.config.sparse_gt_eta_abs)) * abs_bias
            + kappa.new_tensor(float(self.config.sparse_gt_eta_fourier)) * fourier_bias
        )
        return {
            "edge_bias": self._sanitize_tensor(hybrid_bias),
            "abs_bias": self._sanitize_tensor(abs_bias),
            "fourier_bias": self._sanitize_tensor(fourier_bias),
            "hybrid_bias": self._sanitize_tensor(hybrid_bias),
        }

    # ------------------------------------------------------------------
    # Residual sparse graph transformer head
    # ------------------------------------------------------------------

    def _zero_bias_diagnostics(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = self._zero_scalar_like(reference)
        return {name: zero for name in self._BIAS_METRIC_NAMES}

    def _bias_diagnostics(self, bias_parts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bias_mean, bias_std = self._safe_mean_std(bias_parts["edge_bias"])
        abs_mean, abs_std = self._safe_mean_std(bias_parts["abs_bias"])
        fourier_mean, fourier_std = self._safe_mean_std(bias_parts["fourier_bias"])
        hybrid_mean, hybrid_std = self._safe_mean_std(bias_parts["hybrid_bias"])
        return {
            "sparse_gt_curv_bias_mean": bias_mean,
            "sparse_gt_curv_bias_std": bias_std,
            "sparse_gt_abs_bias_mean": abs_mean,
            "sparse_gt_abs_bias_std": abs_std,
            "sparse_gt_fourier_bias_mean": fourier_mean,
            "sparse_gt_fourier_bias_std": fourier_std,
            "sparse_gt_hybrid_bias_mean": hybrid_mean,
            "sparse_gt_hybrid_bias_std": hybrid_std,
        }

    def _segment_softmax(
        self,
        scores: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        """Softmax over edges grouped by source node.

        PyTorch-1.11-compatible implementation.

        ``scores`` has shape [E, H], where E is the number of sparse
        attention edges and H is the number of attention heads.
        Softmax is computed independently for every source-node segment
        and every attention head.

        This implementation intentionally avoids ``scatter_reduce``,
        whose beta API differs across older PyTorch releases.
        """
        scores = self._sanitize_tensor(scores)
        if scores.numel() == 0:
            return scores

        num_heads = int(scores.shape[1])

        # Sort edges by segment so that each source-node segment occupies
        # one contiguous slice.  We then compute the maximum independently
        # for every attention head.
        order = torch.argsort(segment_ids)
        sorted_ids = segment_ids.index_select(0, order)
        sorted_scores = scores.index_select(0, order)

        unique_ids, counts = torch.unique_consecutive(
            sorted_ids,
            return_counts=True,
        )

        split_sizes = counts.detach().cpu().tolist()
        score_groups = torch.split(sorted_scores, split_sizes, dim=0)

        max_values = torch.stack(
            [group.max(dim=0)[0] for group in score_groups],
            dim=0,
        )

        segment_max = scores.new_full(
            (num_segments, num_heads),
            -torch.inf,
        )
        segment_max = segment_max.index_copy(
            0,
            unique_ids,
            max_values,
        )

        gathered_max = segment_max.index_select(0, segment_ids)
        shifted = self._sanitize_tensor(scores - gathered_max)
        exp_scores = self._sanitize_tensor(torch.exp(shifted))

        index = segment_ids.view(-1, 1).expand(-1, num_heads)

        segment_sum = scores.new_zeros(
            (num_segments, num_heads)
        ).scatter_add(
            0,
            index,
            exp_scores,
        )

        denominator = (
            segment_sum.index_select(0, segment_ids)
            + self._eps_like(scores)
        )

        attn = exp_scores / denominator
        return self._sanitize_tensor(attn)

    def _sparse_transformer_layer(
        self,
        hiddens: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        kappa: torch.Tensor,
        layer_idx: int,
        training: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n_nodes = int(hiddens.shape[0])
        num_heads = int(self.config.sparse_gt_num_heads)
        head_dim = int(hiddens.shape[1]) // num_heads
        scale = math.sqrt(float(head_dim))

        q = self.q_layers[layer_idx](hiddens).reshape(n_nodes, num_heads, head_dim)
        k = self.k_layers[layer_idx](hiddens).reshape(n_nodes, num_heads, head_dim)
        v = self.v_layers[layer_idx](hiddens).reshape(n_nodes, num_heads, head_dim)

        q_src = q.index_select(0, edge_src)
        k_dst = k.index_select(0, edge_dst)
        scores = torch.sum(q_src * k_dst, dim=2) / hiddens.new_tensor(scale)

        zero_edge_bias = hiddens.new_zeros((edge_src.numel(),))
        bias_parts = {
            "edge_bias": zero_edge_bias,
            "abs_bias": zero_edge_bias,
            "fourier_bias": zero_edge_bias,
            "hybrid_bias": zero_edge_bias,
        }

        if bool(self.config.sparse_gt_use_curv_bias):
            bias_parts = self._compute_hybrid_bias(kappa, edge_src, edge_dst)
            beta = (
                float(self.config.sparse_gt_curv_bias_beta)
                if self._active_bool(self.config.curvature_bias_active)
                else 0.0
            )
            if beta != 0.0:
                scores = scores + hiddens.new_tensor(beta) * bias_parts["edge_bias"].view(-1, 1)

        attn = self._segment_softmax(scores, edge_src, n_nodes)
        v_dst = v.index_select(0, edge_dst)
        messages = v_dst * attn.unsqueeze(2)

        scatter_index = edge_src.view(-1, 1, 1).expand(-1, num_heads, head_dim)
        head_outputs = hiddens.new_zeros(
            (n_nodes, num_heads, head_dim)
        ).scatter_add(
            0,
            scatter_index,
            messages,
        )

        delta = head_outputs.reshape(n_nodes, int(hiddens.shape[1]))
        delta = self.out_layers[layer_idx](delta)
        dropout_rate = min(max(float(self.config.sparse_gt_dropout), 0.0), 0.95)
        delta = F.dropout(delta, p=dropout_rate, training=bool(training))

        residual_alpha = float(self.config.sparse_gt_residual_alpha)
        if residual_alpha == 0.0:
            output = hiddens
        else:
            output = hiddens + hiddens.new_tensor(residual_alpha) * delta

        if bool(self.config.sparse_gt_use_layernorm):
            output = self.layer_norms[layer_idx](output)

        return self._sanitize_tensor(output), self._bias_diagnostics(bias_parts)

    def _sparse_transformer_refine(
        self,
        hiddens: torch.Tensor,
        training: bool = True,
        query_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Optionally refine one domain with the residual sparse GT head."""
        hiddens = self._sanitize_tensor(hiddens)
        n_nodes = int(hiddens.shape[0])
        zero_kappa = hiddens.new_zeros((n_nodes,))
        zero_bias_metrics = self._zero_bias_diagnostics(hiddens)

        if not bool(self.config.use_sparse_gt_head):
            return hiddens, zero_kappa, zero_bias_metrics
        if n_nodes == 0:
            return hiddens, zero_kappa, zero_bias_metrics

        attention_k = int(self.config.sparse_gt_attention_k)
        if attention_k <= 0:
            attention_k = int(self.config.latent_topk_k)

        edge_src, edge_dst, edge_weight = self._build_latent_topk_edges(
            hiddens,
            topk=attention_k,
            add_self_loops=False,
            query_chunk_size=query_chunk_size,
        )

        if bool(self.config.sparse_gt_use_curv_bias):
            kappa = self._compute_soft_bec_curvature(
                hiddens,
                edge_src,
                edge_dst,
                edge_weight,
                row_chunk_size=query_chunk_size,
            )
        else:
            kappa = zero_kappa

        if bool(self.config.sparse_gt_use_self_loop):
            edge_src, edge_dst, edge_weight = self._build_latent_topk_edges(
                hiddens,
                topk=attention_k,
                add_self_loops=True,
                query_chunk_size=query_chunk_size,
            )

        if edge_src.numel() == 0:
            return hiddens, kappa, zero_bias_metrics

        output = hiddens
        bias_metrics = zero_bias_metrics
        for layer_idx in range(int(self.config.sparse_gt_num_layers)):
            output, bias_metrics = self._sparse_transformer_layer(
                output,
                edge_src,
                edge_dst,
                kappa,
                layer_idx=layer_idx,
                training=training,
            )

        return self._sanitize_tensor(output), self._sanitize_tensor(kappa), bias_metrics
