from __future__ import division
from __future__ import print_function

import argparse
import random

import numpy as np
import torch

from .GRADE_train import GRADE_main
from .utils import (
    create_citation_source_split,
    normalize_citation_domain,
    read_citation_domain,
)


def _add_boolean_pair(parser, enabled_flag, disabled_flag, dest, default):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(enabled_flag, dest=dest, action="store_true")
    group.add_argument(disabled_flag, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce GRADE citation-network transfer experiments."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--source",
        required=True,
        choices=["ACMv9", "DBLPv7", "Citationv1", "A", "D", "C"],
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["ACMv9", "DBLPv7", "Citationv1", "A", "D", "C"],
    )

    # Paper-result GRADE settings retained from the supplied citation code.
    parser.add_argument("--n_hidden", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42, help="Training RNG seed.")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Source 80/10/10 split seed; independent of --seed.",
    )
    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument(
        "--use_becat",
        "--use_geometry_adapter",
        dest="use_becat",
        action="store_true",
        help="Enable BECAT (the old --use_geometry_adapter name is an alias).",
    )
    _add_boolean_pair(
        parser,
        "--becat_use_latent_topk_bec",
        "--no_becat_use_latent_topk_bec",
        "use_latent_topk_bec",
        True,
    )
    parser.add_argument("--becat_latent_topk_k", dest="latent_topk_k", type=int, default=8)
    parser.add_argument("--becat_latent_topk_tau", dest="latent_topk_tau", type=float, default=0.5)
    parser.add_argument("--becat_lambda_latent_bec", dest="lambda_latent_bec", type=float, default=0.001)
    parser.add_argument("--becat_lambda_latent_bec_stat", dest="lambda_latent_bec_stat", type=float, default=0.1)
    parser.add_argument("--becat_latent_bec_use_mmd", dest="latent_bec_use_mmd", action="store_true")
    parser.add_argument("--becat_latent_bec_sample_size", dest="latent_bec_sample_size", type=int, default=128)
    parser.add_argument("--becat_latent_bec_N", dest="latent_bec_N", type=float, default=-1.0)
    parser.add_argument("--becat_latent_bec_mmd_sigma", dest="latent_bec_mmd_sigma", type=float, default=1.0)

    _add_boolean_pair(
        parser,
        "--becat_use_sparse_gt_head",
        "--no_becat_use_sparse_gt_head",
        "use_sparse_gt_head",
        True,
    )
    parser.add_argument("--becat_sparse_gt_num_heads", dest="sparse_gt_num_heads", type=int, default=2)
    parser.add_argument("--becat_sparse_gt_num_layers", dest="sparse_gt_num_layers", type=int, default=1)
    parser.add_argument("--becat_sparse_gt_dropout", dest="sparse_gt_dropout", type=float, default=0.3)
    _add_boolean_pair(
        parser,
        "--becat_sparse_gt_use_curv_bias",
        "--no_becat_sparse_gt_use_curv_bias",
        "sparse_gt_use_curv_bias",
        True,
    )
    parser.add_argument("--becat_sparse_gt_curv_bias_beta", dest="sparse_gt_curv_bias_beta", type=float, default=0.05)
    parser.add_argument("--becat_sparse_gt_curv_bias_mode", dest="sparse_gt_curv_bias_mode", type=str, default="hybrid")
    parser.add_argument("--becat_sparse_gt_fourier_m", dest="sparse_gt_fourier_m", type=int, default=4)
    parser.add_argument("--becat_sparse_gt_fourier_sigma", dest="sparse_gt_fourier_sigma", type=float, default=1.0)
    parser.add_argument("--becat_sparse_gt_fourier_seed", dest="sparse_gt_fourier_seed", type=int, default=1234)
    parser.add_argument("--becat_sparse_gt_eta_abs", dest="sparse_gt_eta_abs", type=float, default=1.0)
    parser.add_argument("--becat_sparse_gt_eta_fourier", dest="sparse_gt_eta_fourier", type=float, default=0.1)
    parser.add_argument("--becat_sparse_gt_bias_norm", dest="sparse_gt_bias_norm", type=str, default="scale")
    parser.add_argument("--becat_sparse_gt_attention_k", dest="sparse_gt_attention_k", type=int, default=8)
    _add_boolean_pair(
        parser,
        "--becat_sparse_gt_use_self_loop",
        "--no_becat_sparse_gt_use_self_loop",
        "sparse_gt_use_self_loop",
        True,
    )
    parser.add_argument("--becat_sparse_gt_residual_alpha", dest="sparse_gt_residual_alpha", type=float, default=0.1)
    parser.add_argument("--becat_sparse_gt_use_layernorm", dest="sparse_gt_use_layernorm", action="store_true")
    return parser


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.patience < 1:
        parser.error("--patience must be at least 1")

    source_domain = normalize_citation_domain(args.source)
    target_domain = normalize_citation_domain(args.target)
    if source_domain == target_domain:
        parser.error("--source and --target must identify different domains")

    args.device = torch.device(
        "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu"
    )
    set_random_seed(args.seed)

    source_data = read_citation_domain(
        args.data_root, source_domain, load_labels=True
    )
    target_data = read_citation_domain(
        args.data_root, target_domain, load_labels=False
    )
    source_masks = create_citation_source_split(
        source_data[1], split_seed=args.split_seed
    )
    source_train_mask, source_val_mask, source_test_mask = source_masks
    print(
        "PROTOCOL,source={source},target={target},seed={seed},split_seed={split_seed},"
        "source_train_nodes={train},source_val_nodes={val},source_test_nodes={test},"
        "use_becat={use_becat}".format(
            source=source_domain,
            target=target_domain,
            seed=args.seed,
            split_seed=args.split_seed,
            train=int(source_train_mask.sum().item()),
            val=int(source_val_mask.sum().item()),
            test=int(source_test_mask.sum().item()),
            use_becat=str(args.use_becat).lower(),
        )
    )

    result = GRADE_main(
        args,
        source_data,
        target_data,
        source_masks=source_masks,
        source_domain=source_domain,
        target_domain=target_domain,
    )
    print(
        "FINAL_RESULT,source={source},target={target},seed={seed},"
        "split_seed={split_seed},best_epoch={best_epoch},"
        "best_source_val_accuracy={best_source_val_accuracy:.6f},"
        "source_val_accuracy={source_val_accuracy:.6f},"
        "source_test_accuracy={source_test_accuracy:.6f},"
        "target_accuracy={target_accuracy:.6f},"
        "target_micro_f1={target_micro_f1:.6f},"
        "target_macro_f1={target_macro_f1:.6f},"
        "use_becat={use_becat},"
        "source_val_micro_f1={source_val_micro_f1:.6f},"
        "source_val_macro_f1={source_val_macro_f1:.6f},"
        "source_test_micro_f1={source_test_micro_f1:.6f},"
        "source_test_macro_f1={source_test_macro_f1:.6f}".format(**result)
    )


if __name__ == "__main__":
    main()
