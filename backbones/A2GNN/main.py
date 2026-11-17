"""Release entrypoint for the six A2GNN-MMD citation experiments."""

import argparse
import time

import torch
import torch.nn.functional as F

from .model import Model
from .utils import (
    CITATION_DOMAINS,
    MMD,
    evaluate_logits,
    load_source_data,
    load_target_labels_for_final_evaluation,
    load_unlabeled_target_data,
    seed_training,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected a boolean value (true/false, yes/no, 1/0)"
    )


def add_bool_argument(parser, *names, dest, default, help_text):
    parser.add_argument(
        *names,
        dest=dest,
        type=str2bool,
        nargs="?",
        const=True,
        default=default,
        help=help_text,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="A2GNN-MMD citation backbone with optional BECAT"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--source", choices=CITATION_DOMAINS, default="DBLPv7")
    parser.add_argument("--target", choices=CITATION_DOMAINS, default="ACMv9")
    parser.add_argument("--seed", type=int, default=200, help="training seed")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="source split seed, independent from --seed",
    )

    # Preserve original A2GNN-MMD defaults.
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight", type=float, default=5.0)
    parser.add_argument("--source_pnum", type=int, default=0)
    parser.add_argument("--target_pnum", type=int, default=30)

    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--encoder_device",
        default=None,
        help="encoder, raw embeddings, and original MMD device",
    )
    parser.add_argument(
        "--geometry_device",
        default=None,
        help="BECAT, classifier, and total-loss device",
    )

    # Release-facing BECAT names retain old geometry aliases for compatibility.
    add_bool_argument(
        parser,
        "--use_becat",
        "--use_geometry_adapter",
        dest="use_becat",
        default=False,
        help_text="enable shared BECAT",
    )
    parser.add_argument(
        "--lambda_becat",
        "--lambda_geometry",
        dest="lambda_becat",
        type=float,
        default=1.0,
    )
    add_bool_argument(
        parser,
        "--use_latent_topk_bec",
        dest="use_latent_topk_bec",
        default=True,
        help_text="enable latent Top-K BEC alignment",
    )
    parser.add_argument("--latent_topk_k", type=int, default=10)
    parser.add_argument("--latent_topk_tau", type=float, default=0.5)
    parser.add_argument("--lambda_latent_bec", type=float, default=0.001)
    parser.add_argument("--lambda_latent_bec_stat", type=float, default=0.1)
    add_bool_argument(
        parser,
        "--latent_bec_use_mmd",
        dest="latent_bec_use_mmd",
        default=False,
        help_text="include the BECAT curvature MMD term",
    )
    parser.add_argument("--latent_bec_sample_size", type=int, default=128)
    parser.add_argument("--latent_bec_N", type=float, default=-1.0)
    parser.add_argument("--latent_bec_mmd_sigma", type=float, default=1.0)
    add_bool_argument(
        parser,
        "--use_sparse_gt_head",
        dest="use_sparse_gt_head",
        default=True,
        help_text="enable the residual sparse graph-transformer head",
    )
    parser.add_argument("--sparse_gt_num_heads", type=int, default=2)
    parser.add_argument("--sparse_gt_num_layers", type=int, default=1)
    parser.add_argument("--sparse_gt_dropout", type=float, default=0.3)
    add_bool_argument(
        parser,
        "--sparse_gt_use_curv_bias",
        dest="sparse_gt_use_curv_bias",
        default=True,
        help_text="enable curvature bias in sparse attention",
    )
    parser.add_argument("--sparse_gt_curv_bias_beta", type=float, default=0.05)
    parser.add_argument(
        "--sparse_gt_curv_bias_mode",
        choices=("abs", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--sparse_gt_fourier_m", type=int, default=4)
    parser.add_argument("--sparse_gt_fourier_sigma", type=float, default=1.0)
    parser.add_argument("--sparse_gt_fourier_seed", type=int, default=1234)
    parser.add_argument("--sparse_gt_eta_abs", type=float, default=1.0)
    parser.add_argument("--sparse_gt_eta_fourier", type=float, default=0.1)
    parser.add_argument(
        "--sparse_gt_bias_norm",
        choices=("none", "scale", "zscore"),
        default="scale",
    )
    parser.add_argument("--sparse_gt_attention_k", type=int, default=10)
    add_bool_argument(
        parser,
        "--sparse_gt_use_self_loop",
        dest="sparse_gt_use_self_loop",
        default=True,
        help_text="include self-loops in sparse attention",
    )
    parser.add_argument("--sparse_gt_residual_alpha", type=float, default=0.1)
    add_bool_argument(
        parser,
        "--sparse_gt_use_layernorm",
        dest="sparse_gt_use_layernorm",
        default=False,
        help_text="apply layer normalization in the sparse head",
    )
    add_bool_argument(
        parser,
        "--loss_active",
        dest="loss_active",
        default=True,
        help_text="enable BECAT geometry loss",
    )
    add_bool_argument(
        parser,
        "--curvature_bias_active",
        dest="curvature_bias_active",
        default=True,
        help_text="activate BECAT curvature bias",
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser


def classifier_edges(args, source_data, target_data):
    if not args.use_becat:
        return source_data.edge_index, target_data.edge_index
    return (
        source_data.edge_index.to(args.geometry_device, non_blocking=True),
        target_data.edge_index.to(args.geometry_device, non_blocking=True),
    )


def source_logits(model, source_data, target_data, source_edge_index):
    """Use the same source/target latent pair without target ground truth."""
    _, _, source_h, _, _ = model.encode_pair(source_data, target_data)
    return model.classify_source(source_h, source_edge_index)


def pair_logits(
    model,
    source_data,
    target_data,
    source_edge_index,
    target_edge_index,
):
    _, _, source_h, target_h, _ = model.encode_pair(source_data, target_data)
    return (
        model.classify_source(source_h, source_edge_index),
        model.classify_target(target_h, target_edge_index),
    )


def clone_state_to_cpu(model):
    """Snapshot every model parameter/buffer without per-epoch checkpoint files."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train(args, source_data, target_data):
    """Train with source-train labels and select solely by source-val accuracy."""
    seed_training(args.seed)
    model = Model(args)
    model.place_modules(args.encoder_device, args.geometry_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    source_edge_index, _ = classifier_edges(args, source_data, target_data)
    source_labels = source_data.y.to(
        args.geometry_device if args.use_becat else args.encoder_device,
        non_blocking=True,
    )
    source_train_mask = source_data.train_mask.to(
        source_labels.device,
        non_blocking=True,
    )

    best_model_state = None
    best_epoch = 0
    best_source_val_accuracy = float("-inf")
    patience_counter = 0
    started_at = time.time()

    for epoch_index in range(args.epochs):
        model.train()
        raw_source_h, raw_target_h, source_h, _, becat_loss = model.encode_pair(
            source_data,
            target_data,
        )

        # The original A2GNN MMD branch remains on raw pre-BECAT H_s/H_t.
        mmd_loss = MMD(raw_source_h, raw_target_h)
        source_output = model.classify_source(source_h, source_edge_index)
        source_supervised_loss = F.nll_loss(
            F.log_softmax(source_output[source_train_mask], dim=1),
            source_labels[source_train_mask],
        )
        if args.use_becat:
            total_loss = (
                source_supervised_loss
                + args.weight * mmd_loss.to(args.geometry_device)
                + args.lambda_becat * becat_loss
            )
        else:
            total_loss = source_supervised_loss + args.weight * mmd_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        # Checkpointing and patience use source validation accuracy only.
        model.eval()
        with torch.no_grad():
            validation_output = source_logits(
                model,
                source_data,
                target_data,
                source_edge_index,
            )
            validation_metrics = evaluate_logits(
                validation_output,
                source_data.y,
                source_data.val_mask,
            )
        source_val_accuracy = validation_metrics["accuracy"]

        if source_val_accuracy > best_source_val_accuracy:
            best_source_val_accuracy = source_val_accuracy
            best_epoch = epoch_index + 1
            best_model_state = clone_state_to_cpu(model)
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            "Epoch: {:04d} source_supervised_loss: {:.6f} "
            "mmd_loss: {:.6f} becat_loss: {:.6f} "
            "source_val_accuracy: {:.6f}".format(
                epoch_index + 1,
                source_supervised_loss.detach().item(),
                mmd_loss.detach().item(),
                becat_loss.detach().item(),
                source_val_accuracy,
            )
        )
        if patience_counter >= args.patience:
            break

    if best_model_state is None:
        raise RuntimeError("training produced no source-validation checkpoint")

    # Restore the complete best state before any final target-label read.
    model.load_state_dict(best_model_state)
    model.place_modules(args.encoder_device, args.geometry_device)
    model.eval()
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_source_val_accuracy": best_source_val_accuracy,
        "elapsed_seconds": time.time() - started_at,
    }


def format_final_result(args, training, source_val, source_test, target):
    fields = (
        ("source", args.source),
        ("target", args.target),
        ("seed", args.seed),
        ("split_seed", args.split_seed),
        ("best_epoch", training["best_epoch"]),
        (
            "best_source_val_accuracy",
            "{:.6f}".format(training["best_source_val_accuracy"]),
        ),
        ("source_val_accuracy", "{:.6f}".format(source_val["accuracy"])),
        ("source_test_accuracy", "{:.6f}".format(source_test["accuracy"])),
        ("target_accuracy", "{:.6f}".format(target["accuracy"])),
        ("target_micro_f1", "{:.6f}".format(target["micro_f1"])),
        ("target_macro_f1", "{:.6f}".format(target["macro_f1"])),
        ("target_weighted_f1", "{:.6f}".format(target["weighted_f1"])),
        ("use_becat", str(args.use_becat).lower()),
    )
    return "FINAL_RESULT," + ",".join(
        "{}={}".format(key, value) for key, value in fields
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.encoder_device = torch.device(args.encoder_device or args.device)
    args.geometry_device = torch.device(
        args.geometry_device or args.encoder_device
    )

    # Source labels/masks are available; target training data is graph-only.
    source_data = load_source_data(args.data_root, args.source, args.split_seed)
    target_data = load_unlabeled_target_data(args.data_root, args.target)
    args.num_features = source_data.x.size(1)
    args.num_classes = int(torch.unique(source_data.y).numel())

    source_data = source_data.to(args.encoder_device)
    target_data = target_data.to(args.encoder_device)
    training = train(args, source_data, target_data)
    model = training["model"]
    source_edge_index, target_edge_index = classifier_edges(
        args,
        source_data,
        target_data,
    )

    # Source diagnostics are masked and occur after best-state restoration.
    with torch.no_grad():
        restored_source_output = source_logits(
            model,
            source_data,
            target_data,
            source_edge_index,
        )
    source_val_metrics = evaluate_logits(
        restored_source_output,
        source_data.y,
        source_data.val_mask,
    )
    source_test_metrics = evaluate_logits(
        restored_source_output,
        source_data.y,
        source_data.test_mask,
    )

    # This is intentionally the first and only target-label read.
    target_labels = load_target_labels_for_final_evaluation(
        args.data_root,
        args.target,
        target_data.num_nodes,
    )
    with torch.no_grad():
        _, restored_target_output = pair_logits(
            model,
            source_data,
            target_data,
            source_edge_index,
            target_edge_index,
        )
    # mask=None means every target node, in unchanged raw-file node order.
    target_metrics = evaluate_logits(restored_target_output, target_labels)
    print(
        format_final_result(
            args,
            training,
            source_val_metrics,
            source_test_metrics,
            target_metrics,
        )
    )


if __name__ == "__main__":
    main()
