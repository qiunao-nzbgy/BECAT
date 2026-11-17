"""Release entry point for the six PairAlign Citation adaptation tasks.

This module intentionally exposes only the common finalized Citation protocol.
Target labels are loaded inside ``train_citation`` after the selected model and
domain-classifier checkpoint has been restored.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


RELEASE_ROOT = Path(__file__).resolve().parents[2]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

try:
    from . import calc_rw, data, models, utils
except ImportError:  # Direct execution by the release shell scripts.
    import calc_rw
    import data
    import models
    import utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def arg_parse():
    parser = argparse.ArgumentParser(
        description="PairAlign Citation release (source-val-accuracy protocol)."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--src_name", required=True)
    parser.add_argument("--tgt_name", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Training seed(s). The final Citation experiments use "
            "1, 2, 3, 4, and 5 with split_seed=42."
        ),
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Source-partition seed only; independent of every training seed.",
    )

    # Original PairAlign model and optimization defaults are retained.
    parser.add_argument("--cls_layers", type=int, default=2)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--dc_layers", type=int, default=3)
    parser.add_argument("--gnn_dim", type=int, default=300)
    parser.add_argument("--cls_dim", type=int, default=300)
    parser.add_argument("--dc_dim", type=int, default=300)
    parser.add_argument("--backbone", choices=("GS", "GCN"), default="GS")
    parser.add_argument("--pooling", choices=("add", "mean"), default="mean")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--gnn_dc", action="store_true")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--opt",
        choices=("adam", "sgd", "rmsprop", "adagrad"),
        default="adam",
    )
    parser.add_argument(
        "--opt_scheduler",
        choices=("none", "step", "multistep", "cos"),
        default="step",
    )
    parser.add_argument("--opt_decay_step", type=int, default=50)
    parser.add_argument("--opt_decay_rate", type=float, default=0.9)
    parser.add_argument("--opt_restart", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--alphatimes", type=float, default=1.0)
    parser.add_argument("--alphamax", type=float, default=1.0)
    parser.add_argument("--dc_coeff", type=float, default=1.0)

    parser.add_argument("--edge_rw", action="store_true")
    parser.add_argument("--label_rw", action="store_true")
    parser.add_argument("--weight_CE_src", action="store_true")
    parser.add_argument("--ew_start", type=int, default=0)
    parser.add_argument("--lw_start", type=int, default=0)
    parser.add_argument("--ew_freq", type=int, default=10)
    parser.add_argument("--lw_freq", type=int, default=10)
    parser.add_argument("--rw_lmda", type=float, default=1.0)
    parser.add_argument("--gamma_reg", type=float, default=1e-4)
    parser.add_argument(
        "--ew_type",
        choices=("pseudobeta", "truth"),
        default="pseudobeta",
        help="Citation rejects truth; it is retained only for an explicit error.",
    )
    parser.add_argument("--ls_lambda", type=float, default=25.0)
    parser.add_argument("--lw_lambda", type=float, default=0.01)

    parser.add_argument("--use_becat", action="store_true")
    parser.add_argument("--lambda_geometry", type=float, default=1e-3)
    parser.add_argument("--geo_k", type=int, default=10)
    parser.add_argument("--geo_tau", type=float, default=0.5)
    parser.add_argument("--geo_lambda_stat", type=float, default=0.1)
    parser.add_argument("--geo_beta", type=float, default=0.05)
    parser.add_argument("--geo_residual_alpha", type=float, default=0.1)
    parser.add_argument("--geo_num_heads", type=int, default=2)
    parser.add_argument("--geo_dropout", type=float, default=0.3)
    return parser.parse_args()


def _cpu_clone_state_dict(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _initialize_citation_reweighting(source_dataset):
    """Initialize diagnostics/weights without any target ground truth."""
    num_classes = source_dataset.num_classes
    source_dataset.true_ew = np.ones((num_classes, num_classes))
    source_dataset.ew = np.ones((num_classes, num_classes))
    source_dataset.true_lw = np.ones(num_classes)
    source_dataset.lw = np.ones(num_classes)
    source_dataset.true_beta = np.ones(num_classes ** 2)


def update_ew(src_data, tgt_data, pred_src, pred_tgt, args):
    if args.ew_type == "truth":
        raise ValueError(
            "Citation forbids --ew_type truth because target labels and "
            "target truth edge statistics are unavailable during training."
        )
    calc_rw.calc_edge_rw_pseudo(
        src_data,
        tgt_data,
        pred_src,
        pred_tgt,
        DEVICE,
        args,
    )


def update_lw(src_data, tgt_data, pred_src, pred_tgt, args):
    # Target class proportions are estimated only from target predictions.
    yhat_tgt = torch.mean(
        F.softmax(pred_tgt, dim=1)[tgt_data.target_mask],
        dim=0,
    )
    y_src_onehot = F.one_hot(
        src_data.y[src_data.source_mask],
        num_classes=src_data.num_classes,
    ).float()
    y_src = torch.mean(y_src_onehot, dim=0)
    y_src_pred = F.softmax(pred_src, dim=1)[src_data.source_mask]
    cov_mat = torch.mm(y_src_pred.T, y_src_onehot) / len(src_data.source_mask)
    return calc_rw.calc_label_rw(y_src, yhat_tgt, cov_mat, args.lw_lambda)


def _pair_forward(model, src_data, tgt_data, use_becat):
    if use_becat:
        return model.forward_pair(src_data, src_data.x, tgt_data, tgt_data.x)
    h_src, logits_src = model(src_data, src_data.x)
    h_tgt, logits_tgt = model(tgt_data, tgt_data.x)
    return {
        "H_s": h_src,
        "H_t": h_tgt,
        "O_s": h_src,
        "O_t": h_tgt,
        "logits_s": logits_src,
        "logits_t": logits_tgt,
        "geometry_loss": h_src.new_tensor(0.0),
    }


def train_one_epoch(model, gnn_dc, src_data, tgt_data, args, opt, scheduler, epoch):
    model.train()
    gnn_dc.train()
    output = _pair_forward(model, src_data, tgt_data, args.use_becat)

    # Domain alignment always consumes raw H; refined O is classifier-only.
    alpha = min(((epoch + 1) / args.epochs), args.alphamax)
    dc_pred_src = gnn_dc(output["H_s"], alpha)[src_data.source_mask]
    dc_pred_tgt = gnn_dc(output["H_t"], alpha)[tgt_data.target_mask]

    source_train_mask = src_data.source_training_mask
    source_train_labels = src_data.y[source_train_mask]
    source_train_logits = output["logits_s"][source_train_mask]
    if args.label_rw:
        cls_loss_src = utils.CE_loss_weight(
            source_train_logits,
            source_train_labels,
            torch.from_numpy(src_data.lw).to(DEVICE),
            args.weight_CE_src,
        )
    elif args.weight_CE_src:
        cls_loss_src = utils.CE_srcweight_loss(
            source_train_logits,
            source_train_labels,
        )
    else:
        cls_loss_src = utils.CE_loss(source_train_logits, source_train_labels)

    loss = cls_loss_src
    if args.use_becat:
        loss = loss + args.lambda_geometry * output["geometry_loss"]

    source_sample_weights = torch.matmul(
        F.one_hot(
            src_data.y[src_data.source_mask],
            num_classes=src_data.num_classes,
        ).float(),
        torch.from_numpy(src_data.lw).float().to(DEVICE),
    )
    weighted_nll_source = -source_sample_weights.reshape(-1, 1) * torch.log(
        dc_pred_src + 1e-12
    )
    nll_target = -torch.log(1 - dc_pred_tgt + 1e-12)
    weighted_dc_loss = (
        torch.mean(weighted_nll_source) + torch.mean(nll_target)
    ) / 2
    if args.gnn_dc:
        loss = loss + args.dc_coeff * weighted_dc_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    if scheduler is not None:
        scheduler.step()

    needs_edge_update = (
        args.edge_rw
        and (epoch + 1) >= args.ew_start
        and (epoch + 1) % args.ew_freq == 0
    )
    needs_label_update = (
        args.label_rw
        and (epoch + 1) >= args.lw_start
        and (epoch + 1) % args.lw_freq == 0
    )
    if needs_edge_update or needs_label_update:
        model.eval()
        with torch.no_grad():
            updated = _pair_forward(model, src_data, tgt_data, args.use_becat)
        tgt_data.y_hat = updated["logits_t"].argmax(dim=1)
        if needs_edge_update:
            update_ew(
                src_data,
                tgt_data,
                updated["logits_s"],
                updated["logits_t"],
                args,
            )
        if needs_label_update:
            src_data.lw = update_lw(
                src_data,
                tgt_data,
                updated["logits_s"],
                updated["logits_t"],
                args,
            )
    return float(loss.detach().cpu())


def source_validation_accuracy(model, source_dataset, target_dataset, args):
    model.eval()
    with torch.no_grad():
        source_logits = _pair_forward(
            model,
            source_dataset,
            target_dataset,
            args.use_becat,
        )["logits_s"]
    mask = source_dataset.source_validation_mask
    return float(
        (
            source_logits[mask].argmax(dim=1) == source_dataset.y[mask]
        ).float().mean().item()
    )


def _classification_metrics(logits, labels, mask=None):
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    y_true = labels.detach().cpu().numpy()
    y_pred = logits.argmax(dim=1).detach().cpu().numpy()
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def evaluate_restored_checkpoint(
    model,
    source_dataset,
    target_dataset,
    target_labels,
    args,
):
    model.eval()
    with torch.no_grad():
        output = _pair_forward(
            model,
            source_dataset,
            target_dataset,
            args.use_becat,
        )
    source_val = _classification_metrics(
        output["logits_s"],
        source_dataset.y,
        source_dataset.source_validation_mask,
    )
    source_test = _classification_metrics(
        output["logits_s"],
        source_dataset.y,
        source_dataset.source_testing_mask,
    )
    target = _classification_metrics(output["logits_t"], target_labels)
    return {
        "source_val_accuracy": source_val["accuracy"],
        "source_val_micro_f1": source_val["micro_f1"],
        "source_val_macro_f1": source_val["macro_f1"],
        "source_test_accuracy": source_test["accuracy"],
        "source_test_micro_f1": source_test["micro_f1"],
        "source_test_macro_f1": source_test["macro_f1"],
        "target_accuracy": target["accuracy"],
        "target_micro_f1": target["micro_f1"],
        "target_macro_f1": target["macro_f1"],
        "target_weighted_f1": target["weighted_f1"],
    }


def train_citation(source_dataset, target_dataset, args):
    if args.ew_type == "truth":
        raise ValueError("Citation release runs forbid --ew_type truth.")
    if "y" in target_dataset or "edge_class" in target_dataset:
        raise RuntimeError("Target training graph must be physically unlabeled.")

    model = models.GNN(
        source_dataset.num_node_features,
        source_dataset.num_classes,
        args,
    ).to(DEVICE)
    gnn_dc = models.GNN_dc(args.gnn_dim, 1, args).to(DEVICE)
    # GNN.__init__ materializes BECAT's lazy transformer before this collection.
    scheduler, opt = utils.build_optimizer(
        args,
        list(model.parameters()) + list(gnn_dc.parameters()),
    )

    source_dataset = source_dataset.to(DEVICE)
    target_dataset = target_dataset.to(DEVICE)
    _initialize_citation_reweighting(source_dataset)

    best_source_val_accuracy = float("-inf")
    best_epoch = 0
    best_state = None
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            gnn_dc,
            source_dataset,
            target_dataset,
            args,
            opt,
            scheduler,
            epoch,
        )
        source_val_accuracy = source_validation_accuracy(
            model,
            source_dataset,
            target_dataset,
            args,
        )
        print(
            f"TRAINING_STATUS,epoch={epoch + 1},"
            f"source_val_accuracy={source_val_accuracy:.6f},"
            f"training_loss={train_loss:.6f}"
        )
        # Strict improvement means exact ties retain the earlier checkpoint.
        if source_val_accuracy > best_source_val_accuracy:
            best_source_val_accuracy = source_val_accuracy
            best_epoch = epoch + 1
            best_state = {
                "model": _cpu_clone_state_dict(model),
                "gnn_dc": _cpu_clone_state_dict(gnn_dc),
            }

    if best_state is None:
        raise RuntimeError("No Citation checkpoint was selected.")
    model.load_state_dict(best_state["model"])
    gnn_dc.load_state_dict(best_state["gnn_dc"])
    model.eval()
    gnn_dc.eval()
    print(
        f"CHECKPOINT_RESTORED,best_epoch={best_epoch},"
        f"best_source_val_accuracy={best_source_val_accuracy:.6f}"
    )

    # This is the only target-label read in the entire training control flow.
    target_labels = data.load_target_labels(
        args.data_root,
        args.tgt_name,
        target_dataset.num_nodes,
        source_dataset.num_classes,
    ).to(DEVICE)
    print("TARGET_LABELS_LOADED,phase=final_evaluation")
    report = evaluate_restored_checkpoint(
        model,
        source_dataset,
        target_dataset,
        target_labels,
        args,
    )
    report["best_epoch"] = best_epoch
    report["best_source_val_accuracy"] = best_source_val_accuracy
    return report


def _becat_result_fields(args):
    if not args.use_becat:
        return ""
    return (
        f",lambda_geometry={args.lambda_geometry}"
        f",geo_k={args.geo_k}"
        f",geo_tau={args.geo_tau}"
        f",geo_lambda_stat={args.geo_lambda_stat}"
        f",geo_beta={args.geo_beta}"
        f",geo_residual_alpha={args.geo_residual_alpha}"
        f",geo_num_heads={args.geo_num_heads}"
        f",geo_dropout={args.geo_dropout}"
    )


def print_final_result(report, args, seed):
    print(
        f"FINAL_RESULT,source={data.canonical_domain_name(args.src_name)},"
        f"target={data.canonical_domain_name(args.tgt_name)},"
        f"seed={seed},split_seed={args.split_seed},"
        f"best_epoch={report['best_epoch']},"
        f"best_source_val_accuracy={report['best_source_val_accuracy']:.6f},"
        f"source_val_accuracy={report['source_val_accuracy']:.6f},"
        f"source_test_accuracy={report['source_test_accuracy']:.6f},"
        f"target_accuracy={report['target_accuracy']:.6f},"
        f"target_micro_f1={report['target_micro_f1']:.6f},"
        f"target_macro_f1={report['target_macro_f1']:.6f},"
        f"target_weighted_f1={report['target_weighted_f1']:.6f},"
        f"source_val_micro_f1={report['source_val_micro_f1']:.6f},"
        f"source_val_macro_f1={report['source_val_macro_f1']:.6f},"
        f"source_test_micro_f1={report['source_test_micro_f1']:.6f},"
        f"source_test_macro_f1={report['source_test_macro_f1']:.6f},"
        f"use_becat={str(args.use_becat).lower()}"
        f"{_becat_result_fields(args)}"
    )


def print_summary(reports, args):
    ddof = 1 if len(reports) > 1 else 0
    micro = np.asarray([report["target_micro_f1"] for report in reports])
    macro = np.asarray([report["target_macro_f1"] for report in reports])
    print(
        f"FINAL_SUMMARY,source={data.canonical_domain_name(args.src_name)},"
        f"target={data.canonical_domain_name(args.tgt_name)},"
        f"runs={len(reports)},split_seed={args.split_seed},"
        f"target_micro_f1_mean={micro.mean():.6f},"
        f"target_micro_f1_std={micro.std(ddof=ddof):.6f},"
        f"target_macro_f1_mean={macro.mean():.6f},"
        f"target_macro_f1_std={macro.std(ddof=ddof):.6f},"
        f"use_becat={str(args.use_becat).lower()}"
    )


def set_training_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = arg_parse()
    if args.split_seed != 42:
        print(
            "WARNING: main-table reproduction uses split_seed=42; "
            f"received {args.split_seed}."
        )
    if args.ew_type == "truth":
        raise ValueError("Citation release runs forbid --ew_type truth.")

    print("PROTOCOL,checkpoint_selection=source_validation_accuracy")
    print("PROTOCOL,target_training_labels=not_loaded")
    print("CONFIG," + json.dumps(vars(args), sort_keys=True))
    reports = []
    for seed in args.seed:
        set_training_seed(seed)
        source_dataset = data.load_source_citation(
            args.data_root,
            args.src_name,
            split_seed=args.split_seed,
        )
        target_dataset = data.load_target_citation_unlabeled(
            args.data_root,
            args.tgt_name,
            source_dataset.num_classes,
        )
        data.validate_citation_pair(source_dataset, target_dataset)
        report = train_citation(source_dataset, target_dataset, args)
        reports.append(report)
        print_final_result(report, args, seed)
    print_summary(reports, args)


if __name__ == "__main__":
    main()
