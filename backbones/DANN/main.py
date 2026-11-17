"""Release entrypoint for the six citation-network DANN experiments."""

import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from becat.torch import GeometryAdapterConfig, LatentGeometryAdapter

from .data import (
    CITATION_DOMAINS,
    create_source_split,
    load_graph_domain,
    load_target_labels_for_final_evaluation,
)
from .model import DomainClassifier, FeatureExtractor, LabelClassifier


def build_parser():
    parser = argparse.ArgumentParser(
        description="Citation DANN backbone with optional canonical BECAT"
    )
    parser.add_argument("--source", required=True, choices=CITATION_DOMAINS)
    parser.add_argument("--target", required=True, choices=CITATION_DOMAINS)
    parser.add_argument(
        "--data_root",
        "--data_dir",
        dest="data_root",
        required=True,
        help="directory containing acmv9.mat, citationv1.mat, and dblpv7.mat",
    )

    # Preserve the supplied graph-DANN training defaults.
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lambda_domain", type=float, default=1.0)
    parser.add_argument(
        "--use_becat",
        "--use_geometry_adapter",
        dest="use_becat",
        action="store_true",
        help="enable the shared canonical BECAT classifier path",
    )
    parser.add_argument(
        "--lambda_becat",
        "--lambda_geometry",
        dest="lambda_becat",
        type=float,
        default=0.001,
        help="outer BECAT loss weight preserved from the supplied DANN code",
    )
    parser.add_argument("--seed", type=int, default=1, help="training seed")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="source split seed, independent from --seed",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_interval", type=int, default=50)
    return parser


def validate_args(args):
    if args.source == args.target:
        raise ValueError("source and target must be different datasets")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in the interval [0, 1)")
    if args.lambda_domain < 0.0 or args.lambda_becat < 0.0:
        raise ValueError("loss weights must be non-negative")
    if args.log_interval < 0:
        raise ValueError("log_interval must be non-negative")


def seed_training(seed):
    """Seed training RNGs; source splitting uses a separate local generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested ({}), but CUDA is not available".format(
                device_name
            )
        )
    return device


def dann_lambda(epoch, epochs):
    """Preserve the original DANN gradient-reversal schedule."""
    progress = float(epoch) / float(epochs)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def build_becat_config():
    """Preserve the supplied DANN outer-loss scaling convention.

    Canonical BECAT defaults are retained except that its internal latent-loss
    scale is one: the unchanged DANN outer ``--lambda_becat`` supplies 0.001.
    """
    return GeometryAdapterConfig(lambda_latent_bec=1.0)


def initialize_becat(becat, hidden_dim, device):
    """Materialize all lazy BECAT parameters before optimizer construction."""
    dummy_source = torch.zeros(2, hidden_dim, device=device)
    dummy_target = torch.zeros(2, hidden_dim, device=device)
    with torch.no_grad():
        becat(dummy_source, dummy_target, training=False)


def clone_state_to_cpu(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def capture_best_state(
    feature_extractor,
    label_classifier,
    domain_classifier,
    becat,
):
    """Capture every trainable module, including the domain critic."""
    return {
        "feature_extractor": clone_state_to_cpu(feature_extractor),
        "label_classifier": clone_state_to_cpu(label_classifier),
        "domain_classifier": clone_state_to_cpu(domain_classifier),
        "becat": clone_state_to_cpu(becat) if becat is not None else None,
    }


def restore_best_state(
    state,
    feature_extractor,
    label_classifier,
    domain_classifier,
    becat,
):
    feature_extractor.load_state_dict(state["feature_extractor"])
    label_classifier.load_state_dict(state["label_classifier"])
    domain_classifier.load_state_dict(state["domain_classifier"])
    if becat is not None:
        if state["becat"] is None:
            raise RuntimeError("BECAT checkpoint state is missing")
        becat.load_state_dict(state["becat"])


def set_evaluation_mode(
    feature_extractor,
    label_classifier,
    domain_classifier,
    becat,
):
    feature_extractor.eval()
    label_classifier.eval()
    domain_classifier.eval()
    if becat is not None:
        becat.eval()


@torch.no_grad()
def pair_classification_logits(
    feature_extractor,
    label_classifier,
    source_features,
    target_features,
    becat,
):
    """Return source/target logits without consulting target ground truth."""
    raw_source_h = feature_extractor(source_features)
    raw_target_h = feature_extractor(target_features)
    if becat is None:
        return (
            label_classifier(raw_source_h),
            label_classifier(raw_target_h),
        )
    refined_source_o, refined_target_o, _, _ = becat(
        raw_source_h,
        raw_target_h,
        training=False,
    )
    return (
        label_classifier(refined_source_o),
        label_classifier(refined_target_o),
    )


def evaluate_logits(logits, labels, indices=None):
    """Compute accuracy and F1 on supplied nodes, or every node if omitted."""
    if indices is not None:
        logits = logits[indices]
        labels = labels[indices]
    if labels.numel() == 0:
        raise ValueError("evaluation selection contains no nodes")

    predictions = logits.argmax(dim=1)
    accuracy = predictions.eq(labels).float().mean().item()
    predictions_cpu = predictions.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    return {
        "accuracy": accuracy,
        "micro_f1": f1_score(
            labels_cpu, predictions_cpu, average="micro", zero_division=0
        ),
        "macro_f1": f1_score(
            labels_cpu, predictions_cpu, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            labels_cpu, predictions_cpu, average="weighted", zero_division=0
        ),
    }


def format_final_result(
    args,
    best_epoch,
    best_source_val_accuracy,
    source_val_metrics,
    source_test_metrics,
    target_metrics,
):
    fields = (
        ("source", args.source),
        ("target", args.target),
        ("seed", args.seed),
        ("split_seed", args.split_seed),
        ("best_epoch", best_epoch),
        ("best_source_val_accuracy", "{:.6f}".format(best_source_val_accuracy)),
        ("source_val_accuracy", "{:.6f}".format(source_val_metrics["accuracy"])),
        ("source_val_micro_f1", "{:.6f}".format(source_val_metrics["micro_f1"])),
        ("source_val_macro_f1", "{:.6f}".format(source_val_metrics["macro_f1"])),
        ("source_test_accuracy", "{:.6f}".format(source_test_metrics["accuracy"])),
        ("source_test_micro_f1", "{:.6f}".format(source_test_metrics["micro_f1"])),
        ("source_test_macro_f1", "{:.6f}".format(source_test_metrics["macro_f1"])),
        ("target_accuracy", "{:.6f}".format(target_metrics["accuracy"])),
        ("target_micro_f1", "{:.6f}".format(target_metrics["micro_f1"])),
        ("target_macro_f1", "{:.6f}".format(target_metrics["macro_f1"])),
        ("target_weighted_f1", "{:.6f}".format(target_metrics["weighted_f1"])),
        ("use_becat", str(args.use_becat).lower()),
    )
    return "FINAL_RESULT," + ",".join(
        "{}={}".format(key, value) for key, value in fields
    )


def train_and_evaluate(args):
    validate_args(args)

    # Source labels are loaded for supervised training and masked diagnostics.
    # Target training data explicitly excludes the MATLAB ``group`` variable.
    source_data = load_graph_domain(
        args.source, args.data_root, load_labels=True
    )
    target_data = load_graph_domain(
        args.target, args.data_root, load_labels=False
    )
    if source_data.features.shape[1] != target_data.features.shape[1]:
        raise ValueError(
            "Feature dimensions differ: {} has {}, {} has {}".format(
                args.source,
                source_data.features.shape[1],
                args.target,
                target_data.features.shape[1],
            )
        )

    source_train_idx, source_val_idx, source_test_idx = create_source_split(
        source_data.features.shape[0], args.split_seed
    )

    # The private split generator above does not consume training RNG state.
    seed_training(args.seed)
    device = resolve_device(args.device)
    source_features = torch.from_numpy(source_data.features).float().to(device)
    source_labels = torch.from_numpy(source_data.labels).long().to(device)
    target_features = torch.from_numpy(target_data.features).float().to(device)
    source_train_idx = source_train_idx.to(device)
    source_val_idx = source_val_idx.to(device)
    source_test_idx = source_test_idx.to(device)

    num_classes = int(source_data.labels.max()) + 1
    feature_extractor = FeatureExtractor(
        input_dim=source_features.size(1),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    label_classifier = LabelClassifier(args.hidden_dim, num_classes).to(device)
    domain_classifier = DomainClassifier(args.hidden_dim, args.dropout).to(device)

    becat = None
    if args.use_becat:
        becat = LatentGeometryAdapter(build_becat_config()).to(device)
        initialize_becat(becat, args.hidden_dim, device)

    model_parameters = (
        list(feature_extractor.parameters())
        + list(label_classifier.parameters())
        + list(domain_classifier.parameters())
    )
    if becat is not None:
        model_parameters += list(becat.parameters())
    optimizer = optim.Adam(
        model_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    domain_labels = torch.cat(
        (
            torch.zeros(
                source_features.size(0), dtype=torch.long, device=device
            ),
            torch.ones(
                target_features.size(0), dtype=torch.long, device=device
            ),
        ),
        dim=0,
    )

    best_state = None
    best_epoch = 0
    best_source_val_accuracy = float("-inf")
    print(
        "RUN_START,source={},target={},seed={},split_seed={},"
        "source_nodes={},source_train_nodes={},source_val_nodes={},"
        "source_test_nodes={},target_nodes={},use_becat={}".format(
            args.source,
            args.target,
            args.seed,
            args.split_seed,
            source_features.size(0),
            source_train_idx.numel(),
            source_val_idx.numel(),
            source_test_idx.numel(),
            target_features.size(0),
            str(args.use_becat).lower(),
        )
    )

    for epoch in range(1, args.epochs + 1):
        feature_extractor.train()
        label_classifier.train()
        domain_classifier.train()
        if becat is not None:
            becat.train()

        lambda_grl = dann_lambda(epoch, args.epochs)
        optimizer.zero_grad()

        # H_s/H_t are the original raw DANN embeddings.
        raw_source_h = feature_extractor(source_features)
        raw_target_h = feature_extractor(target_features)

        geometry_loss = None
        refined_source_o = raw_source_h
        if becat is not None:
            refined_source_o, _, geometry_loss, _ = becat(
                raw_source_h,
                raw_target_h,
                training=True,
            )

        # Supervision is restricted to the 80% source-train partition.
        source_logits = label_classifier(refined_source_o)
        label_loss = criterion(
            source_logits[source_train_idx], source_labels[source_train_idx]
        )

        # The main-table domain critic always consumes raw H_s/H_t.
        raw_domain_h = torch.cat((raw_source_h, raw_target_h), dim=0)
        domain_logits = domain_classifier(raw_domain_h, lambda_grl=lambda_grl)
        domain_loss = criterion(domain_logits, domain_labels)

        total_loss = label_loss + args.lambda_domain * domain_loss
        if becat is not None:
            total_loss = total_loss + args.lambda_becat * geometry_loss
        total_loss.backward()
        optimizer.step()

        # Source validation accuracy is the sole checkpoint criterion.
        set_evaluation_mode(
            feature_extractor,
            label_classifier,
            domain_classifier,
            becat,
        )
        source_val_logits, _ = pair_classification_logits(
            feature_extractor,
            label_classifier,
            source_features,
            target_features,
            becat,
        )
        source_val_metrics = evaluate_logits(
            source_val_logits, source_labels, source_val_idx
        )
        source_val_accuracy = source_val_metrics["accuracy"]
        if source_val_accuracy > best_source_val_accuracy:
            best_source_val_accuracy = source_val_accuracy
            best_epoch = epoch
            best_state = capture_best_state(
                feature_extractor,
                label_classifier,
                domain_classifier,
                becat,
            )

        should_log = args.log_interval > 0 and (
            epoch == 1
            or epoch % args.log_interval == 0
            or epoch == args.epochs
        )
        if should_log:
            message = (
                "Epoch {}/{} total_loss={:.6f} label_loss={:.6f} "
                "domain_loss={:.6f} grl_lambda={:.6f} "
                "source_val_accuracy={:.6f}"
            ).format(
                epoch,
                args.epochs,
                total_loss.detach().item(),
                label_loss.detach().item(),
                domain_loss.detach().item(),
                lambda_grl,
                source_val_accuracy,
            )
            if geometry_loss is not None:
                message += " geometry_loss={:.6f}".format(
                    geometry_loss.detach().item()
                )
            print(message)

    if best_state is None:
        raise RuntimeError("training produced no source-validation checkpoint")

    # Restore every trainable module before any target-label read.
    restore_best_state(
        best_state,
        feature_extractor,
        label_classifier,
        domain_classifier,
        becat,
    )
    set_evaluation_mode(
        feature_extractor,
        label_classifier,
        domain_classifier,
        becat,
    )
    print(
        "CHECKPOINT_RESTORED,best_epoch={},best_source_val_accuracy={:.6f}".format(
            best_epoch, best_source_val_accuracy
        )
    )

    restored_source_logits, _ = pair_classification_logits(
        feature_extractor,
        label_classifier,
        source_features,
        target_features,
        becat,
    )
    source_val_metrics = evaluate_logits(
        restored_source_logits, source_labels, source_val_idx
    )
    source_test_metrics = evaluate_logits(
        restored_source_logits, source_labels, source_test_idx
    )

    # This is intentionally the first and only target-ground-truth read.
    target_labels_array = load_target_labels_for_final_evaluation(
        args.target,
        args.data_root,
        target_features.size(0),
    )
    target_labels = torch.from_numpy(target_labels_array).long().to(device)
    _, restored_target_logits = pair_classification_logits(
        feature_extractor,
        label_classifier,
        source_features,
        target_features,
        becat,
    )
    # indices=None evaluates every target node in unchanged MATLAB node order.
    target_metrics = evaluate_logits(restored_target_logits, target_labels)

    print(
        format_final_result(
            args,
            best_epoch,
            best_source_val_accuracy,
            source_val_metrics,
            source_test_metrics,
            target_metrics,
        )
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
