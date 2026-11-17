from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import torch
from sklearn.metrics import f1_score

from .GRADE import GRADE
from .utils import read_citation_labels


def _masked_accuracy(logits, labels, mask=None):
    predictions = torch.argmax(logits, dim=1)
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    return float((predictions == labels).float().mean().item())


def _masked_f1_scores(logits, labels, mask=None):
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    labels_for_metric = labels.detach().cpu().numpy()
    predictions_for_metric = torch.argmax(logits, dim=1).detach().cpu().numpy()
    micro_f1 = f1_score(
        labels_for_metric,
        predictions_for_metric,
        average="micro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        labels_for_metric,
        predictions_for_metric,
        average="macro",
        zero_division=0,
    )
    return float(micro_f1), float(macro_f1)


def train_citation(
    args,
    data_s,
    data_t,
    source_masks,
    source_domain,
    target_domain,
):
    g_s, source_labels, source_features = data_s
    g_t, target_training_labels, target_features = data_t
    if target_training_labels is not None:
        raise ValueError("Target labels must not be loaded for citation training.")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError(
            "Source and target feature dimensions must match: {} != {}".format(
                source_features.shape[1], target_features.shape[1]
            )
        )

    source_train_mask, source_val_mask, source_test_mask = source_masks
    in_feats = source_features.shape[1]
    n_classes = int(source_labels.max().item()) + 1

    g_s = g_s.to(args.device)
    g_t = g_t.to(args.device)
    source_features = source_features.to(device=args.device)
    source_labels = source_labels.to(device=args.device)
    target_features = target_features.to(device=args.device)
    source_train_mask = source_train_mask.to(device=args.device)
    source_val_mask = source_val_mask.to(device=args.device)
    source_test_mask = source_test_mask.to(device=args.device)

    model = GRADE(
        g_s,
        g_t,
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers,
        args.dropout,
        args=args,
    ).to(device=args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_epoch = None
    best_source_val_accuracy = float("-inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        alpha = 2 / (1 + np.exp(-10 * epoch / args.epochs)) - 1
        loss = model(
            source_features,
            source_labels,
            target_features,
            alpha,
            source_train_mask=source_train_mask,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            source_val_logits = model.inference_source(source_features)
        current_val_accuracy = _masked_accuracy(
            source_val_logits, source_labels, source_val_mask
        )

        improved = current_val_accuracy > best_source_val_accuracy
        if improved:
            best_epoch = epoch + 1
            best_source_val_accuracy = current_val_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            "EPOCH,epoch={epoch},source_val_accuracy={current:.6f},"
            "best_source_val_accuracy={best:.6f},improved={improved}".format(
                epoch=epoch + 1,
                current=current_val_accuracy,
                best=best_source_val_accuracy,
                improved=str(improved).lower(),
            )
        )
        if epochs_without_improvement >= args.patience:
            break

    if best_state_dict is None:
        raise RuntimeError("Citation training did not produce a checkpoint.")

    # Restore the checkpoint chosen only by strictly improving source validation
    # accuracy before any target label file is read.
    model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        source_logits = model.inference_source(source_features)

    source_val_accuracy = _masked_accuracy(
        source_logits, source_labels, source_val_mask
    )
    if not np.isclose(source_val_accuracy, best_source_val_accuracy):
        raise RuntimeError(
            "Restored checkpoint source validation accuracy does not match the "
            "saved selection accuracy."
        )
    source_test_accuracy = _masked_accuracy(
        source_logits, source_labels, source_test_mask
    )
    source_val_micro_f1, source_val_macro_f1 = _masked_f1_scores(
        source_logits, source_labels, source_val_mask
    )
    source_test_micro_f1, source_test_macro_f1 = _masked_f1_scores(
        source_logits, source_labels, source_test_mask
    )

    # This is the first target-label read. Training, early stopping, checkpoint
    # selection, restoration, and all source diagnostics are already complete.
    num_target_nodes = target_features.shape[0]
    target_labels = read_citation_labels(
        args.data_root,
        target_domain,
        num_target_nodes,
    ).to(device=args.device)
    with torch.no_grad():
        target_logits = model.inference_target(target_features)

    # No target mask is applied: every target node is evaluated exactly once.
    target_accuracy = _masked_accuracy(target_logits, target_labels)
    target_micro_f1, target_macro_f1 = _masked_f1_scores(
        target_logits, target_labels
    )

    return {
        "source": source_domain,
        "target": target_domain,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "best_epoch": best_epoch,
        "best_source_val_accuracy": best_source_val_accuracy,
        "source_val_accuracy": source_val_accuracy,
        "source_test_accuracy": source_test_accuracy,
        "target_accuracy": target_accuracy,
        "target_micro_f1": target_micro_f1,
        "target_macro_f1": target_macro_f1,
        "use_becat": str(args.use_becat).lower(),
        "source_val_micro_f1": source_val_micro_f1,
        "source_val_macro_f1": source_val_macro_f1,
        "source_test_micro_f1": source_test_micro_f1,
        "source_test_macro_f1": source_test_macro_f1,
    }


def GRADE_main(
    args,
    data_s,
    data_t,
    source_masks,
    source_domain,
    target_domain,
):
    return train_citation(
        args,
        data_s,
        data_t,
        source_masks,
        source_domain,
        target_domain,
    )
