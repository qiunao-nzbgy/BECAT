"""Train and evaluate DGSDA on one citation-domain transfer task.

Target labels are deliberately unavailable to the training and checkpoint-
selection code. They are read once, after the best source-validation checkpoint
has been restored, for final reporting only.
"""

import argparse
import os
import os.path as osp
from pathlib import Path
import random
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F

from becat.torch import GeometryAdapterConfig
from models import BernNet
from utils import CitationDataset, MMD, load_unlabeled_citation_graph


def seed_training(seed):
    """Seed training randomness; dataset splitting uses a separate generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_classification_metrics(logits, labels, mask=None):
    """Compute detached single-label multiclass metrics."""
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]

    predictions = logits.argmax(dim=1)
    y_true = labels.detach().cpu().numpy()
    y_pred = predictions.detach().cpu().numpy()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "micro_f1": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def validate_model_parallel_devices(backbone_name, becat_name):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "BECAT model parallelism requires at least two visible logical "
            "CUDA devices. Set CUDA_VISIBLE_DEVICES to expose two GPUs."
        )

    backbone_device = torch.device(backbone_name)
    becat_device = torch.device(becat_name)
    if backbone_device.type != "cuda" or becat_device.type != "cuda":
        raise ValueError(
            "--backbone_device and --becat_device must both be CUDA devices "
            "when --becat_model_parallel is enabled."
        )

    backbone_index = (
        torch.cuda.current_device()
        if backbone_device.index is None
        else backbone_device.index
    )
    becat_index = (
        torch.cuda.current_device()
        if becat_device.index is None
        else becat_device.index
    )
    if backbone_index == becat_index:
        raise ValueError(
            "--backbone_device and --becat_device must refer to different "
            "logical CUDA devices."
        )

    visible_count = torch.cuda.device_count()
    for role, index in (("backbone", backbone_index), ("BECAT", becat_index)):
        if not 0 <= index < visible_count:
            raise ValueError(
                f"{role} CUDA device index {index} is not visible; "
                f"{visible_count} logical CUDA devices are available."
            )
    return torch.device(f"cuda:{backbone_index}"), torch.device(
        f"cuda:{becat_index}"
    )


def module_device(module):
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else "no parameters or buffers"


def forward_model_parallel_pair(
    model,
    source_x,
    target_x,
    source_edge_index_backbone,
    target_edge_index_backbone,
    source_edge_index_becat,
    target_edge_index_becat,
    becat_device,
):
    """Run split DGSDA+BECAT without severing cross-device autograd."""
    source_latent = model.get_latent_from_tensors(
        source_x, source_edge_index_backbone, True
    )
    target_latent = model.get_latent_from_tensors(
        target_x, target_edge_index_backbone, False
    )
    source_refined, target_refined, extras = model.adapt_latent_pair(
        source_latent.to(becat_device), target_latent.to(becat_device)
    )
    source_logits = model.classify_latent(
        source_refined, source_edge_index_becat
    )
    target_logits = model.classify_latent(
        target_refined, target_edge_index_becat
    )
    tensors = {
        "source_latent": source_latent,
        "target_latent": target_latent,
        "source_refined": source_refined,
        "target_refined": target_refined,
    }
    return source_logits, target_logits, extras, tensors


def entropy_minimization_loss(output):
    """Original unlabeled-target entropy objective (uses predictions only)."""
    probabilities = F.softmax(output, dim=1)
    log_probabilities = F.log_softmax(output, dim=1)
    class_mass = torch.sum(probabilities, dim=0)
    return -torch.sum(
        probabilities
        * log_probabilities
        / (class_mass / torch.sum(class_mass)),
        dim=1,
    ).mean()


def parse_args():
    parser = argparse.ArgumentParser(
        description="DGSDA citation-domain baseline and BECAT integration"
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--dropout_ratio", type=float, default=0.3)
    parser.add_argument("--dp_ratio", type=float, default=0.3)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--source", default="ACMv9")
    parser.add_argument("--target", default="Citationv1")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="directory containing one subdirectory per citation dataset",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, default=1, help="training randomness seed"
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="source 80/10/10 split seed, independent of --seed",
    )

    parser.add_argument(
        "--use_becat",
        "--use_geometry_adapter",
        dest="use_becat",
        action="store_true",
        help="refine classifier-path latent representations with BECAT",
    )
    parser.add_argument(
        "--lambda_geo",
        "--lambda_geometry",
        dest="lambda_geo",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--becat_residual_alpha",
        "--geo_residual_alpha",
        dest="becat_residual_alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--becat_topk",
        "--geo_topk",
        dest="becat_topk",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--becat_tau", "--geo_tau", dest="becat_tau", type=float, default=0.5
    )
    parser.add_argument(
        "--becat_beta",
        "--geo_beta",
        dest="becat_beta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--becat_lambda_stat",
        "--geo_lambda_stat",
        dest="becat_lambda_stat",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--becat_warmup",
        "--geo_warmup",
        dest="becat_warmup",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--becat_model_parallel",
        "--geometry_model_parallel",
        dest="becat_model_parallel",
        action="store_true",
    )
    parser.add_argument("--backbone_device", default="cuda:0")
    parser.add_argument(
        "--becat_device",
        "--adapter_device",
        dest="becat_device",
        default="cuda:1",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epoch < 1:
        raise ValueError("--epoch must be at least 1")

    seed_training(args.seed)
    model_parallel = args.use_becat and args.becat_model_parallel
    backbone_device = None
    becat_device = None
    if model_parallel:
        backbone_device, becat_device = validate_model_parallel_devices(
            args.backbone_device, args.becat_device
        )

    source_path = osp.join(str(args.data_root), args.source)
    target_path = osp.join(str(args.data_root), args.target)
    source_dataset = CitationDataset(
        source_path, args.source, split_seed=args.split_seed
    )
    source_data = source_dataset[0]
    # This loader admits only graph inputs to the training path. The returned
    # object has no y attribute and cannot expose target ground truth.
    target_data = load_unlabeled_citation_graph(target_path, args.target)

    split_sizes = {
        "train": int(source_data.train_mask.sum()),
        "val": int(source_data.val_mask.sum()),
        "test": int(source_data.test_mask.sum()),
    }
    print(
        f"CONFIG source={args.source} target={args.target} seed={args.seed} "
        f"split_seed={args.split_seed} use_becat={str(args.use_becat).lower()}"
    )
    print(
        "SOURCE_SPLIT "
        f"train={split_sizes['train']} val={split_sizes['val']} "
        f"test={split_sizes['test']} total={source_data.num_nodes}"
    )

    becat_config = None
    if args.use_becat:
        becat_config = GeometryAdapterConfig(
            latent_topk_k=args.becat_topk,
            latent_topk_tau=args.becat_tau,
            lambda_latent_bec_stat=args.becat_lambda_stat,
            sparse_gt_num_heads=2 if args.hidden % 2 == 0 else 1,
            sparse_gt_attention_k=args.becat_topk,
            sparse_gt_curv_bias_beta=args.becat_beta,
            sparse_gt_residual_alpha=args.becat_residual_alpha,
        )

    model = BernNet(
        source_dataset.num_node_features,
        args.hidden,
        source_dataset.num_classes,
        args.dropout_ratio,
        args.dp_ratio,
        args.K,
        use_becat=args.use_becat,
        becat_config=becat_config,
    )

    if model_parallel:
        model.lin1.to(backbone_device)
        model.prop1.to(backbone_device)
        model.prop2.to(backbone_device)
        model.prop4.to(backbone_device)
        model.becat.to(becat_device)
        model.lin2.to(becat_device)
        model.prop3.to(becat_device)

        source_x_backbone = source_data.x.to(backbone_device)
        target_x_backbone = target_data.x.to(backbone_device)
        source_edge_backbone = source_data.edge_index.to(backbone_device)
        target_edge_backbone = target_data.edge_index.to(backbone_device)
        source_edge_becat = source_data.edge_index.to(becat_device)
        target_edge_becat = target_data.edge_index.to(becat_device)
        source_y_becat = source_data.y.to(becat_device)
        source_train_mask_becat = source_data.train_mask.to(becat_device)
        source_val_mask_becat = source_data.val_mask.to(becat_device)
        source_test_mask_becat = source_data.test_mask.to(becat_device)

        print(f"lin1_device={module_device(model.lin1)}")
        print(f"becat_device={module_device(model.becat)}")
        print(f"lin2_device={module_device(model.lin2)}")
    elif torch.cuda.is_available():
        training_device = torch.device(f"cuda:{args.cuda}")
        model = model.to(training_device)
        source_data = source_data.to(training_device)
        target_data = target_data.to(training_device)

    model.set_becat_active(args.becat_warmup <= 0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    best_epoch = None
    best_source_val_accuracy = float("-inf")
    best_model_state = None
    best_becat_active = False

    for epoch in range(1, args.epoch + 1):
        model.train()
        model.set_becat_active(epoch > args.becat_warmup)
        optimizer.zero_grad()
        becat_extras = {}
        parallel_tensors = {}

        if model_parallel:
            output, target_outputs, becat_extras, parallel_tensors = (
                forward_model_parallel_pair(
                    model,
                    source_x_backbone,
                    target_x_backbone,
                    source_edge_backbone,
                    target_edge_backbone,
                    source_edge_becat,
                    target_edge_becat,
                    becat_device,
                )
            )
            source_loss = F.cross_entropy(
                output[source_train_mask_becat],
                source_y_becat[source_train_mask_becat],
            )
        elif args.use_becat:
            output, target_outputs, becat_extras = model.forward_pair(
                source_data, target_data
            )
            source_loss = F.cross_entropy(
                output[source_data.train_mask],
                source_data.y[source_data.train_mask],
            )
        else:
            output = model(source_data)
            target_outputs = model(target_data, False)
            source_loss = F.cross_entropy(
                output[source_data.train_mask],
                source_data.y[source_data.train_mask],
            )

        theta_s = model.prop1.temp
        theta_t = model.prop2.temp
        theta_loss = (
            F.l1_loss(theta_s, theta_t)
            + torch.sum(torch.abs(theta_s))
            + torch.sum(torch.abs(theta_t))
        )
        if model_parallel:
            source_feature = F.relu(model.lin1(source_x_backbone))
            target_feature = F.relu(model.lin1(target_x_backbone))
        else:
            source_feature = F.relu(model.lin1(source_data.x))
            target_feature = F.relu(model.lin1(target_data.x))
        mmd_loss = MMD(source_feature, target_feature)
        entropy_loss = entropy_minimization_loss(target_outputs)
        becat_loss = becat_extras.get("becat_loss")

        if model_parallel:
            if becat_loss is None:
                becat_loss = parallel_tensors["source_refined"].new_zeros(())
            total_loss = (
                source_loss
                + args.alpha * theta_loss.to(becat_device)
                + args.beta * mmd_loss.to(becat_device)
                + args.gamma * entropy_loss
                + args.lambda_geo * becat_loss
            )
        else:
            total_loss = (
                source_loss
                + args.alpha * theta_loss
                + args.beta * mmd_loss
                + args.gamma * entropy_loss
            )
            if args.use_becat and becat_loss is not None:
                total_loss = total_loss + args.lambda_geo * becat_loss

        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if model_parallel:
                source_val_output, _, _, _ = forward_model_parallel_pair(
                    model,
                    source_x_backbone,
                    target_x_backbone,
                    source_edge_backbone,
                    target_edge_backbone,
                    source_edge_becat,
                    target_edge_becat,
                    becat_device,
                )
                val_logits = source_val_output[source_val_mask_becat]
                val_labels = source_y_becat[source_val_mask_becat]
            elif args.use_becat:
                source_val_output, _, _ = model.forward_pair(
                    source_data, target_data
                )
                val_logits = source_val_output[source_data.val_mask]
                val_labels = source_data.y[source_data.val_mask]
            else:
                source_val_output = model(source_data)
                val_logits = source_val_output[source_data.val_mask]
                val_labels = source_data.y[source_data.val_mask]

            source_val_accuracy = (
                val_logits.argmax(dim=1).eq(val_labels).float().mean().item()
            )

        # Strict improvement keeps the earlier checkpoint on an exact tie.
        if source_val_accuracy > best_source_val_accuracy:
            best_epoch = epoch
            best_source_val_accuracy = source_val_accuracy
            best_becat_active = model.becat_active
            best_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        becat_log = ""
        if becat_loss is not None:
            becat_log = f" becat_loss={becat_loss.detach().item():.4f}"
        print(
            f"EPOCH epoch={epoch} source_ce={source_loss.detach().item():.4f} "
            f"theta_loss={theta_loss.detach().item():.4f} "
            f"mmd_loss={mmd_loss.detach().item():.4f} "
            f"target_entropy_loss={entropy_loss.detach().item():.4f}"
            f"{becat_log} source_val_accuracy={source_val_accuracy:.6f} "
            f"best_epoch={best_epoch} "
            f"best_source_val_accuracy={best_source_val_accuracy:.6f}"
        )

    if best_model_state is None:
        raise RuntimeError("No source-validation checkpoint was selected")
    model.load_state_dict(best_model_state)
    model.set_becat_active(best_becat_active)
    model.eval()

    # This is the sole target-ground-truth access in the executable. It occurs
    # only after source-validation checkpoint selection and restoration.
    target_labels = CitationDataset(
        target_path, args.target, split_seed=args.split_seed
    )[0].y
    if model_parallel:
        target_labels = target_labels.to(becat_device)
        source_labels = source_y_becat
        source_val_mask = source_val_mask_becat
        source_test_mask = source_test_mask_becat
    else:
        target_labels = target_labels.to(target_data.x.device)
        source_labels = source_data.y
        source_val_mask = source_data.val_mask
        source_test_mask = source_data.test_mask

    with torch.no_grad():
        if model_parallel:
            final_source_output, final_target_output, _, _ = (
                forward_model_parallel_pair(
                    model,
                    source_x_backbone,
                    target_x_backbone,
                    source_edge_backbone,
                    target_edge_backbone,
                    source_edge_becat,
                    target_edge_becat,
                    becat_device,
                )
            )
        elif args.use_becat:
            final_source_output, final_target_output, _ = model.forward_pair(
                source_data, target_data
            )
        else:
            final_source_output = model(source_data)
            final_target_output = model(target_data, False)

    source_val_metrics = compute_classification_metrics(
        final_source_output, source_labels, source_val_mask
    )
    source_test_metrics = compute_classification_metrics(
        final_source_output, source_labels, source_test_mask
    )
    target_metrics = compute_classification_metrics(
        final_target_output, target_labels
    )

    print("Theta_1:", torch.relu(model.prop1.temp.detach().cpu()).tolist())
    print("Theta_2:", torch.relu(model.prop2.temp.detach().cpu()).tolist())
    print("Theta_3:", torch.relu(model.prop3.temp.detach().cpu()).tolist())
    print(
        f"FINAL_RESULT,source={args.source},target={args.target},seed={args.seed},"
        f"split_seed={args.split_seed},best_epoch={best_epoch},"
        f"best_source_val_accuracy={best_source_val_accuracy:.6f},"
        f"source_val_accuracy={source_val_metrics['accuracy']:.6f},"
        f"source_test_accuracy={source_test_metrics['accuracy']:.6f},"
        f"target_accuracy={target_metrics['accuracy']:.6f},"
        f"target_micro_f1={target_metrics['micro_f1']:.6f},"
        f"target_macro_f1={target_metrics['macro_f1']:.6f},"
        f"target_weighted_f1={target_metrics['weighted_f1']:.6f},"
        f"use_becat={str(args.use_becat).lower()}"
    )


if __name__ == "__main__":
    main()
