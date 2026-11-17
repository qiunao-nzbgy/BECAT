"""Release entrypoint for the six UDAGCN citation-network experiments."""

import argparse
import builtins
import copy
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
import torch.nn.functional as F

from becat.torch import GeometryAdapterConfig, LatentGeometryAdapter
from .dual_gnn.cached_gcn_conv import CachedGCNConv
from .dual_gnn.dataset.DomainData import (
    CITATION_DOMAINS,
    DomainData,
    apply_fixed_split,
    load_target_labels_for_final_evaluation,
)
from .dual_gnn.ppmi_conv import PPMIConv


_grl_rate = 0.0


def _str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def build_parser():
    parser = argparse.ArgumentParser(
        description="UDAGCN citation backbone with optional shared BECAT"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--source", choices=CITATION_DOMAINS, default="ACMv9")
    parser.add_argument("--target", choices=CITATION_DOMAINS, default="DBLPv7")
    parser.add_argument("--name", type=str, default="UDAGCN")
    parser.add_argument("--seed", type=int, default=200, help="training seed")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="source split seed, independent from --seed",
    )

    # Preserve original UDAGCN model and optimization defaults.
    parser.add_argument(
        "--UDAGCN",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)

    # Release-facing BECAT names retain the old geometry aliases.
    parser.add_argument(
        "--use_becat",
        "--use_geometry_adapter",
        dest="use_becat",
        action="store_true",
    )
    parser.add_argument(
        "--lambda_becat",
        "--lambda_geometry",
        dest="lambda_becat",
        type=float,
        default=0.05,
    )
    parser.add_argument("--geo_topk", type=int, default=10)
    parser.add_argument("--geo_tau", type=float, default=0.5)
    parser.add_argument("--geo_beta", type=float, default=0.05)
    parser.add_argument("--geo_residual_alpha", type=float, default=0.1)
    parser.add_argument("--geo_lambda_stat", type=float, default=0.1)
    parser.add_argument("--geo_warmup", type=int, default=0)
    return parser


def seed_training(seed):
    """Seed model/training RNGs without touching the source split generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class GNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_dim,
        base_model=None,
        type="gcn",
        **kwargs
    ):
        super().__init__()
        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = nn.ModuleList([nn.Dropout(0.1) for _ in weights])
        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv
        self.conv_layers = nn.ModuleList(
            [
                model_cls(
                    input_dim,
                    128,
                    weight=weights[0],
                    bias=biases[0],
                    **kwargs
                ),
                model_cls(
                    128,
                    encoder_dim,
                    weight=weights[1],
                    bias=biases[1],
                    **kwargs
                ),
            ]
        )

    def forward(self, x, edge_index, cache_name):
        for index, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if index < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[index](x)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * _grl_rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        return torch.sum(stacked * weights, dim=1)


def classification_metrics(logits, labels):
    """Compute accuracy and F1 scores from the same predicted classes."""
    predictions = logits.argmax(dim=1).detach().cpu()
    labels = labels.detach().cpu()
    if labels.numel() == 0:
        raise ValueError("evaluation selected no nodes")
    predictions_array = predictions.numpy()
    labels_array = labels.numpy()
    return {
        "accuracy": float(predictions.eq(labels).float().mean().item()),
        "micro_f1": float(
            f1_score(
                labels_array,
                predictions_array,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels_array,
                predictions_array,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels_array,
                predictions_array,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def run(args):
    if args.source == args.target:
        raise ValueError("source and target domains must differ")
    if args.epochs < 2:
        raise ValueError("--epochs must be at least 2")

    seed_training(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_udagcn = args.UDAGCN
    use_becat = args.use_becat

    source_root = os.path.join(args.data_root, args.source)
    target_root = os.path.join(args.data_root, args.target)
    for dataset_name, dataset_root in (
        (args.source, source_root),
        (args.target, target_root),
    ):
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(
                "Dataset directory for exact CLI name {!r} does not exist: {}. "
                "Dataset names and Linux paths are case-sensitive.".format(
                    dataset_name,
                    dataset_root,
                )
            )

    # Source labels are loaded now; target training data is graph-only.
    source_dataset = DomainData(
        root=source_root,
        name=args.source,
        load_labels=True,
    )
    target_dataset = DomainData(
        root=target_root,
        name=args.target,
        load_labels=False,
    )
    source_data = apply_fixed_split(
        source_dataset[0],
        split_seed=args.split_seed,
    )
    target_data = target_dataset[0]

    input_dim = int(source_dataset.num_features)
    num_classes = int(source_dataset.num_classes)
    if int(target_dataset.num_features) != input_dim:
        raise ValueError(
            "Incompatible transfer task: source={!r}, target={!r}, source "
            "feature dimension={}, target feature dimension={}. The shared "
            "encoder requires identical feature dimensions.".format(
                args.source,
                args.target,
                input_dim,
                int(target_dataset.num_features),
            )
        )
    if int(source_data.y.min()) < 0 or int(source_data.y.max()) >= num_classes:
        raise ValueError("source labels fall outside the classifier label space")
    if getattr(target_data, "y", None) is not None:
        raise RuntimeError("target training graph must not contain labels")

    source_mask_total = (
        source_data.train_mask.to(torch.int8)
        + source_data.val_mask.to(torch.int8)
        + source_data.test_mask.to(torch.int8)
    )
    if not bool(torch.all(source_mask_total == 1)):
        raise RuntimeError("source masks must be mutually exclusive and exhaustive")

    print(
        "source: {}, target: {}, seed: {}, split_seed: {}, UDAGCN: {}, "
        "encoder_dim: {}, use_becat: {}".format(
            args.source,
            args.target,
            args.seed,
            args.split_seed,
            use_udagcn,
            args.encoder_dim,
            use_becat,
        )
    )
    print("source raw directory: {}".format(source_dataset.raw_dir))
    print("target raw directory: {}".format(target_dataset.raw_dir))
    print("source x shape: {}".format(tuple(source_data.x.shape)))
    print("target x shape: {}".format(tuple(target_data.x.shape)))
    print("source edge_index shape: {}".format(tuple(source_data.edge_index.shape)))
    print("target edge_index shape: {}".format(tuple(target_data.edge_index.shape)))
    print("source num_classes: {}".format(num_classes))
    print("split_seed: {}".format(args.split_seed))
    print(
        "source masks: train={}, val={}, test={}".format(
            int(source_data.train_mask.sum()),
            int(source_data.val_mask.sum()),
            int(source_data.test_mask.sum()),
        )
    )
    if use_becat:
        print(
            "BECAT: enabled, lambda_becat: {}, geo_topk: {}, geo_tau: {}, "
            "geo_beta: {}, geo_residual_alpha: {}, geo_lambda_stat: {}, "
            "geo_warmup: {}".format(
                args.lambda_becat,
                args.geo_topk,
                args.geo_tau,
                args.geo_beta,
                args.geo_residual_alpha,
                args.geo_lambda_stat,
                args.geo_warmup,
            )
        )

    source_data = source_data.to(device)
    target_data = target_data.to(device)
    loss_func = nn.CrossEntropyLoss().to(device)

    encoder = GNN(input_dim, args.encoder_dim, type="gcn").to(device)
    ppmi_encoder = None
    att_model = None
    if use_udagcn:
        ppmi_encoder = GNN(
            input_dim,
            args.encoder_dim,
            base_model=encoder,
            type="ppmi",
            path_len=10,
        ).to(device)
        att_model = Attention(args.encoder_dim).to(device)

    cls_model = nn.Sequential(
        nn.Linear(args.encoder_dim, num_classes),
    ).to(device)
    domain_model = nn.Sequential(
        GRL(),
        nn.Linear(args.encoder_dim, 40),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(40, 2),
    ).to(device)

    models = [encoder, cls_model, domain_model]
    if use_udagcn:
        models.extend([ppmi_encoder, att_model])

    becat_adapter = None
    if use_becat:
        becat_config = GeometryAdapterConfig(
            latent_topk_k=args.geo_topk,
            latent_topk_tau=args.geo_tau,
            lambda_latent_bec_stat=args.geo_lambda_stat,
            sparse_gt_curv_bias_beta=args.geo_beta,
            sparse_gt_attention_k=args.geo_topk,
            sparse_gt_residual_alpha=args.geo_residual_alpha,
        )
        becat_adapter = LatentGeometryAdapter(becat_config).to(device)
        # Preserve the established lazy-parameter initialization before Adam.
        with torch.no_grad():
            becat_init = torch.zeros((1, args.encoder_dim), device=device)
            becat_adapter(becat_init, becat_init, training=False)

    optimizer_models = list(models)
    if becat_adapter is not None:
        optimizer_models.append(becat_adapter)
    params = []
    seen_param_ids = set()
    for model in optimizer_models:
        for parameter in model.parameters():
            parameter_id = builtins.id(parameter)
            if parameter_id not in seen_param_ids:
                params.append(parameter)
                seen_param_ids.add(parameter_id)
    optimizer = torch.optim.Adam(params, lr=3e-3)

    def encode(data, cache_name):
        gcn_output = encoder(data.x, data.edge_index, cache_name)
        if not use_udagcn:
            return gcn_output
        ppmi_output = ppmi_encoder(data.x, data.edge_index, cache_name)
        return att_model([gcn_output, ppmi_output])

    def set_becat_runtime(epoch):
        if becat_adapter is None:
            return
        geometry_active = epoch > args.geo_warmup
        becat_adapter.config.loss_active = geometry_active
        becat_adapter.config.curvature_bias_active = geometry_active

    def adapt_embeddings(raw_source, raw_target, training):
        if becat_adapter is None:
            return raw_source, raw_target, raw_source.new_tensor(0.0)
        adapted_source, adapted_target, geometry_loss, _ = becat_adapter(
            raw_source,
            raw_target,
            training=training,
        )
        return adapted_source, adapted_target, geometry_loss

    def set_modules_mode(training, epoch):
        for model in models:
            model.train(training)
        if becat_adapter is not None:
            becat_adapter.train(training)
            set_becat_runtime(epoch)

    def source_validation_metrics(epoch):
        set_modules_mode(False, epoch)
        with torch.no_grad():
            raw_source = encode(source_data, "source")
            raw_target = encode(target_data, "target")
            adapted_source, _, _ = adapt_embeddings(
                raw_source,
                raw_target,
                training=False,
            )
            source_logits = cls_model(adapted_source)
        return classification_metrics(
            source_logits[source_data.val_mask],
            source_data.y[source_data.val_mask],
        )

    def train_epoch(epoch):
        set_modules_mode(True, epoch)
        optimizer.zero_grad()

        global _grl_rate
        _grl_rate = min((epoch + 1) / args.epochs, 0.05)

        raw_source = encode(source_data, "source")
        raw_target = encode(target_data, "target")
        adapted_source, adapted_target, geometry_loss = adapt_embeddings(
            raw_source,
            raw_target,
            training=True,
        )
        source_logits = cls_model(adapted_source)

        # Source validation/test labels are excluded from every gradient update.
        classification_loss = loss_func(
            source_logits[source_data.train_mask],
            source_data.y[source_data.train_mask],
        )
        for model in models:
            for name, parameter in model.named_parameters():
                if "weight" in name:
                    classification_loss = (
                        classification_loss + parameter.mean() * 3e-3
                    )

        domain_loss = raw_source.new_tensor(0.0)
        target_entropy_loss = raw_source.new_tensor(0.0)
        if use_udagcn:
            # The original domain discriminator remains on raw H_s and H_t.
            source_domain_predictions = domain_model(raw_source)
            target_domain_predictions = domain_model(raw_target)
            source_domain_loss = loss_func(
                source_domain_predictions,
                torch.zeros(source_domain_predictions.size(0))
                .type(torch.LongTensor)
                .to(device),
            )
            target_domain_loss = loss_func(
                target_domain_predictions,
                torch.ones(target_domain_predictions.size(0))
                .type(torch.LongTensor)
                .to(device),
            )
            domain_loss = source_domain_loss + target_domain_loss
            total_loss = classification_loss + domain_loss

            # Unlabeled target entropy uses refined classifier inputs with BECAT.
            target_logits = cls_model(adapted_target)
            target_probabilities = F.softmax(target_logits, dim=-1)
            target_probabilities = torch.clamp(
                target_probabilities,
                min=1e-9,
                max=1.0,
            )
            target_entropy_loss = torch.mean(
                torch.sum(
                    -target_probabilities * torch.log(target_probabilities),
                    dim=-1,
                )
            )
            total_loss = total_loss + target_entropy_loss * (
                epoch / args.epochs * 0.01
            )
        else:
            total_loss = classification_loss

        if becat_adapter is not None:
            total_loss = total_loss + args.lambda_becat * geometry_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return {
            "raw_source_shape": tuple(raw_source.shape),
            "raw_target_shape": tuple(raw_target.shape),
            "adapted_source_shape": tuple(adapted_source.shape),
            "adapted_target_shape": tuple(adapted_target.shape),
            "classification_loss": classification_loss.detach().item(),
            "domain_loss": domain_loss.detach().item(),
            "target_entropy_loss": target_entropy_loss.detach().item(),
            "geometry_loss": geometry_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
        }

    checkpoint_modules = {
        "encoder": encoder,
        "classifier": cls_model,
        "domain_model": domain_model,
    }
    if use_udagcn:
        checkpoint_modules["ppmi_encoder"] = ppmi_encoder
        checkpoint_modules["attention_model"] = att_model
    if becat_adapter is not None:
        checkpoint_modules["becat_adapter"] = becat_adapter

    def capture_checkpoint():
        return {
            name: copy.deepcopy(model.state_dict())
            for name, model in checkpoint_modules.items()
        }

    def restore_checkpoint(checkpoint):
        if set(checkpoint) != set(checkpoint_modules):
            raise RuntimeError("best checkpoint does not match trainable modules")
        for name, model in checkpoint_modules.items():
            model.load_state_dict(checkpoint[name])

    best_source_val_accuracy = -1.0
    best_epoch = None
    best_checkpoint = None
    for epoch in range(1, args.epochs):
        train_metrics = train_epoch(epoch)
        validation_metrics = source_validation_metrics(epoch)
        if epoch == 1:
            print(
                "source raw embedding shape: {}".format(
                    train_metrics["raw_source_shape"]
                )
            )
            print(
                "target raw embedding shape: {}".format(
                    train_metrics["raw_target_shape"]
                )
            )
            print(
                "source adapted embedding shape: {}".format(
                    train_metrics["adapted_source_shape"]
                )
            )
            print(
                "target adapted embedding shape: {}".format(
                    train_metrics["adapted_target_shape"]
                )
            )
        print(
            "Epoch: {}, source_val_accuracy: {:.6f}, "
            "source_val_micro_f1: {:.6f}, source_val_macro_f1: {:.6f}, "
            "classification_loss: {:.6f}, domain_loss: {:.6f}, "
            "target_entropy_loss: {:.6f}, geometry_loss: {:.6f}, "
            "total_loss: {:.6f}".format(
                epoch,
                validation_metrics["accuracy"],
                validation_metrics["micro_f1"],
                validation_metrics["macro_f1"],
                train_metrics["classification_loss"],
                train_metrics["domain_loss"],
                train_metrics["target_entropy_loss"],
                train_metrics["geometry_loss"],
                train_metrics["total_loss"],
            )
        )

        source_val_accuracy = validation_metrics["accuracy"]
        if source_val_accuracy > best_source_val_accuracy:
            best_source_val_accuracy = source_val_accuracy
            best_epoch = epoch
            best_checkpoint = capture_checkpoint()

    if best_checkpoint is None:
        raise RuntimeError("training produced no source-validation checkpoint")

    # Restore every trainable component before the first target-label read.
    restore_checkpoint(best_checkpoint)
    set_modules_mode(False, best_epoch)

    # This is intentionally the first and only target-label read.
    target_labels = load_target_labels_for_final_evaluation(
        root=target_root,
        name=args.target,
        expected_num_nodes=target_data.num_nodes,
        num_classes=num_classes,
    )

    with torch.no_grad():
        raw_source = encode(source_data, "source")
        raw_target = encode(target_data, "target")
        adapted_source, adapted_target, _ = adapt_embeddings(
            raw_source,
            raw_target,
            training=False,
        )
        source_logits = cls_model(adapted_source)
        target_logits = cls_model(adapted_target)

    source_val_metrics = classification_metrics(
        source_logits[source_data.val_mask],
        source_data.y[source_data.val_mask],
    )
    source_test_metrics = classification_metrics(
        source_logits[source_data.test_mask],
        source_data.y[source_data.test_mask],
    )
    # No target mask: final metrics cover every target node in raw-file order.
    target_metrics = classification_metrics(target_logits, target_labels)

    fields = [
        ("source", args.source),
        ("target", args.target),
        ("seed", args.seed),
        ("split_seed", args.split_seed),
        ("best_epoch", best_epoch),
        (
            "best_source_val_accuracy",
            "{:.6f}".format(best_source_val_accuracy),
        ),
        (
            "source_val_accuracy",
            "{:.6f}".format(source_val_metrics["accuracy"]),
        ),
        (
            "source_test_accuracy",
            "{:.6f}".format(source_test_metrics["accuracy"]),
        ),
        ("target_accuracy", "{:.6f}".format(target_metrics["accuracy"])),
        ("target_micro_f1", "{:.6f}".format(target_metrics["micro_f1"])),
        ("target_macro_f1", "{:.6f}".format(target_metrics["macro_f1"])),
        (
            "target_weighted_f1",
            "{:.6f}".format(target_metrics["weighted_f1"]),
        ),
        ("use_becat", str(use_becat).lower()),
        (
            "source_val_micro_f1",
            "{:.6f}".format(source_val_metrics["micro_f1"]),
        ),
        (
            "source_val_macro_f1",
            "{:.6f}".format(source_val_metrics["macro_f1"]),
        ),
        (
            "source_test_micro_f1",
            "{:.6f}".format(source_test_metrics["micro_f1"]),
        ),
        (
            "source_test_macro_f1",
            "{:.6f}".format(source_test_metrics["macro_f1"]),
        ),
        ("source_train_count", int(source_data.train_mask.sum())),
        ("source_val_count", int(source_data.val_mask.sum())),
        ("source_test_count", int(source_data.test_mask.sum())),
        ("target_node_count", int(target_data.num_nodes)),
    ]
    if use_becat:
        fields.extend(
            [
                ("lambda_becat", args.lambda_becat),
                ("geo_topk", args.geo_topk),
                ("geo_tau", args.geo_tau),
                ("geo_beta", args.geo_beta),
                ("geo_residual_alpha", args.geo_residual_alpha),
                ("geo_lambda_stat", args.geo_lambda_stat),
                ("geo_warmup", args.geo_warmup),
            ]
        )
    print(
        "FINAL_RESULT,"
        + ",".join("{}={}".format(key, value) for key, value in fields)
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
