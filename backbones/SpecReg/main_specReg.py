"""Train SpecReg with leakage-free source-validation model selection.

Target labels are absent from the training graph and are read only after the
complete best source-validation checkpoint has been restored.
"""

from argparse import ArgumentParser
import itertools
from pathlib import Path
import random
import sys

RELEASE_ROOT = Path(__file__).resolve().parents[2]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
import torch.nn.functional as F

from becat.torch import GeometryAdapterConfig, LatentGeometryAdapter
from backbones.SpecReg import utils
from backbones.SpecReg.dual_gnn.cached_gcn_conv import CachedGCNConv
from backbones.SpecReg.dual_gnn.dataset.DomainData import DomainData
from backbones.SpecReg.dual_gnn.ppmi_conv import PPMIConv


def parse_args():
    parser = ArgumentParser(
        description="SpecReg citation baseline and BECAT integration"
    )
    parser.add_argument("--source", type=str, default="ACMv9")
    parser.add_argument("--target", type=str, default="DBLPv7")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="external Citation directory containing dataset subdirectories",
    )
    parser.add_argument("--name", type=str, default="UDAGCN")
    parser.add_argument("--UDAGCN", type=bool, default=True)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument(
        "--seed", type=int, default=1, help="model and training randomness"
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="source 80/10/10 split seed, independent of --seed",
    )
    parser.add_argument("--gamma_adv", type=float, default=0.1)
    parser.add_argument("--thr_smooth", type=float, default=-1)
    parser.add_argument("--gamma_smooth", type=float, default=0.01)
    parser.add_argument("--thr_mfr", type=float, default=-1)
    parser.add_argument("--gamma_mfr", type=float, default=0.01)
    parser.add_argument(
        "--use_becat",
        "--use_geometry_adapter",
        dest="use_becat",
        action="store_true",
        help="refine classifier-path SpecReg embeddings with BECAT",
    )
    parser.add_argument("--lambda_latent_bec", type=float, default=0.001)
    parser.add_argument("--lambda_latent_bec_stat", type=float, default=0.1)
    parser.add_argument("--latent_topk_k", type=int, default=10)
    parser.add_argument("--latent_topk_tau", type=float, default=0.5)
    parser.add_argument("--latent_bec_sample_size", type=int, default=128)
    parser.add_argument("--latent_bec_use_mmd", action="store_true")
    parser.add_argument("--latent_bec_mmd_sigma", type=float, default=1.0)
    parser.add_argument("--sparse_gt_attention_k", type=int, default=10)
    parser.add_argument(
        "--sparse_gt_curv_bias_beta", type=float, default=0.05
    )
    parser.add_argument("--sparse_gt_residual_alpha", type=float, default=0.1)
    parser.add_argument("--sparse_gt_num_heads", type=int, default=2)
    parser.add_argument("--sparse_gt_num_layers", type=int, default=1)
    parser.add_argument("--sparse_gt_dropout", type=float, default=0.3)
    parser.add_argument(
        "--sparse_gt_curv_bias_mode",
        choices=("abs", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--sparse_gt_bias_norm",
        choices=("none", "scale", "zscore"),
        default="scale",
    )
    parser.add_argument("--sparse_gt_fourier_m", type=int, default=4)
    parser.add_argument("--sparse_gt_fourier_sigma", type=float, default=1.0)
    parser.add_argument("--sparse_gt_fourier_seed", type=int, default=1234)
    parser.add_argument("--sparse_gt_eta_abs", type=float, default=1.0)
    parser.add_argument("--sparse_gt_eta_fourier", type=float, default=0.1)
    return parser.parse_args()


def seed_training(seed):
    """Seed training RNGs; source splitting has its own CPU generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parse_args()
use_UDAGCN = args.UDAGCN
encoder_dim = args.encoder_dim
use_becat = args.use_becat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if use_becat:
    if args.sparse_gt_num_heads <= 0:
        raise ValueError("sparse_gt_num_heads must be positive.")
    if encoder_dim % args.sparse_gt_num_heads != 0:
        raise ValueError(
            "BECAT attention-head count must divide encoder_dim: "
            f"sparse_gt_num_heads={args.sparse_gt_num_heads}, "
            f"encoder_dim={encoder_dim}."
        )

seed_training(args.seed)

print(
    "PROTOCOL,"
    f"source={args.source},target={args.target},seed={args.seed},"
    f"split_seed={args.split_seed},source_split=80/10/10,"
    "checkpoint=source_validation_accuracy,"
    f"use_becat={str(use_becat).lower()}"
)
if use_becat:
    print(
        "BECAT_CONFIG,"
        f"lambda_latent_bec={args.lambda_latent_bec},"
        f"lambda_latent_bec_stat={args.lambda_latent_bec_stat},"
        f"latent_topk_k={args.latent_topk_k},"
        f"latent_topk_tau={args.latent_topk_tau},"
        f"latent_bec_sample_size={args.latent_bec_sample_size},"
        f"sparse_gt_attention_k={args.sparse_gt_attention_k},"
        f"sparse_gt_curv_bias_beta={args.sparse_gt_curv_bias_beta},"
        f"sparse_gt_residual_alpha={args.sparse_gt_residual_alpha}"
    )

source_dataset = DomainData(
    str(args.data_root / args.source),
    name=args.source,
    split_seed=args.split_seed,
    load_labels=True,
)
target_dataset = DomainData(
    str(args.data_root / args.target),
    name=args.target,
    load_labels=False,
)
source_data = source_dataset[0]
target_data = target_dataset[0]

print(
    f"Source dataset: name={args.source},nodes={source_data.num_nodes},"
    f"features={source_dataset.num_features},"
    f"classes={source_dataset.num_classes}"
)
print(
    f"Target graph (unlabeled): name={args.target},"
    f"nodes={target_data.num_nodes},features={target_dataset.num_features}"
)
print(
    "SOURCE_SPLIT,"
    f"train={int(source_data.train_mask.sum())},"
    f"validation={int(source_data.val_mask.sum())},"
    f"test={int(source_data.test_mask.sum())},"
    f"split_seed={args.split_seed}"
)

if source_dataset.num_features != target_dataset.num_features:
    raise ValueError(
        "Feature dimension mismatch: "
        f"{args.source} has {source_dataset.num_features}, "
        f"{args.target} has {target_dataset.num_features}."
    )

input_dim = source_dataset.num_features
num_classes = source_dataset.num_classes
eival_src = source_dataset.eival.to(device)
eivec_src = source_dataset.eivec.to(device)
eival_tgt = target_dataset.eival.to(device)
eivec_tgt = target_dataset.eivec.to(device)
source_data = source_data.to(device)
target_data = target_data.to(device)


class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
        super().__init__()
        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [layer.weight for layer in base_model.conv_layers]
            biases = [layer.bias for layer in base_model.conv_layers]

        # Register dropout modules so model.eval() makes checkpoint selection
        # and restored inference deterministic without changing training.
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(0.1) for _ in weights]
        )
        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv
        self.conv_layers = nn.ModuleList(
            [
                model_cls(
                    input_dim,
                    128,
                    weight=weights[0],
                    bias=biases[0],
                    **kwargs,
                ),
                model_cls(
                    128,
                    encoder_dim,
                    weight=weights[1],
                    bias=biases[1],
                    **kwargs,
                ),
            ]
        )

    def forward(self, x, edge_index, cache_name):
        for layer_index, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if layer_index < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[layer_index](x)
        return x


rate = 0.0


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * rate, None


class GRL(nn.Module):
    def forward(self, input_tensor):
        return GradReverse.apply(input_tensor)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        return torch.sum(stacked * weights, dim=1)


loss_func = nn.CrossEntropyLoss().to(device)
encoder = GNN(type="gcn").to(device)
ppmi_encoder = None
if use_UDAGCN:
    ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=10).to(device)

cls_model = nn.Sequential(nn.Linear(encoder_dim, num_classes)).to(device)
domain_model = nn.Sequential(
    GRL(),
    nn.Linear(encoder_dim, 40),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(40, 2),
).to(device)
att_model = Attention(encoder_dim).to(device)

models = [encoder, cls_model, domain_model]
if use_UDAGCN:
    models.extend([ppmi_encoder, att_model])

becat_model = None
if use_becat:
    becat_config = GeometryAdapterConfig(
        latent_topk_k=args.latent_topk_k,
        latent_topk_tau=args.latent_topk_tau,
        lambda_latent_bec=args.lambda_latent_bec,
        lambda_latent_bec_stat=args.lambda_latent_bec_stat,
        latent_bec_use_mmd=args.latent_bec_use_mmd,
        latent_bec_sample_size=args.latent_bec_sample_size,
        latent_bec_mmd_sigma=args.latent_bec_mmd_sigma,
        sparse_gt_num_heads=args.sparse_gt_num_heads,
        sparse_gt_num_layers=args.sparse_gt_num_layers,
        sparse_gt_dropout=args.sparse_gt_dropout,
        sparse_gt_curv_bias_beta=args.sparse_gt_curv_bias_beta,
        sparse_gt_curv_bias_mode=args.sparse_gt_curv_bias_mode,
        sparse_gt_fourier_m=args.sparse_gt_fourier_m,
        sparse_gt_fourier_sigma=args.sparse_gt_fourier_sigma,
        sparse_gt_fourier_seed=args.sparse_gt_fourier_seed,
        sparse_gt_eta_abs=args.sparse_gt_eta_abs,
        sparse_gt_eta_fourier=args.sparse_gt_eta_fourier,
        sparse_gt_bias_norm=args.sparse_gt_bias_norm,
        sparse_gt_attention_k=args.sparse_gt_attention_k,
        sparse_gt_residual_alpha=args.sparse_gt_residual_alpha,
    )
    becat_model = LatentGeometryAdapter(becat_config).to(device)

all_models = models + ([becat_model] if becat_model is not None else [])
critic = nn.Sequential(
    nn.Linear(args.encoder_dim, args.encoder_dim),
    nn.ReLU(),
    nn.Linear(args.encoder_dim, args.encoder_dim),
    nn.ReLU(),
    nn.Linear(args.encoder_dim, 1),
).to(device)
relu = nn.ReLU()


def gcn_encode(data, cache_name, mask=None):
    output = encoder(data.x, data.edge_index, cache_name)
    return output if mask is None else output[mask]


def ppmi_encode(data, cache_name, mask=None):
    output = ppmi_encoder(data.x, data.edge_index, cache_name)
    return output if mask is None else output[mask]


def encode(data, cache_name, mask=None):
    gcn_output = gcn_encode(data, cache_name, mask)
    if use_UDAGCN:
        return att_model([gcn_output, ppmi_encode(data, cache_name, mask)])
    return gcn_output


def forward_source_target(training):
    source_raw_embedding = encode(source_data, "source")
    target_raw_embedding = encode(target_data, "target")

    if becat_model is not None:
        (
            source_refined_embedding,
            target_refined_embedding,
            geometry_loss,
            geometry_diagnostics,
        ) = becat_model(
            source_raw_embedding,
            target_raw_embedding,
            training=training,
        )
    else:
        source_refined_embedding = source_raw_embedding
        target_refined_embedding = target_raw_embedding
        geometry_loss = source_raw_embedding.new_zeros(())
        geometry_diagnostics = {}

    if source_raw_embedding.shape != source_refined_embedding.shape:
        raise RuntimeError("BECAT changed the source embedding shape.")
    if target_raw_embedding.shape != target_refined_embedding.shape:
        raise RuntimeError("BECAT changed the target embedding shape.")

    return {
        "source_raw_embedding": source_raw_embedding,
        "target_raw_embedding": target_raw_embedding,
        "source_refined_embedding": source_refined_embedding,
        "target_refined_embedding": target_refined_embedding,
        "source_logits": cls_model(source_refined_embedding),
        "target_logits": cls_model(target_refined_embedding),
        "geometry_loss": geometry_loss,
        "geometry_diagnostics": geometry_diagnostics,
    }


# BECAT creates its transformer parameters lazily. Initialize them with the
# unlabeled graph pair before constructing the optimizer, exactly as in the
# supplied integration.
if becat_model is not None:
    previous_states = [model.training for model in all_models]
    for model in all_models:
        model.eval()
    with torch.no_grad():
        initial_outputs = forward_source_target(training=False)
    for model, was_training in zip(all_models, previous_states):
        model.train(was_training)
    becat_parameter_count = sum(
        parameter.numel()
        for parameter in becat_model.parameters()
        if parameter.requires_grad
    )
    if becat_parameter_count <= 0:
        raise RuntimeError("BECAT is enabled but has no trainable parameters.")
    print(f"BECAT trainable parameters: {becat_parameter_count}")

params = itertools.chain(*[model.parameters() for model in all_models])
optimizer = torch.optim.Adam(params, lr=3e-3)
optimizer_critic = torch.optim.Adam(critic.parameters(), 1e-4)

if becat_model is not None:
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing = [
        name
        for name, parameter in becat_model.named_parameters()
        if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids
    ]
    if missing:
        raise RuntimeError(
            "BECAT parameters are missing from the optimizer: " + ", ".join(missing)
        )


def compute_metrics(predictions, labels):
    y_pred = predictions.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
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


def set_evaluation_mode():
    for model in all_models:
        model.eval()
    critic.eval()


def evaluate_source_validation():
    """Evaluate source validation only; target labels do not exist here."""
    set_evaluation_mode()
    with torch.no_grad():
        outputs = forward_source_target(training=False)
        predictions = outputs["source_logits"][source_data.val_mask].argmax(dim=1)
        labels = source_data.y[source_data.val_mask]
    return compute_metrics(predictions, labels)


epochs = 200


def train(epoch):
    for model in all_models:
        model.train()
    critic.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, 0.05)

    outputs = forward_source_target(training=True)
    source_raw_embedding = outputs["source_raw_embedding"]
    target_raw_embedding = outputs["target_raw_embedding"]
    source_logits = outputs["source_logits"]
    target_logits = outputs["target_logits"]

    # The supervised task loss uses source training nodes only.
    cls_loss = loss_func(
        source_logits[source_data.train_mask],
        source_data.y[source_data.train_mask],
    )
    for model in models:
        for name, parameter in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + parameter.mean() * 3e-3

    if use_UDAGCN:
        # Original domain-adversarial optimization uses raw fused embeddings.
        detached_source = source_raw_embedding.detach()
        detached_target = target_raw_embedding.detach()
        for _ in range(5):
            optimizer_critic.zero_grad()
            wasserstein = (
                critic(detached_source).mean() - critic(detached_target).mean()
            )
            gradient_penalty = utils.calculate_gradient_penalty(
                critic, detached_source, detached_target
            )
            critic_loss = -wasserstein + 10 * gradient_penalty
            critic_loss.backward()
            optimizer_critic.step()

        loss_grl = (
            critic(source_raw_embedding).mean()
            - critic(target_raw_embedding).mean()
        )
        loss = cls_loss + loss_grl * args.gamma_adv

        # Original SSReg/MFRReg calculations remain on raw embeddings.
        x_src = torch.einsum("nm,md->nd", eivec_src, source_raw_embedding)
        x_tgt = torch.einsum("nm,md->nd", eivec_tgt, target_raw_embedding)
        if args.thr_smooth > 0:
            delta_src = (x_src[:-1] - x_src[1:]).abs()
            delta_tgt = (x_tgt[:-1] - x_tgt[1:]).abs()
            loss = loss + (
                relu(delta_src - args.thr_smooth).mean()
                + relu(delta_tgt - args.thr_smooth).mean()
            ) * args.gamma_smooth
        if args.thr_mfr > 0:
            loss = loss + (
                relu(x_src.abs() - args.thr_mfr).mean()
                + relu(x_tgt.abs() - args.thr_mfr).mean()
            ) * args.gamma_mfr

        # The target objective uses predictions from the classifier path only.
        target_probs = F.softmax(target_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
        target_entropy = torch.mean(
            torch.sum(-target_probs * torch.log(target_probs), dim=-1)
        )
        loss = loss + target_entropy * (epoch / epochs * 0.01)
    else:
        loss = cls_loss

    # Canonical BECAT already applies lambda_latent_bec internally.
    loss = loss + outputs["geometry_loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


checkpoint_modules = {
    "encoder": encoder,
    "classifier": cls_model,
    "domain_model": domain_model,
    "critic": critic,
}
if use_UDAGCN:
    checkpoint_modules["ppmi_encoder"] = ppmi_encoder
    checkpoint_modules["attention"] = att_model
if becat_model is not None:
    checkpoint_modules["becat"] = becat_model


def clone_checkpoint_to_cpu():
    return {
        module_name: {
            tensor_name: tensor.detach().cpu().clone()
            for tensor_name, tensor in module.state_dict().items()
        }
        for module_name, module in checkpoint_modules.items()
    }


def restore_checkpoint(checkpoint):
    for module_name, module in checkpoint_modules.items():
        module.load_state_dict(checkpoint[module_name], strict=True)


best_source_val_metrics = None
best_checkpoint = None
best_epoch = None
for epoch in range(1, epochs):
    train(epoch)
    source_val_metrics = evaluate_source_validation()
    improved = (
        best_source_val_metrics is None
        or source_val_metrics["accuracy"]
        > best_source_val_metrics["accuracy"]
    )
    if improved:
        best_source_val_metrics = source_val_metrics.copy()
        best_checkpoint = clone_checkpoint_to_cpu()
        best_epoch = epoch
    print(
        f"Epoch: {epoch}, "
        f"source_val_accuracy: {source_val_metrics['accuracy']:.6f}, "
        f"source_val_micro_f1: {source_val_metrics['micro_f1']:.6f}, "
        f"source_val_macro_f1: {source_val_metrics['macro_f1']:.6f}, "
        f"checkpoint_updated: {str(improved).lower()}"
    )

if best_checkpoint is None:
    raise RuntimeError("Training completed without a valid source checkpoint.")

# Restore first, then recompute source diagnostics. Target labels still have
# not been read and are not present in target_data.
restore_checkpoint(best_checkpoint)
set_evaluation_mode()
with torch.no_grad():
    restored_outputs = forward_source_target(training=False)
    source_val_predictions = restored_outputs["source_logits"][
        source_data.val_mask
    ].argmax(dim=1)
    source_test_predictions = restored_outputs["source_logits"][
        source_data.test_mask
    ].argmax(dim=1)
    source_val_labels = source_data.y[source_data.val_mask]
    source_test_labels = source_data.y[source_data.test_mask]
    final_target_predictions = restored_outputs["target_logits"].argmax(
        dim=1
    ).detach().cpu()

source_val_metrics = compute_metrics(source_val_predictions, source_val_labels)
source_test_metrics = compute_metrics(source_test_predictions, source_test_labels)

# This is the only target-label read. It occurs after complete restoration and
# after restored source diagnostics have been recomputed.
final_target_labels = target_dataset.load_evaluation_labels(
    expected_num_classes=num_classes
)
target_metrics = compute_metrics(final_target_predictions, final_target_labels)

print("=" * 61)
print(
    "FINAL_RESULT,"
    f"source={args.source},"
    f"target={args.target},"
    f"seed={args.seed},"
    f"split_seed={args.split_seed},"
    f"best_epoch={best_epoch},"
    f"best_source_val_accuracy={best_source_val_metrics['accuracy']:.6f},"
    f"source_val_accuracy={source_val_metrics['accuracy']:.6f},"
    f"source_val_micro_f1={source_val_metrics['micro_f1']:.6f},"
    f"source_val_macro_f1={source_val_metrics['macro_f1']:.6f},"
    f"source_test_accuracy={source_test_metrics['accuracy']:.6f},"
    f"source_test_micro_f1={source_test_metrics['micro_f1']:.6f},"
    f"source_test_macro_f1={source_test_metrics['macro_f1']:.6f},"
    f"target_accuracy={target_metrics['accuracy']:.6f},"
    f"target_micro_f1={target_metrics['micro_f1']:.6f},"
    f"target_macro_f1={target_metrics['macro_f1']:.6f},"
    f"target_weighted_f1={target_metrics['weighted_f1']:.6f},"
    f"use_becat={str(use_becat).lower()}"
)
