# coding=utf-8
import argparse
import itertools
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

RELEASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

from becat.torch import GeometryAdapterConfig, LatentGeometryAdapter

if __package__:
    from .GCN_model import Attention, GCN
    from .encode_model import GCNModelVAE, InnerProductDecoder
    from .utils import (
        load_adj_label_for_reconstruction,
        load_data_citation,
        load_target_labels,
        load_unlabeled_data_citation,
        split_source_indices,
    )
else:
    from GCN_model import Attention, GCN
    from encode_model import GCNModelVAE, InnerProductDecoder
    from utils import (
        load_adj_label_for_reconstruction,
        load_data_citation,
        load_target_labels,
        load_unlabeled_data_citation,
        split_source_indices,
    )

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=3e-2)
parser.add_argument("--cuda", type=str, default="0")
parser.add_argument("--n_epoch", type=int, default=2000)
parser.add_argument("--eval_interval", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--split_seed", type=int, default=42)
parser.add_argument(
    "--data_root",
    type=str,
    required=True,
    help="External directory containing acmv9.mat, citationv1.mat, and dblpv7.mat.",
)
parser.add_argument("--hidden", type=int, default=128, help="Number of hidden units.")
parser.add_argument("--gfeat", type=int, default=16, help="Fused ASN embedding dimension.")
parser.add_argument("--nfeat", type=int, default=6775, help="Input feature dimension.")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
parser.add_argument("--data_src", type=str, default="citationv1", help="Source dataset name.")
parser.add_argument("--data_trg", type=str, default="acmv9", help="Target dataset name.")
parser.add_argument("--classes", type=int, default=5, help="Number of classes.")
parser.add_argument("--model_path", type=str, default="models")
parser.add_argument("--lambda_d", type=float, default=0.5, help="Domain loss weight.")
parser.add_argument("--lambda_r", type=float, default=1, help="Reconstruction loss weight.")
parser.add_argument("--lambda_f", type=float, default=0.0001, help="Difference loss weight.")

parser.add_argument(
    "--use_becat",
    "--use_geometry_adapter",
    dest="use_becat",
    action="store_true",
    help="Enable the shared canonical BECAT adapter.",
)
parser.add_argument("--latent_topk_k", type=int, default=12)
parser.add_argument("--latent_topk_tau", type=float, default=0.5)
parser.add_argument("--lambda_latent_bec", type=float, default=0.001)
parser.add_argument("--lambda_latent_bec_stat", type=float, default=0.1)
parser.add_argument("--latent_bec_sample_size", type=int, default=128)
parser.add_argument("--sparse_gt_num_heads", type=int, default=2)
parser.add_argument("--sparse_gt_num_layers", type=int, default=1)
parser.add_argument("--sparse_gt_dropout", type=float, default=0.3)
parser.add_argument("--sparse_gt_curv_bias_beta", type=float, default=0.05)
parser.add_argument(
    "--sparse_gt_curv_bias_mode", type=str, default="hybrid", choices=("abs", "hybrid")
)
parser.add_argument("--sparse_gt_fourier_m", type=int, default=4)
parser.add_argument("--sparse_gt_fourier_sigma", type=float, default=1.0)
parser.add_argument("--sparse_gt_eta_abs", type=float, default=1.0)
parser.add_argument("--sparse_gt_eta_fourier", type=float, default=0.1)
parser.add_argument(
    "--sparse_gt_bias_norm", type=str, default="scale", choices=("none", "scale", "zscore")
)
parser.add_argument("--sparse_gt_attention_k", type=int, default=12)
parser.add_argument("--sparse_gt_residual_alpha", type=float, default=0.1)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * rate


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


class DiffLoss(nn.Module):
    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        return torch.mean((input1_l2.t().mm(input2_l2)).pow(2))


def recon_loss(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    kld = -0.5 / n_nodes * torch.mean(
        torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1)
    )
    return cost + kld


def evaluate_logits(logits, labels, idx=None):
    """Return accuracy and F1 metrics for an arbitrary node subset."""
    if idx is not None:
        logits = logits.index_select(0, idx.to(logits.device))
        labels = labels.index_select(0, idx.to(labels.device))
    pred = logits.argmax(dim=1)
    y_true = labels.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def encode_shared(feature, adj, ppmi):
    """Return the raw attention-fused ASN backbone embedding H."""
    _, basic_encoded, _ = shared_encoder_l(feature, adj)
    _, ppmi_encoded, _ = shared_encoder_g(feature, ppmi)
    return att_model([basic_encoded, ppmi_encoded])


def predict_pair(feature_s, adj_s, ppmi_s, feature_t, adj_t, ppmi_t):
    """Encode both domains jointly, adapting only the classifier branch."""
    encoded_source = encode_shared(feature_s, adj_s, ppmi_s)
    encoded_target = encode_shared(feature_t, adj_t, ppmi_t)
    if geometry_adapter is not None:
        adapted_source, adapted_target, _, _ = geometry_adapter(
            encoded_source,
            encoded_target,
            training=False,
            compute_geometry_loss=False,
        )
    else:
        adapted_source = encoded_source
        adapted_target = encoded_target
    return cls_model(adapted_source), cls_model(adapted_target)


def snapshot_components(components):
    """Clone every module state to CPU for a safe in-memory checkpoint."""
    return {
        component_name: {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        for component_name, model in components.items()
    }


def restore_components(components, checkpoint):
    for component_name, model in components.items():
        model.load_state_dict(checkpoint[component_name])


def metric_fields(prefix, metrics):
    return [
        "{}_{}={:.10f}".format(prefix, metric_name, metrics[metric_name])
        for metric_name in ("accuracy", "micro_f1", "macro_f1", "weighted_f1")
    ]


# Full graphs remain available to every transductive and unsupervised branch.
# Only the labeled source loader deserializes group before training.
adj_s, features_s, labels_s, num_nodes_s, X_n_s = load_data_citation(
    data_root=args.data_root,
    dataset=args.data_src,
)
adj_t, features_t, num_nodes_t, X_n_t = load_unlabeled_data_citation(
    data_root=args.data_root,
    dataset=args.data_trg,
)
idx_train_s, idx_val_s, idx_test_s = split_source_indices(
    num_nodes_s,
    args.split_seed,
)

adj_label_s, pos_weight_s, norm_s = load_adj_label_for_reconstruction(
    args.data_root,
    args.data_src,
)
adj_label_t, pos_weight_t, norm_t = load_adj_label_for_reconstruction(
    args.data_root,
    args.data_trg,
)

adj_s = adj_s.to(device)
adj_t = adj_t.to(device)
features_s = features_s.to(device)
features_t = features_t.to(device)
labels_s = labels_s.to(device)
X_n_s = X_n_s.to(device)
X_n_t = X_n_t.to(device)
idx_train_s = idx_train_s.to(device)
idx_val_s = idx_val_s.to(device)
idx_test_s = idx_test_s.to(device)
adj_label_s = adj_label_s.to(device=device, dtype=torch.float32)
adj_label_t = adj_label_t.to(device=device, dtype=torch.float32)
pos_weight_s = pos_weight_s.to(device=device, dtype=torch.float32)
pos_weight_t = pos_weight_t.to(device=device, dtype=torch.float32)

loss_diff = DiffLoss()
cls_loss = nn.CrossEntropyLoss().to(device)

private_encoder_s_l = GCNModelVAE(args.nfeat, args.hidden, args.gfeat, args.dropout).to(device)
private_encoder_t_l = GCNModelVAE(args.nfeat, args.hidden, args.gfeat, args.dropout).to(device)
private_encoder_s_g = GCNModelVAE(args.nfeat, args.hidden, args.gfeat, args.dropout).to(device)
private_encoder_t_g = GCNModelVAE(args.nfeat, args.hidden, args.gfeat, args.dropout).to(device)
decoder_s = InnerProductDecoder(dropout=args.dropout, act=lambda x: x).to(device)
decoder_t = InnerProductDecoder(dropout=args.dropout, act=lambda x: x).to(device)

shared_encoder_l = GCN(args.nfeat, args.hidden, args.gfeat, args.dropout).to(device)
shared_encoder_g = GCN(args.nfeat, args.hidden, args.gfeat, args.dropout).to(device)

cls_model = nn.Sequential(nn.Linear(args.gfeat, args.classes)).to(device)
domain_model = nn.Sequential(
    GRL(),
    nn.Linear(args.gfeat, 10),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(10, 2),
).to(device)

att_model = Attention(args.gfeat).to(device)
att_model_self_s = Attention(args.gfeat).to(device)
att_model_self_t = Attention(args.gfeat).to(device)

geometry_adapter = None
if args.use_becat:
    geometry_config = GeometryAdapterConfig(
        latent_topk_k=args.latent_topk_k,
        latent_topk_tau=args.latent_topk_tau,
        lambda_latent_bec=args.lambda_latent_bec,
        lambda_latent_bec_stat=args.lambda_latent_bec_stat,
        latent_bec_sample_size=args.latent_bec_sample_size,
        sparse_gt_num_heads=args.sparse_gt_num_heads,
        sparse_gt_num_layers=args.sparse_gt_num_layers,
        sparse_gt_dropout=args.sparse_gt_dropout,
        sparse_gt_curv_bias_beta=args.sparse_gt_curv_bias_beta,
        sparse_gt_curv_bias_mode=args.sparse_gt_curv_bias_mode,
        sparse_gt_fourier_m=args.sparse_gt_fourier_m,
        sparse_gt_fourier_sigma=args.sparse_gt_fourier_sigma,
        sparse_gt_eta_abs=args.sparse_gt_eta_abs,
        sparse_gt_eta_fourier=args.sparse_gt_eta_fourier,
        sparse_gt_bias_norm=args.sparse_gt_bias_norm,
        sparse_gt_attention_k=args.sparse_gt_attention_k,
        sparse_gt_residual_alpha=args.sparse_gt_residual_alpha,
    )
    geometry_adapter = LatentGeometryAdapter(geometry_config).to(device)
    geometry_adapter.initialize_parameters(args.gfeat, device=device, dtype=features_s.dtype)

components = {
    "private_encoder_s_l": private_encoder_s_l,
    "private_encoder_t_l": private_encoder_t_l,
    "private_encoder_s_g": private_encoder_s_g,
    "private_encoder_t_g": private_encoder_t_g,
    "shared_encoder_l": shared_encoder_l,
    "shared_encoder_g": shared_encoder_g,
    "cls_model": cls_model,
    "domain_model": domain_model,
    "att_model": att_model,
    "att_model_self_s": att_model_self_s,
    "att_model_self_t": att_model_self_t,
    "decoder_s": decoder_s,
    "decoder_t": decoder_t,
}
if geometry_adapter is not None:
    components["geometry_adapter"] = geometry_adapter

models = list(components.values())
params = itertools.chain.from_iterable(model.parameters() for model in models)
optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)

best_source_val_accuracy = -1.0
best_epoch = -1
best_checkpoint = None

for epoch in range(args.n_epoch):
    global rate
    rate = min((epoch + 1) / args.n_epoch, 0.05)

    for model in models:
        model.train()
    optimizer.zero_grad()

    recovered_s, mu_s, logvar_s = private_encoder_s_l(features_s, adj_s)
    recovered_t, mu_t, logvar_t = private_encoder_t_l(features_t, adj_t)
    recovered_s_p, mu_s_p, logvar_s_p = private_encoder_s_g(features_s, X_n_s)
    recovered_t_p, mu_t_p, logvar_t_p = private_encoder_t_g(features_t, X_n_t)

    z_s, shared_encoded_source1, shared_encoded_source2 = shared_encoder_l(features_s, adj_s)
    z_t, shared_encoded_target1, shared_encoded_target2 = shared_encoder_l(features_t, adj_t)
    z_s_p, ppmi_encoded_source, ppmi_encoded_source2 = shared_encoder_g(features_s, X_n_s)
    z_t_p, ppmi_encoded_target, ppmi_encoded_target2 = shared_encoder_g(features_t, X_n_t)

    # Raw final ASN backbone embeddings. These remain the domain-branch inputs.
    encoded_source = att_model([shared_encoded_source1, ppmi_encoded_source])
    encoded_target = att_model([shared_encoded_target1, ppmi_encoded_target])

    if geometry_adapter is not None:
        adapted_source, adapted_target, geometry_loss, becat_aux = geometry_adapter(
            encoded_source, encoded_target, training=True
        )
    else:
        adapted_source = encoded_source
        adapted_target = encoded_target
        geometry_loss = encoded_source.new_zeros(())
        becat_aux = {}

    # Original ASN difference branch: no BECAT tensors enter here.
    diff_loss_s = loss_diff(mu_s, shared_encoded_source1)
    diff_loss_t = loss_diff(mu_t, shared_encoded_target1)
    diff_loss_all = diff_loss_s + diff_loss_t

    # Original ASN reconstruction branch: no BECAT tensors enter here.
    z_cat_s = torch.cat(
        (att_model_self_s([recovered_s, recovered_s_p]), att_model_self_s([z_s, z_s_p])), 1
    )
    z_cat_t = torch.cat(
        (att_model_self_t([recovered_t, recovered_t_p]), att_model_self_t([z_t, z_t_p])), 1
    )
    recovered_cat_s = decoder_s(z_cat_s)
    recovered_cat_t = decoder_t(z_cat_t)
    mu_cat_s = torch.cat((mu_s, mu_s_p, shared_encoded_source1, ppmi_encoded_source), 1)
    mu_cat_t = torch.cat((mu_t, mu_t_p, shared_encoded_target1, ppmi_encoded_target), 1)
    logvar_cat_s = torch.cat(
        (logvar_s, logvar_s_p, shared_encoded_source2, ppmi_encoded_source2), 1
    )
    logvar_cat_t = torch.cat(
        (logvar_t, logvar_t_p, shared_encoded_target2, ppmi_encoded_target2), 1
    )
    recon_loss_s = recon_loss(
        recovered_cat_s, adj_label_s, mu_cat_s, logvar_cat_s, features_s.shape[0], norm_s, pos_weight_s
    )
    recon_loss_t = recon_loss(
        recovered_cat_t,
        adj_label_t,
        mu_cat_t,
        logvar_cat_t,
        features_t.shape[0] * 2,
        norm_t,
        pos_weight_t,
    )
    recon_loss_all = recon_loss_s + recon_loss_t

    # Prediction branch: BECAT output is used only when the adapter is enabled.
    source_logits = cls_model(adapted_source)
    target_logits = cls_model(adapted_target)
    cls_loss_source = cls_loss(
        source_logits.index_select(0, idx_train_s), labels_s.index_select(0, idx_train_s)
    )

    # Domain adversarial branch intentionally receives raw ASN embeddings.
    domain_output_s = domain_model(encoded_source)
    domain_output_t = domain_model(encoded_target)
    err_s_domain = cls_loss(
        domain_output_s, torch.zeros(domain_output_s.size(0), dtype=torch.long, device=device)
    )
    err_t_domain = cls_loss(
        domain_output_t, torch.ones(domain_output_t.size(0), dtype=torch.long, device=device)
    )
    loss_grl = err_s_domain + err_t_domain

    # Unsupervised target entropy uses adapted predictions but never target labels.
    target_probs = F.softmax(target_logits, dim=-1)
    target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
    loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

    loss = (
        cls_loss_source
        + args.lambda_d * loss_grl
        + args.lambda_r * recon_loss_all
        + args.lambda_f * diff_loss_all
        + loss_entropy * (epoch / args.n_epoch * 0.01)
        + geometry_loss
    )
    loss.backward()
    optimizer.step()

    should_evaluate = (
        (epoch + 1) % max(args.eval_interval, 1) == 0 or epoch + 1 == args.n_epoch
    )
    if should_evaluate:
        for model in models:
            model.eval()
        with torch.no_grad():
            eval_source_logits, _ = predict_pair(
                features_s, adj_s, X_n_s, features_t, adj_t, X_n_t
            )
            source_train_metrics = evaluate_logits(eval_source_logits, labels_s, idx_train_s)
            source_val_metrics = evaluate_logits(eval_source_logits, labels_s, idx_val_s)

        if source_val_metrics["accuracy"] > best_source_val_accuracy:
            best_source_val_accuracy = source_val_metrics["accuracy"]
            best_epoch = epoch + 1
            best_checkpoint = snapshot_components(components)

        print(
            "epoch: {}, source_train_acc: {:.6f}, source_val_acc: {:.6f}, "
            "loss_class: {:.6f}, loss_domain: {:.6f}, "
            "loss_recon: {:.6f}, loss_diff: {:.6f}, geometry_loss: {:.6f}".format(
                epoch + 1,
                source_train_metrics["accuracy"],
                source_val_metrics["accuracy"],
                cls_loss_source.item(),
                (args.lambda_d * loss_grl).item(),
                (args.lambda_r * recon_loss_all).item(),
                (args.lambda_f * diff_loss_all).item(),
                geometry_loss.item(),
            )
        )

if best_checkpoint is None:
    raise RuntimeError("Training completed without producing a source-validation checkpoint.")

restore_components(components, best_checkpoint)
for model in models:
    model.eval()
print(
    "CHECKPOINT_RESTORED,best_epoch={},best_source_val_accuracy={:.10f}".format(
        best_epoch,
        best_source_val_accuracy,
    )
)

# Target group is physically loaded only here, after restoring the complete
# source-validation-best checkpoint. All target nodes are evaluated.
labels_t = load_target_labels(
    args.data_root,
    args.data_trg,
    expected_num_nodes=num_nodes_t,
).to(device)
with torch.no_grad():
    final_source_logits, final_target_logits = predict_pair(
        features_s, adj_s, X_n_s, features_t, adj_t, X_n_t
    )
    source_train_metrics = evaluate_logits(final_source_logits, labels_s, idx_train_s)
    source_val_metrics = evaluate_logits(final_source_logits, labels_s, idx_val_s)
    source_test_metrics = evaluate_logits(final_source_logits, labels_s, idx_test_s)
    target_metrics = evaluate_logits(final_target_logits, labels_t)

print("best source-validation epoch: {}".format(best_epoch))
print("source train metrics: {}".format(source_train_metrics))
print("source validation metrics: {}".format(source_val_metrics))
print("source test metrics: {}".format(source_test_metrics))
print("target metrics: {}".format(target_metrics))

final_fields = [
    "source={}".format(args.data_src),
    "target={}".format(args.data_trg),
    "seed={}".format(args.seed),
    "split_seed={}".format(args.split_seed),
    "best_epoch={}".format(best_epoch),
    "best_source_val_accuracy={:.10f}".format(best_source_val_accuracy),
]
final_fields.extend(metric_fields("source_train", source_train_metrics))
final_fields.extend(metric_fields("source_val", source_val_metrics))
final_fields.extend(metric_fields("source_test", source_test_metrics))
final_fields.extend(metric_fields("target", target_metrics))

final_fields.append("use_becat={}".format(str(args.use_becat).lower()))

if args.use_becat:
    for name in (
        "latent_topk_k",
        "latent_topk_tau",
        "lambda_latent_bec",
        "lambda_latent_bec_stat",
        "latent_bec_sample_size",
        "sparse_gt_num_heads",
        "sparse_gt_num_layers",
        "sparse_gt_dropout",
        "sparse_gt_curv_bias_beta",
        "sparse_gt_curv_bias_mode",
        "sparse_gt_fourier_m",
        "sparse_gt_fourier_sigma",
        "sparse_gt_eta_abs",
        "sparse_gt_eta_fourier",
        "sparse_gt_bias_norm",
        "sparse_gt_attention_k",
        "sparse_gt_residual_alpha",
    ):
        final_fields.append("{}={}".format(name, getattr(args, name)))

print("FINAL_RESULT," + ",".join(final_fields))
