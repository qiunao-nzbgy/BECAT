import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
import numpy as np
from dgl.nn.pytorch import GraphConv

from becat.torch import GeometryAdapterConfig, LatentGeometryAdapter


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRADE(nn.Module):
    def __init__(self, g_s, g_t, in_feats, n_hidden, n_classes, n_layers, dropout, activation=F.relu, disc="JS", args=None):
        super(GRADE, self).__init__()
        self.disc = disc
        self.g_s = g_s
        self.g_t = g_t
        self.n_classes = n_classes
        self.use_becat = args.use_becat if args is not None else False
        if self.use_becat:
            cfg = GeometryAdapterConfig(
                use_latent_topk_bec=args.use_latent_topk_bec,
                latent_topk_k=args.latent_topk_k,
                latent_topk_tau=args.latent_topk_tau,
                lambda_latent_bec=args.lambda_latent_bec,
                lambda_latent_bec_stat=args.lambda_latent_bec_stat,
                latent_bec_use_mmd=args.latent_bec_use_mmd,
                latent_bec_sample_size=args.latent_bec_sample_size,
                latent_bec_N=args.latent_bec_N,
                latent_bec_mmd_sigma=args.latent_bec_mmd_sigma,
                use_sparse_gt_head=args.use_sparse_gt_head,
                sparse_gt_num_heads=args.sparse_gt_num_heads,
                sparse_gt_num_layers=args.sparse_gt_num_layers,
                sparse_gt_dropout=args.sparse_gt_dropout,
                sparse_gt_use_curv_bias=args.sparse_gt_use_curv_bias,
                sparse_gt_curv_bias_beta=args.sparse_gt_curv_bias_beta,
                sparse_gt_curv_bias_mode=args.sparse_gt_curv_bias_mode,
                sparse_gt_fourier_m=args.sparse_gt_fourier_m,
                sparse_gt_fourier_sigma=args.sparse_gt_fourier_sigma,
                sparse_gt_fourier_seed=args.sparse_gt_fourier_seed,
                sparse_gt_eta_abs=args.sparse_gt_eta_abs,
                sparse_gt_eta_fourier=args.sparse_gt_eta_fourier,
                sparse_gt_bias_norm=args.sparse_gt_bias_norm,
                sparse_gt_attention_k=args.sparse_gt_attention_k,
                sparse_gt_use_self_loop=args.sparse_gt_use_self_loop,
                sparse_gt_residual_alpha=args.sparse_gt_residual_alpha,
                sparse_gt_use_layernorm=args.sparse_gt_use_layernorm,
            )
            self.becat = LatentGeometryAdapter(cfg)
            # Register lazy transformer parameters before GRADE_train creates its optimizer.
            self.becat.initialize_parameters(
                hidden_dim=n_hidden,
                device=torch.device("cpu"),
                dtype=torch.get_default_dtype(),
            )
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # self.layers.append(GraphConv(n_hidden, n_classes, activation=None))
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)

        if disc == "JS":
            self.discriminator = nn.Sequential(
                nn.Linear(n_hidden*n_layers+n_classes, 2)
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(n_hidden * n_layers + n_classes * 2, 2)
            )
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        features_s,
        labels_s,
        features_t,
        alpha=1.0,
        source_train_mask=None,
    ):
        s_f = []
        t_f = []
        for i, layer in enumerate(self.layers):
            features_s = self.dropout(features_s)
            features_t = self.dropout(features_t)
            features_s = layer(self.g_s, features_s)
            features_t = layer(self.g_t, features_t)
            s_f.append(features_s)
            t_f.append(features_t)

        H_s = features_s
        H_t = features_t
        if self.use_becat:
            O_s, O_t, becat_loss, _ = self.becat(
                H_s, H_t, training=self.training
            )
        else:
            O_s, O_t = H_s, H_t
            becat_loss = H_s.new_tensor(0.0)

        domain_features_s = self.fc(H_s)
        domain_features_t = self.fc(H_t)
        if self.use_becat:
            logits_s = self.fc(O_s)
        else:
            logits_s = domain_features_s
        s_f.append(domain_features_s)
        t_f.append(domain_features_t)
        if source_train_mask is None:
            raise ValueError("GRADE citation training requires source_train_mask.")
        class_loss = self.criterion(
            logits_s[source_train_mask],
            labels_s[source_train_mask],
        )

        s_f = torch.cat(s_f, dim=1)
        t_f = torch.cat(t_f, dim=1)
        domain_loss = 0.
        if self.disc == "JS":
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_f, t_f], dim=0), alpha))
            domain_labels = np.array([0] * H_s.shape[0] + [1] * H_t.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=H_s.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
        elif self.disc == "MMD":
            mind = min(s_f.shape[0], t_f.shape[0])
            domain_loss = mmd_rbf_noaccelerate(s_f[:mind], t_f[:mind])
        elif self.disc == "C":
            ratio = 8
            s_l_f = torch.cat([s_f, ratio * self.one_hot_embedding(labels_s)], dim=1)
            t_l_f = torch.cat([t_f, ratio * F.softmax(domain_features_t, dim=1)], dim=1)
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_l_f, t_l_f], dim=0), alpha))
            domain_labels = np.array([0] * H_s.shape[0] + [1] * H_t.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=H_s.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
        loss = class_loss + domain_loss * 0.01
        if self.use_becat:
            loss = loss + becat_loss
        return loss

    def extract_representations(self, x, graph, source_domain):
        """Return the raw and classifier-input representations for one domain.

        The raw representation is the output of the final backbone GraphConv.
        For BECAT, the classifier input is the geometry-adapter output; for the
        baseline it is exactly the raw representation.
        """
        for i, layer in enumerate(self.layers):
            x = layer(graph, x)
        raw = x
        if self.use_becat:
            O_s, O_t, _, _ = self.becat(raw, raw, training=False)
            classifier_input = O_s if source_domain else O_t
        else:
            classifier_input = raw
        return raw, classifier_input

    def extract_target_representations(self, x):
        """Return target ``(H_t, O_t)`` in target feature-row order."""
        return self.extract_representations(x, self.g_t, source_domain=False)

    def _inference(self, x, graph, source_domain):
        _, classifier_input = self.extract_representations(
            x, graph, source_domain=source_domain
        )
        logits = self.fc(classifier_input)
        return torch.log_softmax(logits, dim=-1)

    def inference_source(self, x):
        return self._inference(x, self.g_s, source_domain=True)

    def inference_target(self, x):
        return self._inference(x, self.g_t, source_domain=False)

    def inference(self, x):
        return self.inference_target(x)

    def one_hot_embedding(self, labels):
        y = torch.eye(self.n_classes, device=labels.device)
        return y[labels]
