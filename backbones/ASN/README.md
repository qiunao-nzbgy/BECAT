# ASN citation backbone

The scripts run all six directed transfers among ACMv9, Citationv1, and DBLPv7.

## Fixed citation protocol

- Source split: 80% train, 10% validation, remainder test.
- Split construction: local `torch.Generator` controlled by `--split_seed`
  (default `42`), independent of training `--seed`.
- Source supervision: source train indices only.
- Checkpoint selection: source validation accuracy only. The strict `>` check
  retains the earlier checkpoint on exact ties.
- Checkpoint state: every ASN encoder, attention module, decoder, classifier,
  domain discriminator, and enabled BECAT module is saved and restored.
- Target training data: only `network` and `attrb` are deserialized. Target
  `group` is loaded separately only after the best checkpoint is restored.
- Target evaluation: all target nodes; accuracy, Micro-F1, Macro-F1, and
  weighted-F1 are reported.

Baseline and BECAT use the same source split, ASN hyperparameters, training
seed, checkpoint rule, restoration path, and final target protocol. BECAT
refines only the classifier branch. The domain branch and original ASN
reconstruction/difference objectives continue to use raw ASN representations.

## External data layout

Pass the directory containing the MAT files as `--data_root`:

```text
Citation_mat/
├── acmv9.mat
├── citationv1.mat
└── dblpv7.mat
```

Each MAT file must contain `network`, `attrb`, and `group`. Target training
loads only `network` and `attrb`; final evaluation loads `group` after checkpoint
restoration.

## Running experiments

The final +BECAT settings are:

| Setting | Value |
|---|---:|
| `lambda_latent_bec` | `3.71e-4` |
| `lambda_latent_bec_stat` | `0.095` |
| `latent_topk_k` | `12` |
| `latent_topk_tau` | `0.78` |
| `sparse_gt_attention_k` | `12` |
| `sparse_gt_curv_bias_beta` | `0.00425` |
| `sparse_gt_dropout` | `0.092` |
| `sparse_gt_residual_alpha` | `0.005` |

```bash
DATA_ROOT=/path/to/Citation_mat \
  bash backbones/ASN/scripts/run_baseline.sh

DATA_ROOT=/path/to/Citation_mat \
  bash backbones/ASN/scripts/run_becat.sh
```

## Environment

Framework: Python 3.7 and PyTorch 1.11.0. From the repository root:

```bash
conda env create -f backbones/ASN/environment.yml
conda activate becat-asn
```

Scripts select one CUDA device. See
[docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).
