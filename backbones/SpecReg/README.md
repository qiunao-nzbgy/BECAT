# SpecReg citation main-table experiments

Baseline and SpecReg+BECAT share one protocol for all six directed citation
transfers.

## External data layout

Pass the parent Citation directory with `--data_root` (or as the first
experiment-script argument). The release does not contain datasets.

```text
Citation/
├── ACMv9/raw/
│   ├── ACMv9_docs.txt
│   ├── ACMv9_edgelist.txt
│   └── ACMv9_labels.txt
├── Citationv1/raw/
│   ├── Citationv1_docs.txt
│   ├── Citationv1_edgelist.txt
│   └── Citationv1_labels.txt
└── DBLPv7/raw/
    ├── DBLPv7_docs.txt
    ├── DBLPv7_edgelist.txt
    └── DBLPv7_labels.txt
```

The loader may create `processed/data.pt` for the source and
`processed/unlabeled_data.pt` for the target. The target cache contains no
labels. Source masks are regenerated from `--split_seed` on every load.

## Finalized protocol

- Source split: 80% train, 10% validation, 10% test.
- Training seeds: 23, 202, 11, 27, 201.
- Source split seed: 42, fixed and independent of the training seed.
- Supervised loss: source training mask only.
- Checkpoint: strict improvement in source validation accuracy only; ties keep
  the earlier epoch.
- Epoch evaluation: source validation only.
- Target training object: features, edges, spectral data, embeddings, and
  predictions only; no target `y`.
- Final evaluation: restore the complete best checkpoint, recompute restored
  source diagnostics, then read target labels and evaluate every target node.

## SpecReg and BECAT paths

The GCN and PPMI encoders plus attention produce raw fused embeddings `H_s`
and `H_t`. The original Wasserstein/domain branch and SSReg/MFRReg calculations
remain on those raw embeddings. BECAT, when enabled, maps `H_s,H_t` to refined
`O_s,O_t`; only the refined embeddings enter the classifier. Target entropy is
computed from classifier logits on that same classifier path. The canonical
BECAT module returns a loss already weighted by `lambda_latent_bec`, and that
`geometry_loss` is added exactly once to the unchanged SpecReg objective.
Baseline bypasses BECAT and contributes zero geometry loss.

The final SpecReg citation BECAT configuration is:

| CLI argument | Final value |
|---|---:|
| `--lambda_latent_bec` | 0.001 |
| `--lambda_latent_bec_stat` | 0.1 |
| `--latent_topk_k` | 10 |
| `--latent_topk_tau` | 0.5 |
| `--latent_bec_sample_size` | 128 |
| `--sparse_gt_attention_k` | 10 |
| `--sparse_gt_curv_bias_beta` | 0.05 |
| `--sparse_gt_residual_alpha` | 0.1 |
| `--sparse_gt_num_heads` | 2 |
| `--sparse_gt_num_layers` | 1 |
| `--sparse_gt_dropout` | 0.3 |
| `--sparse_gt_curv_bias_mode` | `hybrid` |
| `--sparse_gt_bias_norm` | `scale` |
| `--sparse_gt_fourier_m` | 4 |
| `--sparse_gt_fourier_sigma` | 1.0 |
| `--sparse_gt_fourier_seed` | 1234 |
| `--sparse_gt_eta_abs` | 1.0 |
| `--sparse_gt_eta_fourier` | 0.1 |
| `--latent_bec_use_mmd` | false (flag omitted) |

## Running experiments

From the release root:

```bash
bash backbones/SpecReg/scripts/run_baseline.sh /absolute/path/to/Citation
bash backbones/SpecReg/scripts/run_becat.sh /absolute/path/to/Citation
```

## Environment

Framework: PyTorch with PyG, torch-scatter, NumPy, scikit-learn, and tqdm.

```bash
conda create -n becat-specreg "python>=3.7" pip
conda activate becat-specreg
python -m pip install -r backbones/SpecReg/requirements.txt
```

See [docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).
