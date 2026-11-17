# GRADE citation experiments

Run from the repository root with `python -m backbones.GRADE.main`.

## Data flow and label isolation

`main.py` loads source labels and an unlabeled target graph. `GRADE_train.py`
selects checkpoints by source validation accuracy; target labels are loaded only
after restoration and evaluated on all target nodes.

## GRADE and BECAT paths

The final GRADE graph-convolution outputs are `H_s` and `H_t`. The original GRADE domain branch continues to concatenate raw per-layer features with classifier projections of raw `H_s`/`H_t`. When BECAT is enabled, the shared adapter refines `H_s,H_t` into `O_s,O_t`; the classifier uses the refined representations, while the original domain branch remains raw. The unchanged BECAT loss is added after `class_loss + 0.01 * domain_loss`. Baseline uses `H_s,H_t` directly and contributes no BECAT loss.

## Arguments and retained settings

Base settings used by both experiment scripts:

| Argument | Value |
|---|---:|
| `--n_hidden` | 8 |
| `--n_layers` | 2 |
| `--epochs` | 100 |
| `--patience` | 20 |
| `--lr` | 0.05 |
| `--weight_decay` | 0.0001 |
| `--dropout` | 0.0 |
| `--split_seed` | 42 |

The final training seeds are `1, 2, 3, 4, 5`. Both scripts use this same list
with the independent source split seed `42`.

The final +BECAT settings are:

| Setting | GRADE CLI argument | Value |
|---|---|---:|
| Latent TopK | `--becat_latent_topk_k` | 8 |
| Sparse-attention TopK | `--becat_sparse_gt_attention_k` | 8 |
| Latent TopK temperature | `--becat_latent_topk_tau` | 1.0 |
| BECAT loss weight | `--becat_lambda_latent_bec` | 0.005 |
| Statistics-alignment weight | `--becat_lambda_latent_bec_stat` | 0.1 |
| Residual alpha | `--becat_sparse_gt_residual_alpha` | 0.15 |
| Curvature-bias beta | `--becat_sparse_gt_curv_bias_beta` | 0.05 |
| Geometry sample size | `--becat_latent_bec_sample_size` | 128 |

`--seed` controls training randomness; `--split_seed` controls a separate source
partition generator.

## Output

The machine-readable final line contains:

```text
FINAL_RESULT,source=...,target=...,seed=...,split_seed=...,best_epoch=...,best_source_val_accuracy=...,source_val_accuracy=...,source_test_accuracy=...,target_accuracy=...,target_micro_f1=...,target_macro_f1=...,use_becat=...
```

## Environment

Framework: PyTorch with DGL, NumPy, and scikit-learn. The code uses the legacy
`DGLGraph` API and `dgl.nn.pytorch.GraphConv`.

```bash
conda create -n becat-grade "python>=3.7" pip
conda activate becat-grade
python -m pip install -r backbones/GRADE/requirements.txt
```

Scripts select one CUDA device. See
[docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).
