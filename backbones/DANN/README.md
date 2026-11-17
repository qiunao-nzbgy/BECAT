# DANN citation backbone

Baseline and `+BECAT` share the same protocol; `--use_becat` enables latent
refinement and geometry loss.

## External data

Pass the directory containing the three MATLAB files through `--data_root`:

```text
Citation_mat/
├── acmv9.mat
├── citationv1.mat
└── dblpv7.mat
```

Each file must contain `network`, `attrb`, and `group`. Target training loads
only `network` and `attrb`; final evaluation loads `group` after restoring the
best state.

## Finalized protocol

- Source nodes are split 80% train, 10% validation, and remainder test by a
  private `torch.Generator` plus deterministic `torch.randperm`.
- `--split_seed` defaults to 42 and is independent of training `--seed`.
- Source supervised cross-entropy indexes source-train nodes only.
- Target features and raw embeddings are available during adaptation; target
  labels are not loaded during training or model selection.
- Strictly greater source-validation accuracy replaces the complete in-memory
  best state. Exact ties retain the earlier epoch.
- The checkpoint includes `FeatureExtractor`, `LabelClassifier`,
  `DomainClassifier`, and BECAT when enabled.
- Source validation/test diagnostics are recomputed after restoration. Only
  then are target labels loaded and all target nodes evaluated.
- Final output is one machine-readable `FINAL_RESULT,...` line.

## DANN and BECAT paths

`FeatureExtractor` produces raw `H_s` and `H_t`. The gradient-reversal
`DomainClassifier` always consumes the concatenated raw embeddings. With BECAT,
the canonical adapter maps `(H_s, H_t)` to `(O_s, O_t)` and the
`LabelClassifier` consumes `O_s` during source supervision and `O_t` during
final target inference. The total loss is:

```text
source classification + lambda_domain * domain adversarial
                      + lambda_becat * BECAT geometry
```

## Running experiments

Both scripts run all six transfers with seeds `1 2 3 4 5` and split seed `42`:

```bash
DATA_ROOT=/datasets/Citation_mat DEVICE=cuda:0 \
  bash backbones/DANN/scripts/run_baseline.sh

DATA_ROOT=/datasets/Citation_mat DEVICE=cuda:0 \
  bash backbones/DANN/scripts/run_becat.sh
```

The final experiment settings are:

| Setting | Value |
|---|---:|
| `epochs` | `1000` |
| `lr` | `0.001` |
| `weight_decay` | `5e-4` |
| `hidden_dim` | `128` |
| `dropout` | `0.5` |
| `lambda_domain` | `1.0` |
| `lambda_becat` | `0.001` |

The domain critic remains on raw `H_s`/`H_t`.

## Environment

Framework: Python 3.7+ with PyTorch, NumPy, SciPy, and scikit-learn:

```bash
conda create -n becat-dann "python>=3.7" pip
conda activate becat-dann
python -m pip install -r backbones/DANN/requirements.txt
```

Scripts default to one CUDA device. See
[docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).
