# AdaGCN/AdaIGCN citation backbone

The TensorFlow 1.x scripts run all six directed citation transfers. Baseline
and `+BECAT` share the same protocol and use `--gnn=gcn`; `--use_becat` selects
the adapted path.

## Data

Pass the external matrix directory through `--data_root`:

```text
Citation_mat/
├── acmv9.mat
├── citationv1.mat
└── dblpv7.mat
```

The source loader reads `attrb`, `network`, and `group`. Target training reads
only `attrb` and `network`; final evaluation loads `group` after restoring the
best checkpoint.

## Citation protocol

- Source split: 80% train, 10% validation, remainder test.
- Split construction: label-independent permutation from a local NumPy
  `RandomState`.
- `--split_seed`: defaults to 42 and controls only source membership.
- `--seed`: independently seeds Python, global NumPy training randomness, and
  every newly reset TensorFlow graph.
- Source classification loss: source training mask only.
- Checkpoint: source validation accuracy only; strict `>` retains the earlier
  checkpoint on an exact tie.
- Source test: reported only after checkpoint restoration.
- Target during training: features and graph only.
- Target final evaluation: load labels after restoration and evaluate all nodes.

## BECAT integration

`models.py` produces raw encoder embeddings as `hiddens_s` and `hiddens_t`.
With `--use_becat=true`, `LatentGeometryAdapter` refines them to
`sparse_gt_outputs_s/t`, which feed the shared classifier. The Wasserstein
critic stays on raw embeddings because `sparse_gt_to_critic` is false. The
adapter geometry loss is added to the generator objective on scheduled BECAT
epochs, and its transformer variables are included in the generator optimizer.
The full `Saver` includes encoder, classifier, critic, BECAT, and optimizer
state.

The warm-up logic is unchanged: an epoch is active when it is at or after
`latent_bec_warmup_epochs` and `(epoch - warmup) % update_interval == 0`.
Curvature-bias activation retains the original `>=` behavior.

## Experiment settings

Training seeds are `1, 2, 3, 4, 5`; the independent split seed is `42`.

The final +BECAT settings are:

| Setting | Value |
|---|---:|
| `latent_topk_k` | `10` |
| `latent_topk_tau` | `0.5` |
| `sparse_gt_curv_bias_beta` | `0.05` |
| `sparse_gt_residual_alpha` | `0.1` |
| `lambda_latent_bec` | `0.001` |
| `lambda_latent_bec_stat` | `0.1` |
| `sparse_gt_fourier_m` | `4` |
| `sparse_gt_eta_abs` | `1.0` |
| `sparse_gt_eta_fourier` | `0.1` |
| `latent_bec_warmup_epochs` | `100` |
| `sparse_gt_curv_bias_warmup` | `100` |

## Running experiments

```bash
DATA_ROOT=/path/to/Citation_mat \
bash backbones/AdaGCN/scripts/run_baseline.sh

DATA_ROOT=/path/to/Citation_mat \
bash backbones/AdaGCN/scripts/run_becat.sh
```

## Environment

Framework: TensorFlow 1.x graph mode through `becat.tensorflow`. The environment
uses Python 3.5.2, `tensorflow-gpu==1.13.0-rc0`, and NumPy 1.16.2.

```bash
conda env create -f backbones/AdaGCN/environment.yml
conda activate becat-adagcn
```

See [docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).
