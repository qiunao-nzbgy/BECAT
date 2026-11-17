# PairAlign Citation experiments

The scripts run all six directed transfers among ACMv9, Citationv1, and DBLPv7.

## External data layout

Pass the directory named `Citation` as `--data_root` (or `DATA_ROOT` in the
shell scripts):

```text
Citation/
├── ACMv9/raw/ACMv9_{docs,edgelist,labels}.txt
├── Citationv1/raw/Citationv1_{docs,edgelist,labels}.txt
└── DBLPv7/raw/DBLPv7_{docs,edgelist,labels}.txt
```

`acm`, `citation`, and `dblp` are accepted as dataset aliases.

## Protocol

`data.py` loads source docs, edges, and labels. A private `torch.Generator`
seeded only with `--split_seed` creates an 80/10/10 source partition. The target
training loader resolves and reads only docs and edges; it neither checks nor
opens a target labels path and creates neither `y` nor `edge_class`.

Supervised loss uses source training nodes. PairAlign reweighting may use source
labels plus target features, graph structure, and predictions. Checkpoints are
selected by strict improvement in source validation accuracy.

The best CPU-cloned state contains both the complete `GNN` (including BECAT
when enabled) and the separately trainable `GNN_dc`. Both are restored and put
in evaluation mode before the target labels file is read. Final target metrics
then cover every target node.

BECAT receives raw PairAlign embeddings `H_s` and `H_t`. Its refined outputs
`O_s` and `O_t` feed the shared classifier. The domain classifier consumes raw
`H_s`/`H_t`; PairAlign reweighting consumes source labels and the classifier
predictions, as in the supplied integration, and never consumes `O` directly.
Geometry loss is added as `lambda_geometry * geometry_loss`; the domain branch
remains on raw embeddings.

## Running experiments

The final training seeds are `1, 2, 3, 4, 5`. Both scripts fix this same seed
list and use the independent source split seed `42`.

The final +BECAT settings are:

| Setting | Value |
|---|---:|
| `lambda_geometry` | `1e-3` |
| `geo_lambda_stat` | `0.1` |
| `geo_beta` | `0.0157` |
| `geo_residual_alpha` | `0.15` |
| `geo_tau` | `0.856` |
| `geo_k` | `8` |
| `geo_dropout` | `0.304` |
| `geo_num_heads` | `2` |

```bash
export DATA_ROOT=/path/to/Citation
bash backbones/PairAlign/scripts/run_baseline.sh
bash backbones/PairAlign/scripts/run_becat.sh
```

Both scripts use `split_seed=42`. `PAIRALIGN_ARGS` supplies optional shared
PairAlign arguments.

## Environment

Framework: PyTorch with PyG. `requirements.txt` records PyTorch
1.12.1+cu113, PyG 2.2.0, torch-sparse 0.6.16+pt112cu113, NumPy 1.23.5,
SciPy 1.12.0, scikit-learn 1.2.0, and CVXPY 1.4.1.

```bash
conda create -n becat-pairalign python=3.10 pip
conda activate becat-pairalign
python -m pip install -r backbones/PairAlign/requirements.txt
```

See [docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).
