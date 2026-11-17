# UDAGCN citation backbone

The scripts run all six directed transfers among ACMv9, Citationv1, and DBLPv7.

## Fixed citation protocol

- Source split: 80% train, 10% validation, remainder test.
- Split construction: local `torch.Generator` controlled by `--split_seed`
  (default `42`), independent of training `--seed`.
- Supervision: source train mask only.
- Checkpoint selection: source validation accuracy only; strict improvement
  retains the earlier checkpoint on exact ties.
- Target training input: `x`, `edge_index`, and `num_nodes` only.
- Target ground truth: first loaded after the complete best checkpoint is
  restored, then evaluated on all target nodes.
- Required target metrics: Micro-F1 and Macro-F1. Accuracy and weighted-F1 are
  also reported.

Baseline and BECAT use the same split, training, selection, restoration, and
final-evaluation protocol. BECAT refines the classifier path only; the original
UDAGCN domain discriminator continues to consume raw fused embeddings.

## External data layout

Pass the directory named `Citation` as `--data_root`:

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

The loader may generate a graph/features-only `processed/graph.pt`. Source
masks are regenerated on every invocation; target training does not load a
label-bearing cache.

## Running experiments

Training seeds default to `5 7 11 23 2026`; `SEEDS` can override the list.

The final +BECAT settings are:

| Setting | Value |
|---|---:|
| `lambda_geometry` | `0.0228` |
| `geo_residual_alpha` | `0.00584` |
| `geo_beta` | `0.004` |
| `geo_lambda_stat` | `0.1` |
| `geo_warmup` | `0` |
| `geo_topk` | `10` |
| `geo_tau` | `0.5` |

```bash
DATA_ROOT=/path/to/Citation \
  bash backbones/UDAGCN/scripts/run_baseline.sh

DATA_ROOT=/path/to/Citation \
  bash backbones/UDAGCN/scripts/run_becat.sh
```

## Environment

Framework: PyTorch with PyG, torch-scatter, NumPy, scikit-learn, and tqdm.

```bash
conda create -n becat-udagcn "python>=3.7" pip
conda activate becat-udagcn
python -m pip install -r backbones/UDAGCN/requirements.txt
```

See [docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).
