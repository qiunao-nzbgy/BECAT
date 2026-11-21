# A2GNN citation backbone

Baseline and `+BECAT` share one entrypoint; `--use_becat` enables latent
refinement and geometry loss.

## Data layout

Pass the parent `Citation/` directory through `--data_root`:

```text
Citation/
├── ACMv9/raw/ACMv9_{docs,edgelist,labels}.txt
├── Citationv1/raw/Citationv1_{docs,edgelist,labels}.txt
└── DBLPv7/raw/DBLPv7_{docs,edgelist,labels}.txt
```

The runtime cache is `<data_root>/<domain>/processed/graph.pt`. It contains only
`x`, `edge_index`, and `num_nodes`. Source labels are read separately and masks
are regenerated at every load from `--split_seed`; target labels are first read
only after the best source-validation checkpoint has been restored. Legacy
`processed/data.pt` files are ignored and never deserialized.

## Protocol

- Source split: deterministic `torch.randperm` using a private generator,
  80% train, 10% validation, remainder test.
- `--split_seed` defaults to `42` and is independent of training `--seed`.
- Source supervised loss uses only `source_data.train_mask`.
- Original A2GNN MMD uses all raw, pre-BECAT source/target embeddings.
- Strict source-validation accuracy improvement selects the in-memory best
  state; validation ties retain the earlier state and patience is validation
  based.
- Target training data has no labels. Final metrics use all target nodes only
  after restoration.

## Running experiments

```bash
DATA_ROOT=/datasets/Citation \
  bash backbones/A2GNN/scripts/run_baseline.sh

DATA_ROOT=/datasets/Citation \
ENCODER_DEVICE=cuda:0 GEOMETRY_DEVICE=cuda:1 \
  bash backbones/A2GNN/scripts/run_becat.sh
```

The scripts run all six directed transfers in fixed order with split seed 42.

## Environment

Framework: PyTorch 1.11.0 with PyG 2.1.0. From the repository root:

```bash
conda env create -f backbones/A2GNN/environment.yml
conda activate becat-a2gnn
```

The scripts allow encoder and geometry work on separate CUDA devices. See
[docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).
