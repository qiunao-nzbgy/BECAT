# DGSDA citation backbone

Baseline and DGSDA+BECAT share the backbone, adaptation objectives, optimizer,
and training protocol.

## Data

Place each dataset under `backbones/DGSDA/data/<dataset>/`. The loader expects
the original DGSDA raw filenames in PyG's `raw/` directory, for example
`ACMv9/raw/ACMv9_docs.txt`, `ACMv9/raw/ACMv9_edgelist.txt`, and
`ACMv9/raw/ACMv9_labels.txt`. An alternate parent directory can be passed with
`--data_root` or the scripts' `DATA_ROOT` environment variable.

Source graph tensors may be reused from `processed/data.pt`. Masks inside an
older cache are ignored: each load replaces them in memory with a deterministic
80/10/10 split generated exclusively from `--split_seed` (default `42`).
`--seed` independently controls Python, NumPy, PyTorch CPU, and PyTorch CUDA
training randomness.

## Leakage-free protocol

Every epoch trains on the source training mask and the original unsupervised
adaptation objectives, then selects a checkpoint using source validation
accuracy only. Exact validation ties retain the earlier checkpoint. Target
ground-truth labels are not present in the training `Data` object. The
pre-training target loader always reads features and edges directly from the
raw graph files; it neither reads the target label file nor deserializes a
label-containing `processed/data.pt`. Target labels are loaded only after the
selected checkpoint is restored for final target evaluation.

## BECAT integration

`models.py` first produces raw domain-specific DGSDA embeddings with `lin1` and
`prop1`/`prop2`. The original MMD branch continues to use the unchanged raw
`lin1` representations. When enabled, BECAT refines the domain-specific latent
embeddings only for the classifier path before `lin2` and `prop3`. Its weighted
geometry loss is added to the unchanged DGSDA objective with `--lambda_geo`.

The public `LatentGeometryAdapter.forward_chunked` interface uses 256-node query
chunks while retaining every node in the current domain as the TopK reference
set. It also retains the globally sampled curvature-statistic loss. The two-GPU
layout used for the citation experiments keeps `lin1`/`prop1`/`prop2` on
`cuda:0` and BECAT/`lin2`/`prop3` on `cuda:1`.

## Running experiments

```bash
bash backbones/DGSDA/scripts/run_baseline.sh
bash backbones/DGSDA/scripts/run_becat.sh
```

The scripts run all six transfers with training seeds 1 through 5 and split
seed 42. BECAT uses two logical GPUs, `lambda_geo=0.005`,
`becat_lambda_stat=0.1`, residual alpha `0.15`, curvature-bias beta `0.05`,
TopK `8`, tau `1.0`, and warmup `30`.
Set `GPU_IDS`, `DATA_ROOT`, `LOG_ROOT`, or `PYTHON_BIN` for server-specific
paths without editing the scripts.

## Environment

Framework: Python 3.10, PyTorch 2.1.1+cu121, and PyG 2.6.1:

```bash
conda create -n becat-dgsda python=3.10 pip
conda activate becat-dgsda
python -m pip install -r backbones/DGSDA/requirements.txt
```

See [docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).
