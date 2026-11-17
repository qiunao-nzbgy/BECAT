# BECAT

This repository contains BECAT integrations for the six directed citation
transfers among ACMv9, Citationv1, and DBLPv7. It provides a shared PyTorch
backend, a TensorFlow 1.x backend for AdaGCN, and nine backbone integrations.

## Supported backbones

- DGSDA
- GRADE
- SpecReg
- A2GNN
- UDAGCN
- DANN
- ASN
- PairAlign
- AdaGCN

Each [backbone README](backbones/) contains its commands and settings.

## Repository structure

```text
BECAT/
├── becat/
│   ├── torch/          # canonical PyTorch BECAT backend
│   └── tensorflow/     # TensorFlow 1.x BECAT backend for AdaGCN
├── backbones/          # nine citation backbone integrations
└── docs/               # data, environment, and attribution guidance
```

## Data

Datasets are not distributed in this repository. Six backbones use the raw-text
`Citation/` layout and three use the legacy MATLAB `Citation_mat/` layout. See
[docs/DATA.md](docs/DATA.md) before running a backbone.

## Environments

Backbones use different PyTorch, TensorFlow, DGL, and PyG stacks. See
[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) and the relevant backbone README.

## Quick start

Provide external data and invoke the selected backbone's paired scripts from
the repository root:

```bash
bash backbones/<BACKBONE>/scripts/run_baseline.sh ...
bash backbones/<BACKBONE>/scripts/run_becat.sh ...
```

Arguments, seeds, and BECAT settings are backbone-specific.

## Experimental protocol

For the released citation experiments, source nodes are split 80%/10%/10% into
train/validation/test subsets using an independent split seed. Model
checkpoints are selected using source validation accuracy only. Target labels
are not used for training or model selection and are loaded only for final
target evaluation. The final evaluation covers all target nodes after the best
source-validation checkpoint is restored.

The released scripts use split seed 42; training seeds and BECAT parameters are
backbone-specific.

## Framework backends

### PyTorch

PyTorch backbones import `GeometryAdapterConfig` and
`LatentGeometryAdapter` from `becat.torch`.

### TensorFlow 1.x

AdaGCN uses the TensorFlow 1.x graph-mode implementation in
`becat.tensorflow`.

## Citation and attribution

BECAT does not claim authorship of third-party backbones. See
[docs/BACKBONES.md](docs/BACKBONES.md) for attribution and license status.

## License notes

Backbone components remain subject to their original licenses. See
[docs/BACKBONES.md](docs/BACKBONES.md).
