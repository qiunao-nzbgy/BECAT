# Backbone environments

Each backbone has its own primary installation file. The environments are not
intended to be combined, and this repository intentionally has no root
`requirements.txt`. Run the commands below from the repository root.

Environment evidence is described as **recorded** when an exact local or
upstream stack exists, **compatibility bridge** when static requirements force a
documented adjustment to an upstream stack, and **unpinned** when imports prove
that a package is required but no trustworthy exact version is available. No
environment was installed during repository assembly.

## Installation matrix

| Backbone | Environment file | Framework | Key version notes | Install command |
|---|---|---|---|---|
| A2GNN | [`environment.yml`](../backbones/A2GNN/environment.yml) | PyTorch + PyG | Recorded: PyTorch 1.11.0, PyG 2.1.0, torch-scatter 2.0.9, torch-sparse 0.6.13, torch-cluster 1.6.0; Python minor and CUDA generation not recorded | `conda env create -f backbones/A2GNN/environment.yml`<br>`conda activate becat-a2gnn` |
| ASN | [`environment.yml`](../backbones/ASN/environment.yml) | PyTorch | Compatibility bridge: upstream numeric pins with Python 3.7 and PyTorch 1.11.0 for the shared backend; not server-validated | `conda env create -f backbones/ASN/environment.yml`<br>`conda activate becat-asn` |
| AdaGCN | [`environment.yml`](../backbones/AdaGCN/environment.yml) | TensorFlow 1.x | Upstream-recorded Python 3.5.2, TensorFlow GPU 1.13.0-rc0, NumPy 1.16.2; SciPy required but unpinned | `conda env create -f backbones/AdaGCN/environment.yml`<br>`conda activate becat-adagcn` |
| DANN | [`requirements.txt`](../backbones/DANN/requirements.txt) | PyTorch | PyTorch, NumPy, SciPy, scikit-learn required; exact versions unknown; Python 3.7+ static lower bound | `conda create -n becat-dann "python>=3.7" pip`<br>`conda activate becat-dann`<br>`python -m pip install -r backbones/DANN/requirements.txt` |
| DGSDA | [`requirements.txt`](../backbones/DGSDA/requirements.txt) | PyTorch + PyG | Locally recorded PyTorch 2.1.1+cu121/PyG 2.6.1 stack; Python 3.10 is an installer-compatible choice, not a validated pin | `conda create -n becat-dgsda python=3.10 pip`<br>`conda activate becat-dgsda`<br>`python -m pip install -r backbones/DGSDA/requirements.txt` |
| GRADE | [`requirements.txt`](../backbones/GRADE/requirements.txt) | PyTorch + DGL | Exact versions unknown; released code uses legacy `DGLGraph` APIs | `conda create -n becat-grade "python>=3.7" pip`<br>`conda activate becat-grade`<br>`python -m pip install -r backbones/GRADE/requirements.txt` |
| PairAlign | [`requirements.txt`](../backbones/PairAlign/requirements.txt) | PyTorch + PyG | Locally recorded PyTorch 1.12.1+cu113/PyG 2.2.0/torch-sparse 0.6.16 stack; Python 3.10 is an installer-compatible choice, not a validated pin | `conda create -n becat-pairalign python=3.10 pip`<br>`conda activate becat-pairalign`<br>`python -m pip install -r backbones/PairAlign/requirements.txt` |
| SpecReg | [`requirements.txt`](../backbones/SpecReg/requirements.txt) | PyTorch + PyG | Direct dependencies only; exact versions unknown; Python 3.7+ static lower bound | `conda create -n becat-specreg "python>=3.7" pip`<br>`conda activate becat-specreg`<br>`python -m pip install -r backbones/SpecReg/requirements.txt` |
| UDAGCN | [`requirements.txt`](../backbones/UDAGCN/requirements.txt) | PyTorch + PyG | Direct dependencies only; exact versions unknown; Python 3.7+ static lower bound | `conda create -n becat-udagcn "python>=3.7" pip`<br>`conda activate becat-udagcn`<br>`python -m pip install -r backbones/UDAGCN/requirements.txt` |

## Compatibility notes

- **A2GNN:** the canonical PyTorch backend retains the older-PyTorch segment
  softmax/scatter path required by the recorded PyTorch 1.11.0 stack. The
  environment does not guess a CUDA toolkit generation.
- **ASN:** its upstream environment used Python 3.6 and PyTorch 1.5.1, but the
  shared backend imports stdlib `dataclasses` and has a recorded PyTorch 1.11
  compatibility target. The supplied file is therefore a conservative bridge,
  not a claim that the original ASN stack was validated unchanged.
- **AdaGCN:** this is TensorFlow 1.x graph-mode code through
  `becat.tensorflow`; it must remain separate from all PyTorch environments.
- **DGSDA:** the retained PyTorch 2.1.1+cu121 wheel uses PyTorch's official CUDA
  12.1 index. SciPy 1.15.1 requires Python 3.10 or newer, so Python 3.10 is the
  conservative intersection selected by the installation command. The recorded
  NumPy 2.2.3/PyTorch 2.1.1 combination still requires runtime verification.
- **GRADE:** do not assume that an arbitrary current DGL release supports the
  legacy `DGLGraph` calls. No exact DGL pin is available in the release.
- **PairAlign:** the requirements file declares the official PyTorch CUDA 11.3
  and PyG extension-wheel indexes needed by its local version tags. It also
  retains upstream packages not imported by the released citation path.
- **SpecReg and UDAGCN:** `torch-scatter` must resolve to a build compatible
  with the selected PyTorch and CUDA runtime. No exact pair was recorded.

## Runtime notes

A2GNN can place encoder and geometry work on separate logical GPUs. DGSDA's
+BECAT script explicitly uses two logical GPUs. Other scripts select at most one
GPU or leave device topology to their arguments. CPU training support and
minimum VRAM were not established during assembly.

Training and split seeds are explicitly controlled, but CUDA, cuBLAS, DGL/PyG
scatter operations, and other kernels may remain nondeterministic across
framework, driver, and hardware versions. Bitwise reproducibility across
software or GPU stacks is not guaranteed.
