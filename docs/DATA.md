# Citation dataset provenance and interfaces

No dataset files, processed caches, or MATLAB binaries are bundled in this
repository. Supply the appropriate dataset externally and
pass its root through the selected backbone's documented interface.

## Raw-text Citation layout

The raw-text Citation data used by A2GNN, DGSDA, GRADE, PairAlign, SpecReg, and
UDAGCN comes from the [PyGDA Citation dataset
documentation](https://pygda.readthedocs.io/en/stable/datasets/Citation/).
PyGDA uses the ACMv9, Citationv1, and DBLPv7 citation domains and expects the
document-feature, edge-list, and label files beneath each domain's `raw/`
directory.

Obtain the Citation dataset through the PyGDA dataset resource/documentation
and place the extracted files in this layout:

```text
Citation/
├── ACMv9/
│   └── raw/
│       ├── ACMv9_docs.txt
│       ├── ACMv9_edgelist.txt
│       └── ACMv9_labels.txt
├── Citationv1/
│   └── raw/
│       ├── Citationv1_docs.txt
│       ├── Citationv1_edgelist.txt
│       └── Citationv1_labels.txt
└── DBLPv7/
    └── raw/
        ├── DBLPv7_docs.txt
        ├── DBLPv7_edgelist.txt
        └── DBLPv7_labels.txt
```

No unverified direct-download URL is provided here. The PyGDA loader's
documented download method is empty, so the raw files must be placed manually.
Some loaders create framework-specific processed caches beneath this external
tree. Do not commit those caches.

## MATLAB Citation layout

The `.mat` citation files used by ASN, DANN, and AdaGCN were obtained from the
official [AdaGCN repository](https://github.com/daiquanyu/AdaGCN_TKDE). Place
the three files in this external layout:

```text
Citation_mat/
├── acmv9.mat
├── citationv1.mat
└── dblpv7.mat
```

The released loaders expect the existing `network`, `attrb`, and `group`
fields. Target-training loaders isolate graph/features from `group`, and load
target labels separately for final evaluation as described in each backbone
README.

This repository does not redistribute the `.mat` files and
does not provide an unverified alternative download URL.

## Relationship between the layouts

Both layouts represent the ACMv9, Citationv1, and DBLPv7 citation domains used
in the released experiments. They are consumed through different legacy
loader formats: six backbones read domain-specific raw-text files, while ASN,
DANN, and AdaGCN read MATLAB files. Byte-level equivalence between the two
representations has not been verified and is not claimed.

## Backbone-to-layout mapping

| Backbone | Expected data layout |
|---|---|
| A2GNN | Citation raw-text |
| DGSDA | Citation raw-text |
| GRADE | Citation raw-text |
| PairAlign | Citation raw-text |
| SpecReg | Citation raw-text |
| UDAGCN | Citation raw-text |
| ASN | `Citation_mat` |
| DANN | `Citation_mat` |
| AdaGCN | `Citation_mat` |

## Supplying the external data root

Pass the external root appropriate to the selected backbone, for example:

```text
--data_root /path/to/Citation
```

or:

```text
--data_root /path/to/Citation_mat
```

Consult the selected backbone README for its exact command interface. Keep
datasets and any generated processed caches outside the released repository.
