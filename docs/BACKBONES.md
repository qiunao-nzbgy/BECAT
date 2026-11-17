# Backbone provenance and license audit

The backbone implementations originate from the third-party research
repositories listed below. The BECAT project does not claim authorship of
those implementations. Baseline paper citations, repository links, and
third-party license notices are scientific and legal provenance; they are not
BECAT-author identity information.

The upstream evidence in this document was checked against the linked
repository roots and their README files on 2026-08-24. A missing license entry
means that no license file or GitHub-detected license was present at the
upstream repository root during that inspection. It does not imply any license
or permission, and no license has been inferred.

| Backbone | Upstream paper/citation evidence | Authoritative upstream repository | Upstream license evidence | Preserved local license/notice | Local modifications in this release |
|---|---|---|---|---|---|
| DGSDA | The upstream README names *Disentangled Graph Spectral Domain Adaptation*, but does not provide a formal bibliographic citation. | https://github.com/Hechriver/DGSDA | No license file or GitHub-detected license found at the repository root. | None supplied; license/permission requires author review before public release. | Citation-only extraction, leakage-safe frozen protocol, optional BECAT classifier path, unified package/path assembly. |
| GRADE | The upstream README provides a BibTeX citation for *Non-IID Transfer Learning on Graphs* (AAAI 2023). | https://github.com/junwu6/GRADE | No license file or GitHub-detected license found at the repository root. | None supplied; license/permission requires author review before public release. | Citation-only extraction, target-label isolation/checkpoint protocol, optional BECAT classifier path, unified package/path assembly. |
| SpecReg | The upstream README provides a BibTeX citation for *Graph Domain Adaptation via Theory-Grounded Spectral Regularization* (ICLR 2023). | https://github.com/Shen-Lab/GDA-SpecReg | Apache License 2.0. No upstream `NOTICE` file was found at the repository root. | Apache License 2.0 preserved at `backbones/SpecReg/LICENSE`; no upstream `NOTICE` file was available to copy. | Citation-only extraction, frozen SpecReg/BECAT settings, unified package/path assembly. |
| A2GNN | The upstream README provides a BibTeX citation for *Rethinking Propagation for Unsupervised Graph Domain Adaptation* (AAAI 2024). | https://github.com/Meihan-Liu/24AAAI-A2GNN | No license file or GitHub-detected license found at the repository root. | None supplied; license/permission requires author review before public release. | A2GNN-MMD citation-only extraction, frozen protocol, optional BECAT path, unified package/path assembly. |
| UDAGCN | The upstream README provides a BibTeX citation for *Unsupervised Domain Adaptive Graph Convolutional Networks* (WWW 2020). | https://github.com/TrustAGI-Lab/UDAGCN | MIT License, copyright 2020 mandy976. | Upstream MIT License and copyright notice preserved at `backbones/UDAGCN/LICENSE`. | Citation-only extraction, target-label isolation/checkpoint protocol, optional BECAT classifier path, unified package/path assembly. |
| DANN | The upstream README links and identifies *Unsupervised Domain Adaptation by Backpropagation* (2015) and *Domain-Adversarial Training of Neural Networks* (2016), but does not provide a formal BibTeX citation. | https://github.com/NaJaeMin92/pytorch-DANN | No license file or GitHub-detected license found at the repository root. | None supplied; license/permission requires author review before public release. | Citation-only graph DANN release, frozen validation/checkpoint protocol, optional BECAT classifier path, unified package/path assembly. |
| ASN | The upstream README names *Adversarial Separation Network for Cross-Network Node Classification*, but its `Reference` section is empty. | https://github.com/yuntaodu/ASN | No license file or GitHub-detected license found at the repository root. | None supplied; license/permission requires author review before public release. | Citation-only extraction, target-label isolation/checkpoint protocol, optional BECAT classifier path, unified package/path assembly. |
| PairAlign | The upstream README provides a BibTeX citation for *Pairwise Alignment Improves Graph Domain Adaptation* (ICML 2024). | https://github.com/Graph-COM/Pair-Align | MIT License, copyright 2024 Graph-COM. | Upstream MIT License and copyright notice preserved at `backbones/PairAlign/LICENSE`. | Citation-only extraction, frozen PairAlign/BECAT protocol, unified package/path assembly. |
| AdaGCN | The upstream README provides a full citation and BibTeX entry for *Graph Transfer Learning via Adversarial Domain Adaptation with Graph Convolution* (IEEE TKDE, 2022). | https://github.com/daiquanyu/AdaGCN_TKDE | No license file or GitHub-detected license found at the repository root. | None supplied; license/permission requires author review before public release. | Citation-only TensorFlow 1.x path, target-label isolation/checkpoint protocol, TensorFlow-native BECAT integration, unified package/path assembly. |

## Author-review checklist

- Select the root BECAT project license only for the final public release; the
  anonymous-review artifact intentionally omits one.
- Confirm redistribution permission or obtain license clarification for DGSDA,
  GRADE, A2GNN, DANN, ASN, and AdaGCN before public release. Their upstream
  repository roots did not provide license evidence during this audit.
- Complete formal bibliography entries for DGSDA, DANN, and ASN from
  authoritative paper sources before public release if full citations are
  required; the upstream READMEs do not provide complete citation records.
- Retain the preserved SpecReg, PairAlign, and UDAGCN license files with the
  corresponding third-party code.
