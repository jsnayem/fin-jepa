# Fin-JEPA

[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org/abs/)

**Fin-JEPA: Joint-Embedding Predictive Representation Learning for Financial Time Series**

This repository contains the official implementation of Fin-JEPA, the first application of the Joint-Embedding Predictive Architecture (JEPA) to financial time series.

## Overview

Fin-JEPA learns 64-dimensional latent representations of daily equity features through a lightweight 367K-parameter architecture comprising:
- **PriceEncoder**: MLP with GELU activations and LayerNorm
- **TransformerPredictor**: 4-layer causal Transformer (n_heads=4, embed_dim=64)
- **SIGReg**: Sketched Isotropic Gaussian Regularization for collapse prevention

## Repository Structure

```
├── model.py              # Fin-JEPA model architecture
├── compare_arch.py       # Architecture ablation training script
├── experiment_e.py       # Downstream evaluation (VoE analysis)
├── compare_arch2.py      # Encoder variant + density ablation
├── analyze.py            # Analysis utilities
├── model_lewm.py         # LeWorldModel-aligned variant
├── ijepa.py              # I-JEPA baseline (MNIST)
├── leworldmodel.py       # LeWorldModel baseline (MNIST)
├── benchmark.py          # Benchmark utilities
├── hf/                   # Hugging Face Jobs training scripts
│   ├── train_jepa.py
│   ├── phase1_train.py
│   ├── phase2_train.py
│   ├── submit_hf.sh
│   └── ...
├── output/               # Ablation results (meta.json)
│   └── arch_*/
└── docs/
    ├── jepa_paper.tex    # Paper source (LaTeX)
    └── figures/          # Paper figures
```

## Reproducibility

See the paper's Reproducibility Statement for details. The architecture ablation experiments (Section 4.2) are fully reproducible using publicly available market data.

## Citation

```bibtex
@article{wang2026finjepa,
  title={Fin-JEPA: Joint-Embedding Predictive Representation Learning for Financial Time Series},
  author={Wang, Yihan and others},
  year={2026}
}
```

## License

MIT
