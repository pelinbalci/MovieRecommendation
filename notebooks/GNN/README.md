# 🧠 Graph Neural Networks for Movie Recommendation

A comprehensive learning guide for implementing GNN-based recommendation systems, building upon the existing collaborative filtering approach.

## 📚 Table of Contents

1. [Overview](#overview)
2. [Why GNNs for Recommendations?](#why-gnns-for-recommendations)
3. [Concepts Explained](#concepts-explained)
4. [GNN Architectures](#gnn-architectures)
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [Quick Start](#quick-start)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Comparison: MF vs GNN](#comparison-mf-vs-gnn)
10. [Next Steps](#next-steps)
11. [References](#references)

---

## Overview

This module explores **Graph Neural Networks (GNNs)** as an evolution from traditional matrix factorization for movie recommendations. GNNs model user-movie interactions as a graph, enabling richer representations and better capturing of complex relationships.

### What's Included

| File | Description |
|------|-------------|
| `GNN_Movie_Recommendation_Learning.ipynb` | Complete learning notebook with implementations |
| `README.md` | This documentation |
| `requirements.txt` | Python dependencies |

---

## Why GNNs for Recommendations?

### The Graph Perspective

Traditional collaborative filtering treats user-item interactions as a matrix. GNNs view it as a **graph**:

```
      ┌──────────────────────────────────────┐
      │         User-Movie Graph             │
      │                                      │
      │   User₁ ──(4.5)──► Movie₁ ◄──(3.0)── User₂
      │     │               │                  │
      │   (5.0)           (similar)          (4.0)
      │     │               │                  │
      │     ▼               ▼                  ▼
      │   Movie₂ ◄───────► Movie₃ ◄───────► Movie₄
      │                                      │
      └──────────────────────────────────────┘
```

### Advantages of GNNs

| Aspect | Matrix Factorization | GNN Approach |
|--------|---------------------|--------------|
| **Structure** | Flat user-item matrix | Rich graph with relationships |
| **Information Flow** | Direct interactions only | Multi-hop neighbor information |
| **Features** | Learned latent factors | Can incorporate side information |
| **Cold Start** | Limited (mean normalization) | Better (propagate from neighbors) |
| **Explainability** | Low | Can trace recommendation paths |
| **Scalability** | Excellent | Good (with sampling techniques) |

---

## Concepts Explained

### 1. Bipartite Graph

Our recommendation graph has two types of nodes:
- **User nodes**: Represent users
- **Movie nodes**: Represent movies
- **Edges**: Connect users to movies they've rated (edge weight = rating)

```python
# Example structure
Nodes: [U₁, U₂, U₃, ..., M₁, M₂, M₃, ...]
Edges: [(U₁, M₁, rating=4.5), (U₁, M₃, rating=5.0), ...]
```

### 2. Message Passing

GNNs learn through **message passing**:

1. **Aggregate**: Each node gathers information from neighbors
2. **Update**: Combine aggregated info with own features
3. **Repeat**: Stack multiple layers for multi-hop information

```
Layer 0: Node knows only itself
Layer 1: Node knows its direct neighbors
Layer 2: Node knows neighbors of neighbors
...
```

### 3. Graph Convolution

Similar to image convolution, but on graphs:

```
Image CNN:  pixel value = f(surrounding pixels)
Graph GNN:  node embedding = f(neighbor embeddings)
```

The key difference: graphs have **irregular structure** (variable number of neighbors).

---

## GNN Architectures

### 1. GCN (Graph Convolutional Network)

**Core idea**: Normalize and average neighbor features

```python
h_v = σ(W · MEAN({h_u : u ∈ N(v) ∪ {v}}))
```

**Pros**: Simple, effective
**Cons**: All neighbors weighted equally

### 2. GraphSAGE

**Core idea**: Sample and aggregate from neighbors

```python
h_v = σ(W · CONCAT(h_v, AGG({h_u : u ∈ sample(N(v))})))
```

**Pros**: Scalable (sampling), inductive learning
**Cons**: Information loss from sampling

### 3. GAT (Graph Attention Network)

**Core idea**: Learn attention weights for neighbors

```python
α_vu = attention(h_v, h_u)  # How important is u to v?
h_v = σ(Σ α_vu · W · h_u)
```

**Pros**: Learns importance of neighbors
**Cons**: More parameters, computationally expensive

### 4. LightGCN ⭐ (Recommended for RecSys)

**Core idea**: Simplify GCN by removing unnecessary components

```python
# No feature transformation, no activation!
h_v^(k) = Σ (1/√|N(v)|√|N(u)|) · h_u^(k-1)

# Final embedding: average across all layers
h_v = (h_v^(0) + h_v^(1) + ... + h_v^(K)) / (K+1)
```

**Pros**: Simple, fast, often best performance for recommendations
**Cons**: Limited expressiveness for complex features

### Architecture Comparison

```
                    ┌─────────────────────────────────────────────┐
                    │        GNN Architecture Complexity          │
                    │                                             │
     Simple ◄───────┼─────────────────────────────────────────────┼───────► Complex
                    │                                             │
                    │  LightGCN    GCN    GraphSAGE    GAT       │
                    │     │         │         │         │         │
                    │     ▼         ▼         ▼         ▼         │
                    │   Best for  Good     Scalable  Attention   │
                    │   RecSys   Baseline             -based     │
                    └─────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv gnn_env
source gnn_env/bin/activate  # Linux/Mac
# or: gnn_env\Scripts\activate  # Windows

# Install PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
from torch_geometric.nn import GCNConv

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("PyTorch Geometric: OK")
```

---

## Project Structure

```
gnn_learning/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── GNN_Movie_Recommendation_Learning.ipynb  # Main learning notebook
│                                            # - Data exploration
│                                            # - Graph construction
│                                            # - Model implementations
│                                            # - Training & evaluation
│                                            # - Visualizations
│
├── models/                                # (To be created)
│   ├── __init__.py
│   ├── gcn.py
│   ├── sage.py
│   ├── gat.py
│   └── lightgcn.py
│
├── utils/                                 # (To be created)
│   ├── __init__.py
│   ├── data_utils.py
│   └── eval_utils.py
│
└── saved_models/                          # (Created after training)
    └── lightgcn_model.pt
```

---

## Quick Start

### 1. Open the Notebook

```bash
jupyter notebook GNN_Movie_Recommendation_Learning.ipynb
```

### 2. Run All Cells

The notebook is self-contained and will:
- Load MovieLens data
- Build the user-movie graph
- Train multiple GNN models
- Compare performance
- Visualize embeddings

### 3. Key Outputs

After running, you'll see:
- Training curves for all models
- RMSE/MAE comparison table
- t-SNE embedding visualizations
- Sample recommendations

---

## Evaluation Metrics

### Rating Prediction Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(Σ(y - ŷ)² / n) | Penalizes large errors more |
| **MAE** | Σ\|y - ŷ\| / n | Average absolute error |

### Ranking Metrics (Advanced)

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K items that are relevant |
| **Recall@K** | Fraction of relevant items in top-K |
| **NDCG@K** | Considers ranking position |
| **Hit Rate** | Did the user's item appear in top-K? |

---

## Comparison: MF vs GNN

### Conceptual Differences

```
Matrix Factorization:
┌─────────────────────────────────────┐
│  User Matrix (U) × Movie Matrix (M) │
│         ↓                           │
│     Rating Prediction               │
└─────────────────────────────────────┘

Graph Neural Network:
┌─────────────────────────────────────┐
│  Build User-Movie Graph             │
│         ↓                           │
│  Message Passing (Multiple Layers)  │
│         ↓                           │
│  Get Final Embeddings               │
│         ↓                           │
│     Rating Prediction               │
└─────────────────────────────────────┘
```

### When to Use What

| Scenario | Recommended Approach |
|----------|---------------------|
| Simple, fast baseline | Matrix Factorization |
| Rich side information | GNN with node features |
| Very sparse data | GNN (better propagation) |
| Real-time updates needed | MF (simpler to update) |
| Explainability required | GNN (trace paths) |
| Production at scale | LightGCN or MF |

---

## Next Steps

### For Learning
- [ ] Complete the notebook end-to-end
- [ ] Experiment with hyperparameters (embedding_dim, num_layers, lr)
- [ ] Try different train/test splits
- [ ] Add your own movie ratings and get recommendations

### For Integration
- [ ] Modularize code into separate Python files
- [ ] Create inference-only pipeline for Streamlit
- [ ] Add node features (genres, year, popularity)
- [ ] Implement efficient cold-start handling

### Advanced Topics
- [ ] Heterogeneous GNNs (multiple edge types)
- [ ] Knowledge Graph integration (actors, directors)
- [ ] Temporal dynamics (user preferences change over time)
- [ ] Contrastive learning for better representations

---

## References

### Papers

1. **LightGCN** (Recommended Reading)
   - He, X., et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
   - [Paper](https://arxiv.org/abs/2002.02126) | [Code](https://github.com/gusye1234/LightGCN-PyTorch)

2. **GCN** (Foundation)
   - Kipf, T. N., & Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.
   - [Paper](https://arxiv.org/abs/1609.02907)

3. **GraphSAGE**
   - Hamilton, W., et al. "Inductive Representation Learning on Large Graphs." NeurIPS 2017.
   - [Paper](https://arxiv.org/abs/1706.02216)

4. **GAT**
   - Veličković, P., et al. "Graph Attention Networks." ICLR 2018.
   - [Paper](https://arxiv.org/abs/1710.10903)

### Tutorials & Resources

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Stanford CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
- [Distill.pub: Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/)

### Dataset

- MovieLens Dataset: Harper, F. M., & Konstan, J. A. "The MovieLens Datasets: History and Context." ACM TIIS 2015.
- [Download](https://grouplens.org/datasets/movielens/)

---

## License

This learning module is part of the Movie Recommendation System project and follows the same MIT License.

---

## Questions?

Feel free to experiment, break things, and learn! The notebook includes detailed comments explaining each step.

**Happy Learning! 🎬🧠**
