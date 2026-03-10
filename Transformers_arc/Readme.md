# Understanding Transformer Architecture From Scratch

A practical engineering explanation of the Transformer architecture introduced in the paper "Attention Is All You Need" (2017).

This article explains the Transformer architecture step-by-step, including the intuition, mathematics, and implementation insights required to truly understand how modern AI models work.

Transformers power many of the most advanced AI systems today:
- GPT models
- BERT
- Vision Transformers
- Stable Diffusion
- Whisper

Unlike traditional sequence models, transformers rely entirely on attention mechanisms rather than recurrence or convolution.

## Table of Contents
1. [Why Transformers Were Needed](#1-why-transformers-were-needed)
2. [High Level Transformer Architecture](#2-high-level-transformer-architecture)
3. [Tokenization](#3-tokenization)
4. [Embedding Layer](#4-embedding-layer)
5. [Positional Encoding](#5-positional-encoding)
6. [Query Key Value Representation](#6-query-key-value-representation)
7. [Scaled Dot Product Attention](#7-scaled-dot-product-attention)
8. [Multi Head Attention](#8-multi-head-attention)
9. [Residual Connections and LayerNorm](#9-residual-connections-and-layernorm)
10. [Feed Forward Network](#10-feed-forward-network)
11. [Encoder Architecture](#11-encoder-architecture)
12. [Decoder Architecture](#12-decoder-architecture)
13. [Transformer Training Pipeline](#13-transformer-training-pipeline)
14. [Attention Visualization](#14-attention-visualization)
15. [Computational Complexity](#15-computational-complexity)
16. [Scaling Laws](#16-scaling-laws)
17. [Why Transformers Dominated Deep Learning](#17-why-transformers-dominated-deep-learning)

---

## 1. Why Transformers Were Needed
Before transformers, sequence modeling used:
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)

These models process tokens sequentially: `Word1 → Word2 → Word3 → Word4`

### Problems:
**Sequential Computation**
$$h_t = f(h_{t-1}, x_t)$$
Each token depends on the previous hidden state, making parallelization difficult.

**Vanishing Gradients**
Long-range relationships are difficult to maintain. 
*Example:* "The **animal** that chased the cat **was** hungry" — The model must connect "was" directly to "animal".

**Training Bottleneck**
GPUs cannot fully parallelize sequential models. Transformers solve these problems using self-attention.

---

## 2. High Level Transformer Architecture
The original transformer contains:
- **Encoder Stack**: Processes the input sequence.
- **Decoder Stack**: Generates the output sequence.

<p align="center">
  <img src="images/Architecture Pipeline.png" alt="Architecture Pipeline" width="800">
  <br>
  <b>Architecture Pipeline</b>
</p>

Each stack contains multiple identical layers.

---

## 3. Tokenization
Text must be converted into tokens.
*Example sentence:* "I'm learning transformers"
*Tokenized as:* `["I", "am", "learning", "transformers"]`

Converted into integer IDs:
- I → 12
- am → 47
- learning → 502
- transformers → 819

---

## 4. Embedding Layer
Neural networks operate on vectors, not integers. Each token is mapped to a dense vector.
*Example:* `learning → [0.18, 0.91, 0.02, ...]`

Mathematically:
$$X \in \mathbb{R}^{n \times d}$$
Where:
- $n$ = sequence length
- $d$ = embedding dimension (e.g., 512)

### PyTorch Example
```python
import torch
import torch.nn as nn

embedding = nn.Embedding(10000, 512)
tokens = torch.tensor([12, 47, 502, 819])
embedded = embedding(tokens)
print(embedded.shape) # Output: (4, 512)
```

---

## 5. Positional Encoding
Transformers process tokens in parallel, so they need positional information.
$$Input = Embedding + Positional Encoding$$

### Mathematical Definition
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Where:
- `pos` = token position
- `i` = dimension index
- `d` = embedding size

### Visualization
`Token Embedding + Positional Vector → Position-aware embedding`

---

## 6. Query Key Value Representation
Each embedding is projected into three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**.

Linear projections:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

### Intuition
| Vector | Meaning |
| :--- | :--- |
| **Query** | What information am I searching for? |
| **Key** | What information do I contain? |
| **Value** | The actual information to pass forward. |

### PyTorch Example
```python
Wq = nn.Linear(512, 512)
Wk = nn.Linear(512, 512)
Wv = nn.Linear(512, 512)

Q = Wq(embedded)
K = Wk(embedded)
V = Wv(embedded)
```

---

## 7. Scaled Dot Product Attention
Core transformer computation:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Steps:
1. **Compute similarity**: $QK^T$
2. **Scale**: Divide by $\sqrt{d_k}$
3. **Normalize**: Apply Softmax
4. **Aggregate**: Weighted value sum

---

## 8. Multi Head Attention
Instead of one attention calculation, transformers compute multiple attention heads to capture different types of relationships (syntactic, semantic, positional).

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O$$

### PyTorch Implementation
```python
attention = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8
)
```

---

## 9. Residual Connections and LayerNorm
Transformers use skip connections to stabilize training:
$$y = \text{LayerNorm}(x + \text{Attention}(x))$$

**Benefits:**
- Stabilizes gradients
- Enables deeper networks
- Prevents vanishing gradients

<p align="center">
  <img src="images/Residual Diagram.png" alt="Residual Diagram" width="800">
  <br>
  <b>Residual Diagram</b>
</p>

---

## 10. Feed Forward Network
Each token passes through a small neural network independently:
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Typical dimensions: `512 → 2048 → 512`

### PyTorch Implementation
```python
ffn = nn.Sequential(
    nn.Linear(512, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512)
)
```

---

## 11. Encoder Architecture
Each encoder layer contains:
1. Multi-Head Attention
2. Add & Norm
3. Feed Forward Network
4. Add & Norm

<p align="center">
  <img src="images/Encoder Diagram.png" alt="Encoder Diagram" width="800">
  <br>
  <b>Encoder Diagram</b>
</p>

---

## 12. Decoder Architecture
The decoder includes **Masked Attention**, which prevents the model from seeing future tokens during training.

*Example:* 
- Input: "I love"
- Hidden: "machine learning" (The model cannot see these yet)

<p align="center">
  <img src="images/Decoder Diagram.png" alt="Decoder Diagram" width="800">
  <br>
  <b>Decoder Diagram</b>
</p>

---

## 13. Transformer Training Pipeline
The training process involves a loop of forward passes, loss calculation, and backpropagation.

<p align="center">
  <img src="images/Transformer Training Pipeline.png" alt="Transformer Training Pipeline" width="800">
  <br>
  <b>Transformer Training Pipeline</b>
</p>

### PyTorch Setup
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
```

---

## 14. Attention Visualization
Attention can be visualized using heatmaps to show which tokens influence others.

| Token | I | am | learning | transformers |
| :--- | :---: | :---: | :---: | :---: |
| **learning** | 0.1 | 0.2 | 0.3 | 0.4 |

*Interpretation:* "learning" attends strongly to "transformers".

**Visual tools:** BertViz, Transformer Explainer.

---

## 15. Computational Complexity
Transformer attention complexity is quadratic with respect to sequence length:
$$O(n^2 \cdot d)$$

Where $n$ is sequence length and $d$ is embedding dimension.

| Model | Complexity |
| :--- | :--- |
| RNN | $O(n)$ |
| CNN | $O(n \log n)$ |
| **Transformer** | $O(n^2)$ |

---

## 16. Scaling Laws
Model performance scales predictably with size, data, and compute:
$$\text{Loss} \propto N^{-\alpha}$$
Where $N$ is the number of parameters and $\alpha \approx 0.05 - 0.1$.

---

## 17. Why Transformers Dominated Deep Learning
- **Parallel Computation**: Entire sequences processed at once.
- **Long-range Dependencies**: Any token can attend to any other.
- **Scalability**: Works efficiently at massive scales.
- **Generality**: Works across language, vision, speech, and more.

---

## References
- [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Transformer Tutorial Video](https://youtu.be/psUsPLMKMn0)
- [Vision Transformer Implementation](https://www.youtube.com/watch?v=ZRo74xnN2SI)

### Author
Moniha P S
Bluvern
Designed and documented for AI Research.