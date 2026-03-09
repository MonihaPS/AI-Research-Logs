# The Transformer Architecture: Research & Implementation Notes

Welcome to the research documentation on the **Transformer Architecture**, a groundbreaking model first introduced in the seminal research paper by Vaswani et al. ([Attention Is All You Need, 2017](https://arxiv.org/abs/1706.03762)). This repository combines insights from the original paper with practical handwritten notes and PyTorch implementations, covering both classic NLP Transformers and modern Vision Transformers (ViT).

---

## 📸 Architecture Overview
![Transformer Architecture](Transformer%20Architecture.webp)
*Figure 1: The Encoder-Decoder structure of the original Transformer from "Attention Is All You Need".*

---

## 1. Data Preparation: Tokenization & Embeddings
Before data can be processed by a Transformer, it must be converted from raw text (or images) into mathematical representations.

1. **Tokenization:** Input sentences are broken down into discrete tokens (words or subwords).
2. **Input Embeddings:** These tokens are converted into high-dimensional dense vectors (e.g., 768 dimensions). This embedding step captures the semantic meaning of each token.
3. **Positional Encoding:** Unlike older Sequential models (RNNs, LSTMs, GRUs) that process data sequentially (and suffer from memory loss or vanishing gradients), Transformers process data in **parallel**. Positional encodings act like "GPS coordinates" added to the input embeddings so that the model understands the positional order of the words. Formally, this uses sine and cosine functions of different frequencies for even and odd positions:
   $$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
   $$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}}) $$

---

## 2. Core Engine: Self-Attention & The Q, K, V Matrices
Self-attention is the mechanism that allows the model to weigh the importance of different words in a sentence relative to a specific word. It computes an attention score for each word.

The inputs are separated into three distinct vectors:
* **Query ($Q$):** What am I looking for? (The current focus word)
* **Key ($K$):** What do others have? (The traits of other words in the sentence)
* **Value ($V$):** The actual information contained in the word.

### Scaled Dot-Product Attention
This maps a query and a set of key-value pairs to an output. The formula is:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

By taking the dot product of $Q$ and $K$ (and dividing by $\sqrt{d_k}$ to stabilize gradients), we obtain the "attention scores". We apply a Softmax function to normalize these weights to probabilities (0 to 1). Finally, we multiply these weights by $V$ to emphasize the most contextually relevant words.

![Attention Mechanisms Diagram](Attentions.png)

### Multi-Head Attention
Instead of performing a single attention function, the Q,K,V vectors are linearly projected $h$ times in parallel. The results are concatenated and projected again. This allows the model to jointly attend to information from different representation subspaces at different positions. The formula is defined as:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $$
where $$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

Here, the projections are parameter matrices $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, and $W^O \in \mathbb{R}^{hd_v \times d_{model}}$.

---

## 3. The Architecture Components

### A. The Encoder (Understanding the Input)
The Encoder is responsible for deeply analyzing the input features. It consists of multiple identical layers, each containing:
* **Multi-Head Self-Attention**
* **Position-wise Feed-Forward Networks (FFN):** These process the extracted features from the previous attention stage. It consists of two linear transformations with a ReLU activation in between:
  $$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$
* **Add & Norm (Skip/Residual Connections):** The model adds the input of the layer to its output, protecting against the vanishing gradient problem. Layer Normalization ($Mean=0, Variance=1$) keeps the numbers at a manageable scale to maintain math stability.

![Skip Connections Architecture](Skip%20connections.png)

### B. The Decoder (Generating the Output)
The Decoder is an auto-regressive model that takes the encoder's output and the previously generated outputs to predict the next token. 
* **Masked Multi-Head Attention:** It ensures that the model can only look at *past* words and blocks future tokens to prevent the model from "cheating".
* **Cross-Attention:** Computes attention using the Decoder's $Q$ and the Encoder's $K$ and $V$.
* **Linear & Softmax:** Converts the decoder output into a predicted word probability distribution. The token with the highest probability is generated.

---

## 4. Vision Transformers (ViT) & Data
*"Treat a photograph exactly like a sentence of words."*

A significant comparison to classic Convolutional Neural Networks (CNNs) is the introduction of **Vision Transformers (ViT)**.
1. **Tokenization (Patches):** Instead of processing raw pixels, an image (e.g., $224 \times 224$ pixels) is chopped into a perfect grid of squares called *patches* (e.g., $P \times P = 16 \times 16$). This yields $N = \frac{H \times W}{P^2} = 196$ patches. Each patch is treated as a "word" in a sentence.
2. **Flattening / Patch Embedding (`nn.Conv2d` / `nn.Linear`):** We flatten these 3D blocks of pixels ($16 \times 16 \times 3$ RGB channels) into a single 1D list of $768$ numbers. This linear projection transforms raw color to a rich feature vector.
3. **[CLS] Token Prepending:** Before the 196 tokens are sent to the main engine, we add one more vector called the `[CLS]` (Classification) token. The sequence now has **197 tokens**. The `[CLS]` acts like a "sponge"—it absorbs the global context from all other 196 patches while they exchange information. Ultimately, for image classification, the network just looks at the output of the `[CLS]` token.
4. **Positional Encoding:** Patch vectors + Positional vectors. Added precisely like in text to give spatial context, stamping "GPS coordinates" onto every single patch to generate 197 unique positional vectors.

### Tensors & PyTorch Level Implementation
Tensors represent data across varying dimensions (a grid of numbers):
* **1D:** List (e.g., `[1.0, 2.0, 3.0]`)
* **2D:** Rows and columns (Matrices)
* **3D:** Images with color channels (Height, Width, RGB)
* **4D:** Vision Batches (Batch size, Channels, Height, Width) -> `B, C, H, W`
* **5D:** Video sequences -> `B, T, C, H, W`

**PyTorch functions commonly used:**
* `torch.nn.Linear`: Fully connected layer used to create Q, K, V matrices and classify/shrink data.
* `torch.nn.Conv2d`: Often used practically to fast-chop images into patches natively.
* `torch.nn.MultiheadAttention`: The core engine taking Q, K, V to yield context.
* `torch.nn.LayerNorm`: Normalizes data.
* `Autograd`: PyTorch invisibly records operations on tensors so that calling `.backward()` instantly calculates all required calculus/gradients.

---

## 5. Practical Implementation Flow: ViT on MNIST
*Based on PyTorch Vision Transformer architecture building flows.*

To implement a Vision Transformer from scratch (e.g., classifying handwritten digits from the MNIST dataset), the process is practically broken down into exactly 6 phases:

* **Phase 1: Data Loading (MNIST) & Setup**
  Importing standard `torch` utilities. Creating the base network using `nn.Module`. By calling `super().__init__()`, the child class inherits properties from the parent `nn.Module` layer without overwriting them, so we can add new logic on top.

* **Phase 2: Patch Embedding**
  Slicing the image using `nn.Conv2d`. For a typical $224 \times 224$ image with $16 \times 16$ patches, this step yields an output shape of `(Batch, 196, 768)`. 196 tokens are created by converting the image.

* **Phase 3: Positional Encoding & [CLS] Token**
  Using `nn.Parameter` to generate learnable numbers for our Positional Encoding vectors and prepending the `[CLS]` sequence. 

* **Phase 4: Multi-Head Attention**
  Utilizing `nn.Linear` as the generator to create Query ($Q$), Key ($K$), and Value ($V$) representations. This pushes into the Self-Attention mechanism, where all 197 GPS-stamped vectors look at each other to capture contextual relationships.

* **Phase 5: LayerNorm, MLP & Skip Connections (Add)**
  Keeping the math stable. This acts as a mini-NN enriching the context of the vector while residual skip connections maintain the original information flow without degradation.

* **Phase 6: Model Training Loop**
  The standard PyTorch training loop on MNIST dataset data: loading, train-val split, batching, forward pass, `loss.backward()`, and `optimizer.step()`.

![MNIST Training Loop Results](MNIST%20results.png)

---
*This repository serves as a master reference for understanding transformer dynamics across NLP and Computer Vision fields.*
