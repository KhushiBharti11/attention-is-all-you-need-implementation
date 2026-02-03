# Transformer from Scratch – Attention Is All You Need

## Overview
This project presents a clean, minimal, and explainable implementation of the
Transformer architecture proposed in the paper **“Attention Is All You Need”**
(Vaswani et al., 2017).

The primary goal of this implementation is to demonstrate a strong understanding
of the Transformer’s **architecture**, **mathematical foundations**, and
**practical implementation**, rather than achieving state-of-the-art performance.

The model is implemented from scratch using PyTorch and follows the original
paper closely.

---

## What This Project Demonstrates
- Ability to translate a research paper into a working implementation
- Clear understanding of self-attention and sequence-to-sequence modeling
- Sound engineering judgment and clean code organization
- Explainability and correctness over unnecessary complexity

---

## Implemented Components (Mapped to the Paper)

This implementation follows **Section 3** of the original paper:

- **Scaled Dot-Product Attention** (Section 3.2.1)
- **Multi-Head Attention** (Section 3.2.2)
- **Masked Self-Attention in the Decoder** (Section 3.2.3)
- **Sinusoidal Positional Encoding** (Section 3.5)
- **Encoder Blocks** with residual connections and layer normalization
- **Decoder Blocks** with encoder–decoder attention
- **Full Transformer architecture** (Figure 1 in the paper)

---

## Project Structure

├── Attention_Is_All_You_Need.ipynb
├── README.md
└── requirements.txt

- `Attention_Is_All_You_Need.ipynb`  
  Contains the full implementation, explanations, and test runs.
- `README.md`  
  Project overview and usage instructions.
- `requirements.txt`  
  Minimal dependency list.

---

## Model Description

The model is a **sequence-to-sequence Transformer** consisting of:

- An **Encoder** that processes the entire input sequence in parallel and
  produces contextual representations using self-attention.
- A **Decoder** that generates the output sequence autoregressively using:
  - masked self-attention
  - encoder–decoder attention
- A final linear projection layer that produces a probability distribution
  over the vocabulary for each output position.

The model outputs logits of shape:
(batch_size, sequence_length, vocabulary_size)

---

## Mathematics (Core Equation)

The central operation implemented in this project is **scaled dot-product attention**:

\[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

This formulation enables the model to capture long-range dependencies efficiently
while maintaining stable gradients during training.

---

## How to Run

### Requirements
- Python 3.8+
- PyTorch

Install dependencies:
```bash
pip install -r requirements.txt



