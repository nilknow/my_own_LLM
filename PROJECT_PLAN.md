# Project Plan: Build a Transformer (GPT-like) from Fundamentals

This file tracks the actionable plan, progress and milestones for building a small-scale GPT-like model with minimal libraries.

Goals
- Understand and implement the core components of a Transformer for text generation.
- Focus on a toy model and learning, not production-scale GPT-2 training.

Assumptions
- Python + NumPy (optionally PyTorch for GPU later).
- Basic linear algebra and calculus familiarity.

Milestones
- Phase 1: Foundations & Core Components
- Phase 2: Assemble a Mini GPT-like Model & Basic Training
- Phase 3: Scaling Concepts & Advanced Topics

---

Phase 1: Foundations & Core Components (Weeks 1–4)

Week 1: Python, NumPy, and Data Basics
- [x] Day 1–2: Python & NumPy
  - [x] Review Python OOP, functions, classes
  - [x] Practice NumPy (array ops, matmul, broadcasting)
- [x] Day 3–4: Text Data Handling
  - [x] Understand UTF-8, large text file handling
  - [x] Project: Load text and count char/word frequencies
- [x] Day 5–7: Basic NN Building Blocks (NumPy)
  - [x] Implement Linear layer (forward/backward)
  - [x] Implement activations: ReLU, Softmax
  - [x] Implement Cross-Entropy loss
  - [x] 2-layer NN forward/backward demo

Week 2: Tokenization and Embedding
- [x] Day 1–3: Character-level Tokenization
  - [x] Build char<->id vocab
  - [x] Encode/decode text sequences
  - [x] Project: Simple Bigram model (predict next char)
- [x] Day 4–5: Subword Tokenization Concepts
  - [x] Study BPE/WordPiece and trade-offs
- [x] Day 6–7: Embedding Layer (NumPy)
  - [x] Implement embedding lookup (forward/backward)

Week 3: Attention Mechanism
- [x] Day 1–3: Scaled Dot-Product Attention (NumPy)
  - [x] Implement Q/K/V projections
  - [x] Implement scaled dot-product and softmax
  - [x] Verify with small test matrices
- [x] Day 4–5: Multi-Head Attention (NumPy)
  - [x] Multi-head split/concat + output projection
- [x] Day 6–7: Masked (Causal) Attention
  - [x] Implement causal mask to prevent attending to future tokens

Week 4: Positional Encoding & Normalization & FFN
- [x] Day 1–2: Positional Encoding (sinusoidal)
  - [x] Implement and add to embeddings
- [x] Day 3–4: Layer Normalization
  - [x] Implement LayerNorm (forward/backward)
- [x] Day 5–7: Feed-Forward + Residual (Add & Norm)
  - [x] 2-layer FFN (Linear-ReLU-Linear)
  - [x] Residual connections with LayerNorm
  - [x] Combine into a single Transformer Block

---

Phase 2: Assembling the Model & Basic Training (Weeks 5–8)

Week 5: Decoder Block Assembly & Mini Language Model
- [x] Day 1–3: Full Decoder Block (Masked MHA + Add&Norm + FFN + Add&Norm)
  - [x] Validate forward pass end-to-end
- [x] Day 4–5: Simple Language Model (NumPy)
  - [x] Stack N decoder blocks
  - [x] Final Linear projection -> vocab
  - [x] Softmax for next-token probabilities
  - [x] Verify forward pass on a toy batch
- [x] Day 6–7: Framework Introduction (PyTorch recommended)
  - [x] Learn torch.Tensor, nn.Module, nn.Parameter, optim
  - [x] Re-implement 2-layer NN in PyTorch

Week 6: Port Core Components to PyTorch
- [ ] Day 1–5: Convert components to PyTorch nn.Modules
  - [ ] Embedding, Linear, LayerNorm
  - [ ] MultiHeadAttention, PositionalEncoding, FeedForward
- [ ] Day 6–7: Full Mini-GPT (Decoder-only) in PyTorch
  - [ ] Assemble end-to-end module

Week 7: Data Pipeline for Training
- [ ] Day 1–3: Dataset & Batching
  - [ ] Prepare a tiny text dataset
  - [ ] Character tokenizer integration
  - [ ] Fixed-length sequences with padding
- [ ] Day 4–5: Loss & Optimizer
  - [ ] Cross-Entropy loss (PyTorch)
  - [ ] Adam/AdamW optimizer
- [ ] Day 6–7: Basic Training Loop
  - [ ] Train for a few epochs and log loss

Week 8: Training Refinements & Overfitting
- [ ] Day 1–3: LR Scheduling (warm-up/decay)
- [ ] Day 4–5: Gradient Clipping
- [ ] Day 6–7: Overfit a Single Batch (sanity check)

---

Phase 3: Scaling & Advanced Concepts (Weeks 9–12)

Week 9: Tokenization & Dataset
- [ ] Day 1–3: Simple BPE (conceptual or minimal implementation)
- [ ] Day 4–7: Prepare a small clean dataset (e.g., Gutenberg subset)

Week 10: Model Size & Hyperparameters
- [ ] Day 1–3: Increase model capacity (layers/heads/hidden size)
- [ ] Day 4–7: Manual hyperparameter tuning (LR, batch size, epochs)

Week 11: Generation & Evaluation
- [ ] Day 1–3: Decoding
  - [ ] Greedy
  - [ ] Temperature sampling
  - [ ] Top-k and Top-p sampling
- [ ] Day 4–7: Evaluation
  - [ ] Understand and report perplexity
  - [ ] Qualitative inspection of samples

Week 12: Distributed Training (Conceptual) & Cloud Setup
- [ ] Day 1–3: Data vs Model Parallelism; DDP basics
- [ ] Day 4–7: Optional: Cloud GPU environment setup & run

---

Backlog / Nice to Have
- [ ] Mixed precision training (AMP)
- [ ] Gradient checkpointing
- [ ] Efficient data loading & caching
- [ ] Logging/Visualization (TensorBoard or simple CSV logs)
- [ ] Save/Load checkpoints

Progress Log (append entries below)
- 2025-..-..: ...

Notes
- Keep tasks small and iterative; aim for correctness before speed.
- Prefer unit tests for each component (embedding, attention, layer norm, etc.).
- For this repository, existing files under `text/` and `attention/` can be iteratively replaced or complemented by PyTorch versions during Phase 6.