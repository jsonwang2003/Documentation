---
title: "Artificial Intelligence Index"
description: "Core concepts, algorithms, and architectures for machine learning, deep learning, and AI systems."
tags:
  - CS/index
  - CS/artificial-intelligence
aliases: ["Artificial Intelligence", "AI Index"]
---

> [!abstract] Overview
> Architectures, algorithms, and statistical foundations for Artificial Intelligence. This category covers machine learning paradigms, deep learning architectures, statistical model evaluation, and the systems engineering required to train and deploy them.

---

# Foundational Concepts

## Statistical Inference & Optimization
{{Definitions of probability distributions, gradients, loss functions, and the optimization techniques used to train models — the mathematical engine underneath every architecture below.}}

## Model Evaluation
{{Metrics for success (e.g., accuracy, precision, recall, MSE), validation techniques (cross-validation), and the bias-variance tradeoff.}}

---

# Core Framework / Shared Architecture

## The Learning Pipeline
Nearly every model in this section follows a shared computational lifecycle:
- **Data Representation:** Structuring inputs (tensors, embeddings, feature vectors) for computation.
- **Forward Pass (Inference):** The mathematical transformation from input to prediction.
- **Loss Calculation:** Quantifying the error against ground truth or expected reward.
- **Backward Pass (Optimization):** Updating weights/parameters to minimize future error (e.g., gradient descent, backpropagation).

---

# Notes in This Section

| Note | One-line description |
| :--- | :--- |
| [[{{Supervised Learning}}]] | {{Algorithms that learn from labeled training data (e.g., linear regression, SVMs, decision trees)}} |
| [[{{Unsupervised Learning}}]] | {{Algorithms that discover inherent patterns in unlabeled data (e.g., K-means clustering, PCA)}} |
| [[{{Deep Learning}}]] | {{Neural network architectures (CNNs, RNNs, Transformers) and backpropagation mechanics}} |
| [[{{Reinforcement Learning}}]] | {{Agents learning optimal policies via cumulative rewards in structured environments}} |
| [[{{AI Systems & Ethics}}]] | {{Hardware constraints, ML systems engineering, bias mitigation, and policy}} |

---

# Related Categories

- [[Discrete Structures Index]] — discrete probability and matrix mathematics foundational to statistical inference.
- [[Algorithms]] — optimization techniques and state-space search algorithms (e.g., A*, Minimax) often utilized in classical AI.
- [[Computer Systems Index]] — low-level hardware constraints and memory management impacting ML systems engineering.