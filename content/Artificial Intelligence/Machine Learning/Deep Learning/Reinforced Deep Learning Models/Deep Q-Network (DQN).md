---
title: "Deep Q-Network (DQN)"
description: "Combines Q-learning with Deep Neural Networks to handle high-dimensional state spaces."
aliases:
  - DQN
tags:
  - CS/artificial-intelligence
  - machine-learning
  - reinforcement-learning
---

> [!abstract] Abstract
> Combines [[Q-Learning]] with [[Deep Neural Network (DNN)]] to handle high-dimensional state spaces[cite: 41]. It uses experience replay and fixed Q-targets to stabilize training, allowing the model to efficiently learn from high-dimensional data[cite: 41].

---
## Architecture & Algorithm

**Training Workflow:**
1. Initialize replay buffer and networks (main and target)[cite: 41].
2. For each step:
   - Select action using $\epsilon$-greedy policy[cite: 41].
   - Store transition in replay buffer[cite: 41].
   - Sample mini-batch and compute loss[cite: 41].
   - Update main network via gradient descent[cite: 41].
   - Periodically sync target network[cite: 41].

---
## Mathematical Formulation

> [!info] Bellman Loss Optimization:
> Uses the Bellman equation to compute temporal difference (TD) error:
> 
> $$
>	L(\theta) = \mathbb{E}[(r + \gamma \max_{\alpha'} Q(s',a'; \theta^{-}) - Q(s,a; \theta))^{2}]
> $$

---
## Model Evaluation & Features

- **Function Approximation with [[Deep Neural Network (DNN)]]:** Replaces traditional [[Q-tables]] with a neural network to estimate $Q(s, a)$, enabling learning in environments with large or continuous state spaces[cite: 41].
- **Experience Replay:** Stores transitions $(s, a, r, s')$ in a buffer and randomly samples mini-batches to break temporal correlations, which improves data efficiency and stabilizes training[cite: 41].
- **Target Network:** Maintains a separate, slowly-updated copy of the [[Q-network]][cite: 41]. It reduces oscillations and divergence during training by syncing target network parameters $\theta ^-$ every $N$ steps[cite: 41].
- **$\epsilon$-Greedy Exploration:** Balances exploration and exploitation[cite: 41]. Starts with a high $\epsilon$ (more exploration) and decays over time to favor the learned policy[cite: 41].
- **Scalability:** Can process raw image frames using convolutional layers[cite: 41].

---
# Related Notes

- [[Artificial Intelligence/Machine Learning/Deep Learning/Reinforced Deep Learning Models/index\|Reinforced Deep Learning Models]]