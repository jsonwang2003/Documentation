---
title: "Deep Deterministic Policy Gradient (DDPG)"
description: "An off-policy, actor-critic algorithm designed for continuous action spaces."
aliases:
  - DDPG
tags:
  - CS/artificial-intelligence
  - machine-learning
  - reinforcement-learning
---

> [!abstract] Abstract
> An off-policy, actor-critic algorithm designed for environments with continuous action spaces. It blends the strength of [[Deep Q-Network (DQN)\|DQN]] and [[Policy Gradient Methods]] to learn both a **[[Q-Learning]]** and a deterministic policy simultaneously.

---
## Architecture & Algorithm

**Model Components:**
- **Actor Network:** Learns a deterministic policy $\mu(s\|\theta^\mu)$ that maps states to actions.
- **Critic Network:** Estimates the Q-value $Q(s, a\|\theta^Q)$ for given state-action pairs.
- **Target Networks:** Stabilize training by slowly updating target versions of actor and critic.
- **Replay Buffer:** Stores transitions $(s, a, r, s')$ for off-policy learning.
- **Exploration Noise:** Adds noise (e.g. Ornstein-Uhlenbeck) to actions for exploration in continuous space.

---
## Model Evaluation & Features

- **Deterministic Policy:** Unlike stochastic Policy Gradient methods, Deep Deterministic Policy Gradient uses a deterministic actor. It is efficient in high-dimensional action spaces where sampling is costly.
- **Off-Policy Learning:** Learns from stored experiences, improving sample efficiency and enabling the reuse of past trajectories for training.
- **Actor-Critic Architecture:** Actor proposes actions, critic evaluates them, and critic guides actor updates via the gradient of Q-values.
- **Target Networks:** Reduce training instability by slowing updating target parameters: 

$$
	\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-
$$

- **Exploration via Noise:** Adds temporary correlated noise (e.g., Ornstein-Uhlenbeck) to encourage exploration, which is especially useful in physical control tasks.

---
# Related Notes

- [[Artificial Intelligence/Machine Learning/Deep Learning/Reinforced Deep Learning Models/index\|Reinforced Deep Learning Models]]