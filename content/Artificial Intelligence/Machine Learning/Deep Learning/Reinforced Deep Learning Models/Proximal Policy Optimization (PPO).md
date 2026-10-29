---
title: "Proximal Policy Optimization (PPO)"
description: "A policy optimization algorithm using clipped surrogate objectives for stability."
aliases:
  - PPO
tags:
  - CS/artificial-intelligence
  - machine-learning
  - reinforcement-learning
---

> [!abstract] Abstract
> Simplifies trust region policy optimization by constraining updates within a predefined clipping threshold, providing stability and reliability during training[cite: 43]. It is a more advanced variant of Deep Reinforcement Learning that significantly reduces sensitivity to hyperparameter selection and ensures consistent learning across diverse environments[cite: 43].

---
## Architecture & Algorithm

**Model Components:**
- **Stochastic Policy:** Learns $\pi_\theta(a|s)$, a probability distribution over actions[cite: 43].
- **Clipped Objective:** Limits how much the new policy can deviate from the old one during updates[cite: 43].
- **Actor-Critic Architecture:** The actor decides what action to take and updates the policy; the critic evaluates how good that action was and estimates the value function $V(s)$[cite: 43]. This setup makes learning more stable and reduces the randomness in the updates[cite: 43].
- **Advantage Estimation:** Uses [[Generalized Advantage Estimation (GAE)]] for variance reduction[cite: 43].
- **On-Policy Learning:** Uses fresh trajectories from the current policy for updates[cite: 43].

---
## Mathematical Formulation

> [!info] Clipped Surrogate Objective:
> Prevents large, destabilizing policy updates:
> $$ L^{CLIP}(\theta) = \mathbb{E}_t \left[\min \left(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)A_t \right) \right] $$[cite: 43]
> Where: $$ r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} $$[cite: 43]

---
## Model Evaluation & Features

- **Trust Region Approximation:** Inspired by [[TRPO]] but avoids second-order derivatives by using clipping instead of [[KL-divergence]] constraints[cite: 43].
- **Sample Efficiency:** More efficient than vanilla policy gradients because it can reuse mini-batches for multiple epochs[cite: 43].

---
## Systems & Applications

> [!example] Deployment Contexts:
> - Works well in high-dimensional, continuous control tasks[cite: 43].
> - Used extensively in robotics, games, and simulated physics[cite: 43].

---
# Related Notes

- [[Artificial Intelligence/Machine Learning/Deep Learning/Reinforced Deep Learning Models/index\|Reinforced Deep Learning Models Index]]