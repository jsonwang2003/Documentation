
>[!INFO]
> An **off-policy**, **actor-critic algorithm** designed for environments with **continuous action spaces**.

- Blends the strength of [[Deep Q-Network (DQN)|DQN]] and [[#Policy Gradient Methods]] to learn both a [[Q-function]] and a **deterministic policy** simultaneously

## Components

- **Actor Network**: Learns a deterministic policy $\mu(s\|\theta^\mu)$ that maps states to actions
- **Critic Network**: Estimates the Q-value $Q(s, a\|\theta^Q)$ for given state-action pairs
- **Target Networks**: Stabilize training by **slowly updating target versions** of _actor_ and _critic_
- **Replay Buffer**: Stores transitions $(s, a, r, s')$ for **off-policy** learning
- **Exploration Noise**: Adds noise (e.g. [[Ornstein-Uhlenbeck]]) to actions for exploration in continuous space
## Key Features

1. **Deterministic Policy**
	- Unlike stochastic **Policy Gradient** methods, **Deep Deterministic Policy Gradient** uses a deterministic actor
	- Efficient in high-dimensional action spaces where sampling is costly
2. **Off-Policy Learning**
	- Learns from stored experiences, improving sample efficiency
	- Enables reuse of past trajectories for training
3. **[Actor-Critic Architecture](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/)[^2]**
	- **Actor** proposes actions
	- **Critic** evaluates them
	- **Critic** guides **actor** updates via _gradient of Q-values_
4. **Target Networks**
	- Reduce training instability by slowing updating target parameters
$$
\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-
$$
5. **Exploration via Noise**
	- Adds temporary correlated noise (e.g. [[Ornstein-Uhlenbeck]]) to encourage exploration
	- especially useful in **physical control tasks**