
> [!INFO]
> Leverages [[neural networks]] to model complex decision-making process


# Deep Q-Network (DQN)

> [!INFO]
> Combines [[Q-learning]] with [[Deep Neural Networks]] to handle high-dimensional state spaces

- Uses experience replay and fixed Q-targets to stabilize training
	- Allows model to efficiently learn from high-dimensional data

## Training Workflow

1. Initialize replay buffer and networks (main and target)
2. For each step: 
	- Select action using $\epsilon$-greedy policy
	- Store transition in replay buffer
	- Sample mini-batch and compute loss
	- Update main network via gradient descent
	- Periodically sync target network
## Key Feature

- **Function Approximation with [[Deep Neural Networks]]**
	- Replaces traditional [[Q-tables]] with a neural network to estimate $Q(s, a)$
	- Enables learning in environments with **large** or **continuous state spaces**
- **Experience Replay**
	- Stores transitions $(s, a, r, s')$ in a buffer
	- Randomly samples mini-batches to **break temporal correlations**
	- Improves data efficiency and stabilizes training
- **Target Network**
	- Maintains a separate, slowly-updated copy of the [[Q-network]]
	- Reduces oscillations and divergence during training
	- Target network parameters $\theta ^-$ are synced every $N$ steps
- **$\epsilon$-Greedy Exploration**
	- Balances _exploration_ and _exploitation_
	- Starts with high $\epsilon$ (more exploration), decays over time to favor learned policy
- **Bellman Loss Optimization**
	- Uses the Bellman equation to compute temporal difference (TD) error
$$
L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]
$$
- **Scalability to Visual Input**
	- Can process raw image frames using convolutional layers

# Policy Gradient Methods

>[!INFO]
> A class of **reinforcement learning algorithms** that directly optimize the **policy** $\pi_\theta(a|s)$ using **gradient ascent** on expected rewards, rather than estimating value functions like DQN does.

- Exemplified by algorithms like
	- **[REINFORCE](https://www.geeksforgeeks.org/machine-learning/reinforce-algorithm/)**[^1]
	- **[Actor-Critic Methods](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/)**[^2]
	- [[#Proximal Policy Optimization (PPO)]][^3]
- Directly parameterize policy functions
	- Allow agent to select actions based on probabilities derived from neural networks
- Useful for environments with
	- high-dimensional action spaces
	- When the goal is to **optimize directly for policy performance**
## Key Features

1. **Direct Policy Optimization**
	- Learns the policy directly without relying on value functions
	- Ideal for **continuous or high-dimensional action spaces**
2. **Stochastic Policies**
	- Naturally supports exploration via [[probabilistic action selection]]
	- Useful in partially observable or noisy environments
3. **High Variance, Low Bias**
	- Monte Carlo estimates can be noisy but unbiased
	- Techniques like [[baseline subtraction]] or [[advantage functions]] help reduce variance
4. **Compatible with [Actor-Critic Architectures](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/)[^2]**
	- Combines policy (actor) and value function (critic) for more stable learning
	- Critic estimates value to guide actor’s updates

# Deep Deterministic Policy Gradient (DDPG)

>[!INFO]
> An **off-policy**, **actor-critic algorithm** designed for environments with **continuous action spaces**.

- Blends the strength of [[DQN]] and [[#Policy Gradient Methods]] to learn both a [[Q-function]] and a **deterministic policy** simultaneously

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

# Proximal Policy Optimization (PPO)

>[!INFO]
> Simplifies trust region policy optimization by **constraining updates** within a predefined _clipping_ threshold &rarr; providing stability and reliability during training

- A more advanced variant of Deep Reinforcement Learning Method
- significantly reduces **sensitivity to hyperparameter selection** and **ensures consistent learning across diverse environments**

## Components

- **Stochastic Policy**: Learns $\pi_\theta(a|s)$, a probability distribution over actions
- **Clipped Objective**: Limits how much the new policy can deviate from the old one during updates
- **[Actor-Critic Architecture](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/)[^2]**: **Actor** updates the policy; **critic** estimates value function $V(s)$
- **Advantage Estimation**: Uses [[Generalized Advantage Estimation (GAE)]] for variance reduction
- **On-Policy Learning**: Uses fresh trajectories from the current policy for updates

## Key Features

1. **Clipped Surrogate Objective**
	- Prevents **large**, **destabilizing policy updates**
	- Objective:
$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[\min \left(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)A_t \right) \right]
$$
	- Where:
$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$
2. **Trust Region Approximation**
	- Inspired by **TRPO** but avoids second-order derivatives
	- Uses clipping instead of **KL-divergence** constraints
3. **Sample Efficiency**
	- More efficient than vanilla policy gradients
	- Can reuse mini-batches for multiple epochs
4. **Wide Applicability**
	- Works well in high-dimensional, continuous control tasks
	- Used in **robotics**, **games**, and **simulated physics**


# Monte Carlo Tree Search (MCTS)

> [!INFO]
> a **heuristic search algorithm** used for decision-making in large, complex environments.

- builds the search tree **incrementally** using [[random simulations (rollouts)]] to evaluate actions, balancing **exploration** and **exploitation** statistically

## How it Works

1. **Selection**: Traverse the tree from `root` using a policy to select promising `nodes`
2. **Expansion**: Add one or more `child nodes` to the tree from the **selected node**
3. **Simulation**: Run a **random** or **heuristic-based playout** from the new node to a terminal state
4. **Backpropagation**: Propagate the simulation result back up the tree, updating visit counts and win rates

## Key Features

1. **Exploration vs Exploitation via UCB1**
	- Uses **Upper Confidence Bound (UCB1)** to select nodes: 
$$
\text{UCB1}(i) = \bar{X}_i + c\sqrt{\frac{\ln N}{n_i}}
$$
		- $\bar{X}_i$: average reward of node $i$
		- $N$: total visits to parent
		- $n_i$: visits to node $i$
		- $c$: exploration constant (typically $\sqrt{2}$)
2. **Anytime Algorithm**
	- Can be stopped at any time and still return the best-known action
	- Ideal for **real-time decision-making** under computational constraints
3. **Domain-Agnostic**
	- Requires no **domain-specific evaluation function**
	- Works well in environments where simulation is cheap but evaluation is hard
4. **Scalable and Parallelizable**
	- Can be distributed across cores or machines for faster search

[^1]: A simple Monte Carlo method that directly estimates the policy gradient using **complete episodes from the environment**. It updates the policy parameters based on the **log probability of actions taken**, weighted by the return (cumulative reward) from those actions. While simple it suffer from **high variance** in the gradient estimates.

[^2]: Uses 2 parts:
	- **Actor**: decides what action to take
	- **Critic**: evaluates how good that action was
	
	The **critic** provides feedback to the **actor** help to improve its decisions. This setup makes learning more ***stable*** and ***reduces the randomness*** in the updates

[^3]: A method that carefully updates the decision-making rules. It avoids making **big changes** at once which helps keep training steady &rarr; makes PPO reliable and popular for tough problems
