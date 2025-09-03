
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