
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
4. **Compatible with [Actor-Critic Architectures](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/)**[^2]
	- Combines policy (actor) and value function (critic) for more stable learning
	- Critic estimates value to guide actor’s updates

[^1]: A simple Monte Carlo method that directly estimates the policy gradient using **complete episodes from the environment**. It updates the policy parameters based on the **log probability of actions taken**, weighted by the return (cumulative reward) from those actions. While simple it suffer from **high variance** in the gradient estimates.

[^2]: Uses 2 parts:
	- **Actor**: decides what action to take
	- **Critic**: evaluates how good that action was
	
	The **critic** provides feedback to the **actor** help to improve its decisions. This setup makes learning more ***stable*** and ***reduces the randomness*** in the updates

[^3]: A method that carefully updates the decision-making rules. It avoids making **big changes** at once which helps keep training steady &rarr; makes PPO reliable and popular for tough problems
