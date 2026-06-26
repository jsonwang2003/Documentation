
>[!INFO]
> Simplifies trust region policy optimization by **constraining updates** within a predefined _clipping_ threshold &rarr; providing stability and reliability during training

- A more advanced variant of Deep Reinforcement Learning Method
- significantly reduces **sensitivity to hyperparameter selection** and **ensures consistent learning across diverse environments**

## Components

- **Stochastic Policy**: Learns $\pi_\theta(a|s)$, a probability distribution over actions
- **Clipped Objective**: Limits how much the new policy can deviate from the old one during updates
- **[Actor-Critic Architecture](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/)[^1]**: **Actor** updates the policy; **critic** estimates value function $V(s)$
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
	- Inspired by [[TRPO]] but avoids second-order derivatives
	- Uses **clipping** instead of [[KL-divergence]] constraints
3. **Sample Efficiency**
	- More efficient than vanilla policy gradients
	- Can reuse mini-batches for multiple epochs
4. **Wide Applicability**
	- Works well in high-dimensional, continuous control tasks
	- Used in **robotics**, **games**, and **simulated physics**

[^1]: Uses 2 parts:
		- **Actor**: decides what action to take
		- **Critic**: evaluates how good that action was
		
	The **critic** provides feedback to the **actor** help to improve its decisions. This setup makes learning more ***stable*** and ***reduces the randomness*** in the updates
