
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