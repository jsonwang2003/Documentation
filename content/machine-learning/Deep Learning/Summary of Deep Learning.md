# Supervised Deep Learning

- Trains on **labeled datasets**
- Process:
	- Model learns to map input data to specific outputs based on provided annotations
	- Widely used in applications where predefined categories guide the learning process
## Advantage:

- High accuracy when trained on large, well-labeled datasets
## Disadvantage:

- Requires extensive computational resources and substantial amounts of annotated data for optimal performance

# Unsupervised Deep Learning

> [!INFO]
> Focuses on extracting meaningful patterns and representations from data without explicit labels

- Techniques (autoencoders and Generative Adversarial Networks - GANs) allow for
	- Dimensionality reduction
	- Data augmentation
	- Synthetic data generation
	- Enhancing learning in domains with limited labeled datasets

# Reinforcement Learning (RL)

> [!INFO]
> Integrates **Deep Learning** to enable agents to make **sequential decisions based on interactions with an environment**

- Effective in fields where agent learns optimal strategies through trail and error by maximizing cumulative rewards
	- Robotics
	- Game playing
	- Autonomous systems
- Deep Q-Networks (DQNs) and policy gradient
	- Enable handling high-dimensional inputs
	- Suitable for complex decision-making scenarios

# Hybrid Deep Learning

> [!INFO]
> Incorporates differential equations to guide the training of neural networks to ensure the predictions adhere to established principles

- Example:
	- Physics-informed neural networks integrated neural networks with scientific computing by
		- Embedding physical laws into the learning process

- Useful in solving partial differential equations (PDEs) in these fields where traditional numerical methods can be computationally expensive
	- Physics
	- Engineering
	- Climate modeling

# Semi-Supervised and Self-Supervised Learning

> [!INFO]
> Leverage small amount of **labeled data** combined with **large volumes of unlabeled data** to improve model efficiency

- Hybrid approaches
## Semi-Supervised Learning

- Useful in scenarios where **obtaining labeled data is costly**

## Self-Supervised Learning

- Enables models to generate their own labels from raw data