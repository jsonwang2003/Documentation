- Algorithms that identify these within datasets without relying on labeled input data
	- Patterns
	- Features
	- Relationships
- Autonomously explores underlying structures, enabling it to uncover hidden
	- Groupings
	- Associations
	- Latent factors
- Types:
	- [[Unsupervised Deep Learning Models#Generative Adversarial networks (GANs)|Generative Adversarial Networks (GANs)]]
	- [[Unsupervised Deep Learning Models#Self-Organizing Maps (SOMs)|Self-Organizing Maps (SOMs)]]

# Autoencoders

> [!INFO]
> A type of neural network designed to learn efficient representations of data by compressing input into a lower-dimensional form (encoding) and then reconstructing it back (decoding) as closely as possible to the original.

- Represent one of the foundational unsupervised deep learning architectures
## Components:
- **Compression (Encoder)**: Transforms input data into a compact latent representation
- **Latent Space (Bottleneck)**: The compressed form that captures the most essential features
- **Decompression (Decoder)**: Reconstructs the original data from the latent representation
## How it Works

- uses [[machine-learning/Unsupervised Learning/index|Unsupervised Learning]], minimizing a **reconstruction loss** between the input and output
## Goal
- [[Summary of Unsupervised Learning#Dimensionality Reduction|Dimensionality Reduction]]: distill complex, high-dimensional data into a reduced, meaningful representation or latent space
## Reconstruction error
- Discrepancy between **original data** and **decoded reconstruction**
- When minimized will
	- Uncover intrinsic features
	- Enable businesses to process large datasets efficiently
- Variants enhances capability
	- Denoising autoencoders: facilitating noise removal
	- Variational autoencoders: probabilistic generation of new data

# Generative Adversarial Networks (GANs)

> [!INFO]
> A class of machine learning models where two neural networks — the **generator** and the **discriminator** — compete in a game-like setup to produce highly realistic synthetic data.

## Components

- **Generator**: Produce realistic synthetic data instances from random noise
- **Discriminator**: Differentiate between real and artificially generated outputs
## How it Works

- **Generator** improves by _learning to fool the discriminator_
- **Discriminator** gets better at _spotting fakes_
- Loop continues until the **generated data becomes indistinguishable from real data**
## Examples

- Businesses
	- Data augmentation
	- Synthetic data creation for privacy protection
	- Generate realistic multimedia content
- Fashion
	- Generate virtual clothing designs
	- Simulate fashion styles
		- Reducing cost and time associated with physical prototyping
- Finance
	- Fraudulent cases
		- Provided an alternative by identifying unusual patterns or behaviors in transaction data without prior fraud labels
		- Flags transactions that significantly deviate from typical encoded representations
		- Can generate synthetic yet realistic examples of fraudulent transactions to enhance the robustness of detection models

# Self-Organizing Maps (SOMs)

## Components

- [[machine-learning/Unsupervised Learning/index|Unsupervised Learning]]: SOMs don't require labeled data &rarr; **learns patterns and structure from raw data**
- **Competitive Learning**: Neurons compete to represent input data
	- _winning_ neuron and its neighbors update their weights to better match the input
- **Topology Preservation**: Similar data points are mapped to nearby neurons, making SOMs excellent for _visualizing clusters and relationships_

## How it Works

1. **Initialization**: Each neuron in the grid starts with random weights
2. **Best Matching Unit (BMU)**: For each input, the neuron whose _weights are closest to the input is selected_
3. **Weight Update**: The BMU and its neighbors adjust their weights to better match the input
4. **Decay**: Learning rate and neighborhood size shrink over time for convergence
## Goal

- [[Summary of Unsupervised Learning#Clustering|Clustering]] - Grouping similar data points
- [[Summary of Unsupervised Learning#Dimensionality Reduction|Dimensionality Reduction]] - Visualizing complex datasets in 2D
- [[Summary of Supervised Learning#Anomaly detection|Anomaly Detection]]
- Utilize competitive learning to **map high-dimensional input data onto a typically 2D grid**
	- Preserves topological relationships in the process
		- Similar inputs are grouped closer together in the map
		- Dissimilar ones are placed farther apart
	- Crucial in exploratory data analysis
- Examples
	- Businesses
		- Segment customers based on
			- Purchasing patterns
			- Demographic attributes
			- Preferences
	- Retail analytics (customer segmentation and personalization)
		- Often collect extensive, unlabeled transactional and behavioral data from customers
			- Require advanced analytical methods for actionable insights
		- Use autoencoders
			- effectively compress transaction histories and browsing data into concise representations
	- Following dimensionality reduction
		- Further categorize customers into distinct segments
		- Facilitates the
			- Precise targeting of marketing campaigns
			- Personalized product recommendations
			- Efficient inventory management