
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

- [[machine-learning/Unsupervised Learning/Clustering Models/index|Clustering]] - Grouping similar data points
- [[machine-learning/Unsupervised Learning/Dimensionality Reduction Models/index|Dimensionality Reduction]] - Visualizing complex datasets in 2D
- [[machine-learning/Supervised Learning/Summary of Supervised Learning#Anomaly Detection|Anomaly Detection]]
- Utilize competitive learning to **map high-dimensional input data onto a typically 2D grid**
	- Preserves topological relationships in the process
		- Similar inputs are grouped closer together in the map
		- Dissimilar ones are placed farther apart
	- Crucial in exploratory data analysis
## Examples
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