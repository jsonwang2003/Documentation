> [!INFO]
> A **probabilistic clustering technique** that models data as a **mixture of multiple Gaussian distributions**

- **Developed by**: 
	- **Karl Pearson** (1894, foundational concept)
	- Modern EM-based clustering formalized in the 1970s
- **Core Principle**: Uses **soft clustering** by assigning probabilities to each point belonging to different clusters
- **Search Strategy**:
	- **Expectation-Maximization (EM) algorithm**
	- Define parameters for each Gaussian component:
		- **Mean**
		- **Covariance**
		- **Weight**
	- Iteratively refine parameters via alternating steps:
		- **Expectation step**: assign probabilities to data points
		- **Maximization step**: update parameters based on assigned probabilities

## Workflow

1. **Initialization**
	- Choose number of Gaussian components
	- Initialize **means**, **covariances**, and **weights**
2. **EM Iteration**
	- **Expectation step**: compute probability of each point belonging to each component
	- **Maximization step**: update parameters to maximize likelihood

## Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic data
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

## Advantages

- Models **elliptical and overlapping clusters**
- **Soft clustering** provides insight into **degree of belongingness**
- Handles **varying cluster densities** via covariance adjustment

## Disadvantages

- **Sensitive to parameter initialization**
	- May converge to **local optima**
	- Mitigated by **multiple initializations**
- Assumes data follows **Gaussian mixture distribution**
- **Computationally expensive** due to iterative EM steps