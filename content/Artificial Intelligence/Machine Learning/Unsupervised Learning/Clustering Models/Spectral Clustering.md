> [!INFO]
> A powerful **unsupervised learning technique** that uses **graph theory** and **eigenvalues of matrices** to partition data into clusters  
> Performs **dimensionality reduction** via the **similarity matrix** before applying clustering algorithms  
> Useful for datasets with **complex structures**, **non-convex shapes**, or data points that do not conform well to **Euclidean distance-based clustering**

- **Developed by**: **Shi and Malik** (2000, normalized cuts formulation)
- **Core Principle**: Transforms data into a lower-dimensional space using **eigenvector decomposition** of the **graph Laplacian**, then applies clustering
- **Search Strategy**:
	- Construct **affinity matrix** representing pairwise similarities
	- Compute **graph Laplacian**
	- Perform **eigen decomposition** to embed data into a space where clusters are more separable

## Workflow

1. **Affinity Construction**
	- Build similarity graph using nearest neighbors or radial basis functions
2. **Spectral Embedding**
	- Compute **graph Laplacian**
	- Extract **eigenvectors** to reduce dimensionality
3. **Clustering**
	- Apply standard clustering (e.g., K-Means) in the embedded space

## Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

# Generate synthetic dataset with non-linearly separable clusters
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
labels = spectral.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.title("Spectral Clustering on Non-Convex Data")
plt.show()
```

## Advantages

- Handles **complex and non-convex cluster structures**
- Effective for **graph-based data**
- Does not assume **cluster shape or distribution**

## Disadvantages

- **Computationally expensive** — $O(n^3)$
- Requires careful tuning of **hyperparameters**
	- Affinity matrix construction method
	- Number of clusters
- Relies heavily on the **similarity graph**