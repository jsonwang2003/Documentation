> [!INFO]
> State-of-the-art dimensionality reduction technique for **visualizing high-dimensional data while preserving its structure**

- **Developed by**: **Leland McInnes**, **John Healy**, and **James Melville** (2018)
- **Core Principle**: Uses **topological data analysis** and **manifold learning** to construct a high-dimensional graph and optimize a low-dimensional representation
- **Search Strategy**:
	- Build a **fuzzy topological graph** from high-dimensional data
	- Optimize a low-dimensional graph to approximate the original structure
	- Preserves **both global and local relationships**

## Workflow

1. **Graph Construction**
	- Define **number of neighbors** and **minimum distance**
	- Build high-dimensional graph representing data topology
2. **Low-Dimensional Optimization**
	- Embed data in 2D or 3D space
	- Optimize layout using **stochastic gradient descent**

## Code Example

```python
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate synthetic customer data
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, n_classes=4, n_informative=5, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

# Convert results to DataFrame
df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
df_umap['Cluster'] = y

# Plot the UMAP results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_umap['UMAP1'], df_umap['UMAP2'], c=df_umap['Cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Cluster")
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Projection of Customer Data')
plt.show()
```

## Advantages

- Preserves **local and global structure** in high-dimensional data
- **Computationally efficient** and scalable
- Useful as a **preprocessing step for clustering**
- Supports both **supervised and unsupervised learning**

## Disadvantages

- **Highly sensitive to hyperparameters**
	- Number of neighbors
	- Minimum distance
- Does not provide **explicit feature importance** or **variance explained**