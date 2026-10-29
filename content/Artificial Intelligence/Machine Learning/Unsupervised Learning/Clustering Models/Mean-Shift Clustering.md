> [!INFO]
> A **non-parametric**, **iterative clustering algorithm** that does not require the **pre-specification of the number of clusters**.  
> Identifies **dense areas in the feature space** by **shifting data points toward regions of higher density**

- **Developed by**: **Yizong Cheng** (1995)
- **Core Principle**: Uses **kernel density estimation** to iteratively shift data points toward **local maxima** in the density function
- **Search Strategy**:
	- Estimate density using a **Gaussian kernel**
	- Shift each point toward the **mean of its neighborhood**
	- Continue until points converge to **stable positions**

## Workflow

1. **Initialization**
	- Define **bandwidth**: radius for neighborhood density estimation
2. **Iterative Shifting**
	- Update each point’s position based on **local density gradients**
	- Points converge to **high-density regions**, forming clusters

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# Generate synthetic customer data (spending, frequency)
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Estimate the optimal bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)

# Apply Mean-Shift clustering
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# Convert to DataFrame for analysis
df = pd.DataFrame(X, columns=["Spending", "Frequency"])
df["Cluster"] = labels

# Display cluster statistics
import ace_tools as tools
tools.display_dataframe_to_user(name="Mean-Shift Clustering Results", dataframe=df)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k', alpha=0.7)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Mean-Shift Clustering of Customer Segments')
plt.xlabel('Spending')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

## Advantages

- Discovers the **optimal number of clusters** without prior specification
- Detects **arbitrary shaped clusters**
- Does not assume **spherical cluster geometry**
- Avoids **local minima** due to non-reliance on initialization

## Disadvantages

- **High computational complexity**
- **Sensitive to bandwidth selection**
- May struggle with **high-dimensional data**