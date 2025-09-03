> [!INFO]
> A powerful [[machine-learning/Unsupervised Learning/index|Unsupervised Learning]] technique used for grouping data points into a **hierarchy of nested clusters**

- **Developed by**: **Robert Farris Lenz** (conceptual roots trace back to the 1950s)
- **Core Principle**: Builds a tree-like structure of clusters without requiring a predefined number of clusters
- **Search Strategy**:
	- [[agglomerative clustering]] (bottom-up)
	- [[divisive clustering]] (top-down)
	- [[linkage criterion]] (single, complete, average)

## Workflow

1. **Agglomerative Approach**
	- Treat each data point as an individual cluster
	- Merge clusters based on [[Euclidean distance]] and chosen [[linkage criterion]]
2. **Divisive Approach**
	- Start with one cluster containing all data points
	- Recursively split clusters based on dissimilarity
	- Evaluate using metrics:
		- **Support**: Not typically used in hierarchical clustering
		- **Confidence**: Not applicable
		- **Lift**: Not applicable

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Creating a synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    'Annual_Spending': np.random.randint(500, 5000, 20),  # Simulated spending amounts
    'Purchase_Frequency': np.random.randint(1, 50, 20)  # Simulated frequency of purchases
})

# Standardizing data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Plotting the dendrogram
plt.figure(figsize=(8, 5))
dendrogram = sch.dendrogram(sch.linkage(data_scaled, method='ward'))
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Applying hierarchical clustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
clusters = hc.fit_predict(data_scaled)

# Adding cluster labels to the dataset
data['Cluster'] = clusters
import ace_tools as tools
tools.display_dataframe_to_user(name="Hierarchical Clustering Results", dataframe=data)
```
## Advantages

- **Interpretability** through **dendrogram** visualization
- **No prior specification** of cluster count
- **Deterministic** results (given fixed metric and linkage)

## Disadvantages

- **Computationally expensive** — time complexity $O(n^3)$
- **Sensitive to outliers** — affects [[linkage criterion]] calculations
- **Irreversible decisions** — merges/splits cannot be undone