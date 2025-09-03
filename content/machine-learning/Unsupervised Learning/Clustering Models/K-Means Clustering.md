> [!INFO]
> partitions a dataset into a **pre-defined number of clusters (k)** by minimizing **intra-cluster variance** based on **feature similarity**

- **Developed by**: **James MacQueen** (1967)
- **Core Principle**: Iteratively refine **cluster centroids** to minimize the **sum of squared distances** between data points and their assigned centroids
- **Search Strategy**:
	- **k-means++**[^1] initialization
	- [[Lloyd’s algorithm]] for iterative refinement
	- Random centroid selection

## Workflow

1. **Initialization**
	- Select **k initial centroids** randomly or using **k-means++**[^1]
	- Assign each data point to the **nearest centroid**
2. **Iterative Refinement**
	- Update centroids based on the **mean of assigned points**
	- Repeat assignment and update steps until **centroids stabilize**

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Generate synthetic customer data
np.random.seed(42)
data = {
    "Annual_Spending": np.random.randint(1000, 5000, 100),
    "Purchase_Frequency": np.random.randint(1, 20, 100)
}

df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualizing clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Annual_Spending'], y=df['Purchase_Frequency'], hue=df['Cluster'], palette='viridis')
plt.xlabel('Annual Spending')
plt.ylabel('Purchase Frequency')
plt.title('Customer Segmentation using K-Means')
plt.show()

# Display the clustered data
import ace_tools as tools
tools.display_dataframe_to_user(name="K-Means Clustered Data", dataframe=df)
```
## Advantages

- **Computationally efficient** and scalable
- Easy to implement and **interpret**
- Flexible across domains and data types

## Disadvantages

- **Sensitive to initial centroid selection**
- Requires manual tuning of **k**
	- Use [[elbow method]] or [[silhouette score]] to determine optimal k
- Assumes **spherical clusters** with **equal variance**
- **Outliers** can significantly distort result

[^1]: an enhancement of the standard k-means clustering algorithm. If focuses on **improving the initialization of cluster centroids**, which is a critical step in achieving better clustering results
