> [!INFO]
> Clustering algorithm that identifies **exemplars** (representative data points) through a **message-passing mechanism**

- **Developed by**: **Brendan J. Frey** and **Delbert Dueck** (2007)
- **Core Principle**: Uses pairwise similarity and iterative message exchange to identify **exemplars** without requiring the number of clusters
- **Search Strategy**:
	- Exchange two types of messages between data points:
		- **Responsibility**: how well a point should serve as an exemplar
		- **Availability**: how appropriate it is for a point to choose another as its exemplar
	- Update messages iteratively until **convergence**

## Workflow

1. **Initialization**
	- Compute **similarity matrix** between all data points
2. **Message Passing**
	- Update **responsibility** and **availability** values
	- Assign exemplars based on combined scores
	- Repeat until stable cluster assignments are reached

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

# Generate synthetic customer purchase data
np.random.seed(42)
data = np.array([
    [500, 20, 3], [550, 18, 2], [600, 22, 5],  # High spenders
    [100, 5, 0], [120, 4, 1], [130, 6, 0],     # Low spenders
    [300, 10, 2], [310, 12, 3], [320, 9, 1],   # Mid-range spenders
    [400, 15, 4], [420, 14, 3], [430, 16, 4]   # Upper-mid-range spenders
])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply Affinity Propagation
aff_prop = AffinityPropagation(random_state=42)
clusters = aff_prop.fit_predict(data_scaled)

# Convert results to a DataFrame
customer_df = pd.DataFrame(data, columns=['Annual Spend ($)', 'Purchases per Month', 'Returns'])
customer_df['Cluster'] = clusters

# Display the clustered data
import ace_tools as tools
tools.display_dataframe_to_user(name="Affinity Propagation Clustering Results", dataframe=customer_df)

# Plot clusters
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', edgecolors='k', s=100)
plt.xlabel('Annual Spend ($)')
plt.ylabel('Purchases per Month')
plt.title('Customer Clustering using Affinity Propagation')
plt.show()
```

## Advantages

- Does **not require pre-specifying the number of clusters**
- Effectively identifies **representative exemplars** within clusters
- Performs well with **non-linearly separable data**

## Disadvantages

- **High computational complexity**
- Requires careful tuning of **preference parameter**
- **Sensitive to noise and outliers**