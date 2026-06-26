> [!INFO]
> An unsupervised machine learning algorithm used for clustering data points based on **density**

- **Developed by**: **Martin Ester**, **Hans-Peter Kriegel**, **Jörg Sander**, and **Xiaowei Xu** (1996)
- **Core Principle**: Groups points into clusters based on the density of surrounding data, identifying core, border, and noise points
- **Search Strategy**:
	- Define **epsilon** ($\epsilon$) — the neighborhood radius
	- Define **minPts** — minimum number of points to form a dense region
	- Expand clusters from **core points** using density reachability

## Workflow

1. **Parameter Definition**
	- Set **$\epsilon$**: radius around each point
	- Set **minPts**: minimum number of neighbors required
2. **Cluster Formation**
	- Identify **core points** with ≥ minPts neighbors within \epsilon
	- Label **border points** that are within \epsilon of a core point but not core themselves
	- Mark remaining points as **noise**

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# Generate synthetic transaction data (two interleaving patterns)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Convert results to DataFrame for analysis
df = pd.DataFrame(X_scaled, columns=['Transaction Amount', 'Transaction Frequency'])
df['Cluster'] = labels

# Visualizing the results
plt.figure(figsize=(8,6))
unique_labels = set(labels)
colors = [plt.cm.jet(float(i) / max(unique_labels + {1})) for i in unique_labels]

for label, color in zip(unique_labels, colors):
    subset = df[df['Cluster'] == label]
    plt.scatter(subset['Transaction Amount'], subset['Transaction Frequency'], color=color, label=f'Cluster {label}', edgecolors='k')

plt.xlabel("Transaction Amount (Standardized)")
plt.ylabel("Transaction Frequency (Standardized)")
plt.title("DBSCAN Clustering of Financial Transactions")
plt.legend()
plt.show()
```

## Advantages

- Detects **clusters of arbitrary shapes**
- **Robust to noise and outliers**
- **Density-based** approach uncovers meaningful structure without requiring number of clusters

## Disadvantages

- **Sensitive to $\epsilon$ and minPts** selection
- Struggles with **high-dimensional data**
- **Computationally expensive** for large datasets due to nearest-neighbor searches