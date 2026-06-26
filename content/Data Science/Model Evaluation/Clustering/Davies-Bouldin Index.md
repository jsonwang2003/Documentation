> [!INFO]
> Calculates the ratio between intra-cluster and inter-cluster distances. A lower index value indicates better clustering.

## How It Works

$$
DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \ne i} \left( \frac{S_i + S_j}{M_{ij}} \right)
$$

- $S_i$: Average distance within cluster $i$  
- $M_{ij}$: Distance between centroids of clusters $i$ and $j$  
- $k$: Number of clusters  

## What to Look For

- **Lower DBI = better clustering**  
- Sensitive to both **compactness** and **separation**  
- Useful for *comparing* clustering algorithms

## Application Models

- [[K-Means Clustering]]
- [[Hierarchical Clustering]]
- [[Density-Based Spatial Clustering of Application with Noise (DBSCAN)]]
- [[Gaussian Mixture Models]]