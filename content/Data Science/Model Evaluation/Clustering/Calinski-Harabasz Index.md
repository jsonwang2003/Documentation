> [!INFO]
> Measures the ratio of the sum of between-cluster dispersion to within-cluster dispersion. A higher index value indicates better clustering.

## How It Works

$$
CH = \frac{Tr(B_k)}{Tr(W_k)} \cdot \frac{N - k}{k - 1}
$$

- $Tr(B_k)$: Between-cluster dispersion  
- $Tr(W_k)$: Within-cluster dispersion  
- $N$: Total number of samples  
- $k$: Number of clusters  

## What to Look For

- **Higher CH = better-defined clusters**  
- Sensitive to **cluster density** and **separation**  
- Works well for ***convex clusters***

## Application Models

- [[K-Means Clustering]]
- [[Hierarchical Clustering]]
- [[Density-Based Spatial Clustering of Application with Noise (DBSCAN)]]
- [[Gaussian Mixture Models]]