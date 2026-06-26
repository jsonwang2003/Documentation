> [!INFO]
> Measures the minimum inter-cluster distance relative to the maximum intra-cluster distance. A higher index value suggests better separation.

## How It Works

$$
D = \frac{\min_{i \ne j} \delta(C_i, C_j)}{\max_k \Delta(C_k)}
$$

- $\delta(C_i, C_j)$: Distance between clusters $i$ and $j$  
- $\Delta(C_k)$: Diameter of cluster $k$  

## What to Look For

- **Higher Dunn Index = better clustering**  
- Sensitive to **noise** and **outliers**  
- Best for compact, well-separated clusters

## Application Models

- [[K-Means Clustering]]
- [[Hierarchical Clustering]]
- [[Density-Based Spatial Clustering of Application with Noise (DBSCAN)]]
- [[Gaussian Mixture Models]]