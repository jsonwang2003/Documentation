> [!INFO]
> Measures how similar a data point is to its own cluster compared to other clusters. A higher score indicates better-defined clusters.

## How It Works

For a data point $i$:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

- $a(i)$: Average distance from $i$ to other points in the same cluster  
- $b(i)$: Minimum average distance from $i$ to points in a different cluster  

## What to Look For

- Range: $-1$ to $+1$
- Close to ($+1$): Well-clustered  
- Close to ($0$): On the boundary  
- Close to ($-1$): Possibly misclassified

## Application Models

- [[K-Means Clustering]]
- [[Hierarchical Clustering]]
- [[Density-Based Spatial Clustering of Application with Noise (DBSCAN)]]
- [[Gaussian Mixture Models]]