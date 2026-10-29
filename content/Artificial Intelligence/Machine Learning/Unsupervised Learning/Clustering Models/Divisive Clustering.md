> **Top-down hierarchical clustering** method that recursively **splits data into increasingly homogeneous subgroups**.

## How It Works

1. **Start with all data points** in a single cluster.
2. **Recursively split** the cluster using a flat clustering algorithm (e.g. k-means, bisecting k-means).
3. **Evaluate splits** based on dissimilarity metrics (e.g. variance, distance).
4. **Repeat** until stopping criteria are met (e.g. minimum dispersion, max depth, or target number of clusters).

## What to Look For

- **Global structure awareness**: Splits are informed by overall data distribution.
- **More scalable** than agglomerative methods for large datasets.
- **Useful for semantic taxonomies** and progressive refinement.
- **Ideal for documentation scaffolding** where top-down modularity is key.

## Application Models

- [[Hierarchical Clustering]] (top-down variant)