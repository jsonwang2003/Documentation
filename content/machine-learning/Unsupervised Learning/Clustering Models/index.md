> [!INFO]
> **Clustering** is an unsupervised learning technique used to group similar data points based on shared characteristics. It helps uncover hidden structures in data without relying on predefined labels. Clustering is foundational in exploratory data analysis, anomaly detection, and segmentation tasks.

## Overview

Clustering focuses on partitioning datasets into groups (clusters) where data points within the same group are more similar to each other than to those in other groups. It is widely used in domains where understanding natural groupings or patterns is essential.

Key tasks include:

- Identifying inherent structures in data  
- Segmenting populations or behaviors  
- Detecting anomalies or outliers  
- Reducing data complexity for visualization  
- Supporting downstream supervised models

These algorithms are often used in **customer segmentation**, **image analysis**, **genomics**, and **network traffic monitoring**.

---

## Included Algorithms

- ### [[K-Means Clustering]]
> Partitions data into k clusters by minimizing intra-cluster variance. Fast and scalable, but sensitive to initialization and assumes spherical clusters.

- ### [[Hierarchical Clustering]]
> Builds nested clusters using either agglomerative or divisive strategies. Useful for dendrogram-based analysis and does not require pre-specifying the number of clusters.

- ### [[Density-Based Spatial Clustering of Application with Noise (DBSCAN)]]
> Groups data based on density and identifies noise points. Effective for discovering clusters of arbitrary shape and handling outliers.

- ### [[Gaussian Mixture Models]]
> Models data as a mixture of multiple Gaussian distributions. Probabilistic and flexible, but assumes underlying distributional structure.

- ### [[Mean-Shift Clustering]]
> Iteratively shifts data points toward the mode of the distribution. Does not require specifying the number of clusters but can be computationally intensive.

- ### [[Affinity Propagation]]
> Exchanges messages between data points to find exemplars. Automatically determines the number of clusters but may be sensitive to input preferences.

- ### [[Spectral Clustering]]
> Uses graph-based techniques and eigenvalues to identify clusters. Effective for non-convex shapes but computationally demanding for large datasets.

---

## Key Concepts

- **Distance Metrics**: Measures of similarity (e.g., Euclidean, cosine, Manhattan)  
- **Centroids**: Representative points for clusters  
- **Inertia**: Sum of squared distances within clusters (used in K-Means)  
- **Silhouette Score**: Evaluates how well each point fits within its cluster  
- **Dendrogram**: Tree-like diagram used in hierarchical clustering  
- **Noise Points**: Outliers not assigned to any cluster (e.g., in DBSCAN)

---

## Applications

- Customer segmentation in marketing  
- Image compression and object recognition  
- Anomaly detection in cybersecurity  
- Gene expression pattern analysis  
- Traffic pattern clustering in smart cities

---

## Suggested Links

- [[Unsupervised Learning/index|Unsupervised Learning]] — Broader context for pattern discovery without labels  
- [[Model Evaluation]] — Metrics like silhouette score, Davies-Bouldin index, and inertia  
- [[Dimensionality Reduction Models/index|Dimensionality Reduction Models]] — Often used before clustering to simplify high-dimensional data