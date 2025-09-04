---
title: Association Rule Learning
---

> [!INFO]
> **Association Rule Learning** is a data mining technique used to uncover interesting relationships, patterns, or associations among variables in large datasets. These models are foundational in market basket analysis, recommendation systems, and behavioral pattern discovery.

## Overview

Association Rule Learning focuses on identifying frequent itemsets and generating rules that describe how items co-occur. Unlike supervised learning, it does not require labeled data—only transactional or categorical datasets.

## Key tasks include:

- Frequent itemset mining  
- Rule generation and pruning  
- Pattern discovery in categorical data  
- Market basket analysis  
- Recommendation modeling

These algorithms are often used in **retail analytics**, **web usage mining**, and **bioinformatics** to extract actionable insights from co-occurrence patterns.

---

## Included Algorithms

- ### [[machine-learning/Unsupervised Learning/Association Rule Learning Models/Apriori Algorithm]]
> Uses breadth-first search and candidate generation to find frequent itemsets. Simple and interpretable, but computationally expensive.

- ### [[machine-learning/Unsupervised Learning/Association Rule Learning Models/Equivalence Class Clustering and Bottom-Up Lattice Traversal (ECLAT)]]
> Groups itemsets into equivalence classes for efficient rule generation. Often used in vertical data formats.

- ### [[machine-learning/Unsupervised Learning/Association Rule Learning Models/Frequent Pattern Growth]]
> Avoids candidate generation by using a compact prefix-tree structure. Fast and memory-efficient for large datasets.

---

## Key Concepts

- **Support**: Frequency of an itemset in the dataset  
- **Confidence**: Likelihood that item Y appears when item X is present  
- **Lift**: Measures how much more likely item Y is given item X, compared to random chance  
- **Itemset Mining**: Identifying sets of items that frequently occur together  
- **Rule Pruning**: Removing redundant or low-value rules based on thresholds

---

## Applications

- Product bundling and cross-selling strategies  
- Web clickstream analysis  
- Medical diagnosis pattern mining  
- Fraud detection in transaction logs  
- Personalized recommendation engines

---

## Suggested Links

- [[machine-learning/Unsupervised Learning/index|Unsupervised Learning]] — Broader context for pattern discovery without labels  
- [[machine-learning/Model Evaluation|Model Evaluation]] — Metrics like support, confidence, lift, and conviction  
- [[machine-learning/Deep Learning/Hybrid Deep Learning Models|Hybrid Deep Learning Models]] — For combining rule-based and neural architectures