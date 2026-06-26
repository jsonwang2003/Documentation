---
tags:
  - PrefixSum
---
> [!INFO]
> A **Prefix Sum** is a technique used to preprocess cumulative totals in an array, enabling **constant-time range queries**. It’s widely used in competitive programming, algorithm design, and data analysis.

### [[Computer Science/Algorithms/Prefix Sum|Prefix Sum]]

## Properties

- **Preprocessing-Based**: Computes cumulative totals in advance
- **Efficient Range Queries**: Enables `O(1)` access to subarray sums
- **Space-Time Tradeoff**: Uses extra space to reduce query time
- **Immutable Input Assumption**: Works best when array doesn’t change frequently
- **Extensible**: Can be adapted for multidimensional arrays (e.g., 2D prefix sum)

## Common Operations

- **Build Prefix Array**:  
    `prefix[i] = prefix[i-1] + arr[i]`  
    (with `prefix[0] = arr[0]` or `prefix[0] = 0` depending on convention)
- **Range Sum Query**:  
    `sum(i, j) = prefix[j] - prefix[i-1]`  
    (handles inclusive ranges efficiently)
- **Update (Naive)**:  
    Recompute prefix array after modifying `arr[i]`  
    (or use advanced structures like Fenwick Tree for dynamic updates)
- **2D Prefix Sum**:  
    `prefix[i][j] = grid[i][j] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]`  
    (used in image processing, matrix queries)