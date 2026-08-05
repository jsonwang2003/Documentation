---
title: Breadth First Search (BFS)
tags:
  - BreadthFirstSearch
---
> [!INFO]  
> A traversal algorithm that explores **all neighbors at the current depth** before moving to the next level. Breadth First Search (BFS) is widely used in **graph traversal**, **shortest path algorithms**, and **level-order tree traversal**.

> More Details here: [[Breadth First Search (BFS)|Breadth First Search (BFS)]]
## Properties

- Explores nodes level-by-level from the starting point  
- Uses a **queue** to maintain traversal order (FIFO)  
- Guarantees the **shortest path** in unweighted graphs  
- Requires a **visited set** to avoid revisiting nodes  
- Time complexity: O(V + E) for graphs with V vertices and E edges

## Common Operations

- **Traversal**: Visit all nodes in a graph or tree  
- **Shortest Path**: Find minimum steps between two nodes  
- **Level Order Traversal**: Traverse trees by depth  
- **Connected Components**: Identify clusters in undirected graphs  
- **Cycle Detection**: Check for loops in undirected graphs