---
title: Depth First Search (DFS)
tags:
  - DepthFirstSearch
---
> [!INFO]  
> A traversal algorithm that explores **as deep as possible** along each branch before backtracking. Depth-First Search (DFS) is widely used in **graph traversal**, **tree algorithms**, **cycle detection**, and **topological sorting**.

### [[Depth First Search (DFS)]]
## Properties

- Explores nodes by diving deep before moving laterally  
- Can be implemented recursively or iteratively (using a stack)  
- Requires a visited set to avoid revisiting nodes  
- Not guaranteed to find the shortest path  
- Space complexity depends on depth of recursion or stack size

## Common Variants

- Pre-order DFS: Visit node → left → right  
- Post-order DFS: Left → right → visit node  
- In-order DFS (binary trees): Left → node → right  
- Reverse DFS: Used in Kosaraju’s algorithm for strongly connected components

## Common Operations

- Traversal: Visit all nodes in a graph or tree  
- Pathfinding: Explore paths from source to destination  
- Cycle Detection: Identify loops in directed or undirected graphs  
- Topological Sort: Order nodes based on dependencies  
- Connected Components: Group nodes reachable from each other
