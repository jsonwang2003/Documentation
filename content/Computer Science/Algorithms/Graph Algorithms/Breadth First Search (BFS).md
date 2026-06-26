---
tags:
  - BreadthFirstSearch
---

> [!ABSTRACT]
> 
> **Breadth-First Search (BFS)** is the primary algorithm for finding the **shortest path in an unweighted graph**. It explores a graph layer-by-layer, ensuring that it visits every node at distance $k$ before moving on to any node at distance $k+1$.

---
## 1. The Core Logic: Layer-by-Layer Exploration

The intuition behind BFS is similar to a ripple in a pond. Starting from a source node, the search expands outward in concentric circles:

1. **Level 0:** The starting node $s$.
2. **Level 1:** All immediate neighbors of $s$.
3. **Level 2:** All neighbors of Level 1 nodes that haven't been visited yet.

> [!Example]- Why does including an "early out" option **not** improve the worst-case time complexity of a shortest path algorithm (or any algorithm, really)?
> Including an "early out" doesn't change the **worst-case** complexity because, in the worst case, the destination node is the very last node visited (or is unreachable), forcing the algorithm to traverse the entire graph anyway.

---
## 2. Implementation with a Queue

To maintain this specific "level-by-level" order, BFS uses a **Queue** (First-In, First-Out). This ensures that nodes discovered first are processed first.

```Python
BFS(u, v):
    q = an empty queue
    add (0, u) to q               # (distance from start, current vertex)
    visited = set()               # keep track of where we've been
    
    while q is not empty:
        (dist, curr) = q.dequeue()
        
        if curr not in visited:
            mark curr as visited
            
            if curr == v:         # "Early out" if destination found
                return dist
                
            for neighbor in adjacency_list[curr]:
                if neighbor not in visited:
                    q.enqueue((dist + 1, neighbor))
                    
    return "FAIL"                 # No path exists
```

---
## 3. Complexity Analysis

The efficiency of BFS makes it a gold standard for unweighted graphs:

- **Time Complexity:** $O(|V| + |E|)$. We visit every vertex once and check every edge once (using an adjacency list).
    
- **Space Complexity:** $O(|V|)$. In the worst case, we might need to store a large portion of the vertices in the queue.
    

> [!Example]- Say we have a dense graph (i.e., roughly $|V|^2$ edges) with $1,000,000,000$ nodes on which we wish to perform BFS. If each edge in the queue could be represented by even just one byte, roughly how much space would the BFS queue require at its peak?
> For a dense graph with $10^9$ nodes, the number of edges $|E|$ is roughly $|V|^2$, which is $10^{18}$. If the queue holds edges, even at 1 byte per edge, $10^{18}$ bytes is **1 Exabyte**. This is far beyond the RAM capacity of any modern consumer or enterprise computer, highlighting a major weakness of BFS: its high memory consumption.

---
## 4. BFS Summary Table

| **Feature**         | **Description**                                                         |
| ------------------- | ----------------------------------------------------------------------- |
| **Data Structure**  | **Queue** (FIFO)                                                        |
| **Shortest Path?**  | Yes, but **only for unweighted graphs**                                 |
| **Search Strategy** | Level-by-level (Wide)                                                   |
| **Ideal Use Case**  | Finding the minimum number of steps/hops (e.g., Degrees of Kevin Bacon) |