> [!ABSTRACT]
> 
> While **[[Breadth First Search (BFS)|BFS]]** finds the shortest path in unweighted graphs (least number of edges), it fails when edges have **varying costs**. **Dijkstra's Algorithm** is a [[Computer Science/Algorithms/Greedy Algorithms/index|Greedy Algorithm]] that solves the **Single-Source Shortest Path** problem for weighted graphs, provided all edge weights are **non-negative**.

---
## 1. Why BFS Fails on Weighted Graphs

BFS assumes that every edge has a cost of $1$. In a weighted graph, a path with **more edges** might actually have a **lower total weight** than a direct edge.

- **BFS Path:** $A \to C$ (Total Weight: 30)
    
- **Shortest Weighted Path:** $A \to B \to C$ (Total Weight: $12 + 5 = 17$)
    

Dijkstra's Algorithm accounts for these costs by prioritizing paths with the smallest cumulative weight.

---
## 2. The Greedy Strategy

Dijkstra's is a **greedy algorithm**. It makes the optimal choice at each step—picking the closest unvisited vertex—and assumes that this choice will lead to the overall shortest path.

### The Core Logic:

1. Assign a **Distance** of infinity to all nodes, except the start node (which is 0).
    
2. Maintain a **Priority Queue (PQ)** to store `(distance, vertex)` pairs.
    
3. Always "relax" the neighbor: If the path to a neighbor through the current node is shorter than its previously known distance, update its distance and add it to the PQ.
    
4. Once a node is "Done" (dequeued), its shortest path is guaranteed.
    

---
## 3. Pseudocode Implementation

Dijkstra's uses a **Priority Queue** to efficiently find the next vertex with the minimum distance.

```C++
dijkstra(s):
    pq = empty priority queue
    for each vertex v:
        v.dist = infinity, v.prev = NULL, v.done = False
    
    s.dist = 0
    enqueue (0, s) onto pq
    
    while pq is not empty:
        dequeue (weight, v) from pq
        if v.done == False:
            v.done = True
            for each edge (v, w, d):
                if w.done is False:
                    c = v.dist + d
                    if c < w.dist:     // Relaxation step
                        w.prev = v
                        w.dist = c
                        enqueue (c, w) onto pq
```

---
## 4. Constraints: The Negative Weight Problem

Dijkstra’s Algorithm **does not work with negative edge weights**.

- **The Reason:** Dijkstra assumes that once a node is marked "Done," no future path can possibly be shorter.
    
- **The Failure:** A negative edge could "reduce" the cost of a path discovered later, breaking the greedy assumption. For graphs with negative weights, you must use the **Bellman-Ford Algorithm**.
    

---

## 5. Complexity Analysis

The efficiency of Dijkstra depends heavily on the Priority Queue implementation (typically a Binary Heap).

| **Operation**             | **Complexity**            | **Notes**                                                      |
| ------------------------- | ------------------------- | -------------------------------------------------------------- |
| **Initialization**        | $O(\|V\|)$                | Setting $\text{dist}=\infty$, $\text{prev}=NULL$ for all nodes |
| **Priority Queue Insert** | $O(\log \|E\|)$           | Performed for each edge processed                              |
| **Priority Queue Delete** | $O(\log \|E\|)$           | Performed once per vertex                                      |
| **Total Time**            | **$O(\|E\| \log \|E\|)$** | Dominant factor is *Priority Queue Operations*                 |
| **Total Space**           | **$O(\|V\| + \|E\|)$**    | Sorting Vertices and the adjacency list                        |



> [!NOTE]
> 
> In very dense graphs, $|E|$ approaches $|V|^2$. In such cases, the $O(|E| \log |V|)$ complexity is nearly $O(|V|^2 \log |V|)$.

---

### Comparison of Shortest Path Algorithms

|**Algorithm**|**Graph Type**|**Guaranteed Shortest Path?**|
|---|---|---|
|**BFS**|Unweighted|Yes (by edge count)|
|**DFS**|Any|No|
|**Dijkstra**|Weighted (Positive only)|**Yes** (by total weight)|