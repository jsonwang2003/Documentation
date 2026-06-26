---
tags:
  - DepthFirstSearch
---
> [!ABSTRACT]
> 
> **Depth-First Search (DFS)** is a graph traversal algorithm that explores as far as possible along each branch before backtracking. While [[Breadth First Search (BFS)|BFS]] explores "wide," DFS explores "deep," making it highly memory-efficient for specific graph structures.

---
## 1. The Core Logic: Go Deep, then Backtrack

DFS starts at a source node and dives into the first available neighbor it finds. It continues moving to unvisited neighbors until it hits a "dead end" (a node with no unvisited neighbors). At that point, it backtracks to the most recent node that still has unexplored paths.

### Comparison of Exploration:

- **BFS (Level-order):** Explores all neighbors at distance 1, then all at distance 2.
- **DFS (Path-order):** Follows a single path until it hits a leaf or a visited node.

---
## 2. Implementation: The Power of the Stack

While BFS uses a **Queue** (FIFO), DFS uses a **Stack** (LIFO). This "last-in, first-out" behavior is what forces the algorithm to focus on the most recently discovered node.

```python
DFS(u, v):
    s = an empty stack
    push (0, u) to s               # (distance, current_vertex)
    visited = set()
    
    while s is not empty:
        (dist, curr) = s.pop()     # Pop the MOST RECENTLY added node
        if curr not in visited:
            mark curr as visited
            if curr == v:
                return dist        # Note: This is A path, not necessarily the shortest!
            for neighbor in adjacency_list[curr]:
                if neighbor not in visited:
                    s.push((dist + 1, neighbor))
    return "FAIL"
```

---
## 3. Recursive Implementation

DFS is naturally recursive. Because recursion uses the **System Call Stack**, we don't even need to declare a stack data structure manually.

```cpp
void DFSRecursion(Node s) {
    s.visited = true;
    for (Node v : s.neighbors) {
        if (!v.visited) {
            DFSRecursion(v);
        }
    }
}
```

> **STOP and Think Answer:** BFS cannot be implemented with simple recursion because the "Search Tree" of a recursive function naturally follows a LIFO (Stack) order. To do BFS, you must process nodes in the order they were discovered (FIFO), which requires an explicit Queue to break away from the natural recursive flow.

---

## 4. Space Complexity: BFS vs. DFS

Both algorithms have a time complexity of **$O(|V| + |E|)$**, but their memory usage differs based on the graph's shape:

- **In a wide, shallow tree:** DFS uses very little memory (height of the tree), while BFS must store the entire wide layer in the queue.
- **In a narrow, deep tree:** BFS uses very little memory, while DFS must store the entire long path in the stack.

---

## 5. Summary Comparison

|**Feature**|**BFS**|**DFS**|
|---|---|---|
|**Data Structure**|**Queue** (FIFO)|**Stack** (LIFO) or Recursion|
|**Shortest Path?**|**Yes** (unweighted)|No (finds any path)|
|**Space Complexity**|$O(\text{Width})$|$O(\text{Height})$|
|**Best Use Case**|Finding the shortest path|Cycle detection, Maze solving, Topological sort|

---
### Important Drawback

While DFS is great for exploration, it has a significant weakness: **It is not guaranteed to find the shortest path.** If DFS picks a "deep" branch that eventually leads to your destination, it will take it, even if a "shorter" branch was available right at the start.