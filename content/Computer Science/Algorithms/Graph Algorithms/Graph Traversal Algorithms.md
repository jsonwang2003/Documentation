# **Summary Description**

- In all graph traversal algorithms we discussed, we choose a specific vertex at which to begin our traversal
- In Breadth First Search, we explore the starting vertex, then its neighbors, then their neighbors, etc. In other words, we explore the graph in layers spreading out from the starting vertex
- Breadth First Search can be easily implemented using a Queue to keep track of vertices to explore
- In Depth First Search, we explore the current path as far as possible before going back to explore other paths
- Depth First Search can be easily implemented using a Stack to keep track of vertices to explore
- In Dijkstra's Algorithm, we explore the shortest possible path at any given moment
- Dijkstra's Algorithm can be easily implemented using a Priority Queue, ordered by shortest distance from starting vertex, to keep track of vertices to explore
- For our purposes in this text, we disallow "multigraphs" (i.e., we are disallowing "parallel edges": multiple edges with the same start and end node), meaning our graphs have at most |_V_|² edges  

## Time/Space Complexities of Breadth First Search

**Time Complexity (Breadth First Search)**

- Exploring the entire graph using Breadth First Search would take time **O(|**_**V**_**|+|**_**E**_**|)** because we have to potentially visit all |_V_| vertices and traverse all |_E_| edges, where each visit/traversal is O(1)

**Space Complexity (Breadth First Search)**  

- **$O(|V| + |E|)$** — We might theoretically have to keep track of every possible vertex and edge in the graph during our exploration  
    
- If we wanted to keep track of the entire current path of every vertex in our Queue, the space complexity would blow up

  

## Time/Space Complexities of Depth First Search  

**Time Complexity (Depth First Search)**

- Exploring the entire graph using Depth First Search would take time $O(|V| + |E|)$ because we have to potentially visit all $|V|$ vertices and traverse all $|E|$ edges, where each visit/traversal is $O(1)$

**Space Complexity (Depth First Search)**  

- **$O(|V| + |E|)$** — We might theoretically have to keep track of every possible vertex and edge in the graph during our exploration  
    
- Because we are only exploring a single path at a time, even if we wanted to keep track of the entire current path, the space required to do so would only be $O(|E|)$ because a single path can have at most $|E|$ edges

  

## Time/Space Complexities of Dijkstra's Algorithm  

**Time Complexity (Dijkstra's Algorithm)**

- We must initialize each of our $|V|$ vertices, and in the worst case, we will insert (and remove) one element into a Priority Queue for each of our $|E|$ edges, resulting in an overall worst-case time complexity of **$O(|V| + |E|\log|E|)$** overall if our Priority Queue is implemented intelligently (e.g. using a Heap)

**Space Complexity (Dijkstra's Algorithm)**  

- **$O(|V| + |E|)$** — We might theoretically have to keep track of every possible vertex and edge in the graph during our exploration


# Minimum Spanning Trees: Prim's and Kruskal's Algorithms

**Summary Description**

- Given a graph _G_, a Spanning Tree of _G_ is a tree that hits every node in _G_
- Given a graph _G_, a Minimum Spanning Tree of _G_ is a Spanning Tree of _G_ that has the minimum overall cost (i.e., that minimizes the sum of all edge weights)
- We discussed two algorithms that can find a Minimum Spanning Tree in an arbitrary graph _G_ equally efficiently
- Prim's Algorithm starts with a one-node tree and repeatedly finds a minimum-weight edge that connects a node in the tree to a node that is not in the tree, and adds that connecting edge to the tree
- Kruskal's Algorithm starts with a forest of one-node trees and repeatedly finds a minimum-weight edge that connects two previously unconnected trees in the forest and merges the two trees using the edge

  

## Time Complexity of Prim's Algorithm

**Worst-Case Time Complexity (Prim's Algorithm)**

- **$O(|V| + |E|\log|E|)$** — We need to initialize all $|V|$ vertices, and we may need to add each of our $|E|$ edges to a Priority Queue (which should be implemented using a Heap).

  

## Time Complexity of Kruskal's Algorithm  

**Time Complexity (Adjacency List)**

- **$O(|V| + |E|\log|E|)$** — We need to initialize all $|V|$ vertices, and we need to sort all $|E|$ edges. Note that the fastest algorithms to sort a list of $n$ elements are $O(n\log n)$.