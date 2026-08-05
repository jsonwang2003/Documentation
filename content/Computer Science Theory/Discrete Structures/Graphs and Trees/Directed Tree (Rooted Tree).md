> [!ABSTRACT]
> 
> A rooted tree is a specific type of [[Graph Reachability#Directed Acyclic Graphs (DAG)|Directed Acyclic Graph (DAG)]] where one vertex is designated as the root. It serves as the fundamental structure for hierarchical data representation.

![[Pasted image 20251204112813.png]]

---
## Structure and Properties
In a rooted tree:
- **Root**: The source vertex. It has an in-degree of 0 (no incoming edges).
- **Internal Vertices & Leaves**: Every vertex other than the root has **exactly one incoming edge** from its parent.
- **Directionality**: Edges are directed away from the root toward the leaves.
- **Parent Function**: For any vertex $v$ (except the root), $p(v)$ denotes its unique parent.
### Height
The **height of a vertex** $v$ is the length of the unique path from the root to $v$:
- $h(r) = 0$
- $h(v) = h(p(v)) + 1$
The **height of the tree** is defined as $\max(h(v))$ for all $v$ in the tree.

![[Pasted image 20251204172716.png]]

---
## Binary Tree
A binary tree is a rooted tree where each vertex has at most two children, typically identified as the **left child** and the **right child**.
### Capacity and Limits
- **Maximum Vertices**: A tree of height $h$ can contain at most $2^{h+1} - 1$ vertices.
- **Maximum Height**: For $n$ vertices, the height is $n-1$ (a degenerate tree or "line").
- Minimum Height: To keep operations efficient, we aim for a height of:
    $$
    h = \lceil \log_2(n+1) - 1 \rceil
    $$
    

---
## Binary Search Tree (BST)
A BST is a binary tree that maintains a specific ordering property to allow for fast lookups, insertions, and deletions.

![[Pasted image 20251204173940.png]]
### The BST Property
For any node $x$:
- All values in the **left subtree** of $x$ are $\leq x.value$.
- All values in the **right subtree** of $x$ are $\geq x.value$.
### Operations
1. **Search**: Compare the target with the current node; move left if smaller, right if larger. Repeat until found or a `NIL` pointer is reached.
2. **Insertion**: Follow the search logic until a `NIL` position is found, then attach the new node as a leaf.
3. **Deletion**:
    - **Leaf**: Remove the node and set the parent's pointer to `NIL`.
    - **Internal Node**: Find a successor (usually the smallest value in the right subtree) to replace the deleted node, then re-link the subtrees.

---
## Performance Analysis
The efficiency of a BST depends entirely on its height. If the tree is balanced, operations are logarithmic. If it is unbalanced (skewed), performance degrades to linear.

| **Operation** | **Array (Unsorted)** | **BST (Balanced)** | **BST (Worst Case/Skewed)** |
| ------------- | -------------------- | ------------------ | --------------------------- |
| **Search**    | $O(n)$               | $O(\log n)$        | $O(n)$                      |
| **Insert**    | $O(1)$               | $O(\log n)$        | $O(n)$                      |
| **Delete**    | $O(n)$               | $O(\log n)$        | $O(n)$                      |

---
## Self-Balancing Trees
To prevent the "line" scenario and ensure $O(\log n)$ performance, various self-balancing algorithms adjust the tree structure during insertion and deletion:
- **[[AVL Tree]]**: Maintains height balance strictly.
- **Red-Black Tree**: Uses color properties to ensure the path to the furthest leaf is no more than twice the path to the nearest.
- **Splay Tree**: Moves frequently accessed nodes closer to the root.
- **2-3 Tree**: A non-binary approach where nodes can have multiple keys and children.

---
## Related Notes
- **[[Graphs]]**: Rooted trees are a specialized subset of directed graphs.
- **[[Graph Reachability]]**: Trees are a type of Directed Acyclic Graph (DAG) where every node is reachable from the root.
- **[[Computer Science Theory/Discrete Structures/Discrete Algorithms/Searching/Binary Search]]**: BSTs are the dynamic data structure implementation of the binary search principle.
- **[[Computer Science Theory/Discrete Structures/Discrete Algorithms/Recursive Algorithms/index|Recursive Algorithms]]**: Most tree operations, including height calculation and BST traversal, are defined and analyzed using recursion.
- **[[Huffman Code]]**: Uses the rooted tree structure to create optimal prefix-free variable-length codes.