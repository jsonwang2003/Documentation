---
description: "A foundational hierarchical data structure where every node holds at most two children, serving as the blueprint for search trees and heaps."
aliases:
  - Binary Tree
  - BT
  - Hierarchical Tree
tags:
  - data-structures
  - trees
  - binary-trees
---
> [!abstract] Abstract 
> A Binary Tree is a non-linear hierarchical data structure in which each element (node) has at most two children, conventionally referred to as the **left child** and **right child**. It serves as the core foundational blueprint for specialized search engines, heaps, expression parsers, and prefix trees.
> 
> - **Category:** Hierarchical Node Network
> - **Branching Bound:** At most 2 outgoing child pointers per node ($0, 1, \text{ or } 2$).
> - **Core Usage:** Backbone for [[Binary Search Tree (BSTs)|Binary Search Tree]], [[Heap|Heaps]], and expression syntax trees.

---

# Key Structural Terminology & Definitions

To analyze tree geometry, several foundational structural parameters are evaluated:

*   **Root:** The single top-most node in the hierarchy with no parent reference.
*   **Leaf Node:** A terminal node with zero children ($\text{leftChild} == \text{NULL} \text{ and } \text{rightChild} == \text{NULL}$).
*   **Depth of Node $u$:** The number of edges along the path from the root node to $u$. The root sits at depth 0.
*   **Height of Node $u$:** The number of edges on the longest downward path from $u$ to a leaf node. Leaf nodes sit at height 0.
*   **Height of Tree ($h$):** The height of the root node (or length of the longest path from root to leaf).

---

# Structural Taxonomies & Classifications

Binary trees are classified according to their topological completeness and balance parameters:

| Classification               | Structural Requirement                                                                                                                                                       |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Full Binary Tree**         | Every internal node has **exactly 0 or 2** children (no node has only 1 child).                                                                                              |
| **Complete Binary Tree**     | Every horizontal level is completely filled, except possibly the bottom-most level, which must be filled sequentially from **left to right**.                                |
| **Perfect Binary Tree**      | All internal nodes have exactly two children, and all leaf nodes reside at the exact same depth level. Total node count equals $2^{h+1} - 1$.                                |
| **Balanced Binary Tree**     | The height of the left and right subtrees for every node differs by at most a defined constant factor (e.g., $\|h_{L} - h_{R}\| \leq 1$) in [[AVL Tree\|AVL Trees]]          |
| **Degenerate (Skewed) Tree** | Every internal node has only one child, causing the tree to degrade into a linear [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List\|Linked List]] |

---

# Sequence Traversals

Traversing a binary tree involves systematically visiting every node in the network. The four standard traversal protocols operate as follows:

### Depth-First Traversals (DFS)
1.  **In-Order Traversal (Left $\to$ Root $\to$ Right):**
    Processes the left subtree, evaluates the active node, then processes the right subtree. On a [[Binary Search Tree (BSTs)|Binary Search Tree]], this yields elements in sorted order.
2.  **Pre-Order Traversal (Root $\to$ Left $\to$ Right):**
    Evaluates the active node first before processing left and right subtrees. Ideal for copying or serializing tree structures.
3.  **Post-Order Traversal (Left $\to$ Right $\to$ Root):**
    Processes left and right subtrees before evaluating the active node. Essential for bottom-up cleanup or expression evaluation.

```pseudo
	\begin{algorithm}
	\caption{Recursive Binary Tree Traversals}
	\begin{algorithmic}
		\Procedure{PreOrder}{node}
			\If{node $\neq \text{NULL}$}
				\State \Call{Output}{node.data}
				\State \Call{PreOrder}{node.leftChild}
				\State \Call{PreOrder}{node.rightChild}
			\EndIf
		\EndProcedure

		\Procedure{InOrder}{node}
			\If{node $\neq \text{NULL}$}
				\State \Call{InOrder}{node.leftChild}
				\State \Call{Output}{node.data}
				\State \Call{InOrder}{node.rightChild}
			\EndIf
		\EndProcedure

		\Procedure{PostOrder}{node}
			\If{node $\neq \text{NULL}$}
				\State \Call{PostOrder}{node.leftChild}
				\State \Call{PostOrder}{node.rightChild}
				\State \Call{Output}{node.data}
			\EndIf
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Breadth-First Traversal (BFS / Level-Order)
Visits nodes horizontally level by level from top to bottom, left to right. This algorithm utilizes a [[Queues|Queue]] data structure to track frontier nodes.

```pseudo
	\begin{algorithm}
	\caption{Level-Order Tree Traversal (BFS)}
	\begin{algorithmic}
		\Procedure{LevelOrder}{root}
			\If{root == $\text{NULL}$}
				\Return
			\EndIf
			\State $q \gets \text{Initialize empty Queue}$
			\State \Call{Enqueue}{q, root}
			\While{\Call{IsEmpty}{q} == $\text{false}$}
				\State $curr \gets$ \Call{Dequeue}{q}
				\State \Call{Output}{curr.data}
				\If{curr.leftChild $\neq \text{NULL}$}
					\State \Call{Enqueue}{q, curr.leftChild}
				\EndIf
				\If{curr.rightChild $\neq \text{NULL}$}
					\State \Call{Enqueue}{q, curr.rightChild}
				\EndIf
			\EndWhile
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# Memory Allocation & Representations

Binary trees are implemented in hardware memory using two primary architectural approaches:

1.  **Dynamic Pointer-Based Nodes:**
    Nodes hold a data value alongside dynamic heap pointers (`leftChild`, `rightChild`, optional `parent`). This is the default structure for general dynamic trees like [[AVL Tree|AVL Trees]] and [[Binary Search Tree (BSTs)|BSTs]].
2.  **Array-Based Contiguous Index Mapping:**
    Used when the tree satisfies the Complete Tree property (such as [[Heap|Binary Heaps]]). Left and right children map to array offsets via constant index arithmetic ($2i+1$ and $2i+2$), bypassing pointer storage overhead entirely.

---

# Related Notes

- [[Binary Search Tree (BSTs)|Binary Search Tree]]
- [[AVL Tree|AVL Tree]]
- [[Heap|Heap]]
- [[Abstract Data Types (ADT)|Abstract Data Types (ADT)]]