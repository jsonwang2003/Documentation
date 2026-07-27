---
description: "A hybrid tree-heap structure assigning randomized priorities to elements to simulate random insertion orders and guarantee average-case logarithmic complexity."
aliases:
  - Randomized Search Tree
  - RST
  - Treap
tags:
  - data-structures
  - trees
  - probabilistic
---
> [!abstract] Abstract 
> A Randomized Search Tree (RST) is a specialized Treap structure where priorities are randomly generated upon insertion. This application of randomness simulates a completely uniform random insertion sequence, successfully securing an $O(\log n)$ average-case time complexity across all operations regardless of the actual order in which keys are provided by the user.
> 
> - **Category:** Probabilistic Self-Balancing Tree
> - **Core Composition:** Dual-attribute node tracking mapping a unique search key alongside a priority score.
> - **Balancing Invariant:** Simultaneously maintains binary search tree and max-heap properties.

---

# The Treap (Tree + Heap) Core Architecture

A Treap is a binary tree structure where each node explicitly encapsulates two structural attributes: a **Key** and a **Priority**. To protect the integrity of the architecture, it must satisfy two data layout properties simultaneously:

1.  **BST Operational Invariant:** The tree is sorted horizontally by keys ($\text{Left} < \text{Node} < \text{Right}$).
2.  **Heap Operational Invariant:** The tree is ordered vertically by node priorities ($\text{Parent Priority} \ge \text{Child Priority}$, operating under max-heap rules).

![[Pasted image 20260114111803.png]]

---

# Fundamental Operations

## `Find(element)`
Since a Randomized Search Tree functions as a valid, standard [[Tree Structures/Binary Search Tree|Binary Search Tree]], lookups navigate down branches using key comparisons alone, bypassing priority scores entirely.

- **Time Complexity:** Bounded by tree height: $O(h)$ operations.

## `Insert(key, priority)`
Insertion updates the tree topology through a two-phase lifecycle:

1.  **BST Insertion Phase:** Walk down the branches based solely on key comparisons, appending the new item as a terminal leaf node.
2.  **Heap Fix Phase (Bubble Up):** While the new node's priority is greater than its parent's priority, execute tree rotations to move the node up the structure without breaking the underlying left-to-right BST sequence.

![[Pasted image 20260114112415.png]]

```pseudo
	\begin{algorithm}
	\caption{RST Node Insertion}
	\begin{algorithmic}
		\Procedure{Insert}{key, priority, root}
			\State $node \gets$ \Call{PerformBSTInsertion}{key, priority, root}
			\While{$node \neq root $\and$ node.\text{priority} > node.\text{parent}.\text{priority}$}
				\If{$node == node.\text{parent}.\text{leftChild}$}
					\State \Call{AVLRight}{$node.\text{parent}$}
				\Else
					\State \Call{AVLLeft}{$node.\text{parent}$}
				\EndIf
			\EndWhile
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Remove(key)`
1.  **BST Removal Phase:** Trace down the branches to isolate the target node matching the input key.
2.  **Heap Fix Phase (Trickle Down):** If the substitute successor node brought into the position violates the priority hierarchy, run tree rotations to shift it down until max-heap ordering properties are restored.

---

# The Structural Tool: Tree Rotations

Rotations are constant-time $O(1)$ pointer-swapping operations that modify the physical layout of tree branches while preserving the relative left-to-right sorted sequence of the keys.

![[Pasted image 20260114111911.png]]

| Rotation Style | Operational Description | Trigger Condition |
|---|---|---|
| **Right Rotation** | Promotes a left child into its parent's structural position. | Triggered when a left child node registers a higher priority score than its parent. |
| **Left Rotation** | Promotes a right child into its parent's structural position. | Triggered when a right child node registers a higher priority score than its parent. |
**RST Balance Rotations**
```pseudo
	\begin{algorithm}
	\caption{RST Right Rotation}
	\begin{algorithmic}
		\Procedure{AVLRight}{b}
			\State $a \gets b.\text{leftChild}$
			\State $y \gets a.\text{rightChild}$
			\State $p \gets b.\text{parent}$
			\If{$p \neq \text{NULL} $\and$ b == p.\text{rightChild}$}
				\State $p.\text{rightChild} \gets a$
			\ElseIf{$p \neq \text{NULL} $\and$ b == p.\text{leftChild}$}
				\State $p.\text{leftChild} \gets a$
			\EndIf
			\State $a.\text{parent} \gets p$
			\State $b.\text{leftChild} \gets y$
			\If{$y \neq \text{NULL}$}
				\State $y.\text{parent} \gets b$
			\EndIf
			\State $a.\text{rightChild} \gets b$
			\State $b.\text{parent} \gets a$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```
```pseudo
	\begin{algorithm}
    \caption{RST Left Rotation}
    \begin{algorithmic}
		\Procedure{AVLLeft}{a}
			\State $b \gets a.\text{rightChild}$
			\State $y \gets b.\text{leftChild}$
			\State $p \gets a.\text{parent}$
			\If{$p \neq \text{NULL} $\and$ a == p.\text{rightChild}$}
				\State $p.\text{rightChild} \gets b$
			\ElseIf{$p \neq \text{NULL} $\and$ a == p.\text{leftChild}$}
				\State $p.\text{leftChild} \gets b$
			\EndIf
			\State $b.\text{parent} \gets p$
			\State $a.\text{rightChild} \gets y$
			\If{$y \neq \text{NULL}$}
				\State $y.\text{parent} \gets a$
			\EndIf
			\State $b.\text{leftChild} \gets a$
			\State $a.\text{parent} \gets b$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# Why Use Randomness?

In a native, non-balancing BST, introducing sorted data in sorted sequences (such as `[1, 2, 3, 4, 5]`) causes the nodes to stack into a single linear branch line. This degrades lookups to an expensive $O(n)$ search cost. An RST resolves this sorting risk through a distinct pipeline:

1. Accepts the incoming key value from the input stream.
2. Generates an independent, random priority score from a uniform distribution.
3. Inserts the key-priority pair into the Treap container.

Because the assigned priority weights are randomly distributed, the nodes bubble up into a balanced layout that mimics a standard tree built from a random insertion sequence. This keeps the branch height balanced on average, even if the input keys themselves are sorted or patterned.

---

# Performance Complexity Summary

| Execution Case | Time Complexity | Structural Behavior Profile |
|---|---|---|
| **Average-Case** | $O(\log n)$ | Maintained via randomized priority distributions. |
| **Worst-Case** | $O(n)$ | Occurs if random priority assignments happen to generate a sorted list layout (statistically rare). |

> [!warning] Worst-Case Mitigation
> While an RST fixes average-case degradation, its absolute worst-case boundary remains $O(n)$. In real-time production systems where worst-case delays are unacceptable, developers instead select strict, deterministic height-balanced models like the [[Tree Structures/AVL Tree|AVL Tree]] or [[Tree Structures/Red-Black Tree|Red-Black Tree]].

---

# Related Notes

- [[Tree Structures/Binary Search Tree|Binary Search Tree]]
- [[Tree Structures/AVL Tree|AVL Tree]]
- [[Tree Structures/Heap|Heap]]
- [[Tree Structures/Red-Black Tree|Red-Black Tree]]