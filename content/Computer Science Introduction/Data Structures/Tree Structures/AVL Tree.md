---
description: "A self-balancing binary search tree enforcing a strict height-invariant constraint via node rotations to secure absolute logarithmic time bounds."
aliases:
  - AVL Tree
  - Self-Balancing BST
  - Adelson-Velsky and Landis Tree
tags:
  - data-structures
  - trees
  - avl
---
> [!abstract] Abstract 
> Named after inventors Adelson-Velsky and Landis, the AVL Tree is a self-balancing [[Binary Search Tree (BSTs)|Binary Search Tree (BST)]] that guarantees a worst-case time complexity of $O(\log n)$ for search, insertion, and deletion operations. It achieves this performance by enforcing a strict structural balance property maintained through deterministic node rotations.
> 
> - **Category:** Balanced Hierarchical Tree
> - **Core Invariant:** Balance factors across all nodes must strictly reside within the set $\{-1, 0, 1\}$.
> - **Balancing Strategy:** Localized single or double pointer rotations executed during stack rollbacks.

---

# Core Structural Properties

To prevent height degradation into a linear $O(n)$ chain, an AVL tree enforces the **Balance Condition** at every internal node coordinate:

> [!important] The Balance Invariant
> For every individual node $u$ inside the tree, the structural heights of its left and right subtrees can differ by at most 1.

The mathematical representation of this tracking metric is the **Balance Factor (BF)**:

$$ \text{BF}(u) = \text{Height}(\text{RightSubtree}(u)) - \text{Height}(\text{LeftSubtree}(u)) $$

A node state is considered structurally valid if and only if:

$$ \text{BF}(u) \in \{-1, 0, 1\} $$

| Valid AVL Tree | Invalid AVL Tree |
|---|---|
| ![[Pasted image 20260116111242.png]] | ![[Pasted image 20260116111257.png]] |

If an insertion or erasure causes $\text{BF}(u)$ to drift to $\pm 2$, the node is flagged as imbalanced, immediately triggering structural rebalancing.

---

# Mathematical Proof of Bounded Height

We can prove that the maximum height $h$ of an AVL tree containing $n$ nodes is bounded at $O(\log n)$ by determining the minimum number of nodes $N_h$ required to form a valid AVL tree of height $h$.

To construct the most sparse AVL tree possible of height $h$, we provide one child with the minimum valid height $h-1$ and the opposing child with the minimum valid height $h-2$, plus the root node itself:

$$ N_h = N_{h-1} + N_{h-2} + 1 $$

Using a 1-based height index framework where base cases resolve to $N_1 = 1$ and $N_2 = 2$, this recurrence relation matches the growth trajectory of the Fibonacci sequence. Because Fibonacci terms scale exponentially relative to the golden ratio ($\phi \approx 1.618$), we establish that:

$$ N_h \approx \phi^h \implies h \approx \log_{\phi}(n) $$

This mathematical relationship confirms that the height of an AVL tree is strictly bounded at $h \le 1.44 \log_2 n$, ensuring guaranteed $O(\log n)$ performance.

---

# Rebalancing: Structural AVL Rotations

When mutations push an asset's balance factor to $\pm 2$, pointer adjustments are executed to restore the structural balance of the tree.

### 1. Single Rotations (The Straight-Line Cases)
Single rotations resolve imbalances caused by insertions or removals occurring on the outer margins of a node's extended subtree lineage (Left-Left or Right-Right configurations).

*   **Right Rotation:** Corrects a Left-Left ($\text{L-L}$) linear imbalance.
*   **Left Rotation:** Corrects a Right-Right ($\text{R-R}$) linear imbalance.

![[Pasted image 20260116111325.png]]

```pseudo
	\begin{algorithm}
	\caption{AVL Single Right Rotation}
	\begin{algorithmic}
		\Procedure{AVLRight}{b}
			\State $a \gets b.\text{leftChild}$
			\State $y \gets a.\text{rightChild}$
			\State $p \gets b.\text{parent}$
			\If{$p \neq \text{NULL}$ \and $b == p.\text{rightChild}$}
				\State $p.\text{rightChild} \gets a$
			\ElseIf{$p \neq \text{NULL}$ \and $b == p.\text{leftChild}$}
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
	\caption{AVL Single Left Rotation}
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

### 2. Double Rotations (The Kink Cases)
Double rotations correct zig-zag imbalances caused by mutations nested deep inside inner child coordinates (Left-Right or Right-Left configurations). A single rotation cannot resolve a zig-zag imbalance.

![[Pasted image 20260116120605.png]]

*   **Left-Right Double Rotation:** Executes a primary left rotation on the child node, transforming the zig-zag into a straight line, followed by a right rotation on the parent node.
*   **Right-Left Double Rotation:** Executes a primary right rotation on the child node, transforming the zig-zag into a straight line, followed by a left rotation on the parent node.

![[Pasted image 20260116120703.png]]

```pseudo
	\begin{algorithm}
	\caption{AVL Double Right Rotation}
	\begin{algorithmic}
		\Procedure{DoubleAVLRightKink}{a}
			\State \Call{AVLRight}{$a.\text{rightChild}$}
			\State \Call{AVLLeft}{a}
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

```pseudo
	\begin{algorithm}
	\caption{AVL Double Left Rotation}
	\begin{algorithmic}
		\Procedure{DoubleAVLLeftKink}{a}
			\State \Call{AVLLeft}{$a.\text{leftChild}$}
			\State \Call{AVLRight}{a}
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Insertion Example Requiring Double Rotation
If we append a value of $10$ into our active tree array:

![[Pasted image 20260116120820.png]]

Following a traditional BST insertion path yields an imbalanced parent node structured in a zig-zag "kink" configuration:

![[Pasted image 20260116120831.png]]

To resolve this imbalance, a double rotation sequence is triggered. First, we execute a left rotation on child node $5$ to unroll the kink structure into a clean straight line:

![[Pasted image 20260116120846.png]]

With the straight line achieved, we complete a right rotation on the root ancestor node to restore absolute tree height balance parameters:

![[Pasted image 20260116120900.png]]

---

# Data Structure Operations

Every mutations pipeline couples traditional binary search tree logic with an integrated upward rebalancing sweep to maintain tree balance.

## `Find(element)`
Operates identically to a standard [[Binary Search Tree (BSTs)|BST]] lookup. The engine traverses down tree branches by comparing target values against active node keys.
- **Time Complexity:** Guaranteed $O(\log n)$ since tree height is strictly controlled.

## `Insert(element)`
1.  **BST Phase:** Trace downward to find the target leaf slot and insert the element.
2.  **Update Phase:** Walk back up toward the root starting from the new leaf node.
3.  **Balance Phase:** Recalculate balance factors at each ancestor node. If any ancestor registers $\text{BF} = \pm 2$, execute the appropriate single or double rotation.

- **Time Complexity:** $O(\log n)$ to search downward plus $O(\log n)$ for the upward rebalancing path.

![[Pasted image 20260116111623.png]]

### Complex Insertion Walkthrough
Consider inserting item $20$ into the following initial state:

![[Pasted image 20260116115439.png]]

A basic insertion drops the new leaf to the right margin, breaking balance codes up the chain:

![[Pasted image 20260116115455.png]]

The engine immediately runs a left rotation centered at the root. Node $10$ assumes the root position, node $7$ is re-assigned to the right child slot of node $5$, and node $5$ shifts into the left child coordinate of node $10$:

![[Pasted image 20260116115506.png]]

## `Remove(element)`
1.  **BST Phase:** Execute standard BST node removal rules (managing the 0, 1, or 2-child structural configurations).
2.  **Update Phase:** Start at the parent coordinate of the physically removed item and trace upward to the root.
3.  **Balance Phase:** Check balance factors at every level. Unlike insertion (where a single rotation fix is guaranteed to restore balance across the entire tree), removal mutations can alter heights globally, occasionally requiring multiple independent rotations along the path to the root.

- **Time Complexity:** $O(\log n)$ search cost plus an $O(\log n)$ multi-step balancing sweep.

![[Pasted image 20260116111630.png]]

---

# Balanced Structural Performance Matrix

| Evaluation Metric | Standard Binary Search Tree | AVL Self-Balancing Tree |
|---|---|---|
| **Average Search Time** | $O(\log n)$ | $O(\log n)$ |
| **Worst-Case Search Time** | $O(n)$ (Degrades into a linear list) | $O(\log n)$ (Strictly enforced) |
| **Balancing Strategy** | None | Height-based balance factors ($\pm 1$) |
| **Operational Passes** | 1 Pass (Downward trajectory only) | 2 Passes (Downward mutation + Upward repair) |

> [!note] Architectural Design Trade-off
> While AVL trees offer excellent lookup speeds due to their strict balance property, they require a two-pass update cycle (down to mutate, then back up to balance). In heavy write-dominated production libraries, developers often select **Red-Black Trees**, which compromise on strict height balancing to complete structural repairs in a single pass.

---

# Related Notes

- [[Binary Search Tree Implementation|Binary Search Tree Implementation]]
- [[Binary Search Tree (BSTs)]]
- [[Priority Queue|Priority Queue]]