---
description: "A hybrid structure combining Multiway Trie prefix searching with Binary Search Tree memory efficiency by using three child branches per node."
aliases:
  - Ternary Search Tree
  - TST
tags:
  - data-structures
  - trees
  - strings
---
> [!abstract] Abstract 
> A Ternary Search Tree (TST) is a hybrid data structure that combines the prefix-searching logic of a [[Tree Structures/Multiway Trie|Multiway Trie]] with the space-efficient storage of a [[Tree Structures/Binary Search Tree|Binary Search Tree (BST)]]. Each node stores a single character and has exactly three potential children: **Left**, **Middle**, and **Right**.
> 
> - **Category:** Hybrid Character Branching Tree
> - **Branching Factor:** Exactly 3 pointers per node (`leftChild`, `middleChild`, `rightChild`).
> - **Key Advantage:** Eliminates the empty pointer array overhead of Multiway Tries while retaining prefix searching capabilities.

---

# Structural Logic

In a TST, the relationship between a node and its children is determined by character comparisons and word progression:

*   **Left Child:** Stores characters that are alphabetically *smaller* than the current node's character.
*   **Right Child:** Stores characters that are alphabetically *larger* than the current node's character.
*   **Middle Child:** Represents the *next character* in the current word string.
*   **Word Nodes:** Nodes representing the end of a valid word are marked (e.g., colored blue).

![[Pasted image 20260126124053.png]]

---

# Core Operations

## `Find(key)`
To search for a key, compare the current character of the key with the current node's label:

*   **Key Char < Node Label:** Move to the **Left** child.
*   **Key Char > Node Label:** Move to the **Right** child.
*   **Key Char == Node Label:** 
    *   If this is the last character of the key, check if the node is a word-node.
    *   Otherwise, move to the **Middle** child and advance to the next character in the key.

```pseudo
	\begin{algorithm}
	\caption{Ternary Search Tree Find Operation}
	\begin{algorithmic}
		\Procedure{Find}{key, root}
			\State $node \gets root$
			\State $idx \gets 0$
			\While{$node \neq \text{NULL} $\and$ idx < \text{length}(key)$}
				\State $c \gets key[idx]$
				\If{$c < node.\text{label}$}
					\State $node \gets node.\text{leftChild}$
				\ElseIf{$c > node.\text{label}$}
					\State $node \gets node.\text{rightChild}$
				\Else
					\If{$idx == \text{length}(key) - 1$}
						\Return node.isWordNode
					\EndIf
					\State $node \gets node.\text{middleChild}$
					\State $idx \gets idx + 1$
				\EndIf
			\EndWhile
			\Return $\text{false}$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Walk-Through Example (Success)
Searching for the word `"mid"`:

![[Pasted image 20260126124507.png]]

1. Start with `node` as root (`'c'`) and letter as `'m'`.
2. `'m' > 'c'`, so move to right child (`'m'`).
3. `'m' == 'm'`, advance to middle child (`'e'`) and letter `'i'`.
4. `'i' > 'e'`, so move to right child (`'i'`).
5. `'i' == 'i'`, advance to middle child (`'n'`) and letter `'d'`.
6. `'d' < 'n'`, so move to left child (`'d'`).
7. `'d' == 'd'`, last letter reached, and node is marked as word-node $\to$ **Success!**

### Walk-Through Example (Failure)
Searching for the word `"cme"`:

![[Pasted image 20260126124559.png]]

1. Start with `node` as root (`'c'`) and letter as `'c'`.
2. `'c' == 'c'`, move to middle child (`'a'`) and letter `'m'`.
3. `'m' > 'a'`, but node (`'a'`) has no right child $\to$ **Failure!**

---

## `Insert(key)`
Compare the current character with the node's label:

*   **Key Char < Node Label:** Move Left. If `NULL`, create a new Left child and build a Middle-child "spine" for remaining characters.
*   **Key Char > Node Label:** Move Right. If `NULL`, create a new Right child and build a Middle-child "spine" for remaining characters.
*   **Key Char == Node Label:** If last character, mark as word-node. Otherwise, move Middle and advance character.

```pseudo
	\begin{algorithm}
	\caption{Ternary Search Tree Insertion}
	\begin{algorithmic}
		\Procedure{Insert}{key, root}
			\State $node \gets root$
			\State $idx \gets 0$
			\While{$idx < \text{length}(key)$}
				\State $c \gets key[idx]$
				\If{$c < node.\text{label}$}
					\If{$node.\text{leftChild} == \text{NULL}$}
						\State $node.\text{leftChild} \gets \text{CreateNode}(c)$
					\EndIf
					\State $node \gets node.\text{leftChild}$
				\ElseIf{$c > node.\text{label}$}
					\If{$node.\text{rightChild} == \text{NULL}$}
						\State $node.\text{rightChild} \gets \text{CreateNode}(c)$
					\EndIf
					\State $node \gets node.\text{rightChild}$
				\Else
					\If{$idx == \text{length}(key) - 1$}
						\State $node.\text{isWordNode} \gets \text{true}$
						\Return
					\EndIf
					\If{$node.\text{middleChild} == \text{NULL}$}
						\State $node.\text{middleChild} \gets \text{CreateNode}(key[idx+1])$
					\EndIf
					\State $node \gets node.\text{middleChild}$
					\State $idx \gets idx + 1$
				\EndIf
			\EndWhile
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Walk-Through Example
Inserting the word `"cabs"` into an existing tree:

![[Pasted image 20260126154049.png]]

1. Start at root (`'c'`) with letter `'c'`. Match $\to$ move to middle (`'a'`), next letter `'a'`.
2. Match at (`'a'`) $\to$ move to middle (`'l'`), next letter `'b'`.
3. `'b' < 'l'`, but `'l'` has no left child $\to$ create left child (`'b'`).
4. Move to (`'b'`), create middle child (`'s'`), mark as word-node.

---

## `Remove(key)`
Use search logic to locate the target node representing the last character of the key. Unmark its `isWordNode` status.

```pseudo
	\begin{algorithm}
	\caption{Ternary Search Tree Removal}
	\begin{algorithmic}
		\Procedure{Remove}{key, root}
			\State $targetNode \gets$ \Call{LocateTerminalNode}{key, root}
			\If{$targetNode \neq \text{NULL} $\and$ targetNode.\text{isWordNode}$}
				\State $targetNode.\text{isWordNode} \gets \text{false}$
			\EndIf
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# Advanced Features

### Alphabetical Iteration
Because TSTs maintain the BST property ($\text{Left} < \text{Node} < \text{Right}$), an **In-Order Traversal** retrieves words in sorted order.

```pseudo
	\begin{algorithm}
	\caption{TST Ascending In-Order Traversal}
	\begin{algorithmic}
		\Procedure{AscendingInOrder}{node}
			\If{node == $\text{NULL}$}
				\Return
			\EndIf
			\State \Call{AscendingInOrder}{$node.\text{leftChild}$}
			\If{node.isWordNode}
				\State \Call{Output}{$node.\text{word}$}
			\EndIf
			\State \Call{AscendingInOrder}{$node.\text{middleChild}$}
			\State \Call{AscendingInOrder}{$node.\text{rightChild}$}
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Auto-Complete
To find all words starting with a prefix:
1. Traverse the TST to the node representing the end of the prefix.
2. Perform an `AscendingInOrder` traversal on that node's **middle child** subtree.

---

# Evaluation for Lexicon ADT

| Feature | BST ([[Tree Structures/AVL Tree\|AVL]]) | [[Tree Structures/Multiway Trie\|Multiway Trie]] | Ternary Search Tree |
|---|---|---|---|
| **Search (Avg)** | $O(\log n)$ | $O(k)$ | $O(k + \log n)$ |
| **Space Efficiency** | High | Very Low | High |
| **Alphabetical** | Yes | Yes | Yes |
| **Auto-Complete** | No | Yes | Yes |

> [!summary] Key Takeaways
> * **Space over Speed:** TSTs avoid the "wasted pointer" problem of Multiway Tries because each node only has 3 pointers instead of $|\Sigma|$ (e.g., 26 or 256).
> * **Balance Matters:** Like a BST, a TST can become skewed if words are inserted in a poor order. Shuffling words before insertion is a common optimization.
> * **The Middle Ground:** It provides the prefix-matching power of a Trie with the memory footprint of a Tree.

---

# Related Notes

- [[Tree Structures/Multiway Trie|Multiway Trie]]
- [[Tree Structures/Binary Search Tree|Binary Search Tree]]
- [[Lexicon/index|Lexicon ADT]]