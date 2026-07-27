---
description: "A character-path search tree mapping keys along edges rather than inside node bodies to achieve deterministic linear string queries."
aliases:
  - Multiway Trie
  - Prefix Tree
  - Standard Trie
tags:
  - data-structures
  - trees
  - strings
---
> [!abstract] Abstract 
> A Trie (derived from "retrieval") is a tree structure designed to store a set of strings. Unlike standard [[Tree Structures/Binary Search Tree|Binary Search Trees]], the keys are not stored within the nodes themselves; instead, a key is defined by the concatenation of labels along the path from the root to a specific node.
> 
> - **Category:** Character-Path Digital Search Tree
> - **Input Constraints:** Expands across an explicit alphabet size $|\Sigma|$ to map contiguous character paths.
> - **Key Advantage:** Run times scale purely with string character length $k$, decoupling latency from total dictionary size $n$.

---

# Structural Properties

The Multiway Trie expands the concept of a Binary Trie to support any arbitrary alphabet $\Sigma$ (such as English letters, DNA base pairs, or numerical digits).

![[Pasted image 20260124161122.png]]

*   **Edge Labels:** Characters are assigned exclusively to the **edges**, not the nodes themselves.
*   **Word Nodes:** Since a path can represent an internal prefix that isn't a full standalone word (for example, the path `ca` is a valid prefix for the complete word `car`), specific nodes are marked with a boolean flag to indicate the end of a valid word. In layout diagrams, these are highlighted as distinct blue nodes.
*   **The Root:** An empty Trie consists of a single root node with no outgoing edges. The root node represents an empty string.

---

# Core Operations

Multiway Tries provide highly efficient operations based on the length of the string ($k$) rather than the number of stored items ($n$).

## `Find(word)`
Starts at the root node and sequentially follows the edge labeled with each consecutive letter of the target word string.

*   **Success Condition:** The search successfully evaluates all characters of the string and lands on an active node marked as a valid word node.
*   **Failure Condition:** The tracking pointer hits a `NULL` edge (indicating the sequence path does not exist) or the final node reached lacks the explicit word node flag.

![[Pasted image 20260124161200.png]]

```pseudo
	\begin{algorithm}
	\caption{Multiway Trie Find Operation}
	\begin{algorithmic}
		\Procedure{Find}{word, root}
			\State $curr \gets root$
			\For{each character $c$ in word}
				\If{curr does not have an outgoing edge labeled by $c$}
					\Return $\text{false}$
				\Else
					\State $curr \gets \text{child of curr along edge labeled by } c$
				\EndIf
			\EndFor
			\Return curr.isWordNode
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Insert(word)`
Traces the character edge path from the root, creating a new child node and a labeled edge whenever a character transition is missing, and marks the final terminal node as a valid word node.

![[Pasted image 20260124161223.png]]

```pseudo
	\begin{algorithm}
	\caption{Multiway Trie Insertion}
	\begin{algorithmic}
		\Procedure{Insert}{word, root}
			\State $curr \gets root$
			\For{each character $c$ in word}
				\If{curr does not have an outgoing edge labeled by $c$}
					\State \Call{CreateChildNode}{curr, c}
				\EndIf
				\State $curr \gets \text{child of curr along edge labeled by } c$
			\EndFor
			\If{curr.isWordNode $\neq$ \True}
				\State curr.isWordNode $\gets$ \True
			\EndIf
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!important] The Edges-Not-Nodes Invariant
> In a Multiway Trie, characters label the edges, not the nodes themselves. This is a crucial distinction: inserting a single-letter word like `"a"` into an empty root requires creating a second child node so the character `"a"` can label the connecting link edge between them.

![[Pasted image 20260124161359.png]]

## `Remove(word)`
Locates the targeted word sequence using the standard `find` algorithm and unmarks its word-node flag.

*   **Structural Preservation:** The physical node containers and character edges typically remain intact after removal to support other independent words that share those prefix branches.

![[Pasted image 20260124161208.png]]

```pseudo
	\begin{algorithm}
	\caption{Multiway Trie Removal}
	\begin{algorithmic}
		\Procedure{Remove}{word, root}
			\State $curr \gets root$
			\For{each character $c$ in word}
				\If{curr does not have an outgoing edge labeled by $c$}
					\Return
				\Else
					\State $curr \gets \text{child of curr along edge labeled by } c$
				\EndIf
			\EndFor
			\If{curr.isWordNode == \True}
				\State curr.isWordNode $\gets$ \False
			\EndIf
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# Physical Implementation: The Array Strategy

To ensure that each character transition is as fast as possible, Multiway Tries prioritize lookup speed over memory efficiency.

![[Pasted image 20260124161549.png]]

*   **The Node Array:** Every individual node contains an array of raw pointer references equal to the total size of the alphabet ($|\Sigma|$). For the English alphabet, each node encapsulates 26 slots.
*   **Constant Time Access:** Because each character maps directly to an array index via constant offset arithmetic (such as `'a' \to 0, \text{ 'b'} \to 1$), the time to follow an edge is constant.
*   **Complexity Bounds:** This structure results in a deterministic worst-case time complexity of $O(k)$ for search, insert, and remove operations.

---

# Alphabetical Iteration and Auto-Complete

Because a Multiway Trie is naturally organized by character, it is inherently sorted. This allows us to perform operations that are impossible in an unordered structure like a [[Hashing/Hash Tables|Hash Table]].

### Alphabetical Iteration
By performing a recursive traversal, we can output all words in the lexicon in perfect alphabetical order:

*   **Ascending Order (A-Z):** Use a **Pre-Order Traversal**. The engine checks if the current node is flagged as a word-node first, then visits the child branches in alphabetical order (from slot $0$ through $\vert{}\Sigma\vert{}-1$).
*   **Descending Order (Z-A):** Use a **Post-Order Traversal**. The engine visits child branches in reverse alphabetical order (from slot $\vert{}\Sigma\vert{}-1$ down to $0$) before evaluating the current node.

```pseudo
	\begin{algorithm}
	\caption{Trie Sorted Iterations Preorder}
	\begin{algorithmic}
		\Procedure{AscendingPreOrder}{node}
			\If{node.isWordNode}
				\State \Call{Output}{node.word}
			\EndIf
			\For{each child of node in ascending alphabetical order}
				\State \Call{AscendingPreOrder}{child}
			\EndFor
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
	\begin{algorithm}
	\caption{Trie Sorted Iterations Preorder}
	\begin{algorithmic}
		\Procedure{DescendingPostOrder}{node}
			\For{each child of node in descending alphabetical order}
				\State \Call{DescendingPostOrder}{child}
			\EndFor
			\If{node.isWordNode}
				\State \Call{Output}{node.word}
			\EndIf
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Auto-Complete (Prefix Search)
The Trie is often called a Prefix Tree because every node represents a distinct prefix shared by all its structural descendants. This makes prefix search highly efficient:

1.  **Traverse to Prefix:** Start at the root and follow character edges to match the given prefix string (such as `"cat"`).
2.  **Subtree Search:** Once the pointer reaches the node representing that prefix, perform an `AscendingPreOrder` traversal on that isolated subtree.
3.  **Result:** This routine outputs every word in the Trie that begins with those characters.

---

# Architectural Trade-Offs Summary

### Advantages
*   **Deterministic Speed:** Operational performance depends entirely on the character count of the word ($k$), completely independent of the total word count ($n$).
*   **Alphabetical Ordering:** A Pre-order traversal walks the structure in sorted sequence natively.
*   **Prefix Matching Efficiency:** Uniquely optimized for auto-complete routines because all words sharing a common prefix cluster under the same subtree.

### Disadvantages
*   **Space Inefficiency:** This is the primary drawback of a Multiway Trie. Because every node allocates space for an entire alphabet's worth of pointers, a sparse trie (where many characters don't follow others) wastes a significant amount of memory on `NULL` pointers.

---

# Related Notes

- [[Lexicon/Multiway Trie Implementation|Multiway Trie Implementation]]
- [[String Searching Data Structures/Aho-Corasick Automaton|Aho-Corasick Automaton]]
- [[Tree Structures/Binary Search Tree|Binary Search Tree]]
- [[Tree Structures/Ternary Search Tree|Ternary Search Tree]]