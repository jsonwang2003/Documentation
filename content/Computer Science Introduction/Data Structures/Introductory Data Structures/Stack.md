---
description: "A linear restricted ADT enforcing the strict Last In, First Out (LIFO) operational protocol."
aliases:
  - Stack
  - Stack ADT
  - LIFO Container
tags:
  - data-structures
  - adt
  - stack
---
> [!abstract] Abstract 
> A Stack is an [[Abstract Data Types (ADT)|Abstract Data Type]] that enforces the Last In, First Out ($\text{LIFO}$) operational protocol. Elements are added and extracted strictly from a single boundary margin called the **top**.
> 
> - **Category:** Boundary Restricted ADT
> - **Core Rule:** The most recently added element is always the first one removed.
> - **Common Backing Implementations:** [[Array Lists|Array Lists]], [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Singly-Linked Lists]], or [[Deques|Deques]].

---

# Core Functional Interface

A compliant Stack interface exposes three primary operations:

| Operation | Detailed Functional Execution |
|---|---|
| `push(element)` | Places a new element onto the top boundary of the Stack. |
| `pop()` | Extracts and removes the top-most element from the Stack. |
| `peek()` / `top()` | Evaluates and returns the top-most element without removing it. |

---

# Implementation Frameworks

A Stack interface can be efficiently implemented using several concrete backing data structures:

### 1. Array List Backbone
*   **`push(element)`:** Appends to the trailing array index in amortized $O(1)$ time.
*   **`pop()`:** Decrements size and removes the trailing element in $O(1)$ time without requiring data shifting.
*   **`peek()`:** Accesses the trailing index directly in $O(1)$ constant time.

### 2. Singly-Linked List Backbone
*   **`push(element)`:** Prepends a new node at the `head` in $O(1)$ time.
*   **`pop()`:** Advances the `head` pointer to `head.next` in $O(1)$ time.
*   **`peek()`:** Inspects `head.data` in $O(1)$ constant time.

---

# Operational Complexity Analysis

$$\begin{array}{ccc} \mathbf{Operation} & \mathbf{Array\ List\ Backbone} & \mathbf{Singly\text{-}Linked\ List\ Backbone} \\  \hline  \text{push(element)} & \text{Amortized } O(1) & O(1) \\ \text{pop()} & O(1) & O(1) \\ \text{peek() / top()} & O(1) & O(1)  \end{array}$$

```pseudo
	\begin{algorithm}
	\caption{Stack Interface Operations (Linked List Implementation)}
	\begin{algorithmic}
		\Procedure{Push}{$element, top$}
			\State $newNode \gets \text{Allocate new node with } data = element$
			\State $newNode.next \gets top$
			\State $top \gets newNode$
		\EndProcedure

		\Procedure{Pop}{$top$}
			\If{$top == \text{NULL}$}
				\Return $\text{Underflow Error}$
			\EndIf
			\State $poppedData \gets top.data$
			\State $top \gets top.next$
			\Return $poppedData$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# Core Architectural Applications

*   **Function Call Stack & Recursion:** Manages activation records, local variables, and return addresses during nested function calls in programming language runtimes.
*   **Expression Parsing & Evaluation:** Evaluates mathematical expressions and converts infix notation to postfix using algorithms like Dijkstra's Shunting-Yard.
*   **Backtracking Algorithms:** Powers Depth-First Search (DFS) graph exploration routines and maze-solving algorithms.
*   **Undo/Redo History:** Stores historical state snapshots in text editors and browser navigation buffers.

---

# Related Notes

- [[Queues|Queues]]
- [[Deques|Deques]]
- [[Array Lists|Array Lists]]
- [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Linked List]]