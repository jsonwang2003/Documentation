---
description: "A double-ended queue blueprint supporting bidirectional element addition and removal at both boundary margins."
aliases:
  - Deque
  - Double-Ended Queue
  - Bidirectional Queue
tags:
  - data-structures
  - adt
  - queues
---
> [!abstract] Abstract 
> A Deque (Double-Ended Queue) is an [[Introductory Data Structures/Abstract Data Types (ADT)|Abstract Data Type]] that allows for insertion and removal of elements from both the front and the back. It serves as a generalized linear list supporting bi-directional growth, combining [[Introductory Data Structures/Stack|Stack]] and [[Introductory Data Structures/Queues|Queue]] workflows into a single interface.
> 
> - **Category:** Linear Boundary ADT
> - **Primary Interface Capability:** Direct edge manipulation.
> - **Common Structural Implementations:** [[Introductory Data Structures/Linked List|Doubly-Linked Lists]] or [[Introductory Data Structures/Circular Arrays|Circular Arrays]].

---

# Formal Operational Contract

A compliant Deque interface exposes six core operations:

| Function | Operational Action |
|---|---|
| `addFront(element)` | Inserts a new element at the beginning of the Deque. |
| `addBack(element)` | Inserts a new element at the trailing end of the Deque. |
| `peekFront()` | Returns the value of the first element without removing it. |
| `peekBack()` | Returns the value of the last element without removing it. |
| `removeFront()` | Removes the first element from the Deque. |
| `removeBack()` | Removes the trailing element from the Deque. |

---

# Implementation Frameworks

The Deque interface contract can be backed by two primary structures, each imposing unique algorithmic trade-offs:

### 1. Doubly-Linked List Backbone
Maintains global, explicit references to `head` and `tail` node objects.

*   **Boundary Performance:** Guaranteed true $O(1)$ for all six core operations via isolated pointer manipulation.
*   **Memory Footprint:** Fully dynamic; allocates node blocks as needed without wasting capacity.
*   **Trade-off Risk:** Accessing or reading elements situated in the middle of the collection requires an $O(n)$ pointer-chasing traversal loop.

### 2. Circular Array Backbone
Utilizes a bounded flat array paired with wrapping indexing logic.

*   **Boundary Performance:** $O(1)$ for lookup and removal operations; addition steps are amortized $O(1)$ but can occasionally spike to $O(n)$ if a capacity resize event is triggered.
*   **Memory Footprint:** May pre-allocate large continuous memory blocks, introducing memory overhead if that capacity goes unused.
*   **Trade-off Risk:** Provides rapid $O(1)$ random access to middle positions via simple modular index arithmetic.

---

# Backing Structural Trade-offs

| Performance Metric | Doubly-Linked List Implementation | Circular Array Implementation |
|---|---|---|
| **Boundary Operations** | Strictly $O(1)$ constant time | Amortized $O(1)$ constant time |
| **Random Access** | $O(n)$ linear traversal cost | $O(1)$ constant time math |
| **Memory Footprint Style** | Node pointer overhead per item | Unused capacity buffer overhead |
| **Worst-Case Add Latency** | $O(1)$ continuous performance | $O(n)$ transient spikes during resizing |

![[Pasted image 20260104132805.png]]

---

# Operational Complexity Analysis

$$\begin{array}{ccc} \mathbf{Operation} & \mathbf{Doubly\text{-}Linked\ List} & \mathbf{Circular\ Array} \\  \hline  \text{addFront / addBack} & O(1) & \text{Amortized } O(1) \\ \text{removeFront / removeBack} & O(1) & O(1) \\ \text{peekFront / peekBack} & O(1) & O(1)  \end{array}$$

> [!note] Architectural Priority Details
> Under a Doubly-Linked List backbone, `removeBack` runs in $O(1)$ constant time because each node retains an explicit pointer to its predecessor (`node.prev`). In a Singly-Linked List, `removeBack` degrades to $O(n)$ because the system must trace forward from the `head` to locate the node preceding the `tail`. This makes the Doubly-Linked List the standard backing structure for Deque variants.

---

# Related Notes

- [[Introductory Data Structures/Circular Arrays|Circular Arrays]]
- [[Introductory Data Structures/Linked List|Linked List]]
- [[Introductory Data Structures/Queues|Queues]]
- [[Introductory Data Structures/Stack|Stack]]


