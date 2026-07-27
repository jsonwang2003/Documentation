---
title: "Data Structures"
description: "A comprehensive directory covering fundamental storage models, behavioral ADTs, hierarchical tree topologies, and complexity classifications."
aliases:
  - Data Structure
  - Data Structures Index
  - Data Structures Hub
tags:
  - index
  - data-structures
  - computer-science
---
> [!abstract] Overview
> A Data Structure is a specialized format for organizing, processing, retrieving, and storing data in computer memory so that operations can be executed efficiently. This index categorizes foundational linear layouts, abstract data type (ADT) behaviors, hierarchical trees, and computational complexity models.

---

# 1. Fundamentals & Linear Structures

This module covers physical memory organization and core sequential arrangements:

*   **Data Structures vs. ADTs:** Distinguishing physical storage mechanics from logical behavioural contracts.
*   **[[Data Structures/Summary of Data Structures#Array List|Array Lists]]:** Dynamic contiguous memory storage supporting $O(1)$ random indexing.
*   **[[Data Structures/Summary of Data Structures#Linked List|Linked Lists]]:** Sequential node chains connected via explicit pointers.
*   **Circular Arrays:** Memory-efficient ring buffers for fixed-size capacity queues.

---

# 2. Common Abstract Data Types (ADTs)

Logical interfaces that define operational contracts regardless of underlying backing storage:

*   **Stack:** LIFO (Last-In, First-Out) push/pop pipeline.
*   **Queue:** FIFO (First-In, First-Out) enqueue/dequeue pipeline.
*   **Deque:** Double-ended queue allowing operations at both ends.
*   **[[Introductory Data Structures/Priority Queue|Priority Queue]]:** Elements ordered and retrieved based on priority rankings.
*   **[[Data Structures/Set|Set]]:** Collection tracking unique elements.
*   **[[Hashing/Hash Maps (Maps)|Hash Map]]:** Key-value associative container.
*   **[[Data Structures/Pair|Pair]]:** Lightweight two-element heterogeneous container.

---

# 3. Tree & Hierarchical Structures

Data organizations optimized for fast searching, sorting, and priority-based access:

*   **[[Tree Structures/Binary Tree|Binary Tree]]:** Rooted topology limiting child branching to at most 2 nodes per parent.
*   **[[Tree Structures/Binary Search Tree|Binary Search Tree (BST)]]:** Sorted structure providing average $O(\log n)$ search efficiency.
*   **[[Tree Structures/Heap|Heap]]:** Complete binary tree mapped to flat arrays for $O(1)$ priority root access.
*   **[[Tree Structures/Frequent Pattern Tree (FP-Tree)|FP-Tree]]:** Dense prefix tree for mining frequent itemsets without candidate generation.

---

# 4. Summary Index by Categorization

### By Storage Layout
*   **Linear:** Arrays, Linked Lists, Circular Buffers.
*   **Non-Linear:** Trees, Heaps, Graphs, Tries.

### By Access Pattern (ADTs)
*   **Sequential:** Stacks, Queues, Deques.
*   **Associative:** Hash Maps, Sets, Pairs.
*   **Ranked:** Priority Queues, Binary Heaps.

---

# Related Modules

- [[Data Structures/Summary of Data Structures|Summary of Data Structures]]
- [[Data Structures/Classes of Computational Complexity|Classes of Computational Complexity]]
- [[Tree Structures/index|Tree Structures]]
- [[Lexicon/index|Lexicon ADT Implementations]]
- [[String Searching Data Structures/index|String Searching Data Structures]]