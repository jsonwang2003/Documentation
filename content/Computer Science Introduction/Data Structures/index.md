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

*   **[[Data Structures vs. Abstract Data Types|Data Structures vs. ADTs]]:** Distinguishing physical storage mechanics from logical behavioral contracts.
*   **[[Summary of Data Structures#Array List|Array Lists]]:** Dynamic contiguous memory storage supporting $O(1)$ random indexing.
*   **[[Summary of Data Structures#Linked List|Linked Lists]]:** Sequential node chains connected via explicit pointers.
*   **[[Circular Arrays|Circular Array]]:** Memory-efficient ring buffers for fixed-size capacity queues.

---

# 2. Common Abstract Data Types (ADTs)

Logical interfaces that define operational contracts regardless of underlying backing storage:

*   **[[Computer Science Introduction/Data Structures/Introductory Data Structures/Stack|Stacks]]:** LIFO (Last-In, First-Out) push/pop pipeline.
*   **[[Queues]]:** FIFO (First-In, First-Out) enqueue/dequeue pipeline.
*   **[[Deques]]:** Double-ended queue allowing operations at both ends.
*   **[[Priority Queue|Priority Queue]]:** Elements ordered and retrieved based on priority rankings.
*   **[[Set|Set]]:** Collection tracking unique elements.
*   **[[Hash Maps (Maps)|Hash Map]]:** Key-value associative container.
*   **[[Computer Science Introduction/Data Structures/Pair|Pair]]:** Lightweight two-element heterogeneous container.

---

# 3. Tree & Hierarchical Structures

Data organizations optimized for fast searching, sorting, and priority-based access:

*   **[[Binary Tree|Binary Tree]]:** Rooted topology limiting child branching to at most 2 nodes per parent.
*   **[[Binary Search Tree (BSTs)|Binary Search Tree]]:** Sorted structure providing average $O(\log n)$ search efficiency.
*   **[[Heap|Heap]]:** Complete binary tree mapped to flat arrays for $O(1)$ priority root access.
*   **[[Frequent Pattern Tree (FP-Tree)|FP-Tree]]:** Dense prefix tree for mining frequent itemsets without candidate generation.

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

- [[Summary of Data Structures|Summary of Data Structures]]
- [[Classes of Computational Complexity|Classes of Computational Complexity]]
- [[Computer Science Introduction/Data Structures/Tree Structures/index|Tree Structures]]
- [[Computer Science Introduction/Data Structures/Lexicon/index|Lexicon ADT Implementations]]
- [[Computer Science Introduction/Data Structures/String Searching Data Structures/index|String Searching Data Structures]]