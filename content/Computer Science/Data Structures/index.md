---
title: Data Structure
---
> [!INFO] Definition
> 
> Ways to store and organize data in a computer so it can be used efficiently.

---

## 1. Fundamentals and Linear Structures

This section covers the core concepts of memory organization and basic sequential data arrangements.
- **[[Introductory Data Structures/Data Structures vs. Abstract Data Types|Data Structures vs. ADTs]]**: The distinction between logical models and physical implementations.
- **[[Introductory Data Structures/Abstract Data Types (ADT)|Abstract Data Types (ADT)]]**: Formal definitions of behavioral models.
- **[[Introductory Data Structures/Array Lists|Array Lists]]**: Sequential storage using contiguous memory (Static and Dynamic).
- **[[Introductory Data Structures/Linked List|Linked List]]**: Elements stored in nodes with pointers to sequential neighbors.
- **[[Introductory Data Structures/Circular Arrays|Circular Arrays]]**: Memory optimization for fixed-size buffers.

---
## 2. Common ADTs (Behavioral Models)
Logical interfaces that define how data is accessed and manipulated, regardless of the underlying storage.
- **[[Introductory Data Structures/Stack|Stack]]**: LIFO (Last-In, First-Out) access pattern.
- **[[Introductory Data Structures/Queues|Queues]]**: FIFO (First-In, First-Out) access pattern.
- **[[Introductory Data Structures/Deques|Deques]]**: Double-ended queue allowing access from both ends.
- **[[Introductory Data Structures/Priority Queue|Priority Queue]]**: Elements processed based on urgency or value.
- **[[Set]]**: Collection of unique elements.
- **[[Hash Maps (Maps)]]**: Key-value pair associations.
- **[[Pair]]**: A simple container for two related data elements.

---
## 3. Tree Structures

Hierarchical data organizations optimized for searching, sorting, and priority-based access.
- **[[Tree Structures/Binary Tree|Binary Tree]]**: The foundational hierarchical structure where each node has at most two children.
- **[[Binary Search Tree (BSTs)|Binary Search Tree (BST)]]**: Optimized for $O(\log n)$ search by maintaining sorted order.
- **[[Tree Structures/Heap|Heap]]**: A complete tree used for efficient priority-based retrieval.
- **[[Tree Structures/Frequent Pattern Tree (FP-Tree)|FP-Tree]]**: Advanced structure used for mining frequent patterns in datasets.

---
## 4. Summary Index

### By Storage Type
- **Linear**: [[Introductory Data Structures/Array Lists|Arrays]], [[Introductory Data Structures/Linked List|Linked Lists]], [[Introductory Data Structures/Circular Arrays|Circular Buffers]].
- **Non-Linear**: [[Tree Structures/Binary Tree|Trees]], [[Binary Search Tree (BSTs)|BSTs]], [[Tree Structures/Heap|Heaps]].
### By Access Pattern (ADTs)
- **Sequential**: [[Introductory Data Structures/Stack|Stacks]], [[Introductory Data Structures/Queues|Queues]], [[Introductory Data Structures/Deques|Deques]].
- **Associative**: [[Hash Maps (Maps)]], [[Set]].
- **Ranked**: [[Introductory Data Structures/Priority Queue|Priority Queues]].