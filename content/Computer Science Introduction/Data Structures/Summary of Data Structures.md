---
description: "A comprehensive reference manual summarizing time and space complexities, memory layouts, and trade-offs across major data structures."
aliases:
  - Summary of Data Structures
  - Data Structure Complexities
  - Master Complexity Summary
tags:
  - data-structures
  - reference
  - complexity
  - cheat-sheet
---
> [!abstract] Overview
> This document provides a consolidated architectural summary of essential data structures. It details performance complexities, memory management behaviors, and algorithmic trade-offs across arrays, linked lists, skip lists, heaps, search trees, hash tables, tries, disjoint sets, and graphs.

---

# 1. Linear Data Structures

## [[Array Lists|Array List]]
An **Array List** is an ADT wrapper built over a dynamic array that automatically resizes when capacity limits are hit.

*   **Random Access:** Constant-time $O(1)$ indexing via contiguous memory calculation.
*   **Contiguity:** Elements reside in contiguous memory slots with zero gaps.
*   **Resizing Logic:** Doubling strategy allocates a new array of capacity $2m$, copies elements, and frees old memory.

### Complexity Analysis
| Operation | Unsorted Array List | Sorted Array List |
|---|---|---|
| **Find** | Avg $O(n)$, Worst $O(n)$ | Avg $O(\log n)$ (Binary Search), Worst $O(\log n)$ |
| **Insert** | Avg $O(n)$, Best $O(1)$ at end | Avg $O(n)$ (Requires shifting) |
| **Remove** | Avg $O(n)$, Best $O(1)$ at end | Avg $O(n)$ (Requires shifting) |
| **Space** | $O(n)$ | $O(n)$ |

## [[Linked List|Linked List]]
Sequential chains of individual node objects connected via explicit pointers.

*   **Singly-Linked:** Node tracks `data` and a single `next` pointer.
*   **Doubly-Linked:** Node tracks `data`, `next`, and `previous` pointers.
*   **Modification Logic:** Pointer redirection takes $O(1)$ time once the target node is located.

### Complexity Analysis
| Operation              | Singly-Linked List          | Doubly-Linked List          |
| ---------------------- | --------------------------- | --------------------------- |
| **Find**               | Avg $O(n)$, Worst $O(n)$    | Avg $O(n)$                  |
| **Insert (Head/Tail)** | $O(1)$                      | $O(1)$                      |
| **Insert (Middle)**    | $O(n)$ search + $O(1)$ swap | $O(n)$ search + $O(1)$ swap |
| **Space Overhead**     | $O(n)$ (1 pointer/node)     | $O(n)$ (2 pointers/node)    |

## [[Skip Lists|Skip List]]
A probabilistic data structure augmenting a linked list with multi-level forward pointers, enabling $O(\log n)$ logarithmic lookups.

*   **Probabilistic Height:** Node levels are assigned via coin flips (probability $p$).
*   **Multi-Level Traversal:** Search starts at top level of head node, skipping large spans before dropping down levels.

### Complexity Analysis
| Operation | Average Case | Worst Case |
|---|---|---|
| **Find / Insert / Remove** | $O(\log n)$ | $O(n)$ (If coin flips degrade to height 1) |
| **Space Overhead** | Expected $O(n)$ | Worst $O(n \log n)$ |

---

# 2. Priority & Search Trees

## [[Heap]]
A complete binary tree enforcing relative priority ordering between parents and children.

*   **Array Mapping:** Flat array storage where child offsets resolve to $2i+1$ and $2i+2$.
*   **Heap Invariant:** Min-Heap ($\text{Parent} \le \text{Children}$) or Max-Heap ($\text{Parent} \ge \text{Children}$).

### Complexity Analysis
| Operation | Complexity | Operational Detail |
|---|---|---|
| **Peek** | $O(1)$ | Root element lookup at index 0. |
| **Insert (Push)** | $O(\log n)$ | Appends to tail + Bubble-Up rebalancing. |
| **Pop (Extract)** | $O(\log n)$ | Swaps root with tail + Trickle-Down rebalancing. |
| **Space** | $O(n)$ | Flat contiguous storage with no empty slots. |

## Binary Search Tree (BST) Variations

| Structure                                                 | Find (Worst) | Insert (Worst) | Remove (Worst) | Balancing Mechanism                                |
| --------------------------------------------------------- | ------------ | -------------- | -------------- | -------------------------------------------------- |
| **[[Binary Search Tree (BSTs)\|Standard BST]]**           | $O(n)$       | $O(n)$         | $O(n)$         | None (Degenerates on sorted input).                |
| **[[Randomized Search Trees (Treap, RST)\|RST (Treap)]]** | $O(n)$       | $O(n)$         | $O(n)$         | Probabilistic random priorities ($O(\log n)$ avg). |
| **[[AVL Tree]]**                                          | $O(\log n)$  | $O(\log n)$    | $O(\log n)$    | Strict balance factors ($\pm 1$) via rotations.    |
| **[[Red-Black Tree]]**                                    | $O(\log n)$  | $O(\log n)$    | $O(\log n)$    | Relaxed color rules; optimized for writes.         |


## B-Tree & B+ Tree
"Fat" balanced search trees designed for disk storage and database indexing by maximizing branching factor $b$.

*   **B-Tree:** Internal nodes store search keys alongside actual data records.
*   **B+ Tree:** Internal nodes store search keys only; all data records reside exclusively in linked leaf nodes for efficient range sweeps.

| Metric | B-Tree | B+ Tree |
|---|---|---|
| **Find (Worst)** | $O(\log n)$ | $O(\log n + \log L)$ |
| **Data Placement** | Any node level | Leaf nodes exclusively |
| **Range Queries** | Requires tree traversal | Fast sequential leaf list walk |

---

# 3. Hash-Based & String Data Structures

## [[Hash Tables|Hash Table]] & [[Hash Maps (Maps)|Hash Map]]
Associative structures mapping keys to array slots via string hash functions $h(k)$.

*   **Open Addressing:** Linear Probing, Double Hashing, Cuckoo Hashing.
*   **Closed Addressing:** Separate Chaining (Linked lists or BSTs per bucket).

| Strategy                                                         | Find (Avg) | Find (Worst) | Key Characteristics                                            |
| ---------------------------------------------------------------- | ---------- | ------------ | -------------------------------------------------------------- |
| **[[Open Addressing (Linear Probing)\|Linear Probing]]**         | $O(1)$     | $O(n)$       | High cache locality; sensitive to clustering ($\alpha < 0.5$). |
| **[[Closed Addressing (Separate Chaining)\|Separate Chaining]]** | $O(1)$     | $O(n)$       | Handles high load factors ($\alpha > 1.0$) gracefully.         |
| **[[Cuckoo Hashing]]**                                           | $O(1)$     | $O(1)$ worst | Guaranteed $O(1)$ lookups via two hash candidate slots.        |

## [[Computer Science Introduction/Data Structures/String Searching Data Structures/index|String Searching Structures]]

| Structure                        | Find (Avg)                  | Space Complexity            | Primary Advantage                                               |
| -------------------------------- | --------------------------- | --------------------------- | --------------------------------------------------------------- |
| **[[Multiway Trie]]**            | $O(k)$                      | $O(n \cdot k \cdot \Sigma)$ | Fastest prefix queries; memory inefficient for large alphabets. |
| **[[Ternary Search Tree]]**      | $O(k + \log n)$             | $O(n \cdot k)$              | Space-efficient hybrid using 3 child pointers per node.         |
| **[[Disjoint Sets & Up-Trees]]** | $O(\alpha(n)) \approx O(1)$ | $O(n)$                      | Amortized near-constant time dynamic set partitioning.          |

---

# 4. Graph Representations

| Representation       | Edge Lookup      | Find Neighbors     | Space Complexity   | Best Use Case                            |
| -------------------- | ---------------- | ------------------ | ------------------ | ---------------------------------------- |
| **Adjacency Matrix** | $O(1)$           | $O(\|V\|)$         | $O(\|V\|^{2})$     | Dense graphs ($\|E\| \approx \|V\|^{2}$) |
| **Adjacency List**   | $O(\|E\|)$ worst | $O(\text{deg}(u))$ | $O(\|V\| + \|E\|)$ | Sparse graphs (BFS, DFS, Dijkstra).      |

For complete write up, visit [[Graph Representations]]

---

# 5. Master Summary Table

| Data Structure          | Search (Avg)           | Search (Worst)         | Space Complexity                | Primary Optimal Use Case                            |
| ----------------------- | ---------------------- | ---------------------- | ------------------------------- | --------------------------------------------------- |
| **Array List**          | $O(n)$ / $O(\log n)^*$ | $O(n)$ / $O(\log n)^*$ | $O(n)$                          | Fast random indexing ($*$Sorted via Binary Search). |
| **Linked List**         | $O(n)$                 | $O(n)$                 | $O(n)$                          | Frequent $O(1)$ head/tail insertions.               |
| **Skip List**           | $O(\log n)$            | $O(n)$                 | $O(n)$                          | Concurrent logarithmic ordered lookups.             |
| **Heap**                | $O(1)$ root            | $O(n)$ arbitrary       | $O(n)$                          | Priority queue dispatching ($O(1)$ peek).           |
| **AVL Tree**            | $O(\log n)$            | $O(\log n)$            | $O(n)$                          | Read-heavy lookups requiring guaranteed bounds.     |
| **Red-Black Tree**      | $O(\log n)$            | $O(\log n)$            | $O(n)$                          | Write-heavy general purpose maps (`std::map`).      |
| **B+ Tree**             | $O(\log n)$            | $O(\log n)$            | $O(n)$                          | Database indexing and file system storage.          |
| **Hash Table**          | $O(1)$                 | $O(n)$                 | $O(n)$                          | Exact match key-value lookups.                      |
| **Multiway Trie**       | $O(k)$                 | $O(k)$                 | $O(n \cdot k \cdot \|\Sigma\|)$ | High-speed auto-complete with small alphabets.      |
| **Ternary Search Tree** | $O(k + \log n)$        | $O(n)$                 | $O(n \cdot k)$                  | Memory-efficient dictionary auto-complete.          |
| **Disjoint Set**        | $O(\alpha(n))$         | $O(\log n)$            | $O(n)$                          | Kruskal's MST and connected components.             |
| **Adjacency List**      | $O(\text{deg}(u))$     | $O(\|E\|)$             | $O(\|V\| + \|E\|)$              | Graph traversal algorithms on sparse networks.      |

---

# Related Notes

- [[Computer Science Introduction/Data Structures/index|Data Structures Directory]]
- [[Classes of Computational Complexity|Classes of Computational Complexity]]
- [[Computer Science Introduction/Data Structures/Tree Structures/index|Tree Structures]]
- [[Computer Science Introduction/Data Structures/String Searching Data Structures/index|String Searching Data Structures]]