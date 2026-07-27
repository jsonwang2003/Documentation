---
description: "An Abstract Data Type storing unique values without duplicates, optimized for fast membership testing and set-theoretic operations."
aliases:
  - Set
  - Hash Set
  - Tree Set
  - Unique Collection
tags:
  - data-structures
  - adt
  - sets
---
> [!abstract] Abstract 
> A **Set** is an Abstract Data Type (ADT) that stores unique elements without duplicates, modeling the mathematical concept of Set Theory. It is designed for high-speed membership testing (`contains`), uniqueness enforcement, and collection operations like union and intersection.
> 
> - **Category:** Unique Associative ADT
> - **Core Requirement:** Duplicate insertions are rejected or ignored.
> - **Primary Benchmark:** Fast $O(1)$ or $O(\log n)$ membership verification.

---

# Core Architectural Properties

*   **Uniqueness Invariant:** Duplicate elements are prohibited; adding an existing value produces no change.
*   **Unordered vs. Ordered:** Base sets do not guarantee insertion order, though specialized variants (e.g., `TreeSet`) maintain sorted ordering.
*   **Membership Optimization:** Optimized to determine whether an element exists far faster than linear searches.

---

# Common Operations & Complexity

| Operation | Description | Hash Set Complexity | Tree Set Complexity |
|---|---|---|---|
| `add(x)` | Inserts element `x` into the set. | $O(1)$ avg | $O(\log n)$ |
| `remove(x)` | Deletes element `x` from the set. | $O(1)$ avg | $O(\log n)$ |
| `contains(x)` | Checks if `x` exists in the set. | $O(1)$ avg | $O(\log n)$ |
| `size()` | Returns total active element count. | $O(1)$ | $O(1)$ |
| `clear()` | Erases all elements from the set. | $O(n)$ | $O(n)$ |
| `union(B)` | Merges elements from set $A$ and set $B$. | $O(n_A + n_B)$ | $O(n_A + n_B)$ |
| `intersection(B)` | Extracts elements shared by both sets. | $O(\min(n_A, n_B))$ | $O(\min(n_A, n_B))$ |
| `difference(B)` | Extracts elements in $A$ that are not in $B$. | $O(n_A)$ | $O(n_A)$ |

---

# Set Implementation Classifications

1.  **Hash Set:** Backed by a [[Hashing/Hash Tables|Hash Table]]. Delivers $O(1)$ average-case operations; order is arbitrary.
2.  **Tree Set:** Backed by a self-balancing search tree (e.g., [[Tree Structures/Red-Black Tree|Red-Black Tree]]). Maintains elements in sorted order with $O(\log n)$ bounds.
3.  **Linked Hash Set:** Backed by a hash table with an embedded doubly linked list to preserve insertion order.
4.  **Multiset (Bag):** Relaxes uniqueness rules to permit duplicate entries while retaining set operations.

---

# Related Notes

- [[Data Structures/Pair|Pair]]
- [[Hashing/Hash Tables|Hash Tables]]
- [[Tree Structures/Red-Black Tree|Red-Black Tree]]
- [[Data Structures/index|Data Structures Directory]]