---
title: Collision Resolution
description: An index covering foundational strategies, trade-offs, and algorithms for resolving hash table indexing conflicts.
aliases:
  - Hashing Collision Hub
  - Collision Strategies Index
  - Collision Resolution
tags:
  - index
  - hashing
  - collision-resolution
---
> [!abstract] Overview
> When mapping an infinite set of keys into a finite array capacity, multiple unique keys will inevitably hash to the exact same array index. Collision resolution frameworks provide the deterministic logic needed to store, search, and delete conflicting elements without losing data or degrading search performance.

---

# Foundational Concepts

### Closed Addressing (External Storage)
Keys are confined to their initial hashed index address. Colliding items are stored outside the primary table using separate data blocks:

*   **Separate Chaining:** Slots in the array point to external linked data structures, typically dynamic singly linked lists.

### Open Addressing (Internal Storage)
All keys are stored directly within the primary backing array structure. When a target slot is occupied, the table is "open" to positioning the key in alternative array coordinates:

*   **Linear Probing:** Sequentially checks the next consecutive array slot ($\text{index} + 1$) until a vacant spot is found.
*   **Double Hashing:** Computes a custom stride size for each key using a secondary hash function to determine step distance.
*   **Random Hashing:** Walks through a repeatable, pseudorandom probe path seeded by the key value itself.

---

# Strategic Architecture Trade-offs

| Feature | Closed Addressing | Open Addressing (Linear Probing) | Open Addressing (Double / Random) |
|---|---|---|---|
| **Storage Destination** | External data structures | Directly inside array slots | Directly inside array slots |
| **Max Load Factor ($\alpha$)** | Can exceed $1.0$ | Must stay below $1.0$ | Must stay below $1.0$ |
| **Deletion Cost** | Simple (Node unlinking) | Complex (Requires tombstones) | Complex (Requires tombstones) |
| **Cache Line Performance** | Poor (Scattered nodes) | Excellent (Sequential access) | Moderate (Calculated stride jumps) |
| **Primary Weakness** | Extra memory pointer overhead | Primary Clustering clumps | CPU computation cycle overhead |

---

# Notes in This Section

| Note Link                                                                                                     | Description                                                                                             |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [[Hashing/Collision Resolution/Closed Addressing (Separate Chaining)\|Closed Addressing (Separate Chaining)]] | Points occupied array slots to external linked chains, keeping keys at their natural hash addresses.    |
| [[Hashing/Collision Resolution/Open Addressing (Linear Probing)\|Open Addressing (Linear Probing)]]           | Steps sequentially through adjacent slots upon conflict, maximizing hardware cache lines.               |
| [[Hashing/Collision Resolution/Double Hashing\|Double Hashing]]                                               | Eliminates primary clustering by generating a personalized jump offset using a secondary hash function. |
| [[Hashing/Collision Resolution/Random Hashing\|Random Hashing]]                                               | Generates a repeatable pseudorandom sequence seeded by the key value to distribute colliders uniformly. |

---

# Related Categories

- [[Tree Structures/index\|Tree Structures]]
- [[Graphs/index\|Graphs]]