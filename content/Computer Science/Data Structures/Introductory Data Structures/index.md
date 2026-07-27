---
title: "Introductory Data Structures"
description: "A foundational hub breaking down structural memory specifications, boundary collection tracking, and common abstract data interfaces."
aliases:
  - Introductory Data Structures Hub
  - Linear Structures Index
  - Introductory Data Structures Directory
tags:
  - index
  - data-structures
  - linear
---
> [!abstract] Overview
> This module covers the fundamental building blocks of software engineering: the crucial division between logical blueprints ([[Introductory Data Structures/Abstract Data Types (ADT)|Abstract Data Types]]) and their underlying physical implementations (Data Structures).

---

# Architectural Principles

* **[[Introductory Data Structures/Abstract Data Types (ADT)|Abstract Data Types (ADT)]]:** High-level interface definitions specifying external behavioral operations without tying logic to explicit memory management code.
* **[[Introductory Data Structures/Data Structures vs. Abstract Data Types|Data Structures vs. Abstract Data Types]]:** Analyzing the design boundary separating specification models from literal machine memory configurations.

---

# Physical Storage Backbones

* **[[Introductory Data Structures/Array Lists|Array Lists]]:** Bounded, contiguous, homogeneous layouts providing rapid constant-time indexing lookups.
* **[[Introductory Data Structures/Circular Arrays|Circular Arrays]]:** An index-wrapping array optimization that handles high-frequency changes at both ends without requiring data shifts.
* **[[Introductory Data Structures/Linked List|Linked List]]:** Dynamically allocated memory nodes connected sequentially via pointer addresses, bypassing structural resizing penalties.
* **[[Introductory Data Structures/Skip Lists|Skip Lists]]:** A layered probabilistic linked array structure leveraging randomized express layers to achieve $O(\log n)$ search speeds over linked records.

---

# Foundational Linear Interfaces

* **[[Introductory Data Structures/Stack|Stack]]:** A restricted boundary container operating on the Last In, First Out ($\text{LIFO}$) protocol.
* **[[Introductory Data Structures/Queues|Queues]]:** A sequential traffic manager operating on the strict First In, First Out ($\text{FIFO}$) baseline.
* **[[Introductory Data Structures/Deques|Deques]]:** A generalized bidirectional double-ended queue supporting insertion and erasure at both margins.
* **[[Introductory Data Structures/Priority Queue|Priority Queue]]:** An ordered dispatcher that releases items based on assigned urgency metrics rather than raw chronological arrival time.

---

# Notes in This Section

| Note Link | Description |
|---|---|
| [[Introductory Data Structures/Abstract Data Types (ADT)\|Abstract Data Types (ADT)]] | Defines behavioral specifications decoupled from underlying physical memory management. |
| [[Introductory Data Structures/Data Structures vs. Abstract Data Types\|Data Structures vs. Abstract Data Types]] | Compares abstract interface contracts against concrete memory structures. |
| [[Introductory Data Structures/Array Lists\|Array Lists]] | Sequential contiguous memory structures providing constant-time random access. |
| [[Introductory Data Structures/Circular Arrays\|Circular Arrays]] | Array wrapper utilizing modular index wrapping for fast end manipulations. |
| [[Introductory Data Structures/Linked List\|Linked List]] | Dynamic pointer-linked node structures bypassing contiguous allocation constraints. |
| [[Introductory Data Structures/Skip Lists\|Skip Lists]] | Layered probabilistic linked lists providing logarithmic search and insertion bounds. |
| [[Introductory Data Structures/Stack\|Stack]] | LIFO restricted container supporting push, pop, and top operations. |
| [[Introductory Data Structures/Queues\|Queues]] | FIFO restricted container mapping sequential arrival buffers. |
| [[Introductory Data Structures/Deques\|Deques]] | Bidirectional double-ended queue enabling constant-time edge edits. |
| [[Introductory Data Structures/Priority Queue\|Priority Queue]] | Priority-ordered dispatch container commonly backed by binary heaps. |

---

# Related Modules

- [[Tree Structures/index\|Tree Structures]]
- [[Tree Structures/AVL Tree\|AVL Tree]]
- [[Hashing/index\|Hashing]]