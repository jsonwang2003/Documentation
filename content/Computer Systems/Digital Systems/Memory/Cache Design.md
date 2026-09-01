---
description: "Principles of cache memory design: address mapping strategies, block placement, replacement algorithms, write policies, and Average Memory Access Time (AMAT) performance modeling."
aliases:
  - Cache Design
  - Cache Mapping & Policies
  - AMAT
  - Cache Replacement Policies
tags:
  - computer-systems
  - digital-systems
  - cache
  - amat
  - cache-policies
---
> [!abstract] Abstract
> **Cache Design** bridges the performance gap between fast processors and slow main memory by keeping temporal and spatial working sets on-chip. Cache architecture is governed by three fundamental design choices: **Address Mapping** (Direct-Mapped, $N$-Way Set Associative, Fully Associative), **Replacement Policies** (LRU, FIFO, Random), and **Write Policies** (Write-Through vs. Write-Back). Cache efficiency directly dictates the system's **Average Memory Access Time (AMAT)**.

---

## 1. Cache Mapping Strategies

Cache mapping dictates how main memory blocks are placed into cache lines.

```
Main Memory Address: [ Tag Bits | Index Bits | Block Offset ]
```

* **Direct-Mapped Cache:** Each memory block maps to exactly one specific cache line determined by `Index = Block Address % Number of Lines`. Fast tag comparison, but susceptible to conflict misses.
* **Fully Associative Cache:** A memory block can be placed in any cache line. Eliminates conflict misses, but requires complex parallel tag comparators.
* **$N$-Way Set-Associative Cache:** Memory blocks map to a specific set containing $N$ lines. Balances hardware comparator complexity with conflict miss reduction.

---

## 2. Replacement & Write Policies

When a cache miss occurs in a full set, a **Replacement Policy** selects the victim line:
* **Least Recently Used (LRU):** Evicts the block idle for the longest duration.
* **First-In, First-Out (FIFO):** Evicts the block resident in cache for the longest time.
* **Random:** Randomly evicts a line; reduces control logic overhead.

### Write Handling Policies

| Policy | Execution Mechanism | Trade-Offs |
|---|---|---|
| **Write-Through** | Writes update both the cache line and main memory simultaneously. | Simple; keeps main memory consistent; higher write bandwidth demand. |
| **Write-Back** | Writes update only the cache line, setting a **dirty bit**. Main memory is updated only upon line eviction. | Fast write execution; reduces memory bus traffic; risks data inconsistency if unbuffered. |
| **Write-Allocate** | Loads the block from main memory into cache on a write miss. | Typically paired with Write-Back. |
| **No-Write-Allocate** | Writes directly to main memory on a write miss without caching the block. | Typically paired with Write-Through. |

---

## 3. Cache Performance & AMAT Analysis

System performance depends on total cache size, block size (spatial locality), associativity, and hit rates.

### Average Memory Access Time (AMAT) Formula

For a two-level cache hierarchy ($L_1, L_2$) backed by Main Memory:

$$t_{av} = \underbrace{h_1 t_{L_1}}_{\text{Hit in } L_1} + \underbrace{(h_2 - h_1) t_{L_2}}_{\text{Hit in } L_2} + \underbrace{(1 - h_2 - h_1) t_{\text{main}}}_{\text{Penalty to Main Memory}}$$

Where:
* $h_1$ = Local hit rate in $L_1$ cache ($\%$).
* $h_2$ = Global hit rate in $L_2$ cache ($\%$).
* $t_{L_1}, t_{L_2}, t_{\text{main}}$ = Access latencies for $L_1$, $L_2$, and main memory.

---

## Related Notes

- [[Computer Systems/Digital Systems/Memory/Memory Hierarchy|Memory Hierarchy]]
- [[Computer Systems/Digital Systems/Memory/Memory Types|Memory Types]]
- [[Computer Systems/Digital Systems/Memory/index|Memory Index]]