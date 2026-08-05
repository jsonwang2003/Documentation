---
title: "Evolution of Memory Management"
description: "Historical progression of address translation techniques from dynamic contiguous Base & Bound relocation to variable-sized Segmentation and early fixed-size Paging."
aliases:
  - Evolution of Memory Management Hub
tags:
  - index
  - operating-systems
  - memory-management
---
> [!abstract] Overview
> The **Evolution of Memory Management** traces how operating systems transitioned from simple contiguous physical memory allocation to flexible hardware-assisted virtual memory paradigms, solving issues of memory fragmentation, access protection, and dynamic process sizing.

---

## Translation Paradigm Comparison

| Paradigm | Allocation Unit | Hardware Registers | Primary Fragmentation Issue |
|---|---|---|---|
| **Base & Bound** | Single contiguous chunk per process | Base Register, Bound Register | External Fragmentation |
| **Segmentation** | Variable-sized logical segments (Code, Data, Stack) | Segment Table (Base + Bound per segment) | External Fragmentation |
| **Paging** | Fixed-size pages / frames (e.g., 4 KB) | Page Table Base Register (PTBR) | Internal Fragmentation (last page only) |

---
## Module Notes

- [[Base & Bound|Base & Bound]]
- [[Segmentation|Segmentation]]
- [[Paging|Paging]]