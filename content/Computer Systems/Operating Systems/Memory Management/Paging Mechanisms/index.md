---
title: Paging Mechanisms
description: Hardware and software mechanisms for address translation, including historical evolution (Base & Bound, Segmentation, early Paging), Page Table Entries (PTE), memory overhead, and Multi-Level Page Tables.
aliases:
  - Paging Mechanisms Hub
  - Paging Index
tags:
  - index
  - operating-systems
  - memory-management
  - paging
---

> [!abstract] Overview 
> Paging Mechanisms encompass the hardware primitives and memory data structures used by the Memory Management Unit (MMU) and Operating System kernel to translate virtual addresses into physical addresses. By dividing virtual address spaces into uniform, fixed-size **pages** and physical memory into corresponding **page frames**, paging eliminates external fragmentation, decouples contiguous logical address spaces from contiguous physical allocation, and provides fine-grained, per-page memory protection.

---
## Evolution of Memory Management

| Note Link        | Description                                                                                        | Key Concepts                                                                          |
| :--------------- | :------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| [[Base & Bound]] | Hardware hardware-assisted contiguous allocation using relocation and limit registers.             | Physical Contiguity, Internal/External Fragmentation, Dynamic Relocation              |
| [[Segmentation]] | Variable-sized logical memory partitioning reflecting application semantics (code, stack, heap).   | Base/Limit Pairs per Segment, Sparse Addressing, External Fragmentation, Compaction   |
| [[Paging]]       | Fixed-size memory partitioning translating Virtual Page Numbers (VPN) to Page Frame Numbers (PFN). | Pages, Page Frames, Virtual Address Split (VPN + Offset), Zero External Fragmentation |

## Advanced Paging Architecture

| Note Link | Description | Key Concepts |
| :--- | :--- | :--- |
| [[Page Table Entries & Memory Overhead]] | Structure of individual Page Table Entries (PTEs) and linear page table space overhead analysis. | PTE Bits (Valid, Protection, Dirty, Accessed, PFN), Linear Table Size Overhead |
| [[Multi-Level Page Tables]] | Tree-structured hierarchical page tables optimizing physical RAM consumption for sparse address spaces. | Page Directory (PDE), Page Upper Directory, Indirection Overhead, Sparse Allocation |

---
# Related Modules

- **[[Translation Lookaside Buffer (TLB)]]**: Specialized hardware MMU cache storing recent VPN-to-PFN mappings to bypass page table memory lookups, including ASIDs and TLB shootdown mechanisms.
- **[[Virtual Memory]]**: High-level OS abstraction handling demand paging, page fault handling, page replacement policies (LRU, CLOCK), and backing store/swap file management.
- **[[Computer Systems/Operating Systems/Memory Management/index|Memory Management]]**: Low-level kernel physical memory management, frame allocation, buddy system allocators, and slab/kmalloc memory pools.