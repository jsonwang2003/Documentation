---
title: Memory Management
---
> [!ABSTRACT]
> 
> This chapter covers the transition from fixed, stack-allocated data to dynamic, heap-allocated memory. It explores the internal mechanics of how the C Standard Library manages memory, the importance of data alignment, and the architectural abstractions provided by Virtual Memory.
### 1. Fundamentals of Structs
How C groups variables and stores them in memory.
- **[[Structs in C]]**: Defining custom types and the "pass-by-value" behavior that necessitates pointers.
- **[[Fixed-Sized Struct]]**: Using array members and understanding the limitations of the Stack.
- **[[Size of Struct]]**: Exploring padding, alignment, and how the compiler organizes memory for CPU efficiency.

---
### 2. Pointer Mechanics
Deep-diving into how addresses are manipulated and measured.
- **[[Memory and References]]**: Using `&` and `sizeof` to locate and measure data on the stack.
- **[[Pointer Operations]]**: Pointer arithmetic, scaling based on type size, and the relationship between `a[i]` and `*(a + i)`.
- **[[Metadata and Pointer Operations]]**: How allocators use "hidden" bitwise flags in pointers to track memory status.

---
### 3. The Heap & Dynamic Allocation
Managing memory that persists across function calls.
- **[[Dynamic Struct]]**: Using `malloc` to create flexible structures that outlive their scope.
- **[[Heap and Strings]]**: Handling variable-length text data safely using dynamic buffers.
- **[[Memory Leak]]**: The lifecycle of `malloc` and `free`, and the critical rules to avoid "orphaned" memory.

---
### 4. Allocator Implementation
Building a custom version of `malloc` from scratch.
- **[[Implementation of Malloc]]**: Using `mmap` and headers to partition raw memory into usable blocks.
- **[[Coalescing Heap]]**: Advanced techniques for merging adjacent free blocks to reduce memory fragmentation.

---
### 5. Quick Reference: Allocation Comparison

|**Feature**|**Stack Allocation**|**Heap Allocation**|
|---|---|---|
|**Persistence**|Automatic (lost on return)|Manual (lasts until `free`)|
|**Size Limit**|Small (MBs)|Large (System RAM)|
|**Speed**|Extremely Fast|Slower (Management overhead)|
|**Danger**|Stack Overflow|Memory Leaks / Use-After-Free|
