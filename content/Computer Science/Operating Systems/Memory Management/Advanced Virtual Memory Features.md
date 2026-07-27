---
description: "Advanced virtual memory capabilities: Shared Memory across processes, Copy-on-Write (CoW) optimizations, and Memory-Mapped Files (mmap)."
aliases:
  - Shared Memory
  - Copy-on-Write
  - CoW
  - Memory-Mapped Files
  - mmap
tags:
  - operating-systems
  - memory-management
  - virtual-memory
  - copy-on-write
  - mmap
---
> [!abstract] Abstract
> Modern operating systems utilize virtual memory indirection to implement sophisticated performance optimizations beyond standard process isolation. Features like **Shared Memory** enable high-speed inter-process communication, **Copy-on-Write (CoW)** eliminates redundant memory duplication during process creation (`fork`), and **Memory-Mapped Files (`mmap`)** allow file system I/O to be performed directly using standard memory instructions.
> 
> - **Category:** Advanced OS Virtual Memory Capabilities
> - **Primary System Calls:** `shm_open`, `fork`, `mmap`.
> - **Core Driver:** Manipulating [[Operating Systems/Memory Management/Page Table Entries & Memory Overhead|Page Table Entries]] and protection bits to trigger lazy evaluation.

---

## Shared Memory

By default, virtual memory enforces strict process isolation—each process possesses a disjoint set of physical memory frames. **Shared Memory** overrides this default by configuring page table entries in two or more distinct processes to map to the exact same physical page frames in RAM.

![[Pasted image 20260726143725.png]]

*   **API Usage:** Configured via Unix system calls such as `shm_open` and `shm_unlink`.
*   **Address Flexibility:** The shared physical page frame can be mapped at different virtual addresses in each process's address space.
*   **Zero-Copy Efficiency:** Data written by one process is instantly accessible to other processes without passing through kernel buffers.

---

## Copy-on-Write (CoW)

When a process executes `fork()` to create a child process, copying the entire address space in physical RAM is extremely expensive and often wasted if the child immediately calls `exec()`.

![[Pasted image 20260726143923.png]]

**Copy-on-Write (CoW)** optimizes process creation by sharing physical pages lazily:

1.  **Lazy Page Sharing:** During [[Operating Systems/Kernel & Architecture/Process/Process Lifecycle & API|fork()]], parent and child page table entries are set to point to the same physical pages and marked as **Read-Only**.
2.  **Protection Fault:** If either process attempts to write to a shared page, the hardware detects a protection violation and traps to the OS kernel.
3.  **Page Replication:** The kernel allocates a new physical frame, copies the 4 KB page contents, updates the faulting process's PTE to point to the new frame with **Read/Write** permissions, and restarts the write instruction.

---

## Memory-Mapped Files (`mmap`)

Standard file I/O uses `open()`, `read()`, and `write()` system calls, which require copying data between kernel disk buffers and user-space memory buffers.

![[Pasted image 20260726144110.png]]

With **Memory-Mapped Files (`mmap`)**, the OS maps file system blocks directly into the application's virtual address space:

![[Pasted image 20260726144321.png]]

*   **Direct Instruction I/O:** Processes access file contents using standard pointer dereferences and memory instructions (e.g., `lw` and `sw`).
*   **Lazy Demand Paging:** Pages of the file are loaded into RAM lazily via [[Operating Systems/Memory Management/Demand Paging & Page Faults|Demand Paging]] as they are referenced.
*   **Automatic Synchronization:** Modified pages are marked dirty in their PTEs and written back to disk by the kernel page flushing subsystem.

---

## Related Notes

- [[Operating Systems/Memory Management/Page Table Entries & Memory Overhead|Page Table Entries & Memory Overhead]]
- [[Operating Systems/Memory Management/Demand Paging & Page Faults|Demand Paging & Page Faults]]
- [[Operating Systems/Kernel & Architecture/Process/Process Lifecycle & API|Process Lifecycle & API]]