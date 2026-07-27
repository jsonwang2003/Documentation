---
title: "Kernel & Architecture Index"
description: "A comprehensive directory covering operating system kernel architecture, hardware privilege boundaries, event-driven trap mechanisms, process abstractions, thread scheduling, and CPU scheduling."
aliases:
  - Kernel & Architecture Directory
  - Kernel Architecture Index
  - OS Core Index
tags:
  - index
  - operating-systems
  - kernel
  - architecture
---
> [!abstract] Overview
> The **Kernel & System Architecture** module details how an operating system controls physical hardware, enforces isolation, handles external events, and provides abstractions for execution. It covers hardware-level privilege enforcement, event-driven kernel execution, and the primary execution subsystems: **Processes** (resource containers), **Threads** (schedulable execution streams), and **CPU Scheduling** (policy-driven core allocation).

---

# Module Structure & Notes

### 1. Hardware Privilege & Isolation Mechanics

| Note Link | Description | Core Primitives |
|---|---|---|
| **[[Operating Systems/Kernel & Architecture/Dual-Mode Operation & Memory Protection\|Dual-Mode Operation & Memory Protection]]** | Hardware isolation via User/Kernel modes, mode bit registers, privileged instruction sets, and MMU protection. | Mode Bit, Privileged Instructions, MMU |
| **[[Operating Systems/Kernel & Architecture/Interrupts and Exceptions\|Interrupts and Exceptions]]** | Event-driven kernel architecture, handling asynchronous hardware interrupts, synchronous faults, and hardware timer preemption. | Trap Vector Table, ISR, Hardware Timer |
| **[[Operating Systems/Kernel & Architecture/System Calls\|System Calls]]** | Software trap mechanisms (`syscall`), register parameter passing, and descriptor handle translation between user and kernel space. | Software Traps, File Descriptors, Handles |

---

### 2. Execution & Resource Subsystems

#### 📁 [[Operating Systems/Kernel & Architecture/Process/index|Process Management Subsystem]]
*   **[[Operating Systems/Kernel & Architecture/Process/Process Abstraction & PCB\|Process Abstraction & PCB]]:** Memory address space layouts (Text, Data, Heap, Stack), execution states, and Process Control Block (`task_struct`) structures.
*   **[[Operating Systems/Kernel & Architecture/Process/Process Lifecycle & API\|Process Lifecycle & API]]:** Creation models (`fork()` + `exec()` vs. `CreateProcess`), process hierarchies, termination (`exit()`, `wait()`), and Zombie/Orphan handling.

#### 📁 [[Operating Systems/Kernel & Architecture/Thread/index|Thread Management Subsystem]]
*   **[[Operating Systems/Kernel & Architecture/Thread/Thread Abstraction & TCB\|Thread Abstraction & TCB]]:** Decoupling address space containers from execution streams, multithreaded memory layouts, TCBs, and Concurrency vs. Parallelism.
*   **[[Operating Systems/Kernel & Architecture/Thread/Thread Context Switch & Scheduling\|Thread Context Switch & Scheduling]]:** State queues, voluntary `yield()` mechanics, low-level assembly context switches, and hardware timer preemption.
*   **[[Operating Systems/Kernel & Architecture/Thread/Kernel vs User Level Threads\|Kernel vs User Level Threads]]:** Evaluating 1:1 Kernel-Level Threads, M:1 User-Level Threads, and M:N Hybrid Multithreading Models.

#### 📁 [[Operating Systems/Kernel & Architecture/CPU Scheduling/index|CPU Scheduling Subsystem]]
*   **[[Operating Systems/Kernel & Architecture/CPU Scheduling/CPU Scheduling Fundamentals & Metrics\|CPU Scheduling Fundamentals & Metrics]]:** Policy vs. mechanism, dispatcher triggers, scheduling metrics ($T_{\text{turnaround}}$, $T_{\text{response}}$), workload profiles, CPU utilization calculations, and starvation.
*   **[[Operating Systems/Kernel & Architecture/CPU Scheduling/Classic Scheduling Algorithms\|Classic Scheduling Algorithms]]:** FCFS, SJF, SRTCF, Round Robin, and Priority Scheduling algorithm evaluation.
*   **[[Operating Systems/Kernel & Architecture/CPU Scheduling/Multilevel Feedback Queue & Real-World Schedulers\|Multilevel Feedback Queue & Real-World Schedulers]]:** Priority decay in MLFQ, I/O burst handling, and production schedulers (Linux CFS, macOS/Windows MLFQ).

---

# System Architecture Map

```
+-----------------------------------------------------------------------+
| USER SPACE                                                            |
|  [Applications] ---> [C Library (glibc)]                              |
|                          |                                            |
|                    system calls (fork, exec, read, yield)             |
+--------------------------|--------------------------------------------+
| HARDWARE BOUNDARY        v                                            |
|                  [Software Traps / Interrupts / Faults]               |
+--------------------------|--------------------------------------------+
| KERNEL SPACE             v                                            |
|  +-----------------------------------------------------------------+  |
|  | Event Handlers & Syscall Dispatcher                             |  |
|  +-----------------------------------------------------------------+  |
|  | Process Subsystem      | Thread Subsystem     | CPU Scheduler   |  |
|  |  - PCBs (task_struct)  |  - TCBs & Stacks     |  - MLFQ / CFS   |  |
|  |  - Address Spaces      |  - Ready/Wait Queues |  - Policy vs    |  |
|  |  - IPC                 |  - Context Switch    |    Mechanism    |  |
|  +-----------------------------------------------------------------+  |
|  | Memory Management (MMU)         | Device Drivers & I/O Systems  |  |
+-----------------------------------------------------------------------+
```

---

# Related Modules

- [[Operating Systems/Concurrency & Synchronization/index|Concurrency & Synchronization Module]]
- [[Operating Systems/Memory Management/index|Memory Management Module]]
- [[Operating Systems/Storage & I/O Systems/index|Storage & I/O Systems Module]]
- [[Operating Systems/index|Operating Systems Main Directory]]