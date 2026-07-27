---
title: "Concurrency & Synchronization Index"
description: "A directory covering concurrency pitfalls, race conditions, critical sections, synchronization primitives, patterns, and deadlocks."
aliases:
  - Concurrency & Synchronization Directory
  - Synchronization Hub
  - Concurrency Index
tags:
  - index
  - operating-systems
  - concurrency
  - synchronization
---
> [!abstract] Overview
> When multiple threads execute concurrently and share mutable state, non-deterministic instruction interleavings can corrupt data structure invariants. **Synchronization** provides hardware and software mechanisms to restrict instruction interleaving, enforce **Mutual Exclusion**, protect **Critical Sections**, and prevent **Deadlocks**.

---

# Module Structure & Subdirectories

### 1. Fundamentals & Critical Sections

| Note Link | Description | Key Concepts |
|---|---|---|
| **[[Operating Systems/Concurrency & Synchronization/Race Conditions & Shared State\|Race Conditions & Shared State]]** | Analyzes non-determinism in multithreaded execution, interleaving mechanics, race condition definitions, thread-private stacks vs. thread-shared heaps/globals, and instruction atomicity assumptions. | Race Conditions, Interleaving, Non-Determinism, Shared Memory |
| **[[Operating Systems/Concurrency & Synchronization/Critical Sections & Mutual Exclusion\|Critical Sections & Mutual Exclusion]]** | Defines critical sections, mutual exclusion locks, the 4 goals of concurrent algorithm design, and Safety vs. Liveness properties. | Critical Sections, Mutual Exclusion, Progress, Bounded Waiting |

---

### 2. Sub-Modules

#### 📁 [[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/index|Synchronization Primitives Subsystem]]
*   **[[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Locks\|Locks]]:** Explores the Lock ADT (`acquire`/`release`), hardware primitives (`disable interrupts`, atomic `test_and_set`), spinlocks, and guarded blocking locks.
*   **[[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Semaphores\|Semaphores]]:** Details Dijkstra's non-negative integer primitive, binary vs counting semaphores, internal wait queue implementations, and atomic `wait()`/`signal()` operations.
*   **[[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Condition Variables\|Condition Variables]]:** Examines memoryless condition variables, Mesa vs Hoare signal semantics, atomic lock-release sleeping (`wait`), `signal`, and `broadcast`.
*   **[[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Monitors\|Monitors]]:** Language-level constructs encapsulating shared data and procedures with compiler-enforced implicit mutual exclusion.

#### 📁 [[Operating Systems/Concurrency & Synchronization/Synchronization Patterns/index|Synchronization Patterns Subsystem]]
*   **[[Operating Systems/Concurrency & Synchronization/Synchronization Patterns/Producer-Consumer Problem\|Producer-Consumer Problem]]:** Formulates bounded buffer synchronization, lost wakeup flaws in naive sleep/wake attempts, and solutions using semaphores or condition variables.
*   **[[Operating Systems/Concurrency & Synchronization/Synchronization Patterns/Reader-Writer Problem\|Reader-Writer Problem]]:** Explores concurrent reader / exclusive writer access rules and semaphore-based implementation (`read_count`, `block_write`).

#### 📁 [[Operating Systems/Concurrency & Synchronization/Deadlocks/index|Deadlocks Subsystem]]
*   **[[Operating Systems/Concurrency & Synchronization/Deadlocks/Deadlock Fundamentals & Coffman Conditions\|Deadlock Fundamentals & Coffman Conditions]]:** Formal definitions, Dining Philosophers, the 4 Coffman conditions, and Resource Allocation Graph (RAG) cycle analysis.
*   **[[Operating Systems/Concurrency & Synchronization/Deadlocks/Deadlock Handling Strategies\|Deadlock Handling Strategies]]:** Evaluations of Ostrich algorithm, Deadlock Prevention (Resource Ordering), Deadlock Avoidance (Banker's Algorithm), and Detection & Recovery.

---

# Related Modules

- [[Operating Systems/Kernel & Architecture/Thread/index|Thread Management Subsystem]]
- [[Operating Systems/index|Operating Systems Main Directory]]