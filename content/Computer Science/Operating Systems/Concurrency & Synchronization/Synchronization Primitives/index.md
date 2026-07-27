---
title: "Synchronization Primitives Index"
description: "A directory covering core synchronization constructs: Locks, Semaphores, Condition Variables, and Monitors."
aliases:
  - Synchronization Primitives Directory
  - Primitives Index
  - Synchronization Primitives Hub
tags:
  - index
  - operating-systems
  - concurrency
  - synchronization
  - primitives
---
> [!abstract] Overview
> **Synchronization Primitives** are software abstractions built on top of hardware atomic instructions (or interrupt disabling) that allow concurrent threads to safely access shared resources, enforce mutual exclusion, and coordinate execution timing.

---

# Core Primitives

| Primitive Link | Description | Primary Mechanism / API | Primary Use Case |
|---|---|---|---|
| **[[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Locks\|Locks]]** | Enforces strict mutual exclusion across critical sections. Evolves from hardware spinlocks to guarded sleep locks. | `acquire()`, `release()`, `test_and_set()` | Mutual Exclusion (Single Thread in Critical Section) |
| **[[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Semaphores\|Semaphores]]** | Dijkstra's non-negative integer variable. Retains event history to manage resource pools and timing sequences. | `wait()` ($P$), `signal()` ($V$), Counter + Wait Queue | Mutual Exclusion ($N=1$) AND Event Sequencing ($N > 1$) |
| **[[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Condition Variables\|Condition Variables]]** | Memoryless synchronization queues allowing threads to sleep inside critical sections by atomically releasing locks. | `wait()`, `signal()`, `broadcast()` (Mesa Semantics) | Waiting for complex shared state conditions |
| **[[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Monitors\|Monitors]]** | Language-level constructs encapsulating shared data and routines with compiler-enforced implicit mutual exclusion and internal CVs. | Encapsulated Procedures + Implicit Locks + Internal CVs | Structured, language-supported thread synchronization |

---

# Related Modules

- [[Operating Systems/Concurrency & Synchronization/Synchronization Patterns/index|Synchronization Patterns Directory]]
- [[Operating Systems/Concurrency & Synchronization/Deadlocks/index|Deadlocks Directory]]
- [[Operating Systems/Concurrency & Synchronization/index|Concurrency & Synchronization Main Index]]