---
description: "Formal deadlock definitions, the Dining Philosophers problem, the four Coffman conditions, and Resource Allocation Graphs (RAGs)."
aliases:
  - Deadlock Fundamentals
  - Deadlock Conditions
  - Coffman Conditions
  - Resource Allocation Graph
  - RAG
  - Dining Philosophers
tags:
  - operating-systems
  - concurrency
  - deadlocks
---
> [!abstract] Abstract
> A **Deadlock** is an execution state where a set of threads is permanently stalled because every thread is waiting for a resource held by another thread in the set. Deadlocks can exist **if and only if** four necessary and sufficient hardware/software conditions (the **Coffman Conditions**) hold simultaneously: Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.
> 
> - **Category:** Concurrency Failures & Formal Modeling
> - **Classic Analogy:** The Dining Philosophers Problem (Dijkstra, 1971).
> - **Modeling Primitive:** Resource Allocation Graph (RAG).

---

# 1. The Dining Philosophers Problem

Introduced by Edsger Dijkstra in 1971, the **Dining Philosophers Problem** illustrates how competing for limited, shared resources leads to deadlock.

![[Pasted image 20260727011212.png]]

*   Five philosophers sit around a table with 5 forks.
*   Each philosopher alternates between **Thinking** and **Eating**.
*   To eat, a philosopher must acquire **two adjacent forks** (one at a time).

> [!danger] The Deadlock Scenario
> If all 5 philosophers become hungry simultaneously and every philosopher picks up their **right fork** at the exact same instant, all 5 forks are held. When each philosopher attempts to pick up their **left fork**, they wait forever $\implies$ **Deadlock**.

---

# 2. Formal Deadlock Definition

> **Deadlock Definition:** Deadlock exists among a set of threads if **every thread in the set is waiting for an event that can be caused only by another thread in that set.**

Deadlock causes permanent execution starvation, requiring external intervention (such as process termination or system reboot) to resolve.

---

# 3. The Four Coffman Conditions

Deadlock can arise **if and only if** the following four conditions hold simultaneously within the system:

1.  **Mutual Exclusion:** At least one resource must be held in a non-sharable mode (only one thread can use the resource at a time).
2.  **Hold and Wait:** A thread holding at least one resource can request additional resources currently being held by other threads without releasing its existing resources.
3.  **No Preemption:** Resources cannot be forcibly taken away from a thread; they can only be released voluntarily by the thread after it has completed its task.
4.  **Circular Wait:** A closed chain of threads exists ($\{T_0, T_1, \dots, T_n\}$) such that $T_0$ waits for a resource held by $T_1$, $T_1$ waits for $T_2$, and $T_n$ waits for $T_0$.

> [!tip] Breaking Deadlock
> Eliminating **at least one** of these four conditions guarantees that deadlock cannot occur.

---

# 4. Resource Allocation Graphs (RAG)

Deadlocks can be represented visually using a directed graph called a **Resource Allocation Graph (RAG)**:

*   **Nodes:** 
    *   **Threads ($T$):** Represented as circles.
    *   **Resources ($R$):** Represented as squares (dots inside represent resource unit counts).
*   **Edges:**
    *   **Assignment Edge ($R_i \to T_j$):** Resource $R_i$ is currently held by Thread $T_j$.
    *   **Request Edge ($T_j \to R_i$):** Thread $T_j$ is currently blocked waiting for Resource $R_i$.

![[Pasted image 20260722160552.png]] *(Thread A holds Resource R)*
![[Pasted image 20260722160613.png]] *(Thread B requests Resource S)*
![[Pasted image 20260722160714.png]] *(Thread 1 and Thread 2 request each other's locks)*

### Cycle Analysis Rules
| Resource Instances per Type | Graph Cycle Status | Deadlock Status |
|---|---|---|
| **Single-Unit Resources** (1 instance per type) | Cycle Detected | **DEADLOCK EXISTS** |
| **Single-Unit Resources** | No Cycle | No Deadlock |
| **Multi-Unit Resources** ($N > 1$ instances) | Cycle Detected | **DEADLOCK MAY EXIST** (Requires matrix reduction) |

---

# Related Notes

- [[Operating Systems/Concurrency & Synchronization/Deadlocks/Deadlock Handling Strategies|Deadlock Handling Strategies]]
- [[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Locks|Locks]]
- [[Operating Systems/Concurrency & Synchronization/Critical Sections & Mutual Exclusion|Critical Sections & Mutual Exclusion]]