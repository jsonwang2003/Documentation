---
description: "Analysis of non-deterministic execution, race conditions, shared vs thread-private memory spaces, and atomicity assumptions."
aliases:
  - Race Conditions
  - Race Condition
  - Shared State
  - Non-Deterministic Execution
  - Thread Interleaving
tags:
  - operating-systems
  - concurrency
  - threads
  - race-conditions
---
> [!abstract] Abstract
> While single-threaded programs execute deterministically, multi-threaded programs running on shared state are subject to arbitrary instruction interleavings driven by the OS scheduler. A **Race Condition** occurs when the final outcome of execution depends on the exact order or timing of thread instruction execution, leading to silent data corruption and non-deterministic bugs.
> 
> - **Category:** Concurrency Pitfalls & Memory Architecture
> - **Primary Trigger:** Unsynchronized concurrent reads/writes to shared memory.
> - **Key Invariant:** Local thread stacks are private, but global variables, static objects, and heap allocations are shared across all threads.

---

# The Problem with Concurrency

In a single-threaded program, execution is strictly deterministic: given the same input, instructions execute sequentially, producing identical results every time.

In a multithreaded program, thread execution is **interleaved arbitrarily** at runtime based on timer interrupts and CPU scheduling decisions.

### Case Study: Unsynchronized Bank Withdrawals
Consider two concurrent threads attempting to withdraw \$100 simultaneously from a shared bank account holding \$500:

```c
void withdraw(Account* account, int amount) {
    int balance = get_balance(account);
    balance = balance - amount;
    put_balance(account, balance);
    return balance;
}
```

If the CPU context switches between Thread 1 and Thread 2 midway through the function:

![[Pasted image 20260716145640.png]]

*   **Expected Result:** $\$300$ remaining ($\$500 - \$100 - \$100$).
*   **Actual Result:** **\$400 remaining**. Thread 1 overwrote Thread 2's update because its execution was interleaved on an stale read.

---

# Race Conditions Defined

A **Race Condition** is a flaw in a concurrent system where the output is wildly sensitive to the relative timing or execution order of threads.

*   **Non-Determinism:** The exact same program with identical inputs can produce correct results on one run and corrupt memory on the next, depending on microsecond timing variations.
*   **Arbitrary Interleaving:** Programmers do not control when the kernel timer interrupt fires to trigger a context switch. Interleaving can occur at the individual machine instruction level.

---

# Which Memory Resources Are Shared?

To identify race conditions, engineers must distinguish **thread-private** memory from **thread-shared** memory:

![[Pasted image 20260714095236.png]]

| Memory Region | Shared Across Threads? | Safety Profile |
|---|---|---|
| **Local Stack Variables** | **No (Thread-Private)** | Safe. Each thread has its own SP and stack frame. *Never pass/store pointers to stack variables across threads!* |
| **Global & Static Variables** | **Yes (Thread-Shared)** | **Unsafe.** Reside in the shared Data/BSS segment; accessible by any thread. |
| **Heap Memory (`malloc` / `new`)** | **Yes (Thread-Shared)** | **Unsafe.** Dynamic objects are reachable by any thread holding a pointer. |
| **OS Resources (Files, Sockets)** | **Yes (Thread-Shared)** | **Unsafe.** Open descriptors and buffers are shared across all process threads. |

---

# Hardware Atomicity & Interleaving Assumptions

When reasoning about concurrency problems, system developers make three fundamental assumptions:

1.  **Instruction Granularity:** High-level code statements (e.g., `x++` or `balance = balance - amount`) are **not atomic**. They break down into multiple assembly instructions (`LOAD`, `SUB`, `STORE`). Context switches can happen *between* these assembly instructions.
2.  **Arbitrary Context Switches:** A context switch can occur at any instruction boundary, driven by hardware timer interrupts or preemption.
3.  **Arbitrary Execution Delays:** A thread can be delayed indefinitely (due to scheduling priorities or page faults), provided it is not stalled forever.

> [!important] Atomic Operation Definition
> An operation is **Atomic** if it executes completely as an indivisible unit or not at all—it cannot be paused, read, or modified by another thread mid-execution.

---

# Related Notes

- [[Critical Sections & Mutual Exclusion|Critical Sections & Mutual Exclusion]]
- [[Thread Abstraction & TCB|Thread Abstraction & TCB]]
- [[Thread Context Switch & Scheduling|Thread Context Switch & Scheduling]]