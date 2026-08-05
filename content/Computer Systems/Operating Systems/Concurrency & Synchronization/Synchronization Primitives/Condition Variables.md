---
description: "Condition Variable synchronization primitives, Mesa vs Hoare signal semantics, atomic lock-release sleeping, and common programming pitfalls."
aliases:
  - Condition Variables
  - Condition Variable
  - Mesa Semantics
  - Hoare Semantics
  - CV
  - wait and signal
tags:
  - operating-systems
  - concurrency
  - synchronization
  - condition-variables
---
> [!abstract] Abstract
> A **Condition Variable (CV)** is a memoryless synchronization primitive that enables a thread to sleep inside a critical section by **atomically releasing an associated lock and placing the thread on a wait queue**. Used in conjunction with a lock, CVs allow threads to wait for complex state conditions to become true without holding the lock while sleeping.
> 
> - **Category:** High-Level Synchronization Mechanics
> - **Core Operations:** `wait(cv, lock)`, `signal(cv)`, `broadcast(cv)`.
> - **Key Invariant:** Memoryless (signals delivered with no waiters present are lost).
> - **Dominant Standard:** **Mesa Semantics** (requires checking conditions inside a `while` loop).

---
# 1. Why Condition Variables?

When a thread enters a critical section protected by a lock but discovers that a required resource condition is not met (e.g., the input buffer is empty), it must wait.

*   **Why not keep holding the lock and sleep?** If the thread sleeps while holding the lock, no other thread can enter the critical section to produce the missing resource $\implies$ **Deadlock**.
*   **The CV Solution:** `wait(cv, lock)` **atomically** releases the lock and puts the calling thread to sleep in a single indivisible step.

---
# 2. Condition Variable API & Operations

A Condition Variable is always associated with an explicit **Mutex Lock**:

*   **`wait(cv, lock)`:** Atomically releases `lock`, puts the calling thread to sleep on `cv`'s queue, and blocks. When awakened, it re-acquires `lock` before returning.
*   **`signal(cv)`:** Awakens one thread waiting on `cv`'s queue (if any). If no threads are waiting, the signal is discarded (**memoryless**).
*   **`broadcast(cv)`:** Awakens *all* threads currently waiting on `cv`'s queue.

> [!warning] Memoryless Property vs. Semaphores
> Unlike semaphores, Condition Variables have no history counter. Calling `signal()` when no threads are currently waiting does nothing; the signal is permanently lost.

---
# 3. Signaling Semantics: Mesa vs. Hoare

What happens immediately after Thread $A$ calls `signal(cv)` to wake Thread $B$?
### 1. Mesa Semantics (Production Standard)
*   The signaler thread retains the lock and continues executing.
*   The woken thread is moved to the **Ready Queue**.
*   **Impact:** By the time the woken thread actually re-acquires the lock and runs, another thread may have entered the critical section and altered the condition!
*   **Mandatory Code Pattern:** Must re-evaluate condition using a **`while` loop**:

```c
// Correct for Mesa Semantics
acquire(&lock);
while (condition_is_not_met) { // RE-CHECK CONDITION!
    wait(&cond_var, &lock);
}
/* Execute Critical Section */
release(&lock);
```

![[Pasted image 20260727011547.png]]
### 2. Hoare Semantics
*   The signaler thread immediately yields the lock to the woken thread, which runs instantly.
*   **Impact:** The condition is guaranteed to hold, so an `if` statement suffices. However, implementation complexity is extremely high.

![[Pasted image 20260727011611.png]]

---

# 4. Common Pitfalls when Using Condition Variables

### Pitfall 1: Checking CVs Without a Separate Flag
CVs hold no state. You cannot test if a CV is "true". You must maintain a separate shared state variable (e.g., `count` or `flag`) guarded by the lock.

### Pitfall 2: Releasing the Lock Before Calling `wait()`
Releasing the lock *before* calling `wait()` opens a race window where another thread can modify the condition and issue a `signal()` before the waiting thread actually goes to sleep $\implies$ **Lost Wakeup Bug**.

### Pitfall 3: Using `if` Instead of `while` Under Mesa Semantics
Using `if (condition)` under Mesa semantics allows "spurious wakeups" or race conditions where a woken thread acts on an invalid condition. **Always use a `while` loop around `wait()`.**

---

# Related Notes

- [[Producer-Consumer Problem|Producer-Consumer Problem]]
- [[Semaphores|Semaphores]]
- [[Locks|Locks]]