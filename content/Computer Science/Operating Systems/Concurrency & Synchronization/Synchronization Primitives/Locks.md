---
description: "The Lock ADT abstraction, implementation attempts from spinlocks to guarded blocking locks, hardware atomic primitives, and specialized lock designs."
aliases:
  - Locks
  - Lock Primitive
  - Spinlocks
  - Blocking Locks
  - Test-and-Set
  - Guarded Locks
tags:
  - operating-systems
  - concurrency
  - synchronization
  - locks
---
> [!abstract] Abstract
> A **Lock** is a fundamental synchronization object providing two paired operations: `acquire()` and `release()`. Between these calls, a thread holds the lock and executes inside a critical section. To implement locks safely, the lock's internal operations must be **atomic**. This evolution spans low-level hardware primitives (**disabling interrupts**, atomic **Test-And-Set** spinlocks) to efficient multicore **Guarded Blocking Locks** that sleep waiting threads.
> 
> - **Category:** Synchronization Primitives & Implementation
> - **Core Interface:** `acquire()` (or `lock()`) and `release()` (or `unlock()`).
> - **Hardware Primitives:** Disable Interrupts, Atomic `test_and_set()` instruction.
> - **Key Invariant:** Minimize busy-waiting while keeping interrupts enabled inside long critical sections.

---

# 1. The Lock Abstraction

A **Lock** is a memory object used to enforce mutual exclusion around critical sections.

```c
lock_t my_lock;

void worker() {
    acquire(&my_lock);  // Enter Critical Section (Blocks if held by another thread)
    
    /* --- CRITICAL SECTION (Shared State Access) --- */
    
    release(&my_lock);  // Exit Critical Section
}
```

### Operational Rules
*   **Pairing:** Every `acquire()` call must be strictly paired with a `release()` call.
*   **Mutual Exclusion:** Only one thread can hold a lock at any given time.
*   **Blocking Behavior:** `acquire()` will not return until any previous holder has called `release()`.
*   **Unpaired Failure Mode:** If a thread acquires a lock and fails to release it (or hangs), all subsequent threads trying to enter the critical section will wait indefinitely (starvation/deadlock).

---

# 2. Low-Level Implementation Attempts (Spinlocks)

### Attempt 1: Naive Software Lock (Broken)
A naive attempt to build a lock using a boolean flag in user memory:

```c
struct lock {
    bool held = false;
};

void acquire(struct lock* lock) {
    while (lock->held); // Busy-wait (spin) until held is false
    lock->held = true;  // Mark as held
}

void release(struct lock* lock) {
    lock->held = false;
}
```

> [!danger] Why Attempt 1 Fails
> A context switch can occur **after** the `while` loop evaluates to `false` but **before** `lock->held = true` executes. Two threads can both observe `held == false` and simultaneously enter the critical section. **The lock implementation itself contains a race condition!**

---

### Attempt 2: Disabling Interrupts (Kernel-Only)
Involuntary context switches on a single CPU core are triggered by hardware interrupts (e.g., timer ticks). We can attempt atomicity by disabling interrupts:

```c
struct lock {
    // No state needed
};

void acquire(struct lock* lock) {
    disable_interrupts(); // One hardware instruction
}

void release(struct lock* lock) {
    enable_interrupts();
}
```

#### Limitations of Disabling Interrupts
1.  **User-Space Inaccessible:** Disabling interrupts is a **privileged instruction**. Allowing user applications to disable interrupts would allow buggy or malicious code to seize the machine permanently.
2.  **Fails on Multicore Systems:** Disabling interrupts only prevents context switches on the *local CPU core*. Threads running on other physical cores can still access shared memory simultaneously.
3.  **Missing Events:** Disabling interrupts for extended periods can cause the OS to miss or delay crucial hardware I/O and timer events.

---

### Attempt 3: Hardware Spinlock (`test_and_set`)
To build multicore-safe locks, modern CPUs provide dedicated **atomic instructions** executed directly at the hardware bus/cache level.

The **Test-And-Set** instruction reads a memory location and sets it to `true` in a single, indivisible hardware cycle:

```c
// Hardware executes this entire routine ATOMICALLY
bool test_and_set(bool* flag) {
    bool old = *flag;
    *flag = true;
    return old;
}
```

Using `test_and_set()`, we can construct a correct, multicore-safe **Spinlock**:

```c
struct lock {
    bool held = false;
};

void acquire(struct lock* lock) {
    // Spin until test_and_set returns false
    while (test_and_set(&lock->held));
}

void release(struct lock* lock) {
    lock->held = false;
}
```

*   **Correctness:** Works on multicore CPUs and can be safely invoked in user space.
*   **The Busy-Waiting Deficit:** If Thread $A$ holds the lock and Thread $B$ attempts to acquire it, Thread $B$ loops continuously in the `while` check (**busy-waiting**). This wastes CPU cycles that could be used by other productive threads.

---

# 3. High-Level Implementation Attempts (Blocking Locks)

When critical sections are long (e.g., File I/O, database updates), spinlocks waste massive amounts of CPU cycles. High-level locks move waiting threads to a sleep queue.

| Primitive | Mechanism | Performance Characteristic | Best Use Case |
|---|---|---|---|
| **Spinlock** | Busy-wait loop (`while(test_and_set)`) | Wastes 100% CPU on waiting core | Very short critical sections (e.g., OS scheduler state) |
| **Blocking Lock** | Moves waiting TCB to queue & sleeps | Releases CPU to run other threads | Long critical sections (e.g., File I/O, Database writes) |

---

### Attempt 4: Blocking Lock with Interrupt Disabling (Flawed)
To prevent busy-waiting, waiting threads are moved to a sleep queue `Q`:

```c
struct lock {
    bool held = false;
    queue Q;
};

void acquire(struct lock* lock) {
    disable_interrupts();
    if (lock->held) {
        put_current_thread_on(lock->Q);
        block_current_thread(); // Yields CPU & switches thread
    }
    lock->held = true;
    enable_interrupts();
}

void release(struct lock* lock) {
    disable_interrupts();
    if (is_empty(lock->Q)) {
        lock->held = false;
    } else {
        move_waiting_thread_to_ready_queue(lock->Q);
    }
    enable_interrupts();
}
```

> [!danger] Drawbacks of Attempt 4
> 1. Fails on **multicore CPUs** because disabling interrupts on Core 0 does not prevent Core 1 from modifying `lock->held`.
> 2. Cannot be called by **user-level applications**.

---

### Attempt 5: Multicore Guarded Blocking Lock (Correct)
To combine multicore safety with thread sleeping, we introduce a short-duration atomic **`guard`** spinlock that protects access to the lock's internal data structures (`held` flag and wait queue `Q`):

```c
struct lock {
    bool held = false;
    bool guard = false; // Short-duration spinlock protecting internal lock state
    queue Q;
};

void acquire(struct lock* lock) {
    disable_interrupts();
    while (test_and_set(&lock->guard)); // Acquire guard spinlock
    
    if (lock->held) {
        put_current_thread_on(lock->Q);
        lock->guard = false;            // Release guard BEFORE sleeping!
        block_current_thread();         // Sleep (CPU context switches)
    } else {
        lock->held = true;
        lock->guard = false;            // Release guard
    }
    enable_interrupts();
}

void release(struct lock* lock) {
    disable_interrupts();
    while (test_and_set(&lock->guard)); // Acquire guard spinlock
    
    if (is_empty(lock->Q)) {
        lock->held = false;
    } else {
        move_waiting_thread_to_ready_queue(lock->Q);
    }
    
    lock->guard = false;                // Release guard
    enable_interrupts();
}
```

#### Key Architectural Benefits
*   **Multicore Safe:** Uses `test_and_set(&lock->guard)` so multiple cores cannot corrupt the lock queue.
*   **Minimal Spinning:** Threads spin on `guard` for only a few assembly instructions (to update the queue), **not** for the duration of the critical section.
*   **Interrupt-Enabled Critical Section:** Interrupts remain enabled while executing code inside the main critical section.

---

# 4. Specialized Lock Variants

Operating systems optimize locking overhead by employing specialized lock designs tailored to specific workload access patterns:

1.  **Reader-Writer Locks (RW Locks):** Allows multiple concurrent reader threads to execute inside the critical section simultaneously, but grants exclusive access to a single writer thread.
2.  **Read-Copy-Update (RCU):** Optimized for read-dominated data structures (e.g., Linux routing tables). Readers execute with zero lock overhead; writers copy the structure, update it, and swap pointers atomically.
3.  **Distributed Locks:** Designed to avoid cache line bouncing and bus contention across multi-socket NUMA systems.

---

# Related Notes

- [[Operating Systems/Concurrency & Synchronization/Critical Sections & Mutual Exclusion|Critical Sections & Mutual Exclusion]]
- [[Operating Systems/Concurrency & Synchronization/Race Conditions & Shared State|Race Conditions & Shared State]]
- [[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Semaphores|Semaphores]]
- [[Operating Systems/Concurrency & Synchronization/Synchronization Primitives/Condition Variables|Condition Variables]]