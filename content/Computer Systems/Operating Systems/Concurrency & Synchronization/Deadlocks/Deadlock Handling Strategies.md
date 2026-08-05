---
description: "Strategies for managing deadlocks: Ostrich algorithm, Deadlock Prevention, Deadlock Avoidance (Banker's Algorithm), and Detection & Recovery."
aliases:
  - Deadlock Handling
  - Deadlock Prevention
  - Deadlock Avoidance
  - Banker's Algorithm
  - Deadlock Detection and Recovery
tags:
  - operating-systems
  - concurrency
  - deadlocks
---
> [!abstract] Abstract
> Operating systems handle deadlocks using four main strategies: **Ignorance** (Ostrich algorithm), **Prevention** (invalidating one Coffman condition), **Avoidance** (dynamically tracking safe states via the Banker's Algorithm), and **Detection & Recovery** (allowing deadlocks but resolving them via thread termination or resource preemption).
> 
> - **Category:** OS Resource Management & System Recovery
> - **Key Production Strategy:** Resource Ordering for Deadlock Prevention.
> - **Theoretical Standard:** Dijkstra's Banker's Algorithm for Deadlock Avoidance.

---

# 1. Overview of Deadlock Handling Strategies

| Strategy | Operational Mechanism | Pros / Cons | Primary Production Use Case |
|---|---|---|---|
| **Ignorance** | Ignore the problem (**Ostrich Algorithm**) | Zero overhead; system crashes if deadlock occurs | General-purpose OS (Linux, Windows, macOS) when deadlocks are rare |
| **Prevention** | Design system rules to eliminate $\ge 1$ Coffman condition | Guarantees no deadlock; can reduce resource utilization | Lock Ordering in kernel device drivers |
| **Avoidance** | Dynamically evaluate state transitions (Banker's Algorithm) | Flexible resource allocation; requires knowing max needs in advance | Batch processing / Database systems with fixed resource profiles |
| **Detection & Recovery** | Run periodic cycle detection on RAGs; abort/preempt on cycle | High runtime overhead; requires rollbacks/thread kills | High-availability databases, transaction monitors |

---

# 2. Strategy 1: Deadlock Prevention

Deadlock Prevention forces the system to violate at least one of the four Coffman conditions:

### 1. Invalidate Mutual Exclusion
*   Make resources sharable (e.g., read-only files).
*   *Limitation:* Inherent to physical hardware (e.g., printers, write locks cannot be shared).

### 2. Invalidate Hold and Wait
*   Require threads to request and receive **all** required resources simultaneously before starting execution, or release current resources before requesting new ones.
*   *Limitation:* Low resource utilization; starvation for threads requiring many popular resources.

### 3. Invalidate No Preemption
*   If a thread holding resources requests another resource that cannot be immediately allocated, the OS forcibly preempts and releases all resources currently held by that thread.
*   *Limitation:* Only works for state-saveable resources (e.g., CPU registers, RAM memory pages); unworkable for locks or I/O writes.

### 4. Invalidate Circular Wait (Most Practical)
*   Impose a strict **global total ordering** on all system resources.
*   **Rule:** If a thread holds resource $R_i$, it can only request resource $R_j$ if $j > i$.

```c
// Example: Global Lock Ordering prevents circular wait
// Always acquire lock_1 BEFORE lock_2
void worker_thread() {
    acquire(&lock_1);
    acquire(&lock_2);
    /* ... Critical Section ... */
    release(&lock_2);
    release(&lock_1);
}
```

---

# 3. Strategy 2: Deadlock Avoidance & The Banker's Algorithm

Deadlock Avoidance permits resource requests only if allocating them leaves the system in a **Safe State**.

> [!important] Safe State vs. Unsafe State
> A state is **Safe** if there exists at least one execution sequence of threads that allows every thread to claim its maximum declared resources and complete without deadlocking. An **Unsafe State** is *not* a deadlock, but it *can lead* to a deadlock.
### Dijkstra's Banker's Algorithm
The **Banker's Algorithm** evaluates incoming resource requests against total available bank funds (resources). It rejects allocations that push the system into an unsafe state:

| Transaction State | Customer A (Max \$1000) | Customer B (Max \$1000) | Bank Balance Remaining | State Classification |
|---|---|---|---|---|
| **Initial State** | Borrowed \$250 | Borrowed \$500 | **\$250** | **SAFE** (Can satisfy A or B sequentially) |
| **Unsafe Request** | Requests \$750 | Holds \$500 | **-\$500** | **REJECTED** (Bank bankrupt / Unsafe) |
| **Safe Execution** | Retains \$250 | Customer B returns \$500 | **\$750** | **SAFE** (Customer A can now claim remaining) |

*   *Limitation of Avoidance:* Requires processes to declare their maximum resource demands in advance, which is impossible for most interactive software.

---

# 4. Strategy 3: Deadlock Detection and Recovery

Allow deadlocks to occur, periodically execute a cycle-detection algorithm on the Resource Allocation Graph (RAG), and recover when cycles are found.

![[Pasted image 20260723150717.png]]

### Deadlock Recovery Options
1.  **Process / Thread Abort:**
    *   *Abort all deadlocked processes:* Clean, but loses massive computation work.
    *   *Abort one process at a time:* Rerun cycle detection after killing each process until the cycle breaks.
2.  **Resource Preemption & Rollback:**
    *   Forcibly seize a resource from a thread in the cycle.
    *   Roll back the victim thread to a previously saved checkpoint state and restart it.

---

# 5. Checkpoint Case Study: Readers-Writers Flaw

Recall the Readers-Writers semaphore solution:

![[Pasted image 20260723150015.png]]

> [!question] Deadlock Bug Scenario
> What happens if an engineer removes `if (read_count == 1)` so that **every reader** calls `wait(block_write)` directly?

```c
// BROKEN READER ROUTINE
void read() {
    wait(&mutex);
    read_count++;
    wait(&block_write); // BUG: Executed by EVERY reader!
    signal(&mutex);
    
    /* Read Operation */
    
    wait(&mutex);
    read_count--;
    signal(&block_write);
    signal(&mutex);
}
```

*   **Result:** The first reader acquires `mutex` and `block_write`. When a second reader arrives, it acquires `mutex` and blocks on `block_write`. Because the second reader holds `mutex`, the first reader can never re-acquire `mutex` during its exit sequence to decrement `read_count` and call `signal(block_write)` $\implies$ **Instant Deadlock**.

---

# Related Notes

- [[Deadlock Fundamentals & Coffman Conditions|Deadlock Fundamentals & Coffman Conditions]]
- [[Reader-Writer Problem|Reader-Writer Problem]]
- [[Locks|Locks]]