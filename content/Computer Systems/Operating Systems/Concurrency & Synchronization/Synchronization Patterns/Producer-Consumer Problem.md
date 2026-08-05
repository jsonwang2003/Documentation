---
description: "Formulation and solution of the Bounded Buffer Producer-Consumer problem using semaphores and condition variables."
aliases:
  - Producer-Consumer Problem
  - Producer Consumer
  - Bounded Buffer Problem
  - Bounded Buffer
tags:
  - operating-systems
  - concurrency
  - synchronization
  - classical-problems
---
> [!abstract] Abstract
> The **Producer-Consumer Problem** (or **Bounded Buffer Problem**) is a classic synchronization challenge where one or more producer threads generate data items into a fixed-size buffer, and one or more consumer threads remove and process those items. Synchronization must prevent producers from overflowing the buffer, consumers from reading from an empty buffer, and race conditions on buffer pointers.
> 
> - **Category:** Classical Synchronization Problems
> - **Core Invariants:**
>   1. Consumer must wait when the buffer is empty ($count == 0$).
>   2. Producer must wait when the buffer is full ($count == N$).
>   3. Buffer operations must execute under Mutual Exclusion.

---

# 1. Problem Formulation

| Pipeline Stage | System Component     | Operational Role & State                                                                  |
| -------------- | -------------------- | ----------------------------------------------------------------------------------------- |
| **Input**      | **Producer Threads** | Generates data items and inserts them into the buffer                                     |
| **Storage**    | **Bounded Buffer**   | Fixed-capacity queue storing up to $N$ items <br>(`[ Item 1 \| item 2 \| ... \| item N]`) |
| **Output**     | **Consumer Threads** | Removes items from the buffer and processes them                                          |

*   **Producer:** Generates items and inserts them into the shared buffer.
*   **Consumer:** Removes items from the buffer and processes them.
*   **Bounded Buffer:** Fixed capacity array/queue of size $N$.
*   **Challenge:** Producers and consumers execute at independent, unpredictable rates without direct time serialization.

---

# 2. Why Naive & Basic Lock Solutions Fail

### Naive Unsynchronized Attempt (Broken)
Updating `count++` and `count--` concurrently without locks introduces **race conditions** that corrupt the buffer index pointers and item counts.

### Lock-Only Attempt (Broken)
Adding a lock prevents race conditions on buffer operations, but does not provide **event sequencing**. A producer inside the lock cannot know if the buffer is full without polling, and a consumer cannot know if the buffer is empty.

### Sleep/Wakeup Attempt (Lost Wakeup Flaw)
```c
// Flawed Producer Logic
if (count == N) sleep();
acquire(&lock);
insert_item(); count++;
release(&lock);
if (count == 1) wakeup(consumer);
```

> [!danger] The Lost Wakeup Race Condition
> If the consumer checks `if (count == 0)` and is preempted **right before calling `sleep()`**, the producer can execute, insert an item, see `count == 1`, and call `wakeup(consumer)` *before the consumer is actually asleep*. The wakeup signal is lost. The consumer then falls asleep, and eventually the producer fills the buffer and sleeps $\implies$ **Both threads sleep forever!**

---

# 3. Correct Solution 1: Using Semaphores (Hoare Semantics)

To solve the problem with semaphores, we enforce three distinct constraints using three semaphores:
1.  **`empty_count` (Counting Semaphore, Init = $N$):** Tracks available empty slots.
2.  **`full_count` (Counting Semaphore, Init = $0$):** Tracks available filled items.
3.  **`mutex` (Binary Semaphore, Init = $1$):** Enforces mutual exclusion on buffer operations.

```c
semaphore empty_count = N;
semaphore full_count = 0;
semaphore mutex = 1;

void producer() {
    while (1) {
        item_t item = produce_item();
        
        wait(&empty_count); // Decrement empty slots (blocks if full)
        wait(&mutex);       // Enter Critical Section
        
        insert_into_buffer(item);
        
        signal(&mutex);     // Exit Critical Section
        signal(&full_count); // Increment filled items count
    }
}

void consumer() {
    while (1) {
        wait(&full_count);  // Decrement filled items (blocks if empty)
        wait(&mutex);       // Enter Critical Section
        
        item_t item = remove_from_buffer();
        
        signal(&mutex);     // Exit Critical Section
        signal(&empty_count); // Increment empty slots count
        
        consume_item(item);
    }
}
```

---

# 4. Correct Solution 2: Using Condition Variables (Mesa Semantics)

When using Condition Variables, we pair a single **Mutex Lock** with two **Condition Variables**:
1.  **`not_full` (CV):** Signaled by consumers when buffer space becomes available.
2.  **`not_empty` (CV):** Signaled by producers when buffer items become available.

```c
lock_t lock;
cond_t not_full;
cond_t not_empty;
int count = 0;

void producer() {
    while (1) {
        item_t item = produce_item();
        
        acquire(&lock);
        while (count == N) {         // WHILE loop required for Mesa Semantics!
            wait(&not_full, &lock);  // Atomically releases lock and sleeps
        }
        
        insert_into_buffer(item);
        count++;
        
        if (count == 1) {
            signal(&not_empty);      // Wake up waiting consumers
        }
        release(&lock);
    }
}

void consumer() {
    while (1) {
        acquire(&lock);
        while (count == 0) {         // WHILE loop required for Mesa Semantics!
            wait(&not_empty, &lock); // Atomically releases lock and sleeps
        }
        
        item_t item = remove_from_buffer();
        count--;
        
        if (count == N - 1) {
            signal(&not_full);       // Wake up waiting producers
        }
        release(&lock);
        
        consume_item(item);
    }
}
```

---

# Related Notes

- [[Semaphores|Semaphores]]
- [[Condition Variables|Condition Variables]]
- [[Reader-Writer Problem|Reader-Writer Problem]]
- [[Locks|Locks]]