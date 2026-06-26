---
title: Queue
tags:
  - Queue
---
> [!INFO]  
> A linear data structure that follows the **First-In, First-Out (FIFO)** principle. Elements are added at the rear and removed from the front. Queues are widely used in scheduling, buffering, and breadth-first traversal algorithms.

### [[Queues|Queues]]
## Properties

- FIFO ordering: the first element added is the first to be removed  
- Supports dynamic or fixed-size implementations  
- Can be implemented using arrays, linked lists, or built-in language structures  
- Often used in operating systems, simulations, and asynchronous processing

## Common Operations

- **Enqueue**: Add an element to the rear (`queue.append(x)`)  
- **Dequeue**: Remove and return the front element (`queue.pop(0)`)  
- **Peek**: View the front element without removing it  
- **isEmpty**: Check if the queue is empty  
- **Size**: Return the number of elements in the queue

## Variants

- **Circular Queue**: Wraps around to reuse space  
- **Priority Queue**: Elements are dequeued based on priority, not order  
- **Double-Ended Queue (Deque)**: Supports insertion/removal from both ends  
- **Message Queue**: Used in distributed systems for asynchronous communication