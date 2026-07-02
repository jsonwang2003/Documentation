## The Motivation: Beyond FIFO

Traditional **Queues** follow the **First In, First Out (FIFO)** principle. While this works for grocery stores, it fails in environments like Emergency Rooms where urgency varies.
- **The Problem**: A patient with a minor injury arrives first, but a patient with a life-threatening injury arrives later.
- **The Solution**: An ordering system based on **priority** rather than arrival time.
## The Priority Queue ADT

A **Priority Queue** is a **"Highest Priority In, First Out" (HPIFO)** data type. It is defined by three primary operations:

|**Function**|**Description**|
|---|---|
|**`insert(element)`**|Adds a new element to the collection.|
|**`peek()`**|Returns the element with the **highest priority** without removing it.|
|**`pop()`**|Removes the element with the **highest priority** from the collection.|

---
## Implementation Trade-offs
### Array (Hash Table)
While a Priority Queue can be backed by simple linear data structures, they often result in inefficient worst-case time complexities:
#### 1. Linked List / Array (Unsorted)
- **Insertion**: $O(1)$ (just add to the end).
- **Peek/Pop**: $O(n)$ (must scan the entire list to find the highest priority).

#### 2. Linked List / Array (Sorted)
- **Peek/Pop**: $O(1)$ (the highest priority is always at the front/back).
- **Insertion**: $O(n)$ (must scan to find the correct position to maintain order).
### The Optimal Solution: The Heap

To achieve a balance between insertion and removal speeds, developers use a specialized tree-based data structure called a **[[Heap]]**.

> [!TIP] Performance Goal
> 
> By using a Heap, we can achieve the following worst-case complexities:
> 
> - **Peek**: $O(1)$
>     
> - **Insert**: $O(\log n)$
>     
> - **Pop**: $O(\log n)$
>