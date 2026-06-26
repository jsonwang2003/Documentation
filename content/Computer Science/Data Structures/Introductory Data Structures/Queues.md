---
tags:
  - Queue
---
> [!INFO] Definition 
> An **Abstract Data Type (ADT)** where the element that is the **first** to go in is the **first** to be removed. This is known as the **First In, First Out (FIFO)** principle.

### Operations
The Queue ADT is defined by three primary operations:

|Operation|Description|
|---|---|
|`enqueue(element)`|Adds an element to the **back** of the Queue.|
|`peek()`|Returns the value of the element at the **front** without removing it.|
|`dequeue()`|Removes the element at the **front** of the Queue.|

---
### Implementation via Deque
Since a [[Deques]] allows adding/removing from both ends, it serves as an ideal backing structure for a Queue. By "restricting" the Deque's functionality, we can implement Queue operations with minimal code.
#### C++ Implementation

```cpp
class Queue {
    private:
        Deque deque; // Backed by a Doubly-Linked List or Circular Array
    public:
        bool enqueue(Data element) { return deque.addBack(element); }
        Data peek() { return deque.peekFront(); }
        void dequeue() { deque.removeFront(); }
        int size() { return deque.size(); }
};
```

#### Python Implementation

```python
class Queue:
    def __init__(self):
        self.deque = Deque()
    def enqueue(self, element):
        return self.deque.addBack(element)
    def peek(self):
        return self.deque.peekFront()
    def dequeue(self):
        self.deque.removeFront()
    def __len__(self):
        return len(self.deque)
```

> [!WARNING] Implementation Detail 
> Note that `dequeue()` here is `void` (it only removes). In some languages like Java, `dequeue` (or `poll`) removes **and** returns the value. In others like C++, the removal and the access (`front()`) are separate steps.

---
### Visualization: The FIFO Process

![[Pasted image 20260104134953.png]]

In the example above:
1. **Enqueue**: Elements A,B,C are added to the back in order.
2. **State**: The Queue is currently [A,B,C], where A is at the front.
3. **Dequeue**: The operation removes A because it was the first to enter.
4. **Result**: The new front of the Queue becomes B.

---
### Analysis & Patterns
**STOP and Think:** Could we have used `addFront()`, `peekBack()`, and `removeBack()` instead?
- **Yes.** As long as the entry point and the exit point are on **opposite ends**, the FIFO property is maintained. Using one end for insertion and the other for removal is the only requirement.

---
### Applications
Despite its simplicity, the Queue is essential for:
- **Buffer Management**: Organizing people at a cash register or print jobs in a printer spooler.
- **Scheduling**: Keeping track of CPU tasks that need to run in the order they arrived.
- **Graph Exploration**: Powering the **Breadth-First Search (BFS)** algorithm to find the shortest path in unweighted graphs.