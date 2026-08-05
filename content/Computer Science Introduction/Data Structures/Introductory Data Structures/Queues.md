---
description: "A linear restricted container enforcing the strict First In, First Out operational scheduling protocol."
aliases:
  - Queue
  - Queue ADT
  - FIFO Queue
tags:
  - data-structures
  - adt
  - queues
---
> [!abstract] Abstract 
> A Queue is an [[Abstract Data Types (ADT)|Abstract Data Type]] that strictly enforces the First In, First Out ($\text{FIFO}$) operational protocol. It mimics real-world waiting lines: the first element introduced into the data container is guaranteed to be the first element extracted from the structure.
> 
> - **Category:** Bounded Monitored ADT
> - **Core Workflow:** Elements enter at the trailing end and exit at the leading end.
> - **Common Structural Backbones:** [[Deques|Deques]], [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Doubly-Linked Lists]], or [[Circular Arrays|Circular Arrays]].

---

# Core Functional Interface

The Queue ADT contract provides three primary operations:

| Operation | Detailed Functional Execution |
|---|---|
| `enqueue(element)` | Appends a new element to the back of the Queue. |
| `peek()` | Evaluates and returns the item sitting at the front boundary without removing it. |
| `dequeue()` | Removes the item positioned at the front boundary of the Queue. |

---

# Structural Composition via Deques

Because a [[Deques|Deque]] interface natively supports data adjustments at both boundary margins, it serves as an excellent structural backbone for a Queue. Wrapping a Deque within a restricted interface implements Queue behavior with minimal code duplication.

### C++ Language Composition Map
```cpp
class Queue {
private:
    Deque deque; // Backed internally by a Doubly-Linked List or Circular Array
public:
    bool enqueue(Data element) { return deque.addBack(element); }
    Data peek() { return deque.peekFront(); }
    void dequeue() { deque.removeFront(); }
    int size() { return deque.size(); }
};
```

### Python Language Composition Map
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

> [!note] Interface Return Variances
> In this implementation pattern, `dequeue()` behaves as a void operation that modifies state without returning a value. While languages like Java combine removal and value-return into a single function call (such as `poll()`), architectures like C++ separate lookup (`front()`) and removal (`pop()`) into distinct steps for conceptual clarity.

---

# Sequential Processing Pipeline

![[Pasted image 20260104134953.png]]

### Interface Symmetry Challenge
Could a valid Queue be implemented by reversing boundary roles—using `addFront()` for insertion alongside `peekBack()` and `removeBack()` for removal?
*   **Answer:** Yes. As long as entry and exit points are kept on opposite margins of the underlying data structure, the structural $\text{FIFO}$ sequence is preserved.

---

# Core Architectural Applications

*   **Buffer Management:** Directs shared system pipelines, such as network packet routing queues, print spools, or customer support lines.
*   **OS Task Scheduling:** Sequences incoming CPU threads in their exact chronological order of arrival.
*   **Graph Exploration Traversals:** Serves as the foundational container that powers Breadth-First Search (BFS) algorithms to discover shortest paths across unweighted graphs.

---

# Related Notes

- [[Deques|Deques]]
- [[Priority Queue|Priority Queue]]
- [[Computer Science Introduction/Data Structures/Introductory Data Structures/Stack|Stack]]