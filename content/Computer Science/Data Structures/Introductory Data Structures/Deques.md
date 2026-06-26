---
tags:
  - Deques
---
> [!INFO] Definition 
> An [[Abstract Data Types (ADT)]] that allows for insertion and removal of elements from both the front and the back. It is essentially a generalized version of a linear list that supports bi-directional growth.
### Formal Set of Functions
A Deque is defined by its ability to perform the following six operations:

|Function|Description|
|---|---|
|`addFront(element)`|Inserts a new element at the beginning of the Deque.|
|`addBack(element)`|Inserts a new element at the end of the Deque.|
|`peekFront()`|Returns the value of the first element without removing it.|
|`peekBack()`|Returns the value of the last element without removing it.|
|`removeFront()`|Removes the first element from the Deque.|
|`removeBack()`|Removes the last element from the Deque.|

---
### Implementation Strategies
The functionality of a Deque can be supported by two primary data structures: **Doubly-Linked Lists** and **Circular Arrays**.
#### 1. Doubly-Linked List Method
By maintaining a global _head_ and _tail_ pointer, a Doubly-Linked List provides highly efficient end-manipulation.
- **Performance**: Guaranteed **O(1)** for all six core operations via pointer rearrangement.
- **Memory**: No wasted space; memory is allocated dynamically per node.
- **Trade-off**: Accessing elements in the middle of the list (if required) remains **O(n)**.
#### 2. Circular Array Method
Utilizes a backing array with wrapping `head` and `tail` indices.
- **Performance**: **O(1)** for `peek` and `remove`. `add` operations are **Amortized O(1)**, occasionally spiking to O(n) during a resize/copy event.
- **Memory**: May waste memory by pre-allocating large blocks to accommodate future growth.
- **Trade-off**: Provides **O(1)** random access to middle elements using modular arithmetic.

---
### Comparison of Backing Structures

|Feature|Doubly-Linked List|Circular Array|
|---|---|---|
|**End Operations**|Always O(1)|Amortized O(1)|
|**Random Access**|O(n)|O(1)|
|**Memory Overhead**|Pointer storage for every node|Unused allocated capacity|
|**Worst-Case Add**|O(1)|O(n) (during resize)|

---
### Adding and Removing Operations
In a Deque, operations occur at the boundaries of the list. The logic depends on whether you are interacting with the **Head** or the **Tail**.
- **Front Operations**:
    - `addFront`: The new element becomes the new "first" node/index.
    - `removeFront`: The second element in the sequence is promoted to the "first" position.
- **Back Operations**:
    - `addBack`: The new element is appended after the current "last" node/index.
    - `removeBack`: The second-to-last element becomes the new "last" position.

#### Example Deque Operations
![[Pasted image 20260104132805.png]]

---
### Complexity Breakdown

When implementing these operations using the two primary backbones, the performance remains highly efficient:

|Operation|Doubly-Linked List|Circular Array|
|---|---|---|
|**`addFront` / `addBack`**|**O(1)** (Pointer swap)|**Amortized O(1)** (Index shift/Resize)|
|**`removeFront` / `removeBack`**|**O(1)** (Pointer swap)|**O(1)** (Index shift)|
|**`peekFront` / `peekBack`**|**O(1)** (Direct pointer)|**O(1)** (Direct index)|

> [!NOTE] Implementation Detail 
> In a **Doubly-Linked List**, `removeBack` is O(1) because each node knows its predecessor. In a **Singly-Linked List**, `removeBack` would be O(n) because you would have to traverse the entire list to find the node _before_ the tail to update its pointer. This is why a Doubly-Linked List is the preferred backbone for a Deque.

---
## Summary

If the primary requirement is maintaining a list where you only interact with the ends, a **Linked List** is generally the superior implementation for a Deque. It provides strictly constant-time performance and efficient memory usage without the "spiky" latency of array resizing.

Beyond browser history, Deques serve as the foundation for two other critical Abstract Data Types.
