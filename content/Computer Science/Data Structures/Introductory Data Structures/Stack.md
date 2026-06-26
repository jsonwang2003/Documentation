---
tags:
  - Stack
---
> [!INFO] Definition 
> An [[Abstract Data Types (ADT)]] where the element that is the **last** to go in is the **first** to be removed. This is known as the **Last In, First Out (LIFO)** principle—similar to a physical stack of dishes.

### Operations

The Stack ADT is defined by three primary operations:

|Operation|Description|
|---|---|
|**`push(element)`**|Adds an element to the **top** of the Stack.|
|**`top()`**|Returns the value of the element at the **top** without removing it.|
|**`pop()`**|Removes the element at the **top** of the Stack.|

---
### Implementation via Deque

Just like the [[Queue]], a [[Deques]] serves as a perfect backing structure for a Stack. By restricting access to only **one end** of the Deque, we can implement all Stack features trivially.

#### C++ Implementation

```cpp
class Stack {
    private:
        Deque deque; // Backed by a Doubly-Linked List or Circular Array
    public:
        bool push(Data element) { return deque.addBack(element); }
        Data top() { return deque.peekBack(); }
        void pop() { deque.removeBack(); }
        int size() { return deque.size(); }
};
```

#### Python Implementation

```python
class Stack:
    def __init__(self):
        self.deque = Deque()
    def push(self, element):
        return self.deque.addBack(element)
    def top(self):
        return self.deque.peekBack()
    def pop(self):
        self.deque.removeBack()
    def __len__(self):
        return len(self.deque)
```

> [!WARNING] Implementation Detail 
> Note that `pop()` here is `void` (it only removes). While languages like Java combine removal and return into one step, C++ keeps them separate for architectural clarity.

---
### Visualization: The LIFO Process

![[Pasted image 20260104135638.png]]

In the example above:
1. **Push**: Elements are added to the top (A, then B, then C).
2. **State**: The Stack is currently [A,B,C], where C is at the top.
3. **Pop**: The operation removes C first, even though A was there the longest.
4. **Result**: The new top of the Stack becomes B.

---
### Analysis & Patterns

**STOP and Think:** Could we have used `addFront()`, `peekFront()`, and `removeFront()` instead?
- **Yes.** For a Stack, it does not matter which end of the backing structure we use, as long as **both insertion and removal happen at the same end**. This maintains the LIFO property.

---
### Applications
The Stack is a foundational tool for managing nested or recursive logic:
- **Expression Evaluation**: Keeping track of parentheses and operations in complex math (1+(2∗(3+4))).
- **Memory Management**: The "Function Call Stack" in programming languages tracks active subroutines.
- **Graph Exploration**: Powering the **Depth-First Search (DFS)** algorithm.
- **Undo Mechanisms**: Storing a history of actions where the most recent action is the first to be undone.