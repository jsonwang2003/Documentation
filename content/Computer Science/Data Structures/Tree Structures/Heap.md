> [!INFO] Definition
> A **Heap** is a specialized tree-based data structure that satisfies specific structural and ordering properties. It is the primary data structure used to implement the **[[Priority Queue]]** ADT.

## 1. Heap Constraints

A **Binary Heap** must satisfy three core properties:
- **Binary Tree Property**: Every node has at most 2 children ($0, 1, \text{ or } 2$).
- **Heap Property**: For any two nodes $A$ and $B$, if $A$ is the parent of $B$, then $A$ has higher (or equal) priority than $B$.
- **Shape Property**: The heap is a **complete tree**. All levels are fully filled except possibly the last level, which must be filled from **left to right**.

---
## 2. Min-Heaps vs. Max-Heaps

Priority is determined by the value of the elements. Because priority defines the relationship, **duplicate values are allowed**; ties are broken arbitrarily.

|**Feature**|**Min-Heap**|**Max-Heap**|
|---|---|---|
|**Ordering**|Parent $\leq$ Children|Parent $\geq$ Children|
|**Root Node**|Minimum value (Highest Priority)|Maximum value (Highest Priority)|
|**Priority Logic**|Smaller values have higher priority|Larger values have higher priority|

---
## 3. Core Operations

### Peek — $O(1)$
Returns the highest-priority element. Due to the Heap Property, this is always the **root**.

### Push (Insertion) — $O(\log n)$
To add an element while maintaining the Shape and Heap properties:
1. Place the new element in the **next open slot** (maintaining the complete tree shape).
2. **Bubble Up**: Compare the element with its parent. If it has higher priority, swap them. Repeat until the Heap Property is restored or the element becomes the root.

![[Pasted image 20260112154506.png]]
### Pop (Removal) — $O(\log n)$
To remove the highest-priority element:
1. Remove the **root** element.
2. Replace the root with the **last element** (rightmost element of the bottom level).
3. **Trickle Down**: Compare the new root with its children. Swap it with its **highest-priority child**. Repeat until the element is at a valid position or becomes a leaf.

![[Pasted image 20260112154458.png]]

> [!NOTE]
> 
> Why swap with the highest-priority child? > When trickling down, you must swap with the highest-priority child to ensure that the new parent of that subtree remains higher in priority than both of its resulting children.

---
## 4. Array Implementation

Because heaps are **complete trees**, they can be stored contiguously in an array without wasted space or the need for pointers.

For an element at **index $i$** (using 0-based indexing):
- **Parent**: $\lfloor \frac{i-1}{2} \rfloor$
- **Left Child**: $2i + 1$
- **Right Child**: $2i + 2$
- **Next Open Slot**: Index $n$ (where $n$ is the current number of elements).

![[Pasted image 20260112154018.png]]

---
## 5. Complexity Summary

| **Operation**        | **Time Complexity (Worst Case)** |
| -------------------- | -------------------------------- |
| **Peek**             | $O(1)$                           |
| **Insert (Push)**    | $O(\log n)$                      |
| **Remove (Pop)**     | $O(\log n)$                      |
| **Space Complexity** | $O(n)$                           |

