> [!ABSTRACT]
> 
> Invented by William Pugh in 1989, the Skip List uses **multiple layers of forward pointers** and **random number generation** to simulate a binary search over a [[Computer Science/Data Structures/Introductory Data Structures/Linked List|linked structure]]. It avoids the $O(n)$ insertion/removal penalty of sorted arrays while bypassing the $O(n)$ search penalty of standard linked lists.

---
## 1. Core Structure
A Skip List is a collection of sorted nodes arranged in **layers**.
- **Layer 0**: The bottom-most layer is a standard, sorted Linked List containing all elements.
- **Express Layers**: Each higher layer ($1, 2, \dots, h$) contains a subset of the nodes below it, acting as "express lanes" to skip over large sections of the list.
- **Head**: The head node contains an array of pointers, one for each level of the list.

---
## 2. Basic Operations
### Find
To find element $e$, start at the **highest layer** of the head node.
1. Move forward on the current layer until the next node is larger than $e$ or is `NULL`.
2. Drop down one layer.
3. Repeat until $e$ is found or you "fall off" the list at Layer 0.

```C++
find(element):
    current = head
    layer = head.height
    while layer >= 0:
        if current.key == element: return True
        if current.next[layer] == NULL or current.next[layer].key > element:
            layer = layer - 1 // Drop down
        else:
            current = current.next[layer] // Move forward
    return False
```

### Remove
1. Perform the `find` algorithm to locate the node.
2. Update the `next` pointers of the preceding nodes on every layer the target node occupied to point to the target's successors.

### Insert
1. Search for the insertion position.
2. Determine the **height** of the new node using a **coin-flip game**.
3. Update pointers to splice the new node into the appropriate layers.

---
## 3. The Probability Mechanics
The height of a new node is determined randomly to ensure a balanced distribution.

### The Coin-Flip Game
- Start at height 0.
- Flip a coin with probability $p$ of success (Heads).
- If Heads: Increase height by 1 and flip again.
- If Tails: Stop.

This process follows a **Geometric Distribution**: $P(X=k) = p^k(1-p)$, where $k$ is the number of successes before the first failure.
### "Smart Choices" for $p$
The performance of a Skip List depends on the probability $p$ and the maximum height.
- **Average Search Complexity**: $O(\log n)$.
- **Worst-Case Complexity**: $O(n)$ (if all nodes have height 0).
- **Optimal Height**: Usually defined as $\log_{1/p} n$.
- **Choosing $p$**: While $p=0.5$ is common (standard coin), smaller values of $p$ reduce the space overhead (fewer pointers) while slightly increasing the number of comparisons.

---
## 4. Performance Comparison

|**Feature**|**Sorted Array**|**Linked List**|**Skip List (Avg)**|
|---|---|---|---|
|**Search**|$O(\log n)$|$O(n)$|**$O(\log n)$**|
|**Insert**|$O(n)$|$O(1)^*$|**$O(\log n)$**|
|**Remove**|$O(n)$|$O(1)^*$|**$O(\log n)$**|
|**Space**|$O(n)$|$O(n)$|**$O(n)$**|

_*Note: Linked List insert/remove is $O(1)$ only if the position is already known; finding the position is $O(n)$._