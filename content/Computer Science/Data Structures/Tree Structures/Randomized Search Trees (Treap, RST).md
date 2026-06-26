> [!ABSTRACT]
> 
> An RST is a Treap where priorities are randomly generated upon insertion. This clever use of randomness simulates a random insertion order, successfully achieving an average-case time complexity of $O(\log n)$ regardless of the actual order in which keys are provided.

---
## 1. The Treap (Tree + Heap)
A **Treap** is a binary tree where each node has two attributes: a **key** and a **priority**. It must satisfy two simultaneous properties:
1. **[[Binary Search Tree (BSTs)#1. Core Properties|BST Properties]]**: The tree is ordered by **keys** (Left < Node < Right).
2. **[[Heap#1. Heap Constraints|Heap Properties]]**: The tree is ordered by **priorities** (Parent priority > Child priority for a Max-Heap).

![[Pasted image 20260114111803.png]]

---
## 2. Fundamental Operations

### Find
Since a RST is a valid BST, the `find` algorithm is identical to a standard BST.
- **Complexity**: $O(h)$, where $h$ is the tree height.
- `find` algorithm found [[Binary Search Tree (BSTs)#`find(element)`|here]]
### Insertion
Insertion occurs in two distinct phases:
1. **BST Insertion**: Insert the node based solely on its **key** as a leaf.
2. **Heap Fix (Bubble Up)**: While the node’s priority is greater than its parent's, perform **[[#3. The Tool AVL Rotations|AVL Rotations]]** to move the node up without violating the BST property.

![[Pasted image 20260114112415.png]]

```C++
insert(key, priority):
    // Phase 1: Standard BST Insertion
    node = perform_BST_insertion(key, priority)

    // Phase 2: Bubble Up using Rotations
    while node != root and node.priority > node.parent.priority:
        if node == node.parent.leftChild:
            AVLRight(node.parent)
        else:
            AVLLeft(node.parent)
```

### Removal
1. **BST Removal**: Remove the node based on its key.
2. **Heap Fix (Trickle Down)**: If the successor moved into the position violates the Heap Property, use rotations to move it down until the property is restored.

---
## 3. The Tool: [[AVL Tree#3. Rebalancing AVL Rotations|AVL Rotations]]

Rotations are constant-time $O(1)$ operations that change the structure of the tree while preserving the relative sorted order of the keys.

![[Pasted image 20260114111911.png]]

| **Rotation**       | **Description**                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| **Right Rotation** | Moves a left child into the parent's position. Used when the left child has higher priority.   |
| **Left Rotation**  | Moves a right child into the parent's position. Used when the right child has higher priority. |

```pseudocode
AVLRight(b): // Perform a right AVL rotation on node b
    a = left child of b
    y = right child of a (or NULL if a does not have a right child)
    p = parent of b (or NULL if b does not have a parent)
    if p is not NULL and b is the right child of p:
        make a the right child of p
    otherwise, if p is not NULL and b is the left child of p:
        make a the left child of p
    make y the left child of b
    make b the right child of a

AVLLeft(a): // Perform a left AVL rotation on node a
    b = right child of a
    y = left child of b (or NULL if b does not have a left child)
    p = parent of a (or NULL if a does not have a parent)
    if p is not NULL and a is the right child of p:
        make b the right child of p
    otherwise, if p is not NULL and a is the left child of p:
        make b the left child of p
    make y the right child of a
    make a the left child of ﻿b
```

---
## 4. Why Use Randomness?
In a standard BST, if data is inserted in sorted order (1, 2, 3...), the tree becomes a linked list ($O(n)$). In an **RST**, we:
1. Take the user's **key**.
2. Generate a **random priority**.
3. Insert the pair into the Treap.

Because the priorities are random, the nodes "bubble up" into a topology that **mimics a tree built from a random insertion order**. This *guarantees* that the tree stays **balanced on average**, even if the keys themselves are sorted or patterned.

---
## 5. Performance Summary

|**Case**|**Time Complexity**|
|---|---|
|**Average Case**|$O(\log n)$|
|**Worst Case**|$O(n)$|

> [!NOTE]
> 
> While the RST effectively fixes the average case, the worst-case remains $O(n)$ (though it is statistically extremely unlikely). To guarantee $O(\log n)$ in the worst case, we must look toward structures like the **AVL Tree**.

