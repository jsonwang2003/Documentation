> [!ABSTRACT]
> 
> Named after Adelson-Velsky and Landis, the AVL Tree is a self-balancing BST that ensures a worst-case time complexity of $O(\log n)$ for search, insertion, and deletion. It achieves this by maintaining a strict balance property through structural rotations.

---
## 1. Core Properties

An AVL tree must satisfy the **Balance Condition**: For every node $u$, the height of its left and right subtrees can differ by at most **1**.
- **Balance Factor ($BF$):** $BF(u) = Height(RightSubtree) - Height(LeftSubtree)$
- **Validity:** A node is balanced if $BF(u) \in \{-1, 0, 1\}$.

| **Valid AVL Tree**                   | **Invalid AVL Tree**                 |
| ------------------------------------ | ------------------------------------ |
| ![[Pasted image 20260116111242.png]] | ![[Pasted image 20260116111257.png]] |

---
## 2. Mathematical Proof of Height
We can prove the worst-case height $h$ is $O(\log n)$ by finding the minimum number of nodes $N_h$ required to form an AVL tree of height $h$.
- **Recurrence:** $N_h = N_{h-1} + N_{h-2} + 1$
- **Base Cases:** $N_1 = 1$, $N_2 = 2$ (using a 1-based height definition).
- **Result:** This recurrence grows faster than the Fibonacci sequence. Since $N_h \approx \phi^h$ (where $\phi$ is the golden ratio), it follows that $h \approx \log_{\phi}(n)$, confirming $h = O(\log n)$.

---
## 3. Rebalancing: AVL Rotations
When an insertion or removal causes a node to have a $BF$ of $\pm 2$, rotations are used to restore balance.
### Single Rotations (The "Straight Line" Case)

Used when the imbalance is caused by an insertion into the "outside" child (Left-Left or Right-Right).

- **Right Rotation:** Fixes Left-Left imbalance.
- **Left Rotation:** Fixes Right-Right imbalance.

![[Pasted image 20260116111325.png]]

```
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
    make a the left child of b
```
### Double Rotations (The "Kink" Case)
Used when the imbalance is caused by an insertion into the "inside" child (Left-Right or Right-Left). A single rotation cannot fix a "kink" shape.

![[Pasted image 20260116120605.png]]

**Solution: Double Rotations**:
1. **First Rotation:** Rotate the child to transform the "kink" into a "straight line."
2. **Second Rotation:** Rotate the parent to restore overall balance.

![[Pasted image 20260116120703.png]]

```
DoubleAVLLeftKink(a): // Shape: <
    AVLLeft(a.leftChild)
    AVLRight(a)

DoubleAVLRightKink(a): // Shape: >
    AVLRight(a.rightChild)
    AVLLeft(a)
```

> [!Example]- Insertion Example Requiring Double Rotation
> Below is a more complex example in which we insert 10 into the following **AVL Tree**:
> 
> ![[Pasted image 20260116120820.png]]
> 
> Following the traditional **Binary Search Tree** insertion algorithm, we would get the following tree (note that we have updated balance factors as well):
> 
> ![[Pasted image 20260116120831.png]]
> 
> As you can see, the root node is now out of balance, and as a result, we need to fix its balance. However, as we highlighted via the bent red arrow above, this time, our issue is in a "kink" shape (not a "straight line" shape, as it was in the previous complex example). As a result, a single AVL rotation will no longer suffice, and we are forced to perform a double rotation. The first rotation will be a left rotation on node 5 in order to transform this "kink" shape into a "straight line" shape:
> 
> ![[Pasted image 20260116120846.png]]
> 
> Now, we have successfully transformed our problematic "kink" shape into the "straight line" shape we already know how to fix. Thus, we can perform a right rotation on the root to fix our tree:
> 
> ![[Pasted image 20260116120900.png]]

---
## 4. Basic Operations

BST operations in an AVL tree are enhanced with a "rebalancing" phase to ensure the $O(\log n)$ height is maintained.
### `find(element)`
- **Logic:** Identical to the standard **Binary Search Tree** find algorithm.
- **Process:** Compare the target to the current node; traverse left if smaller or right if larger.
- **Complexity:** Guaranteed $O(\log n)$ because the tree height is strictly controlled.
### `insert(element)`
- **BST Phase:** Perform a standard BST insertion to place the new node as a leaf.
- **Update Phase:** Traverse **upwards** from the new leaf toward the root.
- **Balance Phase:** At each ancestor, update the balance factor. If any node reaches a factor of $\pm 2$, perform the appropriate **Single** or **Double Rotation**.
- **Complexity:** $O(\log n)$ for the initial search plus $O(\log n)$ for the upward rebalancing pass.

![[Pasted image 20260116111623.png]]

> [!Example]- More Complex Insert
> Below is a more complex example in which we insert 20 into the following **AVL Tree**:
> 
> ![[Pasted image 20260116115439.png]]
> 
> Following the traditional **[[Binary Search Tree (BSTs)]]** insertion algorithm, we would get the following tree (note that we have updated balance factors as well):
> 
> ![[Pasted image 20260116115455.png]]
> 
>As you can see, the root node is now out of balance, and as a result, we need to fix its balance using an AVL rotation. Specifically, we need to rotate left at the root, meaning 10 will become the new root, 7 will become the right child of 5, and 5 will become the left child of 10:
> 
> ![[Pasted image 20260116115506.png]]
### `remove(element)`
- **BST Phase:** Perform standard BST removal (handling the 0, 1, or 2-child cases).
- **Update Phase:** Start at the parent of the physically removed node and traverse **upwards** to the root.
- **Balance Phase:** Check balance factors at every level. Unlike insertion (where one rotation fix is often enough), removal may require multiple rotations along the path to the root.
- **Complexity:** $O(\log n)$ for both the search and the potential multi-step rebalancing pass.

![[Pasted image 20260116111630.png]]

---
## 5. Performance Comparison

| **Feature**            | **Standard BST** | **AVL Tree**          |
| ---------------------- | ---------------- | --------------------- |
| **Average Search**     | $O(\log n)$      | $O(\log n)$           |
| **Worst Search**       | $O(n)$           | **$O(\log n)$**       |
| **Balancing Strategy** | None             | Height-based (Strict) |
| **Passes per Update**  | 1 (Down)         | 2 (Down and Up)       |

> [!NOTE]
> 
> While AVL trees provide the best lookup times due to their strict balance, they require **two** passes (one down to *find/insert*, one up to *rebalance*). The [[Red-Black Tree]] is often preferred in high-performance libraries because it can rebalance in a **single pass**.
