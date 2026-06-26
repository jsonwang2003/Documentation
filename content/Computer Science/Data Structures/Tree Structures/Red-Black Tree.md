> [!Abstract]
> Derived from the 2-4 Tree, the Red-Black Tree is a self-balancing BST that achieves a worst-case $O(\log n)$ time complexity. Unlike the AVL Tree, which requires two passes (down and up) for updates, the Red-Black Tree is designed to maintain balance in a single pass through the tree.

---
## 1. The Four Properties

For a BST to be a valid Red-Black Tree, it must satisfy these strict coloring and structural rules:

1. **Node Color**: Every node is either **Red** or **Black**.
2. **Root Property**: The root of the tree is always **Black**.
3. **Red Property**: Red nodes cannot have red children (no two red nodes in a row).
4. **Black Property**: For any node, every path from that node to a null reference (leaf) must contain the **same number of black nodes**.

> [!NOTE]
> 
> Null References (empty children) are conceptually treated as Black nodes.

---
## 2. Mathematical Height Guarantee

We can prove that the height $h$ of a Red-Black Tree with $n$ internal nodes is $O(\log n)$.
- **Black Height $bh(x)$**: The number of black nodes from $x$ to a leaf (excluding $x$).
- **Internal Nodes**: A subtree rooted at $x$ has at least $2^{bh(x)} - 1$ internal nodes.
- **Height Relation**: Since red nodes cannot be adjacent, at least half the nodes on any path (excluding the root) must be black. Thus, $bh(root) \geq h/2$.
- **Conclusion**: $n \geq 2^{h/2} - 1 \implies h \leq 2\log(n+1)$. The height is at most roughly twice the optimal BST height.

---
## 3. Insertion Algorithm: The Single-Pass Method

Nodes are always inserted as **Red** by default to preserve the Black Property (Property 4). We then fix any violations of the Red Property (Property 3) as we traverse.
### Case 1: The Root
If the tree is empty, the new node becomes the root. Color it **Black**.

![[Pasted image 20260121113137.png]]
### Case 2: Black Node with Two Red Children (Color Flip)

During your descent, if you encounter a black node with **two red children**:
1. Recolor the parent **Red**.
2. Recolor both children **Black**.
3. If this creates a red-red violation with the grandparent, fix it using rotations (Cases 3 or 4).

![[Pasted image 20260121113248.png]]
### Case 3: Red-Red Violation (Straight Line)

If the new red node and its red parent form a straight line (e.g., both are left children):
1. Perform a [[AVL Tree#Single Rotations (The "Straight Line" Case)|Single AVL Rotation]].
2. **Recolor**: Set the new "top" node of the rotation to Black and its children to Red.

![[Pasted image 20260121113342.png]]
### Case 4: Red-Red Violation (Kink)
If the nodes form a "kink" shape:
1. Perform a [[AVL Tree#Single Rotations (The "Straight Line" Case)|Single Rotation]] to transform the kink into a **Straight Line**.
2. Apply the [[#Case 3 Red-Red Violation (Straight Line)|Case 3 Fix]].

![[Pasted image 20260121113359.png]]

---
## 4. Performance Trade-offs: AVL vs. Red-Black

| **Feature**       | **AVL Tree**                       | **Red-Black Tree**                    |
| ----------------- | ---------------------------------- | ------------------------------------- |
| **Balance**       | Stricter (Factor of $\pm 1$)       | Looser (Black height equality)        |
| **Height**        | $\approx 1.44 \log n$              | $\approx 2 \log n$                    |
| **Find**          | **Faster** (shorter paths)         | Slower                                |
| **Insert/Remove** | Slower (2 passes + more rotations) | **Faster** (1 pass + fewer rotations) |
| **Use Case**      | Read-heavy datasets                | Write-heavy/General purpose           |

> [!TIP]
> 
> This is why Red-Black Trees are used for standard library implementations like std::map and std::set in C++.

---
## 5. Summary of BST Evolution

| **Data Structure** | **Best Case** | **Worst Case** | **Balance Strategy**       |
| ------------------ | ------------- | -------------- | -------------------------- |
| **Basic BST**      | $O(\log n)$   | $O(n)$         | None                       |
| **Randomized BST** | $O(\log n)$   | $O(n)$         | Random Priorities (Treaps) |
| **AVL Tree**       | $O(\log n)$   | $O(\log n)$    | Height-based rotations     |
| **Red-Black Tree** | $O(\log n)$   | $O(\log n)$    | Single-pass color rules    |
