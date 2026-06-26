> [!ABSTRACT]
> 
> In complex social networks or connectivity problems—like CIA agent Sarah Walker tracking connections between Chuck Bartowski and known villains—standard graph traversals (BFS/DFS) can be too slow for frequent queries. The **Disjoint Set** ADT provides a way to merge groups and check connectivity in **near-constant time**.

---
## 1. The Core Operations: Union and Find

A Disjoint Set manages a collection of elements partitioned into non-overlapping (disjoint) subsets. It supports two primary operations:
- **Find($u$):** Determine which set $u$ belongs to. If $Find(u) == Find(v)$, then $u$ and $v$ are connected.
- **Union($u, v$):** Merge the set containing $u$ with the set containing $v$ into a single set.

---
## 2. Implementation: The Up-Tree

The most efficient way to represent these sets is using an **Up-Tree**. Unlike a standard tree where parents point to children, in an Up-Tree, **children point to their parents**.
- **Sentinel Nodes (Roots):** The "representative" or "name" of the set. A node with no parent (or a self-pointer) is the root.
- **Array Representation:** We can store the entire forest in a single array.
    - `Array[i]` stores the index of the parent of $i$.
    - If `Array[i] == -1` (or a negative number), node $i$ is a sentinel node.

> [!Example]
> ![[Pasted image 20260301120131.png]]
> 
> ![[Pasted image 20260301120136.png]]

---
## 3. Optimizing the Union Operation

To keep the trees from becoming too tall (which makes `Find` slow), we use "smart" union strategies:

### Union-by-Size

Always attach the root of the **smaller** tree (fewer nodes) to the root of the **larger** tree.

> [!Example]- $Union(F, E)$
> ![[Pasted image 20260301120257.png]]
> ![[Pasted image 20260301120354.png]]

- **Worst-case Height:** $O(\log n)$.
- **Benefit:** Easy to track; just store the negative size in the sentinel's array slot (e.g., `-5` means it's a root of a set with 5 nodes).
### Union-by-Height (Rank)

Always attach the **shorter** tree to the **taller** tree.

> [!Example]- $Union(A, C)$
> ![[Pasted image 20260301120510.png]]
> 
> ![[Pasted image 20260301120517.png]]

- **Worst-case Height:** $O(\log n)$.
- **Drawback:** Harder to maintain if you also use Path Compression (see below), as heights change during searches.

> [!Question]- If we used **Union-by-Size** instead of **Union-by-Height** on the example above, would the resulting tree be better, worse, or just as good as the one produced by the **Union-by-Height** method?
> In the provided example, **Union-by-Size** would likely produce a tree with the same or better performance than **Union-by-Height**, but in practice, Size is preferred because it is easier to update when we start moving nodes around during "Path Compression."

---
## 4. Optimizing the Find Operation: Path Compression

Every time you perform a `Find(u)`, you traverse from $u$ up to the root. 

**Path Compression** dictates that after finding the root, you go back and reattach $u$ (and every node on the path) **directly to the root**.
- **Result:** The next time you call `Find` on any of those nodes, it takes $O(1)$ time.
- **Self-Adjustment:** This turns the Up-Tree into a **self-adjusting structure**.

> [!Example]- $Find(A)$
> ![[Pasted image 20260301120853.png]]
> 
> Sees the nodes $(B, F)$ along the traversal up
> 
> ![[Pasted image 20260301120937.png]]

---
## 5. Amortized Cost Analysis

### A. The "Investment" Logic

In standard analysis, we say a `Find` operation is $O(h)$, where $h$ is the height of the tree. If you have a poorly shaped tree, one `Find` could take $O(n)$ time.

In **Amortized Analysis**, we considers the time or space cost of doing a _sequence of operations_ (as opposed to a single operation) because the total cost of the entire sequence of operations might be less with the extra initial work than without! Acknowledging that the initial $O(n)$ operation is actually an **investment**. As the algorithm travels up to the root, it performs extra work to reassign every node it touches to point directly to the root.

- **The Cost:** This specific `Find` is "expensive" because of the extra pointer reassignments.
- **The Reward:** Every future `Find` for any of those nodes (and their descendants) is now **$O(1)$**.
### B. The Three Methods of Analysis

To prove that the amortized cost is nearly constant, computer scientists use three formal frameworks:

- **The Aggregate Method:** We show that for a sequence of $m$ operations, the _total_ time $T(m)$ is small. The amortized cost is simply $T(m) / m$.
- **The Accounting (Banker's) Method:** We "charge" each cheap operation a bit more than it actually costs. We save this extra "money" as a credit in a bank account. When an expensive $O(n)$ `Find` occurs, it "withdraws" those credits to pay for the work of compressing the path.
- **The Potential (Physicist's) Method:** We define a "potential function" $\Phi$ that represents the "messiness" or height of the tree. A deep, uncompressed tree has high potential energy. An expensive `Find` operation performs work that significantly **decreases** the potential energy (by flattening the tree), which offsets the high actual cost of the operation.
### C. The Scaling of $\log^* N$

The reason the complexity is $O(M \log^* N)$ rather than just $O(M)$ is because the tree isn't _perfectly_ flat immediately. However, $\log^* N$ (the iterative logarithm) is a function that grows so slowly it is essentially a constant for any data we can physically store.

|**Total Elements (N)**|**log∗N**|
|---|---|
|2|1|
|4 ($2^2$)|2|
|16 ($2^4$)|3|
|65,536 ($2^{16}$)|4|
|$2^{65536}$ (More than atoms in the universe)|5|
### D. The "Nearly Constant" Result

The result of this analysis for Disjoint Sets is that the average cost per operation is $O(\alpha(n))$.

The **Inverse Ackermann Function** $\alpha(n)$ is the mathematical way of saying: "This is technically not constant, but it grows so slowly that for any data set humanity will ever create, it will never exceed 5."

> **Key Takeaway:** Without amortized analysis, we would look at the $O(n)$ worst-case of a single `Find` and wrongly conclude the structure is inefficient. Amortized analysis reveals that the **more** you use the structure, the **faster** it gets, leading to a total runtime that is effectively linear ($O(m)$ for $m$ operations).

| **Operation** | **Naive Implementation** | **With Union-by-Size & Path Compression** |
| ------------- | ------------------------ | ----------------------------------------- |
| **Union**     | $O(n)$                   | **$O(\alpha(n)) \approx O(1)$**           |
| **Find**      | $O(n)$                   | **$O(\alpha(n)) \approx O(1)$**           |

---
## Summary: Worst-Case Complexity of Disjoint Sets

While a single "find" or "union" operation can technically hit a worst-case of $O(\log N)$ (with smart *unioning*) or $O(N)$ (without), we use **Amortized Analysis** to describe the true performance over a sequence of $M$ operations.

- **Total Complexity:** For $M$ operations on a set of $N$ elements, the total worst-case time is __$O(N + M \log^* N)$.
- **Per-Operation Complexity:** The "average" worst-case cost for a single operation is _$O(\log^* N)$_*, which is the **Inverse Ackermann Function** $\alpha(N)$.
- **The Result:** Because $\log^* N$ grows so slowly that it never exceeds **5** for any practical value of $N$, the complexity is considered **effectively $O(1)$** (constant time).