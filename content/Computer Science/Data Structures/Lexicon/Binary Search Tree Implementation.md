> [!ABSTRACT]
> 
> A **[[Binary Search Tree (BSTs)|Self-Balancing Binary Search Tree (BST)]]**, such as an **AVL Trees**, offers a powerful compromise for the Lexicon ADT. It guarantees **$O(\log n)$ worst-case** time complexity for all operations while maintaining the ability to traverse words in alphabetical order.

---
## 1. Choosing the Right Tree
While several types of BSTs exist, only self-balancing versions are suitable for a robust Lexicon:
- **[[AVL Tree]]:** Preferred for Lexicons because they have stricter balance requirements than Red-Black trees. This results in a smaller tree height, leading to **faster "find" operations** in practice.
- **[[Red-Black Tree]]:** Also viable with $O(\log n)$ guarantees, but typically optimized for scenarios with more frequent insertions than a standard Lexicon requires.

---
## 2. Performance Analysis

By using a self-balancing tree, we ensure that the lexicon remains efficient even as it grows to contain hundreds of thousands of words.

|**Operation**|**Worst-Case Complexity**|**Logic**|
|---|---|---|
|**`find(word)`**|**$O(\log n)$**|Logarithmic search based on string comparisons.|
|**`insert(word)`**|**$O(\log n)$**|Search for position + local rotations to maintain balance.|
|**`remove(word)`**|**$O(\log n)$**|BST removal + structural rebalancing.|
|**Space**|**$O(n)$**|Exactly one node per word.|

---
## 3. Ordered Iteration
One major advantage of the BST over unordered structures is the ability to retrieve words in alphabetical order. This is achieved via an **In-Order Traversal**.
- **Ascending Order:** Visit the left child, the current node, then the right child.
- **Descending Order:** Visit the right child, the current node, then the left child.

```C++
ascendingInOrder(node):  // Alphabetical (A-Z)
    ascendingInOrder(node.leftChild)   
    output node.word                   
    ascendingInOrder(node.rightChild)  

descendingInOrder(node): // Reverse Alphabetical (Z-A)
    descendingInOrder(node.rightChild) 
    output node.word                   
    descendingInOrder(node.leftChild)  
```

---
## 4. Evaluation for Lexicon ADT
The BST implementation provides a significant upgrade over the Sorted Array when updates are needed:
- **Consistency:** Unlike the Sorted Array, which struggles with $O(n)$ insertions, the BST handles all operations in $O(\log n)$.
- **Functionality:** It supports range queries (e.g., "find all words between 'apple' and 'banana'") much more naturally than a Hash Table.
- **Dependency:** Note that the speed still depends on $n$ (the number of words). As the lexicon grows, the height of the tree increases.

---
## 5. Comparison: Sorted Array vs. BST (AVL)

|**Feature**|**Sorted Array**|**AVL Tree**|
|---|---|---|
|**Search Speed**|$O(\log n)$|$O(\log n)$|
|**Insertion/Removal**|$O(n)$|**$O(\log n)$**|
|**Memory Efficiency**|High (Contiguous)|Moderate (Node pointers)|
|**Alphabetical Order**|Yes|Yes|
