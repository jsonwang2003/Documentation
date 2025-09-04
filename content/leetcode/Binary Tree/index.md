---
title: "Binary Tree"
---

A Binary Tree is a hierarchical data structure in which each node has **at most two children**, referred to as the left and right child. Unlike Binary Search Trees, Binary Trees do not enforce any ordering constraints between node values.

## Properties
- Each node can have zero, one, or two children
- No requirement for left < root < right relationships
- Can be used to represent structured data like expressions, hierarchies, or traversal paths

## Common Variants
- **Full Binary Tree**: Every node has 0 or 2 children
- **Complete Binary Tree**: All levels are fully filled except possibly the last, which is filled left to right
- **Perfect Binary Tree**: All internal nodes have two children and all leaves are at the same level
- **Balanced Binary Tree**: Height difference between left and right subtrees is minimized

## Common Operations
- Traversal:
  - Inorder: Left → Root → Right
  - Preorder: Root → Left → Right
  - Postorder: Left → Right → Root
  - Level-order: Breadth-first traversal
- Insert/Delete: Depends on specific variant (e.g., heap, expression tree)
- Search: No guarantees on performance unless structured (e.g., BST or heap)

## Use Cases
- Expression parsing (e.g., arithmetic trees)
- Hierarchical data modeling (e.g., file systems)
- Priority queues (via binary heaps)
- Tree-based traversal algorithms

## Related Problems
- [[leetcode/Binary Tree/Symmetric Tree]]
- [[leetcode/Binary Tree/Univalued Binary Tree]]