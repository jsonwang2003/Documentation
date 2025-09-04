---
title: "Binary Search Tree"
---

A Binary Search Tree (BST) is a tree data structure where each node has the following properties:
- **Value Property**:  
    Each node contains a unique value (or key).
- **Left Subtree Rule**:  
	All values in the left subtree of a node are **less than** the node’s value.
- **Right Subtree Rule**:  
	All values in the right subtree of a node are **greater than** the node’s value.
- **Recursive Structure**:  
	Both the left and right subtrees must also be binary search trees.
- **No Duplicate Values** (in standard BSTs):  
	Typically, duplicate values are not allowed, though some implementations handle them with custom rules.
## Properties
- The left subtree contains only nodes with keys less than the node's key
- The right subtree contains only nodes with keys greater than the node's key
- Both left and right subtrees are also binary search trees

## Common Operations
- **Search**: O(log n) average, O(n) worst case
- **Insert**: O(log n) average, O(n) worst case  
- **Delete**: O(log n) average, O(n) worst case

## Related Problems
- [[leetcode/Binary Search Tree/Validate Binary Search Tree]]
- [[leetcode/Binary Search Tree/Recover Binary Search Tree]]
