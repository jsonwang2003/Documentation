---
tags:
  - Tree
  - DepthFirstSearch
  - BinarySearchTree
  - BinaryTree
  - Medium
---

## Description
[Validate Binary Search Tree - Leetcode](https://leetcode.com/problems/validate-binary-search-tree/description/)

Given the `root` of a binary tree, _determine if it is a valid binary search tree (BST)._

A **valid BST** is defined as follows:

- The left subtree[^1] of a node contains only nodes with keys **strictly less than** the node's key.
- The right subtree of a node contains only nodes with keys **strictly greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.

## Examples
#### **Example 1:**

![](https://assets.leetcode.com/uploads/2020/12/01/tree1.jpg)
**Input:** root = [2,1,3]
**Output:** true

#### **Example 2:**

![](https://assets.leetcode.com/uploads/2020/12/01/tree2.jpg)
**Input:** root = [5,1,4,null,null,3,6]
**Output:** false
**Explanation:** The root node's value is 5 but its right child's value is 4.

## Constraints
- The number of nodes in the tree is in the range $[1, 10^{4}]$
- $-2^{31} <= Node.val <= 2^{31} - 1$
## Code
```cpp
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        if (root == nullptr) {return true;}
        bool leftIsBST = isLessThan(root->left, root->val) && isValidBST(root->left);
        bool rightIsBST = isGreaterThan(root->right, root->val) && isValidBST(root->right);
        return leftIsBST && rightIsBST;
    }

    bool isLessThan(TreeNode* node, int val){
        if (node == nullptr) {return true;}
        return node->val < val && isLessThan(node->left, val) && isLessThan(node->right, val);
    }

    bool isGreaterThan(TreeNode* node, int val){
        if (node == nullptr) {return true;}
        return node->val > val && isGreaterThan(node->left, val) && isGreaterThan(node->right, val);
    }
};
```

[^1]: A **subtree** of `treeName` is a tree consisting of a node in `treeName` and all of its descendants.
