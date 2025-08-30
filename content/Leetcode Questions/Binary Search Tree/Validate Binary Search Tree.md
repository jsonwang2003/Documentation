---
tags:
  - BinarySearchTree
  - Medium
---

## Description
[Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/description/)

Given the `root` of a binary tree, _determine if it is a valid binary search tree (BST)._

A **valid BST** is defined as follows:

- The left [subtree]() of a node contains only nodes with keys **strictly less than** the node's key.
- The right subtree of a node contains only nodes with keys **strictly greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.

> Subtree
> 
> A **subtree** of `treeName` is a tree consisting of a node in `treeName` and all of its descendants.

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
