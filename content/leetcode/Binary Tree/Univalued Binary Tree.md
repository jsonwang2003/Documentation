---
tags:
  - BinaryTree
  - Easy
---

## Description
[Univalued Binary Tree](https://leetcode.com/problems/univalued-binary-tree/description/)
A binary tree is **uni-valued** if every node in the tree has the same value.

Given the `root` of a binary tree, return `true` _if the given tree is **uni-valued**, or_ `false` _otherwise._

## Code
```cpp
class Solution {
public:
    bool isUnivalTree(TreeNode* node) {
        // Recursive Implementation
        // First check to see if the current node has the same value as its left/right children
        // go to the children and see if they still have the same value
        if (node == nullptr) {return true;}
        bool leftValOK = node->left == nullptr || node->left->val == node->val;
        bool rightValOK = node->right == nullptr || node->right->val == node->val;
        if (!leftValOK || !rightValOK) {return false;}

        bool leftTreeOK = isUnivalTree(node->left);
        bool rightTreeOK = isUnivalTree(node->right);
        return leftTreeOK && rightTreeOK; 
    }
};
```