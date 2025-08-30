---
tags:
  - BinarySearchTree
  - Medium
---

## Description
[Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/description/)

Your are given the `root` of a binary search tree (BST). where the values of **exactly** two nodes of the tree were swapped by mistake. _Recover the tree without changing its structure._

## Code
```cpp
class Solution {

public:
    TreeNode* first = nullptr;
    TreeNode* second = nullptr;
    TreeNode* prev = nullptr;
    
    void recoverTree(TreeNode* root) {
        traverse(root);
        swap(first->val, second->val);
    }

    void traverse(TreeNode* node){
        if (node == nullptr) {return;}
        traverse(node->left);
        
        if(prev && prev->val > node->val){
            if(first == nullptr) {first = prev;}
            second = node;
        }
  
        prev = node;
        traverse(node->right);
    }
};
```