---
tags:
  - Tree
  - DepthFirstSearch
  - BinarySearchTree
  - BinaryTree
  - Medium
---

## Description
[Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/description/)

Your are given the `root` of a binary search tree (BST). where the values of **exactly** two nodes of the tree were swapped by mistake. _Recover the tree without changing its structure._

## Examples
1. 
	- **Input:** root = [1,3,null,null,2]
	- **Output:** [3,1,null,null,2]
	- **Explanation:** 3 cannot be a left child of 1 because 3 > 1. Swapping 1 and 3 makes the BST valid. 
	![[Pasted image 20250830145118.png]]
2. 
	- **Input:** root = [3,1,4,null,null,2]
	- **Output:** [2,1,4,null,null,3]
	- **Explanation:** 2 cannot be in the right subtree of 3 because 2 < 3. Swapping 2 and 3 makes the BST valid.
	![[Pasted image 20250830145354.png]]

## Constraints
- The number of nodes in the tree is in the range `[2, 1000]`.
- -2^31^ <= `Node.val` <= 2^31^ - 1

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