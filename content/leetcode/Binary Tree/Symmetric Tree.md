---
tags:
  - BinaryTree
  - Easy
---

## Description
[Symmetric Tree](https://leetcode.com/problems/univalued-binary-tree/description/)

Given the `root` of a binary tree, _check whether it is a mirror of itself_ (i.e., symmetric around its center).

## Code
```cpp
class Solution{
public: 
	bool isSymmetric(TreeNode* root){
		if (root == nullptr) { return true; }
		return isMirror(root->left, root->right);
	}
	bool isMirror(TreeNode* left, TreeNode* right){
		if (left == nullptr && right == nullptr) {return true;}
		if (left == nullptr || right == nullptr) {return false;}
		bool sameVal = left->val == right->val;
		bool outerTreeOK = isMirror(left->left, right->right);
		bool innerTreeOK = isMirror(left->right, right->left);
		return sameVal && outerTreeOK && innerTreeOK;
	}
}
```