---
tags:
  - Tree
  - DepthFirstSearch
  - BreadthFirstSearch
  - BinaryTree
  - Easy
---

## Description
[Symmetric Tree - Leetcode](https://leetcode.com/problems/univalued-binary-tree/description/)

Given the `root` of a binary tree, _check whether it is a mirror of itself_ (i.e., symmetric around its center).

## Examples
#### **Example 1:**

![](https://assets.leetcode.com/uploads/2021/02/19/symtree1.jpg)

**Input:** root = [1,2,2,3,4,4,3]
**Output:** true

#### **Example 2:**

![](https://assets.leetcode.com/uploads/2021/02/19/symtree2.jpg)

**Input:** root = [1,2,2,null,3,null,3]
**Output:** false

## Constraints
- The number of nodes in the tree is in the range $[1, 1000]$.
- $-100 <= Node.val <= 100$
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