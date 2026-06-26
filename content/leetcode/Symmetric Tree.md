---
tags:
  - Tree
  - DepthFirstSearch
  - BreadthFirstSearch
  - BinaryTree
Difficulty: Easy
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

## Approach
1. Check if there is a tree
2. Use `isMirror()` helper function to check for mirroring of the Binary Tree
	1. Both left and right subtree does not exist &rarr; is mirrored
	2. Either left or right subtree does not exist &rarr; is not mirrored
	3. Check if the left and right root are same value
	4. Check if the outer tree is mirrored
	5. Check if the inner tree is mirrored
> [!IMPORTANT]
> Since the symmetric line is by the root of the whole tree, we must check for mirroring by checking the inner subtree and outer subtree
> - Inner Subtree: `left right` , `right left`
> - Outer Subtree: `left left` , `right right`
3. return the result