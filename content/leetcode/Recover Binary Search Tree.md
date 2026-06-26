---
tags:
  - Tree
  - DepthFirstSearch
  - BinarySearchTree
  - BinaryTree
Difficulty: Medium
---

## Description
[Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/description/)

Your are given the `root` of a binary search tree (BST). where the values of **exactly** two nodes of the tree were swapped by mistake. _Recover the tree without changing its structure._

## Examples
#### **Example 1:**

![](https://assets.leetcode.com/uploads/2020/10/28/recover1.jpg)

**Input:** root = [1,3,null,null,2]
**Output:** [3,1,null,null,2]
**Explanation:** 3 cannot be a left child of 1 because 3 > 1. Swapping 1 and 3 makes the BST valid.

#### **Example 2:**

![](https://assets.leetcode.com/uploads/2020/10/28/recover2.jpg)

**Input:** root = [3,1,4,null,null,2]
**Output:** [2,1,4,null,null,3]
**Explanation:** 2 cannot be in the right subtree of 3 because 2 < 3. Swapping 2 and 3 makes the BST valid.
## Constraints
- The number of nodes in the tree is in the range $[2, 1000]$.
- $-2^{31} <= Node.val <= 2^{31} - 1$

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

## Approach
1. Create 3 pointers
	- `first`: the first node found that is out of place
	- `second`: the second node found that is out of place
	- `prev`: the node is the previous in sorting order
2. Finding the 2 nodes needed to swap using `traverse` helper function that uses Inorder Traversal
	- check if current root is valid
	- continue to traverse to the left (leaf will be the smallest value)
	- check if the `prev` > `node->val`
		- assign `prev` to `first` if not already
		- assign `node` to `second` the second time the out of place value
	- update the previous to the current node
	- traverse to the right subtree
3. After finding the 2 nodes needed to swap, swap them using `swap()` STL function