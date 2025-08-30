# Validate Binary Search Tree

**Problem:** Given the root of a binary tree, determine if it is a valid binary search tree (BST).

## Approach
Use in-order traversal to validate that nodes are in ascending order.

## Solution
```python
def isValidBST(self, root):
    def inorder(node):
        if not node:
            return True
        
        if not inorder(node.left):
            return False
            
        if self.prev is not None and self.prev >= node.val:
            return False
        self.prev = node.val
        
        return inorder(node.right)
    
    self.prev = None
    return inorder(root)
```

## Time Complexity
- Time: O(n)
- Space: O(h) where h is height of tree
