---
title: Tree
tags:
  - Tree
---
> [!INFO]
> A hierarchical data structure composed of **nodes** connected by **edges**, where each node may have child nodes but only one parent (except the root). Trees are used to model relationships, organize data, and support efficient traversal and search operations.

### [[Computer Science/Data Structures/Tree/index|Tree]]
## Properties

- **Rooted**: One node is designated as the root; all others descend from it
- **Parent–Child Relationship**: Each node links to its children via edges
- **Leaf Node**: A node with no children
- **Height**: The longest path from root to any leaf
- **Depth**: Distance from root to a given node
- **Subtree**: Any node and its descendants form a subtree

## Common Types

- **Binary Tree**: Each node has at most two children
- **Binary Search Tree (BST)**: Left child < parent < right child
- **Balanced Tree**: Height difference between subtrees is minimized
- **AVL / Red-Black Trees**: Self-balancing variants of BSTs
- **N-ary Tree**: Nodes can have more than two children
- **Trie**: Specialized tree for string prefix matching

## Common Operations

- **Traversal**:
  - In-order: Left → Root → Right
  - Pre-order: Root → Left → Right
  - Post-order: Left → Right → Root
  - Level-order: Breadth-first traversal
- **Insertion**: Add a node while maintaining structure
- **Deletion**: Remove a node and restructure if needed
- **Search**: Locate a node by value or condition
- **Height/Depth Calculation**: Measure structural properties

