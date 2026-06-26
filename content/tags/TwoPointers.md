---
title: Two Pointers
tags:
  - TwoPointers
---
> [!INFO]
> A technique that uses **two indices** to traverse a data structure—often in opposite directions or at different speeds—to solve problems more efficiently. It’s commonly used in **arrays**, **strings**, and **linked lists** to reduce time complexity from O(n²) to O(n).

### [[Computer Science/Algorithms/Two Pointers|Two Pointers]]
## Properties

- Uses two pointers (e.g., `left` and `right`, or `slow` and `fast`)
- Often applied to **sorted arrays**, **subarray problems**, and **linked list traversal**
- Can be used for **window-based problems**, **cycle detection**, and **partitioning**
- Reduces nested loops into linear scans

## Common Patterns

- **Converging Pointers**: Start at both ends and move toward the center  
  - Used in palindrome checks, pair sums, and partitioning
- **Sliding Window**: Move both pointers forward to maintain a dynamic range  
  - Used in substring problems, max/min subarrays
- **Fast–Slow Pointers**: One pointer moves faster than the other  
  - Used in cycle detection, finding middle of linked list

## Common Operations

- **Pair Sum**: Find two numbers that add up to a target (`left + right`)
- **Reverse**: Swap elements from both ends (`arr[left], arr[right] = arr[right], arr[left]`)
- **Window Expansion/Contraction**: Adjust pointers based on constraints
- **Cycle Detection**: Use fast and slow pointers to detect loops in linked lists

