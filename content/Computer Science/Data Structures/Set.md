---
tags:
  - Set
---
> [!INFO]
> An [[Data Structures vs. Abstract Data Types]] that can **store unique values** without any particular order. Equivalent to the mathematical [[Set Theory]]

- **Uniqueness**: No duplicate elements allowed.
- **Unordered**: Elements are not stored in any guaranteed order (unless using ordered variants).
- **Membership Testing**: Fast lookup to check if an element exists.
- **Dynamic Size**: Can grow or shrink as elements are added or removed.

To find the **Mathematical Theory and Properties**, visit [[Set Theory]]

---
## Common Operations
| Operation            | Description                              | Typical Complexity |
| -------------------- | ---------------------------------------- | ------------------ |
| `add(x)`             | Inserts element `x` into the set         | O(1) (hash-based)  |
| `remove(x)`          | Deletes element `x` from the set         | O(1)               |
| `contains(x)`        | Checks if `x` is in the set              | O(1)               |
| `size()`             | Returns the number of elements           | O(1)               |
| `clear()`            | Removes all elements                     | O(n)               |
| `union(setB)`        | Combines elements from two sets          | O(n)               |
| `intersection(setB)` | Returns common elements between two sets | O(n)               |
| `difference(setB)`   | Returns elements in A not in B           | O(n)               |

---
## Types of Sets
- **Hash Set**: Backed by a [[Computer Science/Data Structures/Hash Table]]. Fast operations, no order.
- **Tree Set**: Maintains sorted order (e.g., [[Red-Black Tree]]).
- **Linked Set**: Preserves insertion order.
- **Multiset**: Allows duplicates (not a true set, but often grouped here).

--- 
# Language-Specific Implementations
- [[Sets and Multisets|C++ Sets and Multisets]]