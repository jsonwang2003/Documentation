---
tags:
  - LinkedList
---
> [!INFO] Definition 
> A **linked list** is a linear, dynamically-allocated data structure where elements (nodes) are stored in non-contiguous memory locations and connected via **pointers**. Unlike arrays, it grows organically and does not require a single block of memory.
### Properties
- **Non-Contiguous**: Elements are scattered in memory, linked only by addresses.
- **Dynamic Size**: Can grow or shrink at runtime without expensive resizing/copying operations.
- **Node-Based**: Each "Node" contains a data field and at least one reference (pointer) to another node.
- **Sequential Access**: Lacks random access; must be traversed from the **Head** or **Tail**.
---
### Common Patterns
- **Singly-Linked**: Each node points forward to the next; minimizes memory overhead.
- **Doubly-Linked**: Nodes point to both previous and next neighbors; allows bidirectional traversal.
- **Circular**: The last node points back to the first; used in buffer management and round-robin scheduling.
- **Fast–Slow Pointers**: Moving two pointers at different speeds to find midpoints or detect cycles.
---
### Common Operations
- **Head Insertion/Deletion**: Updating the head pointer to add/remove the first element (O(1)).
- **Tail Insertion/Deletion**: Updating the tail and its neighbor (O(1) if tail pointer is maintained).
- **Traversal**: Moving from node to node via `.next` pointers until reaching NULL (O(n)).
- **Reverse**: Iteratively or recursively flipping the direction of all pointers in the list.
- **Cycle Detection**: Using the **Two Pointers** pattern to determine if a node points back to a previous node.

---
### Complexity & Trade-offs

|Operation|Complexity|Implementation Detail|
|---|---|---|
|**Access/Search**|O(n)|Must iterate through nodes; cannot calculate address via index.|
|**Insert/Delete (Ends)**|O(1)|Simple pointer swap; no element shifting required.|
|**Insert/Delete (Middle)**|O(n)|Time is taken by the search; the actual "link" change is O(1).|
|**Space**|O(n)|Higher overhead than arrays due to storing extra pointers.|

---
### Theory Connections
- **[[Array Lists]]**: The primary alternative; provides O(1) access but O(n) head-insertion.
- **[[Two Pointers]]**: Essential technique for solving Linked List problems like "Middle of List" or "Cycle Detection."
- **[[tags/Stack]] / [[tags/Queue]]**: These Abstract Data Types are frequently implemented using Linked Lists to ensure O(1) performance.