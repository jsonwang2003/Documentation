---
tags:
  - HashTable
---
> [!INFO]
> A **Hash Table** is a key-value data structure that enables **constant-time average access** through hashing. It’s foundational in computer science for implementing associative arrays, dictionaries, and caches.

### [[Computer Science/Data Structures/Hash Table|Hash Tables]]

## Properties

- **Key-Value Mapping**: Each value is accessed via a unique key
- **Hash Function**: Transforms keys into indices for storage
- **Constant-Time Access**: Average-case `O(1)` for insert, delete, and lookup
- **Collision Handling**: Uses chaining or open addressing to resolve hash conflicts
- **Dynamic Resizing**: Grows or shrinks based on load factor
- **Unordered**: Does not preserve insertion order (unless using ordered variants)

## Common Operations

- **Insert**: Add a key-value pair (`table["x"] = 42`)
- **Lookup**: Retrieve value by key (`table["x"]`)
- **Delete**: Remove a key (`del table["x"]`)
- **Update**: Modify value for existing key (`table["x"] = 99`)
- **Contains**: Check if key exists (`"x" in table`)
- **Iterate**: Traverse keys, values, or items (`for k, v in table.items()`)

## Collision Resolution Strategies

- **Chaining**: Store multiple values at the same index using [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Linked List]]
- **Open Addressing**: Probe for next available slot (e.g., linear, quadratic, double hashing)
- **Perfect Hashing**: Collision-free hashing for static key sets
- **Rehashing**: Resize and reassign keys when load factor exceeds threshold