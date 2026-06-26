---
title: Stack
tags:
  - Stack
---

> [!INFO]
> A **linear data structure** that follows the **Last In, First Out (LIFO)** principle. The most recently added element is the first to be removed. It’s conceptually similar to a stack of plates: you add to the top and remove from the top.

### [[Computer Science/Data Structures/Introductory Data Structures/Stack|Stack]]
## Properties

- Operates on **LIFO**: Last element added is the first to be removed
- Only the **top element** is accessible at any time
- Supports **constant-time operations** for push and pop
- Can be implemented using [[Array Lists|Array Lists]] or [[linked lists]]

## Common Variants

- **Bounded Stack**: Has a fixed maximum size
- **Dynamic Stack**: Grows as needed (e.g., via dynamic arrays or linked lists)
- **Call Stack**: Used by programming languages to manage function calls and local variables
- **Concurrent Stack**: Thread-safe stack for multi-threaded environments

## Common Operations

- **Push**: Add an element to the top of the stack
- **Pop**: Remove the top element
- **Peek**/**Top**: View the top element without removing it
- **IsEmpty**: Check if the stack is empty
- **Size**: Return the number of elements in the stack

## Use Cases

- **Function call management** (e.g., recursion tracking via call stack)
- **Undo/Redo functionality** in applications
- **Syntax parsing** (e.g., matching parentheses or tags)
- **[[Depth First Search (DFS)|Depth-first search (DFS)]]** in graph traversal
- **Backtracking algorithms** (e.g., maze solving, puzzle solving)