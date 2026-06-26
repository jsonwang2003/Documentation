---
tags:
  - Pair
---

> [!INFO]
> An [[Data Structures vs. Abstract Data Types]] that **stores two heterogeneous values** as a single unit. Commonly used for returning multiple values from functions or representing key-value pairs in associative containers.

- **Two-Element Container**: Holds exactly two values: `.first` and `.second`
- **Heterogeneous Types**: Each element can be of a different type
- **Direct Access**: Members accessed via dot notation
- **Lexicographical Comparison**: Supports relational operators for sorting and equality

---
## Common Operations

| Operation                      | Description                                                  | Typical Complexity |
|-------------------------------|--------------------------------------------------------------|--------------------|
| `pair<T1, T2> p{a, b}`         | Constructs a pair with values `a` and `b`                    | O(1)               |
| `make_pair(a, b)`             | Constructs a pair with type deduction                        | O(1)               |
| `p.first`, `p.second`         | Accesses the first and second elements                       | O(1)               |
| `p = {x, y}`                  | Assigns new values to both elements                          | O(1)               |
| `std::swap(p1, p2)`           | Exchanges contents of two pairs                              | O(1)               |
| `p1 == p2`, `p1 < p2`, etc.   | Compares pairs lexicographically                             | O(1)               |
| `auto [x, y] = p` (C++17)     | Structured binding to unpack pair elements                   | O(1)               |

---
## Use Cases

- Returning multiple values from a function
- Storing key-value pairs in `std::map`, `std::unordered_map`
- Zipping two containers together
- Intermediate results in STL algorithms
- Lightweight tuple alternative for two elements

---
## Comparison Semantics

Relational operators compare pairs **lexicographically**:
- First compare `.first`
- If equal, compare `.second`

```cpp
std::pair<int, int> a = {1, 5};
std::pair<int, int> b = {1, 7};
bool result = a < b; // true
```

---
Language-Specific Implementations

- [[Maps and Multimaps|C++ STL Maps(uses pair internally)]]
- [[Computer Science/Programming Languages/C++/Standard Template Library/Pair|C++ Pair]]
- [[Python TuplePython Tuple (similar concept)]]
- [[Java PairJavaFX Pair Class]]
- [[JavaScript ArrayJavaScript 2-element Array]]

Related Concepts

- [[Computer Science/Discrete Structures/Tuple]]
- [[C++ STL/Utility Functions/make_pair]]
- [[C++ STL/Containers/map]]
- [[C++ STL/Containers/unordered_map]]
- [[C++ STL/Structured Bindings]]