---
description: "A lightweight container storing two heterogeneous elements as a single unit, commonly used for function returns and map entries."
aliases:
  - Pair
  - std::pair
  - Two-Element Container
tags:
  - data-structures
  - adt
  - containers
---
> [!info] Abstract
> A **Pair** is an Abstract Data Type (ADT) wrapper that couples two heterogeneous values into a single compound structure. It is widely utilized for returning multiple values from functions, representing key-value pairs in associative containers (like `std::map`), and zipping parallel datasets.
> 
> - **Category:** Composite Value Container
> - **Primary Members:** `.first` and `.second`
> - **Comparison:** Lexicographical ordering supported natively.

---

# Key Architectural Properties

*   **Two-Element Container:** Explicitly encapsulates exactly two values accessible via `.first` and `.second`.
*   **Heterogeneous Types:** The first and second elements can belong to entirely different data types (e.g., `pair<string, int>`).
*   **Direct Access:** Value members are accessed directly via dot notation without getter overhead.
*   **Lexicographical Comparison:** Evaluates relational operators (`<`, `==`, etc.) by comparing `.first` first, and checking `.second` only if the first elements match.

---

# Common Operations & Complexity

| Operation | Description | Typical Complexity |
|---|---|---|
| `pair<T1, T2> p{a, b}` | Constructs a pair initialized with values `a` and `b`. | $O(1)$ |
| `make_pair(a, b)` | Helper function constructing a pair with type deduction. | $O(1)$ |
| `p.first`, `p.second` | Direct field access to stored elements. | $O(1)$ |
| `p = {x, y}` | Assigns new values to both elements simultaneously. | $O(1)$ |
| `swap(p1, p2)` | Exchanges the contents of two pair structures. | $O(1)$ |
| `p1 == p2`, `p1 < p2` | Compares two pairs lexicographically. | $O(1)$ |
| `auto [x, y] = p` | Unpacks pair elements via structured bindings (C++17). | $O(1)$ |

---

# Comparison Semantics

Relational operators evaluate pairs using strict lexicographical comparisons:

```cpp
std::pair<int, int> a = {1, 5};
std::pair<int, int> b = {1, 7};

// Returns true because a.first == b.first (1 == 1), and a.second < b.second (5 < 7)
bool result = a < b; 
```

---

# Common Use Cases

1.  **Multiple Return Values:** Packages dual function outputs without building custom structs.
2.  **Associative Containers:** Forms key-value entry records inside maps and hash maps.
3.  **Coordinate Representation:** Tracks 2D spatial points $(x, y)$ or grid positions.
4.  **Zipping Collections:** Pairs related elements from separate array streams.

---

# Related Notes

- [[Data Structures/Set|Set]]
- [[Hashing/Hash Maps (Maps)|Hash Maps]]
- [[Data Structures/index|Data Structures Directory]]