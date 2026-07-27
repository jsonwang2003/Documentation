---
description: "An implementation of the Map ADT leveraging hash table mechanisms to map distinct key identifiers to associatively clustered value records."
aliases:
  - Hash Map
  - Associative Array
  - Map ADT
tags:
  - data-structures
  - hashing
  - maps
---
> [!abstract] Abstract 
> The Map ADT (often called an associative array) allows us to map keys to their corresponding values. While a standard Hash Table checks for the presence or absence of a key, a Hash Map associatively pairs each key with a distinct value record, enabling fast average constant-time storage, retrieval, and removal based on key queries.
> 
> - **Category:** Key-Value Storage Structure
> - **Stores:** Associative pairs of hashable key identifiers linked to variable value payloads.
> - **Built on top of:** Arrays, Hash Tables, and Collision Resolution frameworks.
> - **Typical use cases:** Database index records, grade books, caching systems, representing one-to-many relationship maps.

---

# Core Structure

A Hash Map utilizes an underlying backing array of entries where each slot stores a key and its value together as a unified pair. Keys must be hashable and support an equality test to handle uniqueness checks and bucket routing.

```
Index Mapping (via H(key) % m)
[ Index 0 ] ---> | Key: "Kammy"  | Value: 'A' |
[ Index 1 ] ---> | Key: "Alicia" | Value: 'C' |
[ Index 2 ] ---> NULL
```

> [!tip] Key Idea
> When we find, insert, or remove elements in a Hash Map, the operational routing algorithms mirror those of a standard Hash Table exactly, but everything is processed strictly with respect to the key. Once the target key's slot is located, the associated value payload is exposed.

---

# Structural Properties

*   **Invariant:** Every stored value is structurally bound to a unique key identifier. Duplicate keys cannot point to multiple independent primary entries simultaneously within the base map structure.
*   **Overwriting Behavior:** Attempting to insert a key that already exists will not abort the operation. Instead, the original value associated with that key is overwritten by the new input value.
*   **Order Guarantee:** Does NOT guarantee consistent element ordering. Iterating through a Hash Map yields elements in an unstable sequence determined entirely by hash distributions and collision arrangements.

---

# Data Structure Operations

## `Insert(key, value)`
Hashes the key to resolve its backing array index, replacing the old value with the new value if the key already exists.

- **Time Complexity:** $O(1)$ average; $O(N)$ worst-case under severe clustering.

```pseudo
	\begin{algorithm}
	\caption{Hash Map Insertion (Collision-Free Paradigm)}
	\begin{algorithmic}
		\Procedure{Insert}{$key, value, arr$}
			\State $index \gets$ \Call{HashFunction}{$key$}
			\State $returnVal \gets \text{NULL}$
			\If{$arr[index].key == key$}
				\State $returnVal \gets arr[index].value$
			\EndIf
			\State $arr[index] \gets \langle key, value \rangle$
			\Return $returnVal$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Find(key)`
Traverses the probe sequence or chain corresponding to the key's hash index and returns the associated value if found.

- **Time Complexity:** $O(1)$ average case.

```pseudo
	\begin{algorithm}
	\caption{Hash Map Find}
	\begin{algorithmic}
		\Procedure{Find}{$key, arr$}
			\State $index \gets$ \Call{HashFunction}{$key$}
			\If{$arr[index].key == key$}
				\Return $arr[index].value$
			\Else
				\Return $\text{NULL}$
			\EndIf
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Remove(key)`
Locates the key inside the map, isolates its associated value payload for output preservation, and deletes the element entry.

- **Time Complexity:** $O(1)$ average case.

```pseudo
	\begin{algorithm}
	\caption{Hash Map Removal}
	\begin{algorithmic}
		\Procedure{Remove}{$key, arr$}
			\State $index \gets$ \Call{HashFunction}{$key$}
			\State $returnVal \gets \text{NULL}$
			\If{$arr[index].key == key$}
				\State $returnVal \gets arr[index].value$
				\State \Call{Delete}{$arr[index]$}
			\EndIf
			\Return $returnVal$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# C++ Language Implementation Detail

In the C++ Standard Template Library (STL), the primary implementation of a Hash Map is `std::unordered_map`, which relies on Separate Chaining for collision management.

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>

int main() {
    // Initializing a Hash Map
    std::unordered_map<std::string, std::string> gradeBook = {
        {"Kammy", "A"},
        {"Alicia", "C"}
    };

    // Inserting a new pair
    gradeBook.insert({"Bob", "B"});

    // Querying using the bracket operator
    std::cout << gradeBook["Kammy"] << std::endl; // Outputs: A
}
```

### Duplicate Insertion Deviation
C++ deviates slightly from the traditional Map ADT specification during duplicate handling. When inserting a duplicate element using `unordered_map::insert()`, the container protects the original value and ignores the overwrite. To force a replacement value update, you must use the assignment bracket operator instead (`gradeBook["Key"] = newValue`).

---

# Structuring One-to-Many Relationships

Hash Maps are often deployed to manage one-to-many relationship structures by mapping discrete keys onto a collection container value, such as a vector or an inner nested Hash Table.

```cpp
// Mapping drawer labels to a vector of office items
std::unordered_map<std::string, std::vector<std::string>> desk = {
    {"pens", {"favPen", "redPen"}},
    {"personal papers", {"schedule"}}
};

// Modifying the nested vector collection value dynamically
desk["personal papers"].push_back("taxDocument");
```

> [!warning] Cascading Time Complexity Risks
> When wrapping collection structures inside map values, the cost of accessing a nested item requires adding the find complexity of that inner collection container. For an unsorted backing vector containing $n$ elements, locating a nested target adds an internal $O(n)$ linear lookup cost. If absolute constant-time access is required across all nested values, use an `std::unordered_set` as the value type instead of a vector.

---

# Related Notes

- [[Hashing/Hash Functions|Hash Functions]]
- [[Hashing/Collision Resolution/index|Collision Resolution Strategies]]
- [[Hashing/Collision Resolution/Closed Addressing (Separate Chaining)|Closed Addressing (Separate Chaining)]]