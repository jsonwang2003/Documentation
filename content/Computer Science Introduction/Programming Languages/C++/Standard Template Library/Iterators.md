## C++ Standard Template Library: Iterators

The **iterator pattern** permits access to data in a sequential order without exposing the underlying representation of the container. This creates a uniform interface for traversing different data structures, whether they are based on arrays, linked nodes, or trees.

---
## The Anatomy of an Iterator

In C++, an iterator acts as a "mystical entity" that behaves similarly to a pointer. It tracks a position within a container and provides operators to move and access data.

#### Key Functions and Operators

To support the iterator pattern, a data structure must implement specific functions, and its internal iterator class must overload specific operators.

|**Component**|**Member**|**Description**|
|---|---|---|
|**Container**|`begin()`|Returns an iterator to the first element.|
|**Container**|`end()`|Returns an iterator positioned **just past** the last element.|
|**Iterator**|`*`|**Dereference**: Returns a reference to the data at the current position.|
|**Iterator**|`++`|**Increment**: Moves the iterator to the next sequential element.|
|**Iterator**|`==` / `!=`|**Comparison**: Checks if two iterators refer to the same position.|

---
## Iterator Usage Syntax

The C++ STL allows for two primary ways to traverse a container using iterators.

### 1. Manual Loop

This version explicitly manages the iterator object. It is useful when you need to modify the iterator (like skipping elements) during the loop.

```cpp
vector<string>::iterator itr = c.begin();
vector<string>::iterator end = c.end();

while(itr != end) {
    cout << *itr << endl; // Access data
    ++itr;                // Move to next
}
```

### 2. Range-based For Loop (For-each)

This is "syntactic sugar" for the manual loop above. The compiler automatically generates the `begin`, `end`, and `++` logic behind the scenes.

```cpp
for(string s : c) {
    cout << s << endl;
}
```

---
## Implementation Examples

Implementing iterators requires defining a nested `iterator` class and providing the `begin()` and `end()` hooks.
### Linked List Iterator
In a Linked List, the iterator must store a pointer to a **Node**. Because memory for nodes is non-contiguous, the `++` operator must follow the `next` pointer.

```cpp
// Inside LinkedList class
class iterator {
    Node* curr;
    public:
        iterator(Node* x) : curr(x) {}
        reference operator*() const { return curr->value; }
        iterator& operator++() { 
            curr = curr->next; // Jump to next node
            return *this; 
        }
        bool operator!=(const iterator& x) const { return curr != x.curr; }
};

iterator begin() { return iterator(root); }
iterator end() { return iterator(nullptr); }
```

### Array List Iterator
Because an Array List is contiguous in memory, the iterator can simply store a pointer to the array's data type. The `++` operator uses pointer arithmetic to move to the next memory slot.

```cpp
// Inside ArrayList class
iterator begin() { return iterator(&arr[0]); }
iterator end() { return iterator(&arr[size]); } // Pointer just past valid data
```

---
## Why Use Iterators?

1. **Uniformity**: You can swap a `vector` for a `list` without changing the loop logic.    
2. **Encapsulation**: The user doesn't need to know if the data is stored in a tree, an array, or a hash table.
3. **Efficiency**: STL implementations are highly optimized by experts.