> [!INFO]
> Associative container **stores** and **retrieves** element based on he value of "**key**"

- **Key**: Can be **any** data type (int, string, object, etc.)
- Order of insertion **have no bearing on how they are organized**

> [!IMPORTANT]
> Associative container **automatically** recognize data in **ascending order** by default. 

## STL Sets and Multisets
- **Set**: each key **must be unique**
- **Multiset**: can contain **duplicate keys**

> [!NOTE]
> Although elements in set are sorted in **non-continuous memory (similar to [[tags/Linked List]])** they are sorted in order → Can use [[Binary Search Tree (BSTs)]] to look for specific key

---
## Constructors
#### Default Constructor
```cpp
set<int> set1;
```
#### Copy Constructor
```cpp
set<int> set2(set1); // set2 not is a copy of set1
```
#### Initializer List Constructor
```cpp
set<int> set3{10, 20, 30, 40};
set<int> set4 = {1, 2, 3, 4, 5};
```
#### Range Constructor
```cpp
auto iterStart = set3.begin() + 2;
auto iterEnd = set3.end();
set<int> set5(iterStart, iterEnd);
```

---
## Member Functions
- `set::find(const T& val)`:
	- If `val` is found in set → Return the **iterator** pointing to the `val`
	- If `val` is not found in set → Return the `set::end`
- `set::erase()`: Removes element(s) in the set, reducing size of the set
	- `iterator erase(const_iterator position)`
		- Delete by iterator `posiion`
		- Return iterator to the element following the last element removed pointed by `position`
		- **All occurrence of element** are deleted in **multiset**
		- Return `set::end` if **last element in set is removed**
		- `position`: the iterator pointing to the element to be removed
	- `size_type erase(const T& val)`:
		- Return the number of elements deleted
		- `val`: value of element to the deleted
		- more used in multiset
	- `iterator erase(const_iterator first, const_iterator last)`:
		- Return the iterator to the element the last element removed (`set::end` if last element is removed)
		- `first` and `last`: Range to be removed $[\text{first}, \text{last})$