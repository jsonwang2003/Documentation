> [!INFO]
> Associative container that **stores and retrieves** elements based on **value of key**

- Stores **key-value pairs**, which **can be different types**
- Sorted by key in **non-continuous memory (similar to [[tags/Linked List]])**
- Uses **balanced [[Binary Search Tree (BSTs)]]** in implementation
## STL Maps and Multimaps
- **Maps**: each key **must be unique**
- **Multimaps**: can have **duplicate keys**

---
## Initialization
#### Default Constructor
```cpp
map<int, string> map1;
```
#### Copy Constructor
```cpp
map<int, string> map2(map1);
```
#### Initializer List Constructor
```cpp
map<int, string> map3{
	{21, "Jason"},
	{2, "John"},
};
```
#### Range Constructor
```cpp
map<int, string> map4(iter, map3.end());
```

#### Make Pair function
- Use the [[Computer Science/Programming Languages/C++/Standard Template Library/Pair|STL Pair]] member function `make_pair`
```cpp
map<int, double> map5;
for(int i = 0; i < 6; ++i){
	map5.insert(make_pair(i, i / 10.0));
}
```
---
## Member Functions
- `map::insert()`: Insert a [[Computer Science/Programming Languages/C++/Standard Template Library/Pair|Pair]] into the map
	- `map::insert(pair<T, T>{key, value})`
		- Declare and initialize the pair in the argument
	- `map::insert(make_pair(key, value))`
		- Utilize the `make_pair` function from the [[Computer Science/Programming Languages/C++/Standard Template Library/Pair|Pair]] class
- `map::find(const T& key)`: 
	- if `key` is found in map → Return the **iterator** pointing to the **pair** with the `key`
	- if `key` is not found in map → Return the `map::end`
- `map::erase()`: Removes pair(s) in the map, reducing the size of the map
	- `iterator erase(const_iterator position)`:
		- Delete by iterator `position`
		- Return iterator to the element following the last pair removed pointed by `posiiton`
		- **All occurrence of pair with the key** are deleted in **multimap**
		- Return `map::end` if **last pair in map is removed**
		- `position`: the iterator pointing to the element to be removed
	- `size_type erase(const T& key)`
		- Return the number of elements deleted
		- `val`: value of **key in the pair** to be deleted
		- more used in multimap
	- `iterator erase(const_iterator first, const_iterator last)`
		- Return the iterator to the element the last **pair** removed (`map::end` if last element is removed)
		- `first` and `last`: Range to be removed $[\text{first}, \text{last})$