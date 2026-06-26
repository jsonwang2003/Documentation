
> [!INFO]
> Sequence containers that organizes data **sequentially** in the order that it is added. List is different than [[Vectors]] is that it is implemented with [[Doubly Linked List (DLL)]]
- Does not have **random access**[^1]
- `slist`
	- Implemented as [[Singly Linked List (SLL)]]
	- Not **standard** and **not all compilers support it**
---
## Constructors
#### Default Constructor
```cpp
list<int> list1;
```
#### Fill Constructor
```cpp
list<int> list2(4, 100); // 4 ints of value 100
```
#### Range Constructor
```cpp
list<int> list3(list.begin() + 5, list.end()); // same content as list except for the first 5 elements
```
#### Copy Constructor
```cpp
list<int> list4(list); // a copy of list
```

---
## Member Functions
- `list::push_front(const T& value)`: Add new element to the **front**
- `list::pop_front()`: Remove the **front** element
- `list::reverse()`: reverse the order of elements
- `list::sort()`: sort the container in **ascending order**
- `list::remove(const T& value)`: delete a given element in the list
- `list::splice()`: Transfers elements from `otherList` into the container, inserting them at `position`. Effectively removes the elements from `otherList` and alter the size of **both** containers
	- `list::splice(const_iterator position, list& otherList)`
		- `position`: Position within the container where the elements of `otherList` are inserted
		- `otherList`: A list of object of the same type
	- `list::splice(const_iterator position, list& otherList, const_iterator first, const_iterator last)`
		- `position`: Position within the container where the elements of `otherList` are inserted
		- `otherList`: A list of object of the same type
		- `first` and `last`: range of elements in `otherList`
- `list::merge(list& otherList)`: remove all elements from `otherList` and insert them to the container in **ascending order**
[^1]: The ability to retrieve or modify any element in a [[Computer Science/Data Structures/index|Data Structure]] directly, **without needing to traverse other elements**
