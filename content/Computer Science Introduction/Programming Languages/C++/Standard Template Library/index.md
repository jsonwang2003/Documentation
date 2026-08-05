
The library of **classes** and **associated functions** that allows programmers to:
- Develop more easily
- Reliable
- Portable
## Containers
> [!INFO]
> Data Structures capable of sorting objects of almost any type.
- Implemented using Class Templates
- Includes:
	- Sequence Containers
		- [[Vectors]]
		- [[Lists]]
	- Associative Containers
		- [[Sets and Multisets]]
		- [[Maps and Multimaps]]
	- Container Adaptors
		- [[Computer Science Introduction/Programming Languages/C++/Standard Template Library/Stack|Stack]]
		- [[Queues|Queues]]
		- [[Priority Queue]]

> [!IMPORTANT]
> Before choosing containers, need to consider the following
> - **How often** and **in which manner** the container need to be accessed
> - What is the **general purpose** of the container
> 	- For fast searches → [[Sets and Multisets]] and [[Maps and Multimaps]]
> 	- For random access → [[Vectors]]

## Algorithms
> [!INFO]
> **Non-member** functions that can perform **common tasks** on the elements in container
> Container **must provide iterators** that define a **half-open interval** → $[\text{first}, \text{last})$
> need to call `#include <algorithm>` to use the functions

More information on STL algorithms [[STL Algorithms|here]]

---
## Iterators
> [!INFO]
> Used to Look through the elements of a container
- Container classes use **iterators** to facilitate cycling through data in container
- a "***generalization***" of [[pointer]], BUT not a **pointer**(abstraction)
- Iterator variable is located on(points to) **one data entry in container**

### Types of Iterators
- Each container has its **own iterator type**
	- Similar to [[pointer]] → has its own pointer type

### Directions of accessing for Iterators
#### Forward
- Can access the *sequence of elements* in a range of a container in the direction that **goes from its beginning toward its end**
#### Bidirectional
- Can access the *sequence of elements* in a range of a container in **both directions (forward and backward)**
- Supported by
	- [[Lists]]
	- [[Set|Sets]]
	- [[Hash Maps (Maps)|Maps]]
#### Random Access Iterators
- Can access the *elements* at an **arbitrary offset position** relative to the **element to which they point**
- Offers same functionality as **pointers**
- Supported by
	- [[Array Lists|Vectors]]

### Member Functions for Iterators
#### From **Container Class** to get the iterator started:
- `ct.begin()` → returns the iterator for the container `ct` that points to **first data item in `ct`**
- `ct.end()` → A **flag** indicating **the end of the container `ct`**
#### Common Operations on Iterators
- Move Iterator 1 position forward / backward from its current position
	- `++iter`
	- `iter++`
	- `--iter`
	- `iter--`
- Assign position of one iterator to another
	- `iter1 = iter2`
- Compare iterators to (in)equality
	- `iter1 != iter2`
	- `iter1 == iter2`
- Dereferencing an offset from the current iterator position
	- `iter[i]` ⇆ `*(iter + i)`
	- `i` = Number of indices to the right of where `iter` is positioned
	- does not move the iterator
- Dereference the iterator
	- `*iter`
	- Return the value of the element pointed by `iter`
- Increment/Decrement the iterator by a certain position
	- `iter += i` or `iter -= i`
	- `i` = number of indices to move
### Const Iterator
- Can declare iterator as **const** to make sure the element that the iterator is pointing to **is not modified**
- use `cbegin()` and `cend()`
```cpp
vector<char>::const_iterator iter = aVector.cbegin();
```
### Loops with Iterator
```cpp
auto iterStart = ct.begin();
auto iterEnd = ct.end();
while(iterStart != iterEnd){
	// Iterator Operation
	++iterStart;
}
```
- **Do Not** use *for loop* (few cases are safe)
- Declare `iterEnd` outside of the loop
	- No need to call `end()` every time going through the loop
- use `auto` keyword for declaring iterator
### Cycling in Reverse Order
- **CANNOT** start from `end()` and decrement
	- recall `end()` gives a **flag**, not the **last element**
- Use these instead:
	- `rbegin()` or `crbegin()` → return the iterator pointing at the last element in the list
	- `rend()` or `crend()` → return a flag, not the first element
	- use `++iter` to move toward the `front of the list`

### Ostream Iterator
- useful for printing data
```cpp
// Container
vector<int> myVector = {1, 2, 3, 4, 5};

// Printing myVector using ostream iterator
ostream_iterator<int> screen(cout, " ");
copy(myVector.begin(), myVector.end(), screen);
```
---
## Summary
- **Forward**: use `begin()` and `end()`
```cpp
vector<char>::iterator iter = aVector.begin();
```
- **Const Forward**: use `cbegin()` and `cend()`
```cpp
vector<char>::const_iterator constIter = aVector.cbegin();
```
- **Reverse**: use `rbegin()` and `rend()`
```cpp
vector<char>::reverse_iterator revIter = aVector.rbegin();
```
- **Const Reverse**: use `crbegin()` and `crend()`
```cpp
vector<char>::const_reverse_iterator constRevIter = aVector.crbegin();
```
