
> [!INFO]
> A sequence container that organizes data sequentially by index
> In other programming languages, mostly called Array
- Organized in **the order that they are added**
- Element is retrieved **using an index** that specifies its place in the sequence
- Implemented as a [[Dynamic Array]]
- Allows **random access**[^1]

---
## Constructors
#### Default Constructor
```cpp
vector<int> vector1; // construct empty vector
vector1.reserve(5); // reserve 5 element space in vector
vector1.push_back(10); // add the value 10 into vector1
```
#### Copy Constructor
```cpp
vector<int> vector1 = {1, 2, 3};
vector<int> vector2(vector1); // preferred when want to explicit about constructor call
vector<int> vector3 = vector1; // also calls the copy constructor
```
#### Initializer list
```cpp
// both of the following syntax is valid
vector<int> vector1{3, 4, 5, 6, 7};
vector<int> vector2 = {3, 5, 7, 9};
```
#### Range Constructor
```cpp
// creates a new vector that has all elements from index2 to the end of the vector in vector1
auto iter = vector1.begin() + 2;
vector<int> vector2(iter, vector1.end());

int arr[] = {10, 11, 12, 13};
vector<int> vector3(arr, arr + 3); // contains {10, 11, 12};
```

---
## Size, Capacity, and Efficiency Issues
- Vector has **capacity**(correspond to **how many memory is allocated**)
- **Size**: denotes the **number of elements**
- Vectors grow **automatically**
- `vector::push_back()` can be inefficient
	- If **no more space to fit new element** → **new dynamic array** formed with **larger capacity** → all elements are **copied** into the new dynamic array
- Capacity of vector should be **set** using default constructor
```cpp
vector<int> vector1;
vector1.reserve(10);
```

---
## Inserting and Replacing Element
#### Single Element
```cpp
iterator insert(const_iterator position, const value_type& val);
```
#### N Elements of the Same Value
```cpp
iterator insert(const_iterator position, size_t n, const value_type& val);
```
#### Initializer List
```cpp
iterator insert(const_iterator position, initializer_list<value_type> list);
```

---
## Initialize Vector with 0's
#### Overloaded Constructor
```cpp
vector<int> vector1(8); // 0, 0, 0, 0, 0, 0, 0, 0
```
#### Resize Member Function
```cpp
vector1.resize(10); // 1, 2, 3, 4, 5, 6, 7, 0, 0, 0
```

---
## Member Functions
- `vector::push_back(const T& val)`: Add Element `val` at the end
- `vector::front()`: Access first element
- `vector::back()`: Access last element
- `vector::erase(iterator position)`: Erase element at position
- `vector::insert(iterator position, const T& value)`: Insert value at position
- `vector::pop_back()`: delete last element


[^1]: The ability to retrieve or modify any element in a [[Computer Science Introduction/Data Structures/index|Data Structure]] directly, **without needing to traverse other elements**
