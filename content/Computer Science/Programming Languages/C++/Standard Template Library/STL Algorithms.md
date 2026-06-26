## Functions
### Count
- `typename iterator_traits<InputIterator>::difference_type count(InputIterator first, InputIterator last, const T& val)`: Count occurrences of `val` in range $[\text{first}, \text{last})$
	- `first` and `last`: the range of the list
	- `val`: the value to search for
	- Returns number of elements in `range` that equals to `value`

> [!IMPORTANT]
> `iterator_traits<InputIterator>::difference_type` is a **type alias** that represents the **difference between 2 iterators** → result type of expressions like `last - first`
> Extracts the `difference_type` from the iterator type `InputIterator`, using the `iterator_traits` template
### Sort
- `void sort(RandomAccessIterator first, RandomAccessIterator last)`: Sorts a container
	- [[Lists]] has its own sorting function since **lists is implemented with [[Doubly Linked List (DLL)]]** → no random access capability which the STL sort function uses
### Remove
- `template <class ForwardIterator, class T> ForwardIterator remove(ForwardIterator first, ForwardIterator last, const T& val);`
	- `first` and `last`: range to search through in the container
	- `val`: the value to search for in the container
	- Returns an iterator to the element that **follows the last element not removed**

> [!IMPORTANT]
> The function **cannot alter the properties of the object** containing the range of elements. The removal is done by **replacing the elements that compare equal to `val`** by the **next element that does not** and signaling the **new size of the shortened range**

### For Each
- `template <class InputIterator, class Function> Function for_each(InputIterator first, InputIterator last, Function fn)`: Applies the function `fn` to range $[\text{first}, \text{last})$
	- `first` and `last`: Range to apply through in the container
	- `fn`: A **Unary** function accepts as argument an element in range
---
## Understanding Algorithm Description
#### Need of Overloading Operators that STL Uses
Given an example class `Student`
```cpp
class Student{
public:
	Student(): id(0), grade('X'){}
	Student(int newId, char newGrade)
		: id(newId), grade(newGrade){}
	... // other functions
	~Student(){};
private:
	int id;
	char grade;
}
```
Assume we create a vector of students
```cpp
Student bob(111, 'A'),
		jane(222, 'B'),
		jill(333, 'A');
vector<Student> students = {bob, jane, jill};
```
If we are using STL algorithm `std::count` on `students` to count all A students in the group
```cpp
cout << count(students.begin(), students.end(), 'A') << endl;
```
We will get an error
```bash
binary '==' : 'Student' does not define ...
```

This is due to the algorithm uses **certain operators to develop the algorithm**, which in a user-defined class objects may not have incorporated them. To fix this problem, we can do the following:
- If we have access to the class `Student`
```cpp
// create a overloaded operator for STL Algorithms to use
bool Student::operator==(char aGrade){
	// Implement the == operator
}
```
- If we don't have access to the class `Student`
```cpp
bool operator==(const Student& aStudent, char aGrade){
	// Implement the == operator
}
```
#### Return Values
- Some functions return values we **don't necessarily need to use**
- Example: [[#Remove]]
	- 2 choices to deal with the returned value
		1. Ignore it → `remove(list.being(), list.end(), val)`
		2. Create a variable to store the return value → `auto iter = remove(list.begin(), list.end(), val)`

---
## Summary
### Efficiency
When using STL functions, no need to worry about **calling functions in parameter list**

| Efficient                                    | Inefficient                        |
| -------------------------------------------- | ---------------------------------- |
| `remove(aVector.begin(), aVector.end(), 20)` | `auto iterBegin = aVector.begin()` |
### Improve Readability
- **No Abbreviation** for identifiers for [[Computer Science/Programming Languages/C++/Standard Template Library/index#Containers|STL Containers]]
- If using for **Multiple** types of [[Computer Science/Programming Languages/C++/Standard Template Library/index#Containers|STL Containers]] → choose identifiers **clearly denote** the type of iterator

### Pointer, Reference, and Iterator Invalidation
- Operations that **add or remove** elements from a container can **invalidate** pointers, reference, and iterators
- Always keep in mind that **iterator might not be valid** any longer after **using it**
- Check **narrative of function** you are using

--- 
## Relevant Information
- [[Predicate Functions]]