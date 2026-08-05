
> [!INFO]
> An expression that returns a value that **can be used as a condition**
- **Unary**: Takes **1** parameter
- **Binary**: Takes **2** parameters
- **Tertiary**: Takes **3** parameters

> [!NOTE]
> [[STL Algorithms]] only uses the **Unary** and **Binary** predicate functions
### Example
`std::sort` has 2 templates
- **Default** → `template <class RandomAccessIterator> void sort(RandomAccessIterator first, RandomAccessIterator last)`
- **Custom** → `template <class RandomAccessIterator, class Compare> void sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp);`
Notice how the **Custom** template of the `std::sort` has an extra `comp` argument? This is the **Binary Function that accepts 2 elements in the range** as arguments, and returns a value convertible to `bool`. Also known as **Predicate Functions**

With the **Default** sort (ascending order)
```cpp
vector<int> v = {3, 1, 2, 4, 7, 5, 6};
sort(v.begin(), v.end()); // 1, 2, 3, 4, 5, 6, 7
```
But what if we want to sort the vector `v` in descending order?
```cpp
bool descending(int a, int b) {return a > b;} // The Predicate Function
sort(v.begin(), v.end(), descending); // 7, 6, 5, 4, 3, 2, 1
```
- For each 2 elements (`a`, `b`) in vector `v`, the sort function will **call the predicate function** → compares the elements and return **true** or **false**
- Depending on the returned Boolean value, `sort()` will rearrange elements in vector
	- `if (a > b) {return true;}` → a comes before b