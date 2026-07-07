---
tags:
  - Arrays
  - ArrayLists
---
## Array

> [!INFO] Definition 
> An **array** is a _homogeneous_ data structure where each element is stored in **adjacent memory locations**.
> - **homogeneous**: all elements are of the same type (int, string, etc.)
>     

### Properties of an Array

#### Random Access

Since each cell has the same type (and thus the same size), and the cells are adjacent in memory, it is possible to quickly calculate the address of any array cell given the address of the first cell.

> [!Example] 
> Suppose we allocated an array of n elements, where the elements are of a type that has a size of b bytes, starting at memory address x. Using 0-indexing:
> 
> - The element at index i=0 is at memory address $x$
> - The element at index i=1 is at memory address $x+b$
> - The element at index i is at memory address $x+bi$

This phenomenon allows finding the memory address of any $i^{th}$ element in **constant time**, enabling O(1) access.

> [!Question] Exercise 
> You create an array of integers (assume each integer is exactly 4 bytes) in memory, and the beginning of the array happens to be at memory address 1000 (in decimal). What is the memory address of the start of cell 6 of the array, assuming 0-based indexing?
> 
> > [!Tip]- Answer 
> > $x+bi=1000+(4)(6)=1024​$

---

## Array List (Vector)

> [!INFO] Definition 
> An **Array List** is an array where no empty cells are present between elements. Users can only add elements to indices between 0 to n (inclusive), where n is the number of total elements.

### Dynamic Implementation

When the number of elements is unknown, the system:
1. Allocates a default "large" amount of memory initially.
2. Inserts elements into this initial array.
3. Once the array is full, creates a new larger array and copies the elements over.
4. Replaces the old array reference with the new reference.

> [!Question] Array structures require all elements be the same size. How can they contain strings of varying lengths?
> 
> > [!Tip]- Answer 
> > Strings are stored as **pointers** (fixed size) that indicate the memory location containing the actual string data.

### Inserting
- **To the Back**: If the current count is known → **Constant Time O(1)**.
- **To the Front**: Requires moving all existing elements to create space → **Linear Time O(n)**.

> [!Abstract] Summary **Best Case**: O(1) (appending at the end) **Worst Case**: O(n) (insertion at front or exceeding capacity)

```python
# inserts element into array and returns True on success
insert(element, index):                  
    if index < 0 or index > n:           # invalid indices 
        return False 
        
    if n == array.length:                # if array is full 
        newArray = empty array of length 2*array.length 
        for i from 0 to n-1:             # copy all elements
            newArray[i] = array[i] 
        array = newArray                 
        
    if index == n:                       # insertion at end
        array[index] = element           
    else:                                # general insertion 
        for i from n-1 to index:         # make space
            array[i+1] = array[i] 
        array[index] = element           
        
    n = n+1                              
    return True
```

### Searching

```python
# returns True if element in array, otherwise False
find(element):                       
    for i from 0 to n-1:               # iterate through all n elements
        if array[i] == element:        # match found
            return True
    return False                       # checked all n elements
```

> [!Abstract] Summary **Best Case**: O(1) (element at the first index) **Worst Case**: O(n) (element at the last index or not present)
#### Binary Search
If the array is sorted, searching runtime improves to **O(logn)** via [[Computer Science/Discrete Structures/Discrete Algorithms/Searching/Binary Search]].
### Deletion
- **Last Element**: Simply remove it → **Constant Time O(1)**.
- **First Element**: Requires moving all subsequent elements left one spot → **Linear Time O(n)**.

> [!Question] Can we avoid move operations when removing from the very beginning?
> 
> > [!Tip]- Answer 
> > Yes, by using a [[Circular Array]], which has no fixed beginning or end.

```python
# removes element at position "index"
remove(index): 
    if index < 0 or index >= n:
        return False
        
    remove(array[index])
    if index < n - 1:
        for i from index to n - 1:
            array[i] = array[i + 1]
        remove(array[n - 1])
    
    n = n - 1
    return True
```

---
## Final Summary: Trade-offs

|Feature|Performance / Detail|
|---|---|
|**Random Access**|O(1) — Extremely fast lookup by index|
|**Search (Sorted)**|O(logn) — Efficient with Binary Search|
|**Search (Unsorted)**|O(n) — Slow for large datasets|
|**Insert/Delete**|O(n) — Costly due to shifting (except at the end)|
|**Memory**|Potential waste if allocated space is not fully used|

- **Best Use Case**: Fixed-size data or scenarios where frequent access by index is required.
- **Weakest Use Case**: Frequent insertions/deletions at the beginning or middle of the list.