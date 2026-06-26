> [!ABSTRACT]
> 
> Unlike Fixed-Sized Structs, which have a rigid memory footprint determined at compile-time, Dynamic Structs use the Heap to store data of varying sizes. This allows structs to persist beyond the function that created them and handle data that changes in size during runtime.

### 1. The Power of `malloc`
The `malloc(size_t size)` function is the key to dynamic memory.
- **Heap Allocation**: It returns a pointer to newly allocated memory on the **Heap**.
- **Persistence**: This memory **does not go away when the function returns**. This solves the "Dangling Pointer" problem encountered when returning addresses of stack-allocated local variables.

![[Pasted image 20251029104600.png]]

### 2. Implementation: From Fixed to Dynamic
In your provided example, we move from a rigid structure to one that can accommodate any number of doubles by storing a **pointer** to the data rather than the data itself.

**Refining the Struct Definition:**

```c
typedef struct ListOfDoubles {
    int size;
    double* contents; // Changed from fixed array to a pointer for heap data
} ListOfDoubles;
```

**Revised `concat` Logic:**
When concatenating two lists, we calculate the total bytes needed and request that specific amount from the heap.

```c
ListOfDoubles concat(ListOfDoubles d1, ListOfDoubles d2){
    ListOfDoubles result;
    result.size = d1.size + d2.size;
    
    // Allocate exactly enough space for the combined doubles on the heap
    result.contents = malloc(result.size * sizeof(double));
    
    for(int i = 0; i < d1.size; i++){
        result.contents[i] = d1.contents[i];
    }
    for(int i = 0; i < d2.size; i++){
        result.contents[i + d1.size] = d2.contents[i];
    }
    return result; // The struct is returned by value, but it carries a pointer to heap data
}
```

---
### 3. Memory Lifecycle Comparison
Understanding where your struct data lives is critical for avoiding memory leaks and corruption.

|**Feature**|**Fixed-Sized Struct**|**Dynamic Struct (Heap-based)**|
|---|---|---|
|**Data Location**|Stack (inside the struct)|Heap (pointed to by the struct)|
|**Lifetime**|Ends when function returns|Lasts until manually `free()`-d|
|**Flexibility**|Limited by `MAX_LIST_SIZE`|Limited only by available system RAM|
|**Safety**|Safer for small, local data|Risk of **Memory Leaks** if not freed|

---
### 4. Key Takeaways
- **Pointer as a Bridge**: The struct itself remains small (storing an `int` and a `char*`), but it acts as a handle for a much larger, dynamic block of memory on the heap.
- **Manual Management**: Because `malloc` memory persists, every `malloc` must eventually be matched with a `free` to prevent **Memory Leaks** (covered in [[Memory Leak]]).