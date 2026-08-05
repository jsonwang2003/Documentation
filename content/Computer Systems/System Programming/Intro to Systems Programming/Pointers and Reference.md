> [!ABSTRACT]
> 
> This section covers the relationship between pointers and arrays. We explore why passing arrays in C requires explicit length parameters and how to perform manual memory operations like appending.

---
### 1. Pointers vs. Arrays
In C, when you pass an array to a function, you are actually passing a **pointer** to its first element.
- **`T*`**: Represents an address where we can access data of type `T`.
- **The "Size" Problem**: Unlike C strings, standard arrays are **not null-terminated**. A function receiving `double a[]` has no way of knowing where the array ends unless we tell it.

---
### 2. Implementation: The `append` Pattern
To safely manipulate arrays, our function signatures must include lengths for every array passed.
#### Revised Function Signature

```c
// Takes two double arrays a and b, and changes result to have the concatenated data.
// ASSUMES that result has at least (alen + blen) elements of space.
void append(double a[], int alen, double b[], int blen, double result[]);
```
#### The Logic
We must manually "offset" the second array into the `result` buffer by using the length of the first array.

```c
void append(double a[], int alen, double b[], int blen, double result[]) {
    // Copy first array starting at index 0
    for(int i = 0; i < alen; i++) {
        result[i] = a[i];
    }
    
    // Copy second array starting at index 'alen'
    for(int i = 0; i < blen; i++) {
        result[i + alen] = b[i];
    }
}
```

---
### 3. Usage Example
Notice how we must track `3` and `4` manually in the `main` function.

```c
int main() {
    double n1[] = {4.0, 5.0, 6.0};      // Length: 3
    double n2[] = {0.7, 0.3, 0.1, 0.8}; // Length: 4
    double result[7];                   // 3 + 4 = 7
    
    append(n1, 3, n2, 4, result);
    
    print_list(n1, 3);
    print_list(n2, 4);
    print_list(result, 7);
}
```

---
### 4. Comparison: Strings vs. Arrays

|**Feature**|**C Strings (char*)**|**Numeric Arrays (double[], int[])**|
|---|---|---|
|**Termination**|**Null-terminated** (`\0`)|**No built-in terminator**|
|**Size Tracking**|Functions like `strlen()` work|Length must be passed as an `int`|
|**Memory**|Contiguous in memory|Contiguous in memory|
![[fef28e71-9235-4716-bb24-9a650ecf8cbc.png]]

---
### Key Takeaway for Systems Programming
When working with raw memory or system calls, **always pass the size**. Relying on "assumptions" about array boundaries is a primary cause of **buffer overflows** and security vulnerabilities.