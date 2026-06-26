> [!DANGER]
> 
> Never return the address of a local variable.
> Local variables (including arrays) live on the Stack. When a function returns, its "stack frame" is reclaimed by the **OS**, meaning that memory is now available for the next function call to overwrite.

---
### 1. The "Dangling Pointer" Problem
In your `concat` function, `result` is a local array. It is stored in the stack space allocated specifically for that call to `concat`.
- **Step 1**: `concat(like, c1)` runs. It stores "I like apples" at address `0x7ff...8e0`.
- **Step 2**: The function returns that address to `result1`. However, the `concat` stack frame is now **invalid**.
- **Step 3**: `concat(like, c2)` runs. Because it is the same function, the OS reuses the **exact same stack space**. It overwrites the old data with "I like CSE29".
- **Step 4**: `result2` now points to the same address as `result1`.

---
### 2. Memory Stack Organization Breakdown
Referencing your previous image of stack organization, we can see how the registers interact here.
- **SP (Stack Pointer)**: When `concat` is called, the SP moves to create space for `result[]`.
- **Return**: When `concat` finishes, the SP moves back up. The data ("I like apples") is still physically there for a moment, but the pointer `result1` is now "dangling" because that memory is no longer protected.
- **Overwrite**: The second call to `concat` moves the SP to the same spot, and the **Instructions** (pointed to by the **PC**) write the new string into that same **Data** area.

---
### 3. Analysis of the `concat` Bug

|**Issue**|**Explanation**|
|---|---|
|**Same Address**|Both calls reused the same stack frame offset, returning `0x7ffc215ae8e0` twice.|
|**Overwritten Value**|Because `result1` and `result2` point to the same memory, they both show the result of the _last_ thing written there: `"I like CSE29"`.|
|**Compiler Warning**|If you `return result;`, the compiler warns you because it knows the `result` array's lifetime ends the moment the function returns.|

---
### 4. Correcting the Implementation
To fix this, the calling function (**main**) should provide the memory, or you must use dynamic memory allocation (Heap).
**Preferred Approach: Caller-Provided Buffer**

```c
// The caller provides the 'result' buffer, similar to our previous 'append' logic
void concat_safe(char a[], char b[], char result[]) {
    int alen = strlen(a), blen = strlen(b);
    for(int i = 0; i < alen; i++) result[i] = a[i];
    for(int i = 0; i < blen; i++) result[i + alen] = b[i];
    result[alen + blen] = '\0';
}
```