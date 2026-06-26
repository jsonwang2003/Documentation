The `sizeof` operator is a fundamental tool in C for determining the memory footprint of data types and variables. Understanding its behavior is critical for manual memory management and ensuring your programs interact correctly with the operating system's address space.
### 1. Memory Footprints of Standard Types
The `sizeof` operator returns the number of bytes required to store a type or variable in memory.

|**Input Type**|**Size (Bytes)**|**Description**|
|---|---|---|
|`uint8_t`|**1**|An 8-bit unsigned integer (exactly 1 byte).|
|`uint32_t`|**4**|A 32-bit unsigned integer (exactly 4 bytes).|
|`char*`|**8**|A pointer to a character (memory address).|
|`uint32_t*`|**8**|A pointer to a 32-bit integer (memory address).|

---
### 2. The "Pointer" Rule
Notice that both `char*` and `uint32_t*` return **8 bytes**.

In a 64-bit environment (like the `ieng6` servers), **all pointers are the same size** regardless of what data type they point to. This is because a pointer is simply an address within the process's **Address Space**. Whether you are pointing to a single byte (`char`) or a large structure, the address itself always requires 64 bits (8 bytes) of storage.

---
### 3. Usage with Arrays vs. Pointers
One common trap in C is using `sizeof` on array parameters in functions.
- **Local Arrays**: If you declare `double result[7]` in `main`, `sizeof(result)` will return `56` (7 elements × 8 bytes per double).
- **Function Parameters**: If you pass that array to a function `void func(double arr[])`, `sizeof(arr)` inside the function will return **8**.

This happens because arrays "decay" into pointers when passed to functions. This is why your `append()` and `concat()` functions required an explicit `alen` or `blen` parameter—the function cannot use `sizeof` to figure out how many elements are in the array.

---
### 4. Connection to System Architecture
The sizes returned by `sizeof` are determined by the architecture of the machine and the Operating System.
- **Instructions (Code)**: The size of individual instructions is fixed by the CPU architecture.
- **Stack/Data Segments**: When the OS allocates these sections, it uses the byte sizes defined by these types to ensure variables do not overlap in the **Address Space**.